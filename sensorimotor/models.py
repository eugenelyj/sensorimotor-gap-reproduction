from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from sensorimotor.simulator import Observation


class RunningMeanVariance(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1, *self.mean.shape)
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        updated_mean = self.mean + delta * batch_count / total_count
        previous_moment = self.variance * self.count
        batch_moment = batch_variance * batch_count
        correction = delta.square() * self.count * batch_count / total_count
        self.mean.copy_(updated_mean)
        self.variance.copy_((previous_moment + batch_moment + correction) / total_count)
        self.count.copy_(total_count)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return ((values - self.mean) / torch.sqrt(self.variance + 1e-8)).clamp(-10.0, 10.0)


class ObservationNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.points = RunningMeanVariance((3,))
        self.state_action = RunningMeanVariance((10,))

    @torch.no_grad()
    def update(self, observation: Observation) -> None:
        self.points.update(observation.gap_points)
        self.state_action.update(observation.state_action)

    def normalize(self, observation: Observation) -> Observation:
        return Observation(
            gap_points=self.points.normalize(observation.gap_points),
            state_action=self.state_action.normalize(observation.state_action),
        )


def initialize_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class PointEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_layers = nn.Sequential(
            initialize_layer(nn.Linear(3, 128), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            initialize_layer(nn.Linear(128, 64), gain=nn.init.calculate_gain("relu")),
        )
        self.output = initialize_layer(nn.Linear(64, 32), gain=1.0)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_layers(points)
        features = features.max(dim=-2).values
        return self.output(features)


class TeacherActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = PointEncoder()
        self.policy = nn.Sequential(
            initialize_layer(nn.Linear(42, 256), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            initialize_layer(nn.Linear(256, 128), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            initialize_layer(nn.Linear(128, 4), gain=0.01),
        )
        hover_normalized = (9.81 - 13.0) / 7.0
        self.policy[-1].bias.data[0] = torch.atanh(torch.tensor(hover_normalized))
        self.log_standard_deviation = nn.Parameter(torch.full((4,), -0.5))

    def distribution(self, points: torch.Tensor, state_action: torch.Tensor) -> Normal:
        feature = torch.cat((self.point_encoder(points), state_action), dim=-1)
        mean = self.policy(feature)
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        return Normal(mean, standard_deviation)


class TeacherCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = PointEncoder()
        self.value = nn.Sequential(
            initialize_layer(nn.Linear(42, 256), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            initialize_layer(nn.Linear(256, 128), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            initialize_layer(nn.Linear(128, 1), gain=1.0),
        )

    def forward(self, points: torch.Tensor, state_action: torch.Tensor) -> torch.Tensor:
        feature = torch.cat((self.point_encoder(points), state_action), dim=-1)
        return self.value(feature).squeeze(-1)


@dataclass
class ActionOutput:
    action: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class TeacherActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = TeacherActor()
        self.critic = TeacherCritic()

    def act(self, observation: Observation, deterministic: bool = False) -> ActionOutput:
        distribution = self.actor.distribution(observation.gap_points, observation.state_action)
        action = distribution.mean if deterministic else distribution.sample()
        return ActionOutput(
            action=action,
            log_probability=distribution.log_prob(action).sum(dim=-1),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.critic(observation.gap_points, observation.state_action),
        )

    def evaluate_actions(
        self,
        points: torch.Tensor,
        state_action: torch.Tensor,
        actions: torch.Tensor,
    ) -> ActionOutput:
        distribution = self.actor.distribution(points, state_action)
        return ActionOutput(
            action=actions,
            log_probability=distribution.log_prob(actions).sum(dim=-1),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.critic(points, state_action),
        )


class StudentPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 48, kernel_size=6, stride=4, padding=1),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        )
        self.image_projection = nn.Linear(64 * 4 * 5, 128)
        self.input_projection = nn.Linear(134, 134)
        self.recurrent = nn.GRU(input_size=134, hidden_size=96, num_layers=1)
        self.output_projection = nn.Sequential(nn.Linear(96, 96), nn.ELU())
        self.head = nn.Linear(102, 5)
        hover_normalized = (9.81 - 13.0) / 7.0
        self.head.bias.data[0] = torch.atanh(torch.tensor(hover_normalized))
        self.head.bias.data[4] = -1.0

    def forward(
        self,
        image: torch.Tensor,
        attitude_previous_action: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_feature = self.image_encoder(image).flatten(start_dim=1)
        image_feature = self.image_projection(image_feature)
        recurrent_input = torch.cat((image_feature, attitude_previous_action), dim=-1)
        recurrent_input = self.input_projection(recurrent_input).unsqueeze(0)
        recurrent_output, hidden_state = self.recurrent(recurrent_input, hidden_state)
        output_feature = self.output_projection(recurrent_output.squeeze(0))
        output = self.head(torch.cat((output_feature, attitude_previous_action), dim=-1))
        return output, hidden_state

    def forward_sequence(
        self,
        images: torch.Tensor,
        attitude_previous_action: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps, batch_size = images.shape[:2]
        flat_images = images.flatten(0, 1)
        image_feature = self.image_encoder(flat_images).flatten(start_dim=1)
        image_feature = self.image_projection(image_feature).reshape(steps, batch_size, -1)
        recurrent_input = torch.cat((image_feature, attitude_previous_action), dim=-1)
        recurrent_input = self.input_projection(recurrent_input)
        if episode_starts is None:
            recurrent_output, hidden_state = self.recurrent(recurrent_input, hidden_state)
        else:
            if hidden_state is None:
                hidden_state = torch.zeros(
                    (1, batch_size, self.recurrent.hidden_size),
                    device=images.device,
                    dtype=recurrent_input.dtype,
                )
            recurrent_steps = []
            for step in range(steps):
                hidden_state = hidden_state * (~episode_starts[step])[None, :, None]
                recurrent_step, hidden_state = self.recurrent(
                    recurrent_input[step : step + 1], hidden_state
                )
                recurrent_steps.append(recurrent_step)
            recurrent_output = torch.cat(recurrent_steps, dim=0)
        output_feature = self.output_projection(recurrent_output)
        output = self.head(torch.cat((output_feature, attitude_previous_action), dim=-1))
        return output, hidden_state
