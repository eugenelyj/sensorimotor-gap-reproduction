from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from sensorimotor.config import PPOConfig
from sensorimotor.models import TeacherActorCritic


@dataclass
class RolloutBatch:
    points: torch.Tensor
    state_action: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutStorage:
    def __init__(self, steps: int, num_envs: int, device: torch.device) -> None:
        self.steps = steps
        self.num_envs = num_envs
        self.device = device
        self.points = torch.empty((steps, num_envs, 32, 3), device=device)
        self.state_action = torch.empty((steps, num_envs, 10), device=device)
        self.actions = torch.empty((steps, num_envs, 4), device=device)
        self.log_probabilities = torch.empty((steps, num_envs), device=device)
        self.values = torch.empty((steps, num_envs), device=device)
        self.rewards = torch.empty((steps, num_envs), device=device)
        self.dones = torch.empty((steps, num_envs), device=device, dtype=torch.bool)
        self.advantages = torch.empty((steps, num_envs), device=device)
        self.returns = torch.empty((steps, num_envs), device=device)

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_advantage = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.steps)):
            if step == self.steps - 1:
                following_value = next_value
            else:
                following_value = self.values[step + 1]
            not_done = (~self.dones[step]).float()
            delta = self.rewards[step] + gamma * following_value * not_done - self.values[step]
            last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
            self.advantages[step] = last_advantage
        self.returns.copy_(self.advantages + self.values)

    def flattened(self) -> RolloutBatch:
        return RolloutBatch(
            points=self.points.flatten(0, 1),
            state_action=self.state_action.flatten(0, 1),
            actions=self.actions.flatten(0, 1),
            old_log_probabilities=self.log_probabilities.flatten(0, 1),
            old_values=self.values.flatten(0, 1),
            returns=self.returns.flatten(0, 1),
            advantages=self.advantages.flatten(0, 1),
        )


@dataclass
class PPOStatistics:
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    epochs: int


class PPOOptimizer:
    def __init__(self, model: TeacherActorCritic, config: PPOConfig) -> None:
        self.model = model
        self.config = config
        self.actor_optimizer = torch.optim.Adam(
            model.actor.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            model.critic.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )

    def update(self, storage: RolloutStorage) -> PPOStatistics:
        batch = storage.flattened()
        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std(unbiased=False) + 1e-8
        )
        batch_size = batch.actions.shape[0]
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        completed_epochs = 0
        actor_state_before_update = {
            name: value.detach().clone() for name, value in self.model.actor.state_dict().items()
        }
        actor_optimizer_state_before_update = deepcopy(self.actor_optimizer.state_dict())
        hard_kl_limit = max(4.0 * self.config.target_kl, 0.1)
        rolled_back = False

        for epoch in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=batch.actions.device)
            stop_early = False
            for start in range(0, batch_size, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                output = self.model.evaluate_actions(
                    batch.points[indices],
                    batch.state_action[indices],
                    batch.actions[indices],
                )
                log_ratio = output.log_probability - batch.old_log_probabilities[indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[indices]
                unclipped_loss = -minibatch_advantages * ratio
                clipped_loss = -minibatch_advantages * ratio.clamp(
                    1.0 - self.config.clip_coefficient,
                    1.0 + self.config.clip_coefficient,
                )
                actor_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
                entropy = output.entropy.mean()
                total_actor_loss = actor_loss - self.config.entropy_coefficient * entropy
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_coefficient
                    ).float().mean()
                if approximate_kl > hard_kl_limit:
                    self.model.actor.load_state_dict(actor_state_before_update)
                    self.actor_optimizer.load_state_dict(actor_optimizer_state_before_update)
                    rolled_back = True
                    stop_early = True
                    break
                if approximate_kl > self.config.target_kl:
                    stop_early = True
                    break

                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                if self.config.minimum_action_log_std is not None:
                    with torch.no_grad():
                        self.model.actor.log_standard_deviation.clamp_(
                            min=self.config.minimum_action_log_std
                        )

                value = self.model.critic(batch.points[indices], batch.state_action[indices])
                critic_loss = 0.5 * torch.square(value - batch.returns[indices]).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coefficient * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                approximate_kls.append(approximate_kl.item())
                clip_fractions.append(clip_fraction.item())
            completed_epochs = epoch + 1
            if stop_early:
                break

        if rolled_back:
            approximate_kls.append(approximate_kl.item())
            clip_fractions.append(clip_fraction.item())

        return PPOStatistics(
            actor_loss=sum(actor_losses) / len(actor_losses),
            critic_loss=sum(critic_losses) / len(critic_losses),
            entropy=sum(entropies) / len(entropies),
            approximate_kl=sum(approximate_kls) / len(approximate_kls),
            clip_fraction=sum(clip_fractions) / len(clip_fractions),
            epochs=completed_epochs,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for parameter_group in self.actor_optimizer.param_groups:
            parameter_group["lr"] = self.config.actor_learning_rate
        for parameter_group in self.critic_optimizer.param_groups:
            parameter_group["lr"] = self.config.critic_learning_rate
