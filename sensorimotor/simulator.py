from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sensorimotor.config import EnvironmentConfig
from sensorimotor.geometry import (
    gap_intersection_status,
    gap_rotation,
    make_rectangular_gap_points,
)
from sensorimotor.math import (
    matrix_to_roll_pitch_yaw,
    quaternion_from_euler,
    quaternion_integrate,
    quaternion_to_matrix,
    wrap_angle,
)


@dataclass
class Observation:
    gap_points: torch.Tensor
    state_action: torch.Tensor


class VectorizedGapEnv:
    def __init__(
        self,
        config: EnvironmentConfig,
        device: str | torch.device,
        seed: int = 1,
        auto_reset: bool = True,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.num_envs = config.num_envs
        self.auto_reset = auto_reset
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self.collider_half_size = torch.tensor(config.collider_size, device=self.device) * 0.5
        self.nominal_drag_linear = torch.tensor(config.air_drag_linear, device=self.device)
        self.nominal_drag_quadratic = torch.tensor(config.air_drag_quadratic, device=self.device)
        self.perturbation_max = torch.tensor(config.perturbation_max_accel, device=self.device)
        self.base_delay = torch.tensor(config.action_delay_steps, device=self.device, dtype=torch.long)
        self.response_scale_range = torch.tensor(
            config.response_scale_randomization, device=self.device
        )
        self.gap_points_local = make_rectangular_gap_points(
            config.gap_width,
            config.gap_height,
            points_per_edge=8,
            device=self.device,
        )
        self.history_length = int(self.base_delay.max().item()) + config.action_average_window + 3
        self.informed_trajectories: dict[str, torch.Tensor] | None = None
        if config.informed_trajectory_path:
            trajectory_data = torch.load(
                config.informed_trajectory_path,
                map_location=self.device,
                weights_only=False,
            )
            if not trajectory_data["successes"].all() or trajectory_data["collisions"].any():
                raise ValueError("The informed-reset trajectory library contains invalid paths")
            self.informed_trajectories = {
                key: value.to(self.device)
                for key, value in trajectory_data.items()
                if isinstance(value, torch.Tensor)
            }

        count = self.num_envs
        self.position = torch.zeros((count, 3), device=self.device)
        self.velocity = torch.zeros((count, 3), device=self.device)
        self.quaternion = torch.zeros((count, 4), device=self.device)
        self.quaternion[:, 0] = 1.0
        self.gap_center = torch.zeros((count, 3), device=self.device)
        self.gap_roll = torch.zeros(count, device=self.device)
        self.gap_rotation_matrix = torch.eye(3, device=self.device).expand(count, -1, -1).clone()
        self.previous_action = torch.zeros((count, 4), device=self.device)
        self.action_response = torch.zeros((count, 4), device=self.device)
        self.action_history = torch.zeros((count, self.history_length, 4), device=self.device)
        self.response_delay = torch.zeros((count, 4), device=self.device, dtype=torch.long)
        self.response_scale = torch.ones((count, 4), device=self.device)
        self.response_scale_steps = torch.zeros(count, device=self.device, dtype=torch.long)
        self.drag_linear = torch.zeros((count, 3), device=self.device)
        self.drag_quadratic = torch.zeros((count, 3), device=self.device)
        self.perturbation = torch.zeros((count, 3), device=self.device)
        self.perturbation_steps = torch.zeros(count, device=self.device, dtype=torch.long)
        self.episode_step = torch.zeros(count, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(count, device=self.device)
        self.episode_length = torch.zeros(count, device=self.device, dtype=torch.long)
        self.had_collision = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.had_out_of_bounds = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.informed_reset = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.camera_fov_scale = torch.ones((count, 2), device=self.device)
        self.reset()

    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(
            shape, device=self.device, generator=self.generator
        )

    def _normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=self.device, generator=self.generator)

    def _physical_to_normalized_action(self, action: torch.Tensor) -> torch.Tensor:
        thrust_center = 0.5 * (self.config.thrust_min + self.config.thrust_max)
        thrust_half_range = 0.5 * (self.config.thrust_max - self.config.thrust_min)
        normalized = action.clone()
        normalized[:, 0] = (action[:, 0] - thrust_center) / thrust_half_range
        normalized[:, 1:] = action[:, 1:] / self.config.bodyrate_max
        return normalized

    def action_to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(raw_action)
        thrust_center = 0.5 * (self.config.thrust_min + self.config.thrust_max)
        thrust_half_range = 0.5 * (self.config.thrust_max - self.config.thrust_min)
        physical = bounded.clone()
        physical[:, 0] = thrust_center + thrust_half_range * bounded[:, 0]
        physical[:, 1:] = self.config.bodyrate_max * bounded[:, 1:]
        return physical

    def reset(self, indices: torch.Tensor | None = None) -> Observation:
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        if indices.numel() == 0:
            return self.observe()
        count = indices.numel()

        gap_roll = self._uniform(
            (count,),
            torch.deg2rad(torch.tensor(self.config.gap_roll_min_deg)).item(),
            torch.deg2rad(torch.tensor(self.config.gap_roll_max_deg)).item(),
        )
        gap_center = torch.zeros((count, 3), device=self.device)
        gap_center[:, 1] = self._uniform((count,), -0.25, 0.25)
        gap_center[:, 2] = self._uniform((count,), 1.35, 1.65)
        self.gap_roll[indices] = gap_roll
        self.gap_center[indices] = gap_center
        self.gap_rotation_matrix[indices] = gap_rotation(gap_roll)

        start = gap_center.clone()
        start[:, 0] = self._uniform(
            (count,), self.config.start_x_range[0], self.config.start_x_range[1]
        )
        start[:, 1] += self._uniform(
            (count,), -self.config.start_lateral_offset, self.config.start_lateral_offset
        )
        start[:, 2] += self._uniform(
            (count,), -self.config.start_vertical_offset, self.config.start_vertical_offset
        )
        yaw = torch.atan2(gap_center[:, 1] - start[:, 1], -start[:, 0])
        yaw += self._uniform((count,), -0.12, 0.12)
        roll = self._uniform((count,), -0.04, 0.04)
        pitch = self._uniform((count,), -0.04, 0.04)
        velocity = torch.zeros((count, 3), device=self.device)
        planned_body_rate = torch.zeros((count, 3), device=self.device)
        library_action = torch.zeros((count, 4), device=self.device)
        library_informed = torch.zeros(count, dtype=torch.bool, device=self.device)

        informed = torch.rand((count,), device=self.device, generator=self.generator)
        informed = informed < self.config.informed_reset_probability
        self.informed_reset[indices] = informed
        if informed.any():
            local_indices = torch.nonzero(informed, as_tuple=False).squeeze(-1)
            if self.informed_trajectories is None:
                phase = self._uniform((local_indices.numel(),), 0.06, 0.82)
                smooth_phase = phase * phase * (3.0 - 2.0 * phase)
                informed_start = start[local_indices]
                target = gap_center[local_indices].clone()
                target[:, 0] = 0.35
                displacement = target - informed_start
                lateral_phase = 1.0 - torch.square(1.0 - smooth_phase)
                start[local_indices, 0] = (
                    informed_start[:, 0] + displacement[:, 0] * smooth_phase
                )
                start[local_indices, 1:] = (
                    informed_start[:, 1:] + displacement[:, 1:] * lateral_phase[:, None]
                )
                derivative = 6.0 * phase * (1.0 - phase) / 2.2
                velocity[local_indices, 0] = displacement[:, 0] * derivative
                lateral_derivative = 2.0 * (1.0 - smooth_phase) * derivative
                velocity[local_indices, 1:] = (
                    displacement[:, 1:] * lateral_derivative[:, None]
                )
                roll_progress = ((start[local_indices, 0] + 2.5) / 2.5).clamp(0.0, 1.0)
                alignment = roll_progress.square() * (3.0 - 2.0 * roll_progress)
                roll[local_indices] = gap_roll[local_indices] * alignment
                normalized_phase = phase / 0.82
                pitch[local_indices] = 0.30 * torch.sin(torch.pi * normalized_phase)
                alignment_derivative = (
                    6.0 * roll_progress * (1.0 - roll_progress) / 2.5
                )
                planned_body_rate[local_indices, 0] = (
                    gap_roll[local_indices]
                    * alignment_derivative
                    * velocity[local_indices, 0]
                ).clamp(-self.config.bodyrate_max, self.config.bodyrate_max)
                planned_body_rate[local_indices, 1] = (
                    0.30
                    * torch.pi
                    * torch.cos(torch.pi * normalized_phase)
                    / (0.82 * 2.2)
                )
                yaw[local_indices] = torch.atan2(
                    gap_center[local_indices, 1] - start[local_indices, 1],
                    -start[local_indices, 0].clamp_max(-0.05),
                )
            else:
                trajectories = self.informed_trajectories
                trajectory_angles = trajectories["angles"]
                multi_path = trajectories["positions"].ndim == 4
                if multi_path:
                    path_index = torch.randint(
                        trajectories["positions"].shape[1],
                        (local_indices.numel(),),
                        device=self.device,
                        generator=self.generator,
                    )
                upper = torch.searchsorted(
                    trajectory_angles, gap_roll[local_indices]
                ).clamp(1, trajectory_angles.numel() - 1)
                lower = upper - 1
                angle_fraction = (
                    (gap_roll[local_indices] - trajectory_angles[lower])
                    / (trajectory_angles[upper] - trajectory_angles[lower]).clamp_min(1e-6)
                )
                state_count = trajectories["positions"].shape[2 if multi_path else 1]
                state_index = torch.randint(
                    1,
                    state_count - 1,
                    (local_indices.numel(),),
                    device=self.device,
                    generator=self.generator,
                )
                def interpolate(name: str) -> torch.Tensor:
                    if multi_path:
                        lower_value = trajectories[name][lower, path_index, state_index]
                        upper_value = trajectories[name][upper, path_index, state_index]
                    else:
                        lower_value = trajectories[name][lower, state_index]
                        upper_value = trajectories[name][upper, state_index]
                    fraction = angle_fraction
                    while fraction.ndim < lower_value.ndim:
                        fraction = fraction.unsqueeze(-1)
                    return torch.lerp(lower_value, upper_value, fraction)

                relative_position = interpolate("positions")
                start[local_indices] = gap_center[local_indices] + relative_position
                velocity[local_indices] = interpolate("velocities")
                quaternion = interpolate("quaternions")
                quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                sampled_roll, sampled_pitch, sampled_yaw = matrix_to_roll_pitch_yaw(
                    quaternion_to_matrix(quaternion)
                )
                roll[local_indices] = sampled_roll
                pitch[local_indices] = sampled_pitch
                yaw[local_indices] = sampled_yaw
                action_index = (state_index - 1).clamp_max(
                    trajectories["thrust"].shape[2 if multi_path else 1] - 1
                )
                if multi_path:
                    lower_thrust = trajectories["thrust"][lower, path_index, action_index]
                    upper_thrust = trajectories["thrust"][upper, path_index, action_index]
                else:
                    lower_thrust = trajectories["thrust"][lower, action_index]
                    upper_thrust = trajectories["thrust"][upper, action_index]
                library_action[local_indices, 0] = torch.lerp(
                    lower_thrust, upper_thrust, angle_fraction
                )
                if multi_path:
                    lower_rate = trajectories["body_rate"][lower, path_index, action_index]
                    upper_rate = trajectories["body_rate"][upper, path_index, action_index]
                else:
                    lower_rate = trajectories["body_rate"][lower, action_index]
                    upper_rate = trajectories["body_rate"][upper, action_index]
                library_action[local_indices, 1:] = torch.lerp(
                    lower_rate, upper_rate, angle_fraction[:, None]
                )
                library_informed[local_indices] = True

        self.position[indices] = start
        self.velocity[indices] = velocity
        self.quaternion[indices] = quaternion_from_euler(roll, pitch, yaw)

        hover_action = torch.zeros((count, 4), device=self.device)
        hover_action[:, 0] = 9.81
        if informed.any():
            informed_roll = roll[informed]
            informed_pitch = pitch[informed]
            compensated_thrust = 9.81 / (
                torch.cos(informed_roll).abs() * torch.cos(informed_pitch).abs()
            ).clamp_min(0.45)
            hover_action[informed, 0] = compensated_thrust.clamp(
                self.config.thrust_min, self.config.thrust_max
            )
            hover_action[informed, 1:] = planned_body_rate[informed]
        hover_action[library_informed] = library_action[library_informed]
        self.previous_action[indices] = hover_action
        self.action_response[indices] = hover_action
        self.action_history[indices] = hover_action[:, None, :]

        if self.config.response_randomization:
            delay_jitter = torch.empty((count, 4), device=self.device).uniform_(
                -1.0, 1.0, generator=self.generator
            )
            jitter_scale = torch.tensor([0.40, 0.30, 0.30, 0.30], device=self.device)
            delays = torch.round(self.base_delay * (1.0 + delay_jitter * jitter_scale)).long()
            self.response_delay[indices] = delays.clamp_min(0)
            scale_noise = torch.empty((count, 4), device=self.device).uniform_(
                -1.0, 1.0, generator=self.generator
            )
            self.response_scale[indices] = 1.0 + scale_noise * self.response_scale_range
            self.response_scale_steps[indices] = torch.randint(
                30, 91, (count,), device=self.device, generator=self.generator
            )
        else:
            self.response_delay[indices] = self.base_delay
            self.response_scale[indices] = 1.0
            self.response_scale_steps[indices] = 10_000

        drag_scale = self._uniform((count, 3), 0.5, 1.5)
        self.drag_linear[indices] = self.nominal_drag_linear * drag_scale
        self.drag_quadratic[indices] = self.nominal_drag_quadratic * drag_scale
        self.perturbation[indices] = 0.0
        self.perturbation_steps[indices] = 0
        self.episode_step[indices] = 0
        self.episode_return[indices] = 0.0
        self.episode_length[indices] = 0
        self.had_collision[indices] = False
        self.had_out_of_bounds[indices] = False
        self.camera_fov_scale[indices] = self._uniform((count, 2), 0.95, 1.05)
        return self.observe()

    def termination_label(self) -> torch.Tensor:
        rotation = quaternion_to_matrix(self.quaternion)
        _, fully_passed, _ = gap_intersection_status(
            self.position,
            rotation,
            self.gap_center,
            self.gap_rotation_matrix,
            self.collider_half_size,
            self.config.gap_width * 0.5,
            self.config.gap_height * 0.5,
        )
        return torch.where(fully_passed & ~self.had_collision, 1.0, -1.0)

    def _compute_response(self, command: torch.Tensor) -> torch.Tensor:
        self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
        self.action_history[:, 0] = command
        response = torch.empty_like(command)
        env_indices = torch.arange(self.num_envs, device=self.device)
        for action_index in range(4):
            samples = []
            for offset in range(self.config.action_average_window):
                history_index = self.response_delay[:, action_index] + offset
                samples.append(self.action_history[env_indices, history_index, action_index])
            response[:, action_index] = torch.stack(samples, dim=-1).mean(dim=-1)

        self.response_scale_steps -= 1
        refresh = self.response_scale_steps <= 0
        if self.config.response_randomization and refresh.any():
            count = int(refresh.sum().item())
            noise = torch.empty((count, 4), device=self.device).uniform_(
                -1.0, 1.0, generator=self.generator
            )
            self.response_scale[refresh] = 1.0 + noise * self.response_scale_range
            self.response_scale_steps[refresh] = torch.randint(
                30, 91, (count,), device=self.device, generator=self.generator
            )
        response *= self.response_scale
        response[:, 0].clamp_(self.config.thrust_min, self.config.thrust_max)
        response[:, 1:].clamp_(-self.config.bodyrate_max, self.config.bodyrate_max)
        return response

    def _update_perturbation(self, body_rate: torch.Tensor) -> None:
        active = self.perturbation_steps > 0
        self.perturbation_steps[active] -= 1
        self.perturbation[~active] = 0.0
        if not self.config.perturbation_force:
            return
        distance = -self.position[:, 0]
        eligible = (
            (self.perturbation_steps <= 0)
            & (distance > 1.5)
            & (body_rate.norm(dim=-1) < 3.0)
        )
        trigger = torch.rand(
            (self.num_envs,), device=self.device, generator=self.generator
        ) < self.config.perturbation_probability
        trigger &= eligible
        if trigger.any():
            count = int(trigger.sum().item())
            sampled = torch.empty((count, 3), device=self.device).uniform_(
                -1.0, 1.0, generator=self.generator
            )
            self.perturbation[trigger] = sampled * self.perturbation_max
            self.perturbation_steps[trigger] = self.config.perturbation_steps

    def _integrate(self, response: torch.Tensor) -> None:
        body_rate = response[:, 1:]
        self._update_perturbation(body_rate)
        substep_dt = self.config.control_dt / self.config.integration_substeps
        for _ in range(self.config.integration_substeps):
            self.quaternion = quaternion_integrate(self.quaternion, body_rate, substep_dt)
            rotation = quaternion_to_matrix(self.quaternion)
            body_velocity = torch.einsum("nij,nj->ni", rotation.transpose(-1, -2), self.velocity)
            body_drag = (
                self.drag_linear * body_velocity
                + self.drag_quadratic * body_velocity.abs() * body_velocity
            )
            world_drag = torch.einsum("nij,nj->ni", rotation, body_drag)
            thrust_acceleration = rotation[:, :, 2] * response[:, :1]
            acceleration = thrust_acceleration + self.gravity - world_drag + self.perturbation
            self.velocity += acceleration * substep_dt
            self.position += self.velocity * substep_dt

    def _speed_reward(self, speed: torch.Tensor) -> torch.Tensor:
        if self.config.speed_reward_mode == "overspeed_penalty":
            overspeed = speed > self.config.speed_limit
            return 0.05 * overspeed * (
                1.0
                - torch.exp(
                    (speed - self.config.speed_limit).clamp(max=4.0)
                )
            )
        if self.config.speed_reward_mode == "below_limit_reward":
            below_limit = speed <= self.config.speed_limit
            return 0.05 * below_limit * (
                1.0 - torch.exp(speed - self.config.speed_limit)
            )
        raise ValueError(f"Unknown speed reward mode: {self.config.speed_reward_mode}")

    def _action_rewards(
        self,
        command: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude_weights = torch.tensor(
            [0.04, 0.05, 0.04, 0.02], device=self.device
        )
        variation_weights = torch.tensor(
            [0.06, 0.015, 0.01, 0.0], device=self.device
        )
        if self.config.action_penalty_mode == "paper":
            penalty_command = command
            penalty_previous_action = previous_action
        elif self.config.action_penalty_mode == "physical":
            penalty_command = command.clone()
            penalty_previous_action = previous_action.clone()
            penalty_command[:, 0] -= 9.81
            penalty_previous_action[:, 0] -= 9.81
        elif self.config.action_penalty_mode == "normalized":
            thrust_half_range = 0.5 * (
                self.config.thrust_max - self.config.thrust_min
            )
            penalty_command = command.clone()
            penalty_previous_action = previous_action.clone()
            penalty_command[:, 0] = (command[:, 0] - 9.81) / thrust_half_range
            penalty_previous_action[:, 0] = (
                previous_action[:, 0] - 9.81
            ) / thrust_half_range
            penalty_command[:, 1:] /= self.config.bodyrate_max
            penalty_previous_action[:, 1:] /= self.config.bodyrate_max
        else:
            raise ValueError(
                f"Unknown action penalty mode: {self.config.action_penalty_mode}"
            )
        magnitude_penalty = -(
            penalty_command.abs() * magnitude_weights
        ).sum(dim=-1)
        variation_penalty = -(
            (penalty_command - penalty_previous_action).abs() * variation_weights
        ).sum(dim=-1)
        return magnitude_penalty, variation_penalty

    def observe(self) -> Observation:
        rotation = quaternion_to_matrix(self.quaternion)
        local_points = self.gap_points_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        world_points = torch.einsum("nij,npj->npi", self.gap_rotation_matrix, local_points)
        world_points = world_points + self.gap_center[:, None, :]
        relative_world = world_points - self.position[:, None, :]
        body_points = torch.einsum("nij,npj->npi", rotation.transpose(-1, -2), relative_world)
        body_velocity = torch.einsum("nij,nj->ni", rotation.transpose(-1, -2), self.velocity)
        roll, pitch, _ = matrix_to_roll_pitch_yaw(rotation)
        normalized_response_thrust = self._physical_to_normalized_action(self.action_response)[:, :1]
        normalized_previous_action = self._physical_to_normalized_action(self.previous_action)
        state_action = torch.cat(
            (
                body_velocity,
                roll[:, None],
                pitch[:, None],
                normalized_response_thrust,
                normalized_previous_action,
            ),
            dim=-1,
        )
        return Observation(gap_points=body_points, state_action=state_action)

    def step(
        self, raw_action: torch.Tensor
    ) -> tuple[Observation, torch.Tensor, torch.Tensor, dict[str, Any]]:
        command = self.action_to_physical(raw_action)
        previous_position = self.position.clone()
        previous_action = self.previous_action.clone()
        previous_rotation = quaternion_to_matrix(self.quaternion)
        previous_gap_vector = self.gap_center - previous_position
        previous_gap_direction = previous_gap_vector / previous_gap_vector.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        previous_alignment = (previous_rotation[:, :, 0] * previous_gap_direction).sum(dim=-1)
        previous_distance = previous_gap_vector.norm(dim=-1)
        previous_body_velocity = torch.einsum(
            "nij,nj->ni", previous_rotation.transpose(-1, -2), self.velocity
        )

        response = self._compute_response(command)
        self.action_response = response
        self._integrate(response)
        self.episode_step += 1
        self.episode_length += 1

        rotation = quaternion_to_matrix(self.quaternion)
        collision, fully_passed, _ = gap_intersection_status(
            self.position,
            rotation,
            self.gap_center,
            self.gap_rotation_matrix,
            self.collider_half_size,
            self.config.gap_width * 0.5,
            self.config.gap_height * 0.5,
        )
        self.had_collision |= collision

        previous_x = previous_position[:, 0] - self.gap_center[:, 0]
        current_x = self.position[:, 0] - self.gap_center[:, 0]
        near_plane = current_x.abs() <= 0.2
        traversal_delta = current_x.clamp(max=0.2) - previous_x.clamp(min=-0.2)
        traversal_reward = 10.0 * near_plane * ~self.had_collision * traversal_delta

        current_gap_vector = self.gap_center - self.position
        current_distance = current_gap_vector.norm(dim=-1)
        shaping_reward = 0.3 * (previous_distance - current_distance) * (current_x < 0.0)

        magnitude_penalty, variation_penalty = self._action_rewards(
            command, previous_action
        )

        speed = self.velocity.norm(dim=-1)
        speed_penalty = self._speed_reward(speed)

        gap_direction = current_gap_vector / current_distance[:, None].clamp_min(1e-6)
        alignment = (rotation[:, :, 0] * gap_direction).sum(dim=-1)
        far_from_gap = current_x < -0.6
        distillation_reward = 0.15 * (alignment - previous_alignment) * far_from_gap

        roll, pitch, yaw = matrix_to_roll_pitch_yaw(rotation)
        body_velocity = torch.einsum("nij,nj->ni", rotation.transpose(-1, -2), self.velocity)
        if self.config.attitude_penalty_mode == "time_scaled":
            attitude_penalty_scale = self.config.control_dt
        elif self.config.attitude_penalty_mode == "paper":
            attitude_penalty_scale = 1.0
        else:
            raise ValueError(
                f"Unknown attitude penalty mode: {self.config.attitude_penalty_mode}"
            )
        near_regularization = (
            0.375 * (body_velocity[:, 0] - previous_body_velocity[:, 0])
            - 0.25 * pitch.abs() * attitude_penalty_scale
            - 0.5 * wrap_angle(yaw).abs() * attitude_penalty_scale
        ) * ~far_from_gap

        reward = (
            traversal_reward
            + shaping_reward
            + magnitude_penalty
            + variation_penalty
            + speed_penalty
            + distillation_reward
            + near_regularization
        )
        out_of_bounds = (
            (self.position[:, 0] < -6.0)
            | (self.position[:, 0] > 2.0)
            | ((self.position[:, 1] - self.gap_center[:, 1]).abs() > 3.0)
            | (self.position[:, 2] < 0.15)
            | (self.position[:, 2] > 3.5)
        )
        self.had_out_of_bounds |= out_of_bounds
        success = fully_passed & ~self.had_collision & ~self.had_out_of_bounds
        timeout = self.episode_step >= self.config.max_episode_steps
        if self.config.failure_termination_mode == "immediate":
            done = collision | success | out_of_bounds | timeout
        elif self.config.failure_termination_mode == "traversal_or_horizon":
            done = fully_passed | timeout
        else:
            raise ValueError(
                "Unknown failure termination mode: "
                f"{self.config.failure_termination_mode}"
            )
        self.episode_return += reward

        info: dict[str, Any] = {
            "success": success.clone(),
            "collision": self.had_collision.clone(),
            "timeout": timeout.clone(),
            "out_of_bounds": self.had_out_of_bounds.clone(),
            "episode_return": torch.where(done, self.episode_return, torch.nan),
            "episode_length": torch.where(done, self.episode_length, 0),
            "gap_roll": self.gap_roll.clone(),
            "informed_reset": self.informed_reset.clone(),
            "reward/traversal": traversal_reward.mean(),
            "reward/shaping": shaping_reward.mean(),
            "reward/magnitude": magnitude_penalty.mean(),
            "reward/variation": variation_penalty.mean(),
            "reward/speed": speed_penalty.mean(),
            "reward/distillation": distillation_reward.mean(),
            "reward/near_regularization": near_regularization.mean(),
        }
        self.previous_action = command
        if self.auto_reset and done.any():
            self.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
        return self.observe(), reward, done, info
