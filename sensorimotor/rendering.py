from __future__ import annotations

import torch
import torch.nn.functional as functional

from sensorimotor.math import quaternion_to_matrix
from sensorimotor.simulator import VectorizedGapEnv


class BinaryGapRenderer:
    def __init__(
        self,
        height: int = 256,
        width: int = 320,
        horizontal_fov_degrees: float = 82.0,
        vertical_fov_degrees: float = 72.0,
        frame_thickness: float = 0.04,
        max_distance: float = 10.0,
        edge_randomization: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self.vertical_fov_degrees = vertical_fov_degrees
        self.frame_thickness = frame_thickness
        self.max_distance = max_distance
        self.edge_randomization = edge_randomization
        self._pixel_x: torch.Tensor | None = None
        self._pixel_y: torch.Tensor | None = None
        self._device: torch.device | None = None

    def _pixels(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._device != device or self._pixel_x is None or self._pixel_y is None:
            horizontal = torch.linspace(-1.0, 1.0, self.width, device=device)
            vertical = torch.linspace(1.0, -1.0, self.height, device=device)
            self._pixel_y, self._pixel_x = torch.meshgrid(vertical, horizontal, indexing="ij")
            self._device = device
        return self._pixel_x, self._pixel_y

    @torch.no_grad()
    def render(self, environment: VectorizedGapEnv, randomize: bool = True) -> torch.Tensor:
        device = environment.device
        pixel_x, pixel_y = self._pixels(device)
        count = environment.num_envs
        horizontal_tangent = torch.tan(
            torch.deg2rad(
                torch.tensor(self.horizontal_fov_degrees * 0.5, device=device)
            )
            * environment.camera_fov_scale[:, 0]
        )
        vertical_tangent = torch.tan(
            torch.deg2rad(torch.tensor(self.vertical_fov_degrees * 0.5, device=device))
            * environment.camera_fov_scale[:, 1]
        )
        ray_body = torch.empty((count, self.height, self.width, 3), device=device)
        ray_body[..., 0] = 1.0
        ray_body[..., 1] = pixel_x * horizontal_tangent[:, None, None]
        ray_body[..., 2] = pixel_y * vertical_tangent[:, None, None]

        body_rotation = quaternion_to_matrix(environment.quaternion)
        ray_world = torch.einsum("nij,nhwj->nhwi", body_rotation, ray_body)
        plane_normal = environment.gap_rotation_matrix[:, :, 0]
        numerator = ((environment.gap_center - environment.position) * plane_normal).sum(dim=-1)
        denominator = torch.einsum("ni,nhwi->nhw", plane_normal, ray_world)
        distance = numerator[:, None, None] / denominator.clamp(min=1e-6)
        intersection = environment.position[:, None, None, :] + distance[..., None] * ray_world
        gap_relative = torch.einsum(
            "nij,nhwj->nhwi",
            environment.gap_rotation_matrix.transpose(-1, -2),
            intersection - environment.gap_center[:, None, None, :],
        )
        half_width = environment.config.gap_width * 0.5
        half_height = environment.config.gap_height * 0.5
        inside_outer = (
            (gap_relative[..., 1].abs() <= half_width + self.frame_thickness)
            & (gap_relative[..., 2].abs() <= half_height + self.frame_thickness)
        )
        inside_opening = (
            (gap_relative[..., 1].abs() < half_width)
            & (gap_relative[..., 2].abs() < half_height)
        )
        valid = (distance > 0.0) & (distance < self.max_distance) & (denominator > 1e-6)
        mask = (inside_outer & ~inside_opening & valid).float().unsqueeze(1)
        if randomize and self.edge_randomization:
            mask = self._randomize_edges(mask, environment.generator)
        return mask

    def _randomize_edges(
        self, mask: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        count = mask.shape[0]
        block_four = torch.rand((count,), device=mask.device, generator=generator) < 0.5
        use_max = torch.rand((count,), device=mask.device, generator=generator) < 0.5
        results = []
        for block_size in (2, 4):
            maximum = functional.max_pool2d(mask, block_size, stride=block_size)
            minimum = -functional.max_pool2d(-mask, block_size, stride=block_size)
            mode = use_max[:, None, None, None]
            pooled = torch.where(mode, maximum, minimum)
            results.append(functional.interpolate(pooled, size=(self.height, self.width), mode="nearest"))
        return torch.where(block_four[:, None, None, None], results[1], results[0])


class SyntheticRGBRenderer(BinaryGapRenderer):
    @torch.no_grad()
    def render(self, environment: VectorizedGapEnv) -> torch.Tensor:
        device = environment.device
        pixel_x, pixel_y = self._pixels(device)
        count = environment.num_envs
        horizontal_tangent = torch.tan(
            torch.deg2rad(
                torch.tensor(self.horizontal_fov_degrees * 0.5, device=device)
            )
            * environment.camera_fov_scale[:, 0]
        )
        vertical_tangent = torch.tan(
            torch.deg2rad(torch.tensor(self.vertical_fov_degrees * 0.5, device=device))
            * environment.camera_fov_scale[:, 1]
        )
        ray_body = torch.empty((count, self.height, self.width, 3), device=device)
        ray_body[..., 0] = 1.0
        ray_body[..., 1] = pixel_x * horizontal_tangent[:, None, None]
        ray_body[..., 2] = pixel_y * vertical_tangent[:, None, None]
        body_rotation = quaternion_to_matrix(environment.quaternion)
        ray_world = torch.einsum("nij,nhwj->nhwi", body_rotation, ray_body)
        ray_direction = ray_world / ray_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        sky_amount = (0.55 + 1.2 * ray_direction[..., 2]).clamp(0.0, 1.0)
        horizon = torch.tensor([0.72, 0.82, 0.91], device=device)
        zenith = torch.tensor([0.18, 0.42, 0.72], device=device)
        rgb = torch.lerp(
            horizon,
            zenith,
            sky_amount[..., None],
        ).expand(count, -1, -1, -1).clone()

        ground_z = ray_world[..., 2]
        safe_ground_z = torch.where(
            ground_z.abs() > 1e-6,
            ground_z,
            torch.ones_like(ground_z),
        )
        ground_distance = (
            -environment.position[:, None, None, 2] / safe_ground_z
        )
        ground_valid = (
            (ground_z < -1e-6)
            & (ground_distance > 0.0)
            & (ground_distance < self.max_distance * 3.0)
        )
        ground_point = (
            environment.position[:, None, None, :]
            + ground_distance[..., None] * ray_world
        )
        checker = (
            torch.floor(ground_point[..., 0] * 2.0)
            + torch.floor(ground_point[..., 1] * 2.0)
        ).remainder(2.0)
        floor_dark = torch.tensor([0.26, 0.30, 0.27], device=device)
        floor_light = torch.tensor([0.40, 0.44, 0.38], device=device)
        floor_color = torch.lerp(floor_dark, floor_light, checker[..., None])
        grid_x = torch.remainder(ground_point[..., 0], 0.5).minimum(
            torch.remainder(-ground_point[..., 0], 0.5)
        )
        grid_y = torch.remainder(ground_point[..., 1], 0.5).minimum(
            torch.remainder(-ground_point[..., 1], 0.5)
        )
        grid_line = (grid_x < 0.015) | (grid_y < 0.015)
        floor_color = torch.where(
            grid_line[..., None],
            floor_color * 0.45,
            floor_color,
        )
        rgb = torch.where(ground_valid[..., None], floor_color, rgb)

        plane_normal = environment.gap_rotation_matrix[:, :, 0]
        numerator = (
            (environment.gap_center - environment.position) * plane_normal
        ).sum(dim=-1)
        denominator = torch.einsum("ni,nhwi->nhw", plane_normal, ray_world)
        safe_denominator = torch.where(
            denominator.abs() > 1e-6,
            denominator,
            torch.ones_like(denominator),
        )
        distance = numerator[:, None, None] / safe_denominator
        intersection = (
            environment.position[:, None, None, :]
            + distance[..., None] * ray_world
        )
        gap_relative = torch.einsum(
            "nij,nhwj->nhwi",
            environment.gap_rotation_matrix.transpose(-1, -2),
            intersection - environment.gap_center[:, None, None, :],
        )
        half_width = environment.config.gap_width * 0.5
        half_height = environment.config.gap_height * 0.5
        inside_opening = (
            (gap_relative[..., 1].abs() < half_width)
            & (gap_relative[..., 2].abs() < half_height)
        )
        inside_frame = (
            (gap_relative[..., 1].abs() <= half_width + self.frame_thickness)
            & (gap_relative[..., 2].abs() <= half_height + self.frame_thickness)
            & ~inside_opening
        )
        plane_valid = (
            (distance > 0.0)
            & (distance < self.max_distance)
            & (denominator > 1e-6)
        )
        ground_in_front = ground_valid & (ground_distance < distance)
        wall_hit = plane_valid & ~inside_opening & ~ground_in_front
        frame_hit = plane_valid & inside_frame & ~ground_in_front

        concrete_noise = (
            0.06
            * torch.sin(gap_relative[..., 1] * 17.0)
            * torch.sin(gap_relative[..., 2] * 23.0)
        )
        concrete_brightness = (0.67 + concrete_noise).clamp(0.48, 0.78)
        wall_color = concrete_brightness[..., None] * torch.tensor(
            [0.82, 0.86, 0.90], device=device
        )
        block_y = torch.remainder(gap_relative[..., 1] + 20.0, 0.55)
        block_z = torch.remainder(gap_relative[..., 2] + 20.0, 0.28)
        mortar = (block_y < 0.025) | (block_z < 0.018)
        wall_color = torch.where(mortar[..., None], wall_color * 0.55, wall_color)
        distance_light = (1.0 - 0.035 * distance).clamp(0.72, 1.0)
        wall_color *= distance_light[..., None]
        rgb = torch.where(wall_hit[..., None], wall_color, rgb)

        frame_texture = 0.88 + 0.12 * torch.sin(
            (gap_relative[..., 1] + gap_relative[..., 2]) * 38.0
        )
        frame_color = frame_texture[..., None] * torch.tensor(
            [1.0, 0.43, 0.035], device=device
        )
        rgb = torch.where(frame_hit[..., None], frame_color, rgb)

        vignette = (
            1.0
            - 0.16
            * (pixel_x.square() + pixel_y.square()).clamp(0.0, 1.0)
        )
        rgb *= vignette[None, ..., None]
        return rgb.clamp(0.0, 1.0).permute(0, 3, 1, 2)
