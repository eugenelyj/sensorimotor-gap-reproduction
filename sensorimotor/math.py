from __future__ import annotations

import torch


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    scalar = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    vector = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    return torch.cat((scalar, vector), dim=-1)


def quaternion_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_integrate(quaternion: torch.Tensor, body_rate: torch.Tensor, dt: float) -> torch.Tensor:
    angle = body_rate.norm(dim=-1, keepdim=True) * dt
    half_angle = angle * 0.5
    axis = body_rate / body_rate.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    delta = torch.cat((torch.cos(half_angle), axis * torch.sin(half_angle)), dim=-1)
    small = angle < 1e-7
    identity = torch.zeros_like(delta)
    identity[..., 0] = 1.0
    delta = torch.where(small.expand_as(delta), identity, delta)
    return normalize_quaternion(quaternion_multiply(quaternion, delta))


def matrix_to_roll_pitch_yaw(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitch = torch.asin((-rotation[..., 2, 0]).clamp(-1.0, 1.0))
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return roll, pitch, yaw


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))
