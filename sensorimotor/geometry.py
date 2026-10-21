from __future__ import annotations

import torch


def gap_rotation(roll: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(roll)
    sine = torch.sin(roll)
    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)
    return torch.stack(
        (
            ones,
            zeros,
            zeros,
            zeros,
            cosine,
            -sine,
            zeros,
            sine,
            cosine,
        ),
        dim=-1,
    ).reshape(roll.shape + (3, 3))


def make_rectangular_gap_points(
    width: float,
    height: float,
    points_per_edge: int,
    device: torch.device,
) -> torch.Tensor:
    edge_parameter = torch.linspace(-1.0, 1.0, points_per_edge + 1, device=device)[:-1]
    half_width = width * 0.5
    half_height = height * 0.5
    zeros = torch.zeros_like(edge_parameter)
    top = torch.stack((zeros, edge_parameter * half_width, torch.full_like(zeros, half_height)), dim=-1)
    right = torch.stack((zeros, torch.full_like(zeros, half_width), -edge_parameter * half_height), dim=-1)
    bottom = torch.stack((zeros, -edge_parameter * half_width, torch.full_like(zeros, -half_height)), dim=-1)
    left = torch.stack((zeros, torch.full_like(zeros, -half_width), edge_parameter * half_height), dim=-1)
    return torch.cat((top, right, bottom, left), dim=0)


def projected_collider_half_extents(
    body_rotation: torch.Tensor,
    gap_rotation_matrix: torch.Tensor,
    collider_half_size: torch.Tensor,
) -> torch.Tensor:
    body_in_gap = gap_rotation_matrix.transpose(-1, -2) @ body_rotation
    return body_in_gap.abs() @ collider_half_size


def gap_intersection_status(
    position: torch.Tensor,
    body_rotation: torch.Tensor,
    gap_center: torch.Tensor,
    gap_rotation_matrix: torch.Tensor,
    collider_half_size: torch.Tensor,
    gap_half_width: float,
    gap_half_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = projected_collider_half_extents(body_rotation, gap_rotation_matrix, collider_half_size)
    signs = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        device=position.device,
        dtype=position.dtype,
    )
    local_vertices = signs * collider_half_size
    world_vertices = position[:, None, :] + torch.einsum(
        "nij,vj->nvi", body_rotation, local_vertices
    )
    gap_vertices = torch.einsum(
        "nij,nvj->nvi",
        gap_rotation_matrix.transpose(-1, -2),
        world_vertices - gap_center[:, None, :],
    )
    signed_distance = gap_vertices[..., 0]
    fully_passed = signed_distance.min(dim=-1).values > 0.0
    intersects_plane = (
        (signed_distance.min(dim=-1).values <= 0.0)
        & (signed_distance.max(dim=-1).values >= 0.0)
    )

    edges = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        device=position.device,
    )
    start = gap_vertices[:, edges[:, 0]]
    end = gap_vertices[:, edges[:, 1]]
    start_distance = start[..., 0]
    end_distance = end[..., 0]
    denominator = start_distance - end_distance
    crosses = (
        (start_distance * end_distance <= 0.0)
        & (denominator.abs() > 1e-7)
    )
    safe_denominator = torch.where(crosses, denominator, torch.ones_like(denominator))
    interpolation = start_distance / safe_denominator
    intersection = start + interpolation[..., None] * (end - start)
    outside_opening = (
        (intersection[..., 1].abs() > gap_half_width)
        | (intersection[..., 2].abs() > gap_half_height)
    )
    collision = intersects_plane & (crosses & outside_opening).any(dim=-1)
    return collision, fully_passed, support
