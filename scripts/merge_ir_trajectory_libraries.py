from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sensorimotor.geometry import gap_intersection_status, gap_rotation
from sensorimotor.math import quaternion_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge informed-reset trajectory libraries and validate angle interpolation."
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-step-degrees", type=float, default=1.0)
    return parser.parse_args()


def interpolate_angle(
    values: torch.Tensor,
    angles: torch.Tensor,
    query_angles: torch.Tensor,
) -> torch.Tensor:
    upper = torch.searchsorted(angles, query_angles).clamp(1, angles.numel() - 1)
    lower = upper - 1
    fraction = (query_angles - angles[lower]) / (
        angles[upper] - angles[lower]
    ).clamp_min(1e-6)
    while fraction.ndim < values.ndim:
        fraction = fraction.unsqueeze(-1)
    return torch.lerp(values[lower], values[upper], fraction)


@torch.no_grad()
def validate_interpolation(
    library: dict[str, object], step_degrees: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = library["angles"]
    query_degrees = torch.arange(
        0.0, 90.0 + 0.5 * step_degrees, step_degrees, device=angles.device
    )
    query_angles = torch.deg2rad(query_degrees)
    positions = interpolate_angle(library["positions"], angles, query_angles)
    quaternions = interpolate_angle(library["quaternions"], angles, query_angles)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angle_count, path_count = positions.shape[:2]
    flat_angles = query_angles[:, None].expand(-1, path_count).reshape(-1)
    gap_centers = torch.zeros((angle_count * path_count, 3), device=angles.device)
    gap_rotations = gap_rotation(flat_angles)
    collider_half_size = torch.tensor([0.17, 0.17, 0.055], device=angles.device)
    collisions = torch.zeros(
        (angle_count * path_count,), dtype=torch.bool, device=angles.device
    )
    successes = torch.zeros_like(collisions)
    flat_positions = positions.flatten(0, 1)
    flat_quaternions = quaternions.flatten(0, 1)
    for state_index in range(1, positions.shape[2]):
        active = ~collisions & ~successes
        collision, fully_passed, _ = gap_intersection_status(
            flat_positions[:, state_index],
            quaternion_to_matrix(flat_quaternions[:, state_index]),
            gap_centers,
            gap_rotations,
            collider_half_size,
            0.30,
            0.10,
        )
        collisions |= active & collision
        successes |= active & fully_passed & ~collision
    return (
        query_angles,
        collisions.reshape(angle_count, path_count),
        successes.reshape(angle_count, path_count),
    )


def main() -> None:
    args = parse_args()
    libraries = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs
    ]
    reference_offsets = libraries[0]["start_offsets"]
    for library in libraries[1:]:
        if library["control_dt"] != libraries[0]["control_dt"]:
            raise ValueError("All libraries must use the same control step")
        if not torch.equal(library["start_offsets"], reference_offsets):
            raise ValueError("All libraries must use the same path offsets")

    angle_fields = (
        "angles",
        "positions",
        "velocities",
        "quaternions",
        "thrust",
        "body_rate",
        "collisions",
        "successes",
    )
    merged: dict[str, object] = {
        "control_dt": libraries[0]["control_dt"],
        "start_offsets": reference_offsets,
    }
    merged_angles = torch.cat([library["angles"] for library in libraries])
    order = merged_angles.argsort()
    if torch.unique(merged_angles).numel() != merged_angles.numel():
        raise ValueError("Input libraries contain duplicate angle anchors")
    for field in angle_fields:
        merged[field] = torch.cat([library[field] for library in libraries])[order]
    if all("losses" in library for library in libraries):
        merged["losses"] = torch.cat(
            [library["losses"] for library in libraries]
        )[order]

    query_angles, collisions, successes = validate_interpolation(
        merged, args.validation_step_degrees
    )
    if collisions.any() or not successes.all():
        failed = torch.nonzero(collisions | ~successes, as_tuple=False)
        descriptions = [
            (
                round(float(torch.rad2deg(query_angles[angle_index]).item()), 2),
                reference_offsets[path_index].tolist(),
            )
            for angle_index, path_index in failed[:20]
        ]
        raise RuntimeError(
            f"Interpolated trajectory validation failed for {failed.shape[0]} paths; "
            f"first failures: {descriptions}"
        )
    merged["validation_angles"] = query_angles
    merged["validation_collisions"] = collisions
    merged["validation_successes"] = successes
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(
        f"Wrote {output} with {merged_angles.numel()} anchors, "
        f"{reference_offsets.shape[0]} paths per anchor, and "
        f"{query_angles.numel()} validated angles"
    )


if __name__ == "__main__":
    main()
