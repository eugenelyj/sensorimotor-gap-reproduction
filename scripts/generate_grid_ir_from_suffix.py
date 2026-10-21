from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.generate_ir_trajectory_library import (
    physical_controls,
    simulate,
    trajectory_losses,
    validate,
)
from sensorimotor.math import (
    matrix_to_roll_pitch_yaw,
    quaternion_integrate,
    quaternion_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend optimized approach motions to a validated traversal library."
    )
    parser.add_argument(
        "--suffix-library", default="artifacts/ir/optimized_single_gap_roll.pt"
    )
    parser.add_argument(
        "--output", default="artifacts/ir/optimized_single_gap_roll_grid_anchors.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--approach-steps", type=int, default=150)
    parser.add_argument(
        "--lateral-offsets", type=float, nargs="+", default=[-1.0, 0.0, 1.0]
    )
    parser.add_argument(
        "--vertical-offsets", type=float, nargs="+", default=[-0.4, 0, 0.4]
    )
    return parser.parse_args()


def controls_to_parameters(thrust: torch.Tensor, body_rate: torch.Tensor) -> torch.Tensor:
    parameters = torch.empty(thrust.shape + (4,), device=thrust.device)
    parameters[..., 0] = torch.logit(((thrust - 6.0) / 14.0).clamp(1e-5, 1.0 - 1e-5))
    parameters[..., 1:] = torch.atanh((body_rate / 6.0).clamp(-0.999, 0.999))
    return parameters


def initialize_approach_parameters(
    initial_positions: torch.Tensor, steps: int, control_dt: float
) -> torch.Tensor:
    settling_steps = min(30, steps // 4)
    motion_steps = steps - settling_steps
    duration = motion_steps * control_dt
    count = initial_positions.shape[0]
    position = initial_positions.clone()
    velocity = torch.zeros_like(position)
    quaternion = torch.zeros((count, 4), device=position.device)
    quaternion[:, 0] = 1.0
    drag_linear = torch.tensor([0.08, 0.08, 0.12], device=position.device)
    drag_quadratic = torch.tensor([0.02, 0.02, 0.03], device=position.device)
    thrust_history = []
    body_rate_history = []
    for step in range(steps):
        phase = torch.tensor(
            min((step + 1) / motion_steps, 1.0), device=position.device
        )
        smooth = 10.0 * phase.pow(3) - 15.0 * phase.pow(4) + 6.0 * phase.pow(5)
        smooth_velocity = (
            30.0 * phase.square() - 60.0 * phase.pow(3) + 30.0 * phase.pow(4)
        ) / duration
        smooth_acceleration = (
            60.0 * phase - 180.0 * phase.square() + 120.0 * phase.pow(3)
        ) / duration**2
        if step >= motion_steps:
            smooth_velocity = torch.zeros_like(smooth_velocity)
            smooth_acceleration = torch.zeros_like(smooth_acceleration)
        desired_position = initial_positions[:, 1:] * (1.0 - smooth)
        desired_velocity = -initial_positions[:, 1:] * smooth_velocity
        desired_acceleration = -initial_positions[:, 1:] * smooth_acceleration
        commanded_acceleration = (
            desired_acceleration
            + 10.0 * (desired_position - position[:, 1:])
            + 6.0 * (desired_velocity - velocity[:, 1:])
        )
        rotation = quaternion_to_matrix(quaternion)
        body_velocity = torch.einsum(
            "nij,nj->ni", rotation.transpose(-1, -2), velocity
        )
        body_drag = (
            drag_linear * body_velocity
            + drag_quadratic * body_velocity.abs() * body_velocity
        )
        world_drag = torch.einsum("nij,nj->ni", rotation, body_drag)
        lateral_force = commanded_acceleration[:, 0] + world_drag[:, 1]
        vertical_force = commanded_acceleration[:, 1] + 9.81 + world_drag[:, 2]
        desired_roll = torch.atan2(-lateral_force, vertical_force)
        current_roll, _, _ = matrix_to_roll_pitch_yaw(rotation)
        body_rate = torch.zeros((count, 3), device=position.device)
        body_rate[:, 0] = ((desired_roll - current_roll) / control_dt).clamp(-6.0, 6.0)
        thrust = torch.sqrt(lateral_force.square() + vertical_force.square()).clamp(
            6.01, 19.99
        )
        thrust_history.append(thrust)
        body_rate_history.append(body_rate)

        quaternion = quaternion_integrate(quaternion, body_rate, control_dt)
        rotation = quaternion_to_matrix(quaternion)
        body_velocity = torch.einsum(
            "nij,nj->ni", rotation.transpose(-1, -2), velocity
        )
        body_drag = (
            drag_linear * body_velocity
            + drag_quadratic * body_velocity.abs() * body_velocity
        )
        world_drag = torch.einsum("nij,nj->ni", rotation, body_drag)
        acceleration = rotation[:, :, 2] * thrust[:, None]
        acceleration[:, 2] -= 9.81
        acceleration -= world_drag
        velocity += acceleration * control_dt
        position += velocity * control_dt
    return controls_to_parameters(
        torch.stack(thrust_history), torch.stack(body_rate_history)
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    library = torch.load(args.suffix_library, map_location=device, weights_only=False)
    if library["positions"].ndim != 3:
        raise ValueError("The suffix library must contain one path per angle")
    if not library["successes"].all() or library["collisions"].any():
        raise ValueError("The suffix library must be collision-free")
    control_dt = float(library["control_dt"])
    angles = library["angles"]
    offsets = torch.cartesian_prod(
        torch.tensor(args.lateral_offsets, device=device),
        torch.tensor(args.vertical_offsets, device=device),
    )
    if offsets.ndim == 1:
        offsets = offsets[None]
    angle_count = angles.numel()
    path_count = offsets.shape[0]
    trajectory_count = angle_count * path_count
    repeated_angles = angles.repeat_interleave(path_count)
    initial_positions = torch.zeros((trajectory_count, 3), device=device)
    initial_positions[:, 0] = -3.5
    initial_positions[:, 1:] = offsets.repeat(angle_count, 1)

    approach_parameters = initialize_approach_parameters(
        initial_positions, args.approach_steps, control_dt
    ).requires_grad_()
    suffix_parameters = controls_to_parameters(
        library["thrust"], library["body_rate"]
    )
    suffix_parameters = suffix_parameters[:, None].expand(
        angle_count, path_count, -1, -1
    ).reshape(trajectory_count, suffix_parameters.shape[1], 4).transpose(0, 1)
    optimizer = torch.optim.Adam([approach_parameters], lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max(1, args.iterations // 3),
            max(2, 2 * args.iterations // 3),
        ],
        gamma=0.3,
    )
    best_loss = torch.inf
    best_parameters = approach_parameters.detach().clone()
    identity = torch.eye(3, device=device)
    for iteration in range(args.iterations + 1):
        all_parameters = torch.cat((approach_parameters, suffix_parameters), dim=0)
        positions, velocities, quaternions, thrust, body_rate = simulate(
            all_parameters, repeated_angles, control_dt, initial_positions
        )
        losses = trajectory_losses(
            positions, velocities, quaternions, thrust, body_rate, repeated_angles
        )
        boundary_position = positions[args.approach_steps]
        boundary_rotation = quaternion_to_matrix(quaternions[args.approach_steps])
        losses += 2000.0 * (boundary_position[:, 0] + 3.5).square()
        losses += 4000.0 * boundary_position[:, 1:].square().sum(dim=-1)
        losses += 1000.0 * velocities[args.approach_steps].square().sum(dim=-1)
        losses += 300.0 * (
            boundary_rotation - identity
        ).square().sum(dim=(-1, -2))
        mean_loss = losses.mean()
        with torch.no_grad():
            if mean_loss < best_loss:
                best_loss = mean_loss.detach().clone()
                best_parameters.copy_(approach_parameters)
        if iteration < args.iterations:
            optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_([approach_parameters], 100.0)
            optimizer.step()
            scheduler.step()
        if iteration % 100 == 0:
            print(
                f"iteration={iteration} mean_loss={mean_loss.item():.4f} "
                f"best_loss={best_loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    all_parameters = torch.cat((best_parameters, suffix_parameters), dim=0)
    positions, velocities, quaternions, thrust, body_rate = simulate(
        all_parameters, repeated_angles, control_dt, initial_positions
    )
    collisions, successes = validate(positions, quaternions, repeated_angles)
    if collisions.any() or not successes.all():
        failed = torch.nonzero(collisions | ~successes, as_tuple=False).squeeze(-1)
        descriptions = [
            (
                round(float(torch.rad2deg(repeated_angles[index]).item()), 1),
                offsets[index % path_count].tolist(),
            )
            for index in failed
        ]
        raise RuntimeError(f"Trajectory validation failed for paths: {descriptions}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "control_dt": control_dt,
            "angles": angles.cpu(),
            "start_offsets": offsets.cpu(),
            "positions": positions.transpose(0, 1)
            .reshape(angle_count, path_count, positions.shape[0], 3)
            .cpu(),
            "velocities": velocities.transpose(0, 1)
            .reshape(angle_count, path_count, velocities.shape[0], 3)
            .cpu(),
            "quaternions": quaternions.transpose(0, 1)
            .reshape(angle_count, path_count, quaternions.shape[0], 4)
            .cpu(),
            "thrust": thrust.transpose(0, 1)
            .reshape(angle_count, path_count, thrust.shape[0])
            .cpu(),
            "body_rate": body_rate.transpose(0, 1)
            .reshape(angle_count, path_count, body_rate.shape[0], 3)
            .cpu(),
            "collisions": collisions.reshape(angle_count, path_count).cpu(),
            "successes": successes.reshape(angle_count, path_count).cpu(),
        },
        output,
    )
    print(f"Wrote {output} with {trajectory_count} validated paths")


if __name__ == "__main__":
    main()
