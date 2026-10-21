from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sensorimotor.geometry import gap_intersection_status, gap_rotation
from sensorimotor.math import quaternion_from_euler, quaternion_integrate, quaternion_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize dynamically feasible trajectories for informed reset."
    )
    parser.add_argument(
        "--output", default="artifacts/ir/optimized_single_gap_roll.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90]
    )
    parser.add_argument("--lateral-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--vertical-offsets", type=float, nargs="+", default=[0.0])
    return parser.parse_args()


def physical_controls(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    thrust = 6.0 + 14.0 * torch.sigmoid(parameters[..., 0])
    body_rate = 6.0 * torch.tanh(parameters[..., 1:])
    return thrust, body_rate


def simulate(
    parameters: torch.Tensor,
    angles: torch.Tensor,
    control_dt: float,
    initial_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    thrust, body_rate = physical_controls(parameters)
    count = angles.numel()
    position = torch.zeros((count, 3), device=angles.device)
    position[:, 0] = -3.5
    if initial_positions is not None:
        position.copy_(initial_positions)
    velocity = torch.zeros_like(position)
    quaternion = torch.zeros((count, 4), device=angles.device)
    quaternion[:, 0] = 1.0
    positions = [position]
    velocities = [velocity]
    quaternions = [quaternion]
    linear_drag = torch.tensor([0.08, 0.08, 0.12], device=angles.device)
    quadratic_drag = torch.tensor([0.02, 0.02, 0.03], device=angles.device)
    gravity = torch.tensor([0.0, 0.0, -9.81], device=angles.device)
    for step in range(parameters.shape[0]):
        quaternion = quaternion_integrate(quaternion, body_rate[step], control_dt)
        rotation = quaternion_to_matrix(quaternion)
        body_velocity = torch.einsum(
            "nij,nj->ni", rotation.transpose(-1, -2), velocity
        )
        body_drag = (
            linear_drag * body_velocity
            + quadratic_drag * body_velocity.abs() * body_velocity
        )
        world_drag = torch.einsum("nij,nj->ni", rotation, body_drag)
        acceleration = rotation[:, :, 2] * thrust[step, :, None] + gravity - world_drag
        velocity = velocity + acceleration * control_dt
        position = position + velocity * control_dt
        positions.append(position)
        velocities.append(velocity)
        quaternions.append(quaternion)
    return (
        torch.stack(positions),
        torch.stack(velocities),
        torch.stack(quaternions),
        thrust,
        body_rate,
    )


def trajectory_losses(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    quaternions: torch.Tensor,
    thrust: torch.Tensor,
    body_rate: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    rotations = quaternion_to_matrix(quaternions)
    target_quaternion = quaternion_from_euler(
        angles, torch.zeros_like(angles), torch.zeros_like(angles)
    )
    target_rotation = quaternion_to_matrix(target_quaternion)
    final_position = positions[-1]
    final_velocity = velocities[-1]
    losses = 200.0 * (final_position[:, 0] - 0.30).square()
    losses += 1200.0 * final_position[:, 1:].square().sum(dim=-1)
    losses += 20.0 * (final_velocity[:, 0] - 3.0).square()
    losses += 80.0 * final_velocity[:, 1:].square().sum(dim=-1)
    losses += 80.0 * (rotations[-1] - target_rotation).square().sum(dim=(-1, -2))

    plane_weight = torch.exp(-torch.square(positions[..., 0] / 0.28)).detach()
    plane_position_error = positions[..., 1:].square().sum(dim=-1)
    plane_rotation_error = (rotations - target_rotation[None]).square().sum(
        dim=(-1, -2)
    )
    weight_sum = plane_weight.sum(dim=0).clamp_min(1e-6)
    losses += 300.0 * (plane_weight * plane_position_error).sum(dim=0) / weight_sum
    losses += 25.0 * (plane_weight * plane_rotation_error).sum(dim=0) / weight_sum

    body_in_gap = torch.einsum(
        "bij,tbjk->tbik", gap_rotation(angles).transpose(-1, -2), rotations
    )
    collider_half_size = torch.tensor([0.17, 0.17, 0.055], device=angles.device)
    support = body_in_gap.abs() @ collider_half_size
    plane_contact = torch.sigmoid((support[..., 0] - positions[..., 0].abs()) / 0.02)
    width_violation = torch.relu(positions[..., 1].abs() + support[..., 1] - 0.285)
    height_violation = torch.relu(positions[..., 2].abs() + support[..., 2] - 0.085)
    losses += 200000.0 * (
        plane_contact * (width_violation.square() + height_violation.square())
    ).sum(dim=0)

    speed = velocities.norm(dim=-1)
    losses += 2.0 * torch.relu(-velocities[..., 0]).square().mean(dim=0)
    losses += 2.0 * torch.relu(speed - 4.0).square().mean(dim=0)
    losses += 0.002 * (thrust - 9.81).square().mean(dim=0)
    losses += 0.002 * body_rate.square().sum(dim=-1).mean(dim=0)
    losses += 0.01 * (body_rate[1:] - body_rate[:-1]).square().sum(dim=-1).mean(dim=0)
    losses += 0.0005 * (thrust[1:] - thrust[:-1]).square().mean(dim=0)
    return losses


def initialize_parameters(
    steps: int,
    angles: torch.Tensor,
    control_dt: float,
    initial_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    parameters = torch.zeros((steps, angles.numel(), 4), device=angles.device)
    parameters[..., 0] = torch.logit(
        torch.tensor((10.5 - 6.0) / 14.0, device=angles.device)
    )
    parameters[:20, :, 2] = torch.atanh(torch.tensor(0.8 / 6.0, device=angles.device))
    parameters[20:45, :, 2] = torch.atanh(torch.tensor(-0.5 / 6.0, device=angles.device))
    if initial_positions is not None:
        correction_steps = min(20, max((steps - 45) // 2, 0))
        if correction_steps:
            lateral_angle = (0.35 * initial_positions[:, 1]).clamp(-0.45, 0.45)
            lateral_rate = lateral_angle / (correction_steps * control_dt)
            normalized_lateral_rate = (lateral_rate / 6.0).clamp(-0.95, 0.95)
            parameters[:correction_steps, :, 1] = torch.atanh(
                normalized_lateral_rate
            )[None]
            parameters[correction_steps : 2 * correction_steps, :, 1] = torch.atanh(
                -normalized_lateral_rate
            )[None]
            initial_thrust = (
                10.5 - 1.5 * initial_positions[:, 2]
            ).clamp(6.1, 19.9)
            parameters[: 2 * correction_steps, :, 0] = torch.logit(
                (initial_thrust - 6.0) / 14.0
            )[None]
    roll_start = max(45, steps - 45)
    roll_steps = min(25, steps - roll_start)
    parameters[roll_start : roll_start + roll_steps, :, 1] = torch.atanh(
        (angles / (roll_steps * control_dt) / 6.0).clamp(-0.95, 0.95)
    )[None]
    return parameters


@torch.no_grad()
def validate(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = angles.numel()
    collisions = torch.zeros(count, dtype=torch.bool, device=angles.device)
    successes = torch.zeros_like(collisions)
    gap_centers = torch.zeros((count, 3), device=angles.device)
    gap_rotations = gap_rotation(angles)
    collider_half_size = torch.tensor([0.17, 0.17, 0.055], device=angles.device)
    for step in range(1, positions.shape[0]):
        active = ~collisions & ~successes
        collision, fully_passed, _ = gap_intersection_status(
            positions[step],
            quaternion_to_matrix(quaternions[step]),
            gap_centers,
            gap_rotations,
            collider_half_size,
            0.30,
            0.10,
        )
        collisions |= active & collision
        successes |= active & fully_passed & ~collision
    return collisions, successes


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    control_dt = 1.0 / 60.0
    angle_values = torch.deg2rad(torch.tensor(args.angles, device=device))
    offsets = torch.cartesian_prod(
        torch.tensor(args.lateral_offsets, device=device),
        torch.tensor(args.vertical_offsets, device=device),
    )
    if offsets.ndim == 1:
        offsets = offsets[None]
    paths_per_angle = offsets.shape[0]
    angles = angle_values.repeat_interleave(paths_per_angle)
    initial_positions = torch.zeros((angles.numel(), 3), device=device)
    initial_positions[:, 0] = -3.5
    initial_positions[:, 1:] = offsets.repeat(angle_values.numel(), 1)
    parameters = initialize_parameters(
        args.steps, angles, control_dt, initial_positions
    ).requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=0.02)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.iterations // 3, 2 * args.iterations // 3],
        gamma=0.35,
    )
    best_losses = torch.full((angles.numel(),), torch.inf, device=device)
    best_parameters = parameters.detach().clone()
    for iteration in range(args.iterations + 1):
        positions, velocities, quaternions, thrust, body_rate = simulate(
            parameters, angles, control_dt, initial_positions
        )
        losses = trajectory_losses(
            positions, velocities, quaternions, thrust, body_rate, angles
        )
        with torch.no_grad():
            improved = losses < best_losses
            best_losses[improved] = losses[improved]
            best_parameters[:, improved] = parameters[:, improved]
        optimizer.zero_grad()
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_([parameters], 100.0)
        optimizer.step()
        scheduler.step()
        if iteration % 250 == 0:
            print(
                f"iteration={iteration} mean_loss={losses.mean().item():.4f} "
                f"best_mean_loss={best_losses.mean().item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    positions, velocities, quaternions, thrust, body_rate = simulate(
        best_parameters, angles, control_dt, initial_positions
    )
    collisions, successes = validate(positions, quaternions, angles)
    if collisions.any() or not successes.all():
        failed_paths = [
            (round(float(torch.rad2deg(angle).item()), 1), offset.tolist())
            for angle, offset, collision, success in zip(
                angles, offsets.repeat(angle_values.numel(), 1), collisions, successes
            )
            if collision or not success
        ]
        raise RuntimeError(
            f"Trajectory validation failed for paths: {failed_paths}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "control_dt": control_dt,
            "angles": angle_values.cpu(),
            "start_offsets": offsets.cpu(),
            "positions": positions.transpose(0, 1)
            .reshape(angle_values.numel(), paths_per_angle, positions.shape[0], 3)
            .cpu(),
            "velocities": velocities.transpose(0, 1)
            .reshape(angle_values.numel(), paths_per_angle, velocities.shape[0], 3)
            .cpu(),
            "quaternions": quaternions.transpose(0, 1)
            .reshape(angle_values.numel(), paths_per_angle, quaternions.shape[0], 4)
            .cpu(),
            "thrust": thrust.transpose(0, 1)
            .reshape(angle_values.numel(), paths_per_angle, thrust.shape[0])
            .cpu(),
            "body_rate": body_rate.transpose(0, 1)
            .reshape(angle_values.numel(), paths_per_angle, body_rate.shape[0], 3)
            .cpu(),
            "losses": best_losses.reshape(angle_values.numel(), paths_per_angle).cpu(),
            "collisions": collisions.reshape(angle_values.numel(), paths_per_angle).cpu(),
            "successes": successes.reshape(angle_values.numel(), paths_per_angle).cpu(),
        },
        output,
    )
    for angle, offset, loss, collision, success in zip(
        angles, offsets.repeat(angle_values.numel(), 1), best_losses, collisions, successes
    ):
        print(
            f"roll={torch.rad2deg(angle).item():5.1f} offset={offset.tolist()} "
            f"loss={loss.item():.4f} "
            f"collision={bool(collision)} success={bool(success)}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
