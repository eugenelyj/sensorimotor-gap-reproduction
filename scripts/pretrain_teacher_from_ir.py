from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from sensorimotor.config import load_config
from sensorimotor.evaluation import evaluate_policy
from sensorimotor.geometry import gap_rotation, make_rectangular_gap_points
from sensorimotor.math import matrix_to_roll_pitch_yaw, quaternion_to_matrix
from sensorimotor.models import ObservationNormalizer, TeacherActorCritic
from sensorimotor.ppo import PPOOptimizer
from sensorimotor.simulator import Observation
from sensorimotor.train_teacher import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a teacher by behavior cloning a validated IR library."
    )
    parser.add_argument("--config", default="configs/single_gap_roll.yaml")
    parser.add_argument(
        "--trajectory", default="artifacts/ir/optimized_single_gap_roll_grid_short.pt"
    )
    parser.add_argument("--output", default="artifacts/teachers/ir_bc_teacher.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--approach-steps", type=int, default=105)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--action-log-std", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def normalized_action(action: torch.Tensor) -> torch.Tensor:
    result = action.clone()
    result[..., 0] = (action[..., 0] - 13.0) / 7.0
    result[..., 1:] = action[..., 1:] / 6.0
    return result


def policy_action_parameters(action: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(action)
    result[..., 0] = torch.atanh(
        ((action[..., 0] - 13.0) / 7.0).clamp(-0.999, 0.999)
    )
    result[..., 1:] = torch.atanh(
        (action[..., 1:] / 6.0).clamp(-0.999, 0.999)
    )
    return result


def build_dataset(
    library: dict[str, object], device: torch.device, approach_steps: int
) -> tuple[Observation, torch.Tensor]:
    positions = library["positions"].to(device)
    velocities = library["velocities"].to(device)
    quaternions = library["quaternions"].to(device)
    thrust = library["thrust"].to(device)
    body_rate = library["body_rate"].to(device)
    angles = library["angles"].to(device)
    angle_count, path_count, state_count = positions.shape[:3]
    count = angle_count * path_count
    rotations = quaternion_to_matrix(quaternions).flatten(0, 1)
    flat_positions = positions.flatten(0, 1)
    flat_velocities = velocities.flatten(0, 1)
    repeated_angles = angles[:, None].expand(-1, path_count).reshape(-1)
    gap_rotations = gap_rotation(repeated_angles)
    local_points = make_rectangular_gap_points(0.60, 0.20, 8, device)
    world_points = torch.einsum("nij,pj->npi", gap_rotations, local_points)
    relative_points = world_points[:, None] - flat_positions[:, :, None]
    body_points = torch.einsum(
        "nsij,nspj->nspi", rotations.transpose(-1, -2), relative_points
    )
    body_velocity = torch.einsum(
        "nsij,nsj->nsi", rotations.transpose(-1, -2), flat_velocities
    )
    roll, pitch, _ = matrix_to_roll_pitch_yaw(rotations)

    physical_actions = torch.cat((thrust[..., None], body_rate), dim=-1).flatten(0, 1)
    previous_action = torch.empty((count, state_count, 4), device=device)
    previous_action[:, 0] = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device)
    previous_action[:, 1:] = physical_actions
    target_action = torch.cat((physical_actions, physical_actions[:, -1:]), dim=1)
    normalized_previous = normalized_action(previous_action)
    state_action = torch.cat(
        (
            body_velocity,
            roll[..., None],
            pitch[..., None],
            normalized_previous[..., :1],
            normalized_previous,
        ),
        dim=-1,
    )
    observation = Observation(
        gap_points=body_points.flatten(0, 1),
        state_action=state_action.flatten(0, 1),
    )
    targets = policy_action_parameters(target_action).flatten(0, 1)
    settling_steps = min(30, approach_steps // 4)
    settling_start = approach_steps - settling_steps
    state_indices = torch.arange(state_count, device=device)
    valid = (state_indices < settling_start) | (state_indices >= approach_steps)
    valid = valid[None].expand(count, -1).clone()
    center_paths = library["start_offsets"].to(device).abs().sum(dim=-1) == 0
    center_paths = center_paths[None].expand(angle_count, -1).reshape(-1)
    valid[center_paths, :approach_steps] = False
    valid = valid.flatten()
    return Observation(
        gap_points=observation.gap_points[valid],
        state_action=observation.state_action[valid],
    ), targets[valid]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)
    config.device = str(device)
    config.seed = args.seed
    config.environment.informed_trajectory_path = args.trajectory
    library = torch.load(args.trajectory, map_location=device, weights_only=False)
    observation, targets = build_dataset(library, device, args.approach_steps)
    normalizer = ObservationNormalizer().to(device)
    normalizer.update(observation)
    normalized = normalizer.normalize(observation)
    model = TeacherActorCritic().to(device)
    model.actor.log_standard_deviation.data.fill_(args.action_log_std)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)
    sample_count = targets.shape[0]
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        losses = []
        for start in range(0, sample_count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            mean = model.actor.distribution(
                normalized.gap_points[indices], normalized.state_action[indices]
            ).mean
            loss = functional.mse_loss(mean, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 3.0)
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} loss={sum(losses) / len(losses):.6f}", flush=True)

    ppo_optimizer = PPOOptimizer(model, config.ppo)
    output = Path(args.output)
    save_checkpoint(output, config, model, normalizer, ppo_optimizer, samples=0)
    result = evaluate_policy(
        model,
        normalizer,
        config.environment,
        device,
        episodes=4096,
        seed=args.seed + 10_000,
    )
    print(
        f"standard_start_success={result.success_rate:.6f} "
        f"collisions={result.collisions} out_of_bounds={result.out_of_bounds}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
