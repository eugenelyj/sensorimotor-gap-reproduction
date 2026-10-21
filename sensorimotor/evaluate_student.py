from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from sensorimotor.config import EnvironmentConfig
from sensorimotor.models import StudentPolicy
from sensorimotor.rendering import BinaryGapRenderer
from sensorimotor.simulator import VectorizedGapEnv
from sensorimotor.train_student import student_proprioception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a distilled image policy.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--angles", type=float, nargs="*", default=[0, 30, 60, 80, 90])
    parser.add_argument("--randomized-masks", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_angle(
    student: StudentPolicy,
    environment_config: EnvironmentConfig,
    device: torch.device,
    episodes: int,
    angle: float,
    randomized_masks: bool,
) -> tuple[float, int, int]:
    config = deepcopy(environment_config)
    config.num_envs = min(episodes, config.num_envs)
    config.informed_reset_probability = 0.0
    config.perturbation_force = False
    config.response_randomization = False
    config.gap_roll_min_deg = angle
    config.gap_roll_max_deg = angle
    environment = VectorizedGapEnv(config, device=device, seed=20_000 + round(angle))
    renderer = BinaryGapRenderer(edge_randomization=randomized_masks)
    hidden = torch.zeros((1, config.num_envs, 96), device=device)
    completed = 0
    successes = 0
    collisions = 0
    while completed < episodes:
        image = renderer.render(environment, randomize=randomized_masks)
        proprioception = student_proprioception(environment)
        output, hidden = student(image, proprioception, hidden)
        _, _, done, info = environment.step(output[:, :4])
        if done.any():
            remaining = episodes - completed
            indices = torch.nonzero(done, as_tuple=False).squeeze(-1)[:remaining]
            completed += indices.numel()
            successes += int(info["success"][indices].sum().item())
            collisions += int(info["collision"][indices].sum().item())
            hidden[:, done] = 0.0
    return successes / completed, successes, collisions


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    student = StudentPolicy().to(device)
    student.load_state_dict(checkpoint["student"])
    student.eval()
    teacher_checkpoint = torch.load(
        checkpoint["config"]["teacher_checkpoint"], map_location="cpu", weights_only=False
    )
    environment_config = EnvironmentConfig(**teacher_checkpoint["config"]["environment"])
    for angle in args.angles:
        rate, successes, collisions = evaluate_angle(
            student,
            environment_config,
            device,
            args.episodes,
            angle,
            args.randomized_masks,
        )
        print(
            f"roll={angle:5.1f} success={rate:.3f} "
            f"({successes}/{args.episodes}) collisions={collisions}"
        )


if __name__ == "__main__":
    main()
