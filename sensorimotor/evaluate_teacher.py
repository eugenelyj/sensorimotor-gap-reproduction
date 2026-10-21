from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sensorimotor.config import EnvironmentConfig
from sensorimotor.evaluation import evaluate_policy
from sensorimotor.models import ObservationNormalizer, TeacherActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a privileged teacher checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--angles", type=float, nargs="*", default=[0, 30, 60, 80, 90])
    parser.add_argument(
        "--random-angles",
        action="store_true",
        help="Evaluate the configured random roll distribution instead of fixed angles.",
    )
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    environment_config = EnvironmentConfig(**checkpoint["config"]["environment"])
    model = TeacherActorCritic().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = ObservationNormalizer().to(device)
    normalizer.load_state_dict(checkpoint["normalizer"])
    angles: list[float | None] = [None] if args.random_angles else args.angles
    for angle in angles:
        result = evaluate_policy(
            model,
            normalizer,
            environment_config,
            device,
            episodes=args.episodes,
            gap_roll_degrees=angle,
            deterministic=not args.stochastic,
        )
        print(
            f"roll={'random' if angle is None else f'{angle:5.1f}'} "
            f"episodes={result.episodes} "
            f"success={result.success_rate:.3f} collisions={result.collisions} "
            f"timeouts={result.timeouts} out_of_bounds={result.out_of_bounds} "
            f"return={result.mean_return:.3f}"
        )


if __name__ == "__main__":
    main()
