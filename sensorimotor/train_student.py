from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from sensorimotor.config import EnvironmentConfig
from sensorimotor.models import ObservationNormalizer, StudentPolicy, TeacherActorCritic
from sensorimotor.rendering import BinaryGapRenderer
from sensorimotor.simulator import VectorizedGapEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the teacher with on-policy DAgger.")
    parser.add_argument("--config", default="configs/distill_single_gap_roll.yaml")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gap-roll-range", type=float, nargs=2)
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def student_proprioception(environment: VectorizedGapEnv) -> torch.Tensor:
    oracle = environment.observe().state_action
    return torch.cat((oracle[:, 3:5], oracle[:, 6:10]), dim=-1)


def save_checkpoint(
    path: Path,
    student: StudentPolicy,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "update": update,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.teacher_checkpoint:
        config["teacher_checkpoint"] = args.teacher_checkpoint
    if args.updates is not None:
        config["updates"] = args.updates
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.gap_roll_range is not None:
        config["gap_roll_range"] = list(args.gap_roll_range)
    if args.run_dir:
        config["run_dir"] = args.run_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.smoke:
        config.update(
            {
                "num_envs": 8,
                "rollout_steps": 8,
                "environment_minibatch_size": 4,
                "learning_epochs": 1,
                "updates": 2,
                "run_dir": "runs/smoke_student",
                "checkpoint_interval": 1,
            }
        )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(config["device"]))
    teacher_checkpoint = torch.load(
        str(config["teacher_checkpoint"]), map_location=device, weights_only=False
    )
    environment_config = EnvironmentConfig(**teacher_checkpoint["config"]["environment"])
    environment_config.num_envs = int(config["num_envs"])
    if "gap_roll_range" in config:
        environment_config.gap_roll_min_deg = float(config["gap_roll_range"][0])
        environment_config.gap_roll_max_deg = float(config["gap_roll_range"][1])
    environment_config.informed_reset_probability = 0.0
    environment_config.perturbation_probability = 0.1
    environment_config.perturbation_max_accel[0] = 1.5
    environment = VectorizedGapEnv(environment_config, device=device, seed=seed)

    teacher = TeacherActorCritic().to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    normalizer = ObservationNormalizer().to(device)
    normalizer.load_state_dict(teacher_checkpoint["normalizer"])
    student = StudentPolicy().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config["learning_rate"]))
    start_update = 0
    if args.resume:
        student_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(student_checkpoint["student"])
        optimizer.load_state_dict(student_checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config["learning_rate"])
        start_update = int(student_checkpoint["update"])
    renderer = BinaryGapRenderer()
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    writer = SummaryWriter(run_dir)

    steps = int(config["rollout_steps"])
    count = int(config["num_envs"])
    images = torch.empty((steps, count, 1, 256, 320), device=device, dtype=torch.uint8)
    proprioception = torch.empty((steps, count, 6), device=device)
    targets = torch.empty((steps, count, 5), device=device)
    episode_starts = torch.zeros((steps, count), device=device, dtype=torch.bool)
    hidden = torch.zeros((1, count, 96), device=device)
    total_samples = start_update * steps * count
    start_time = time.perf_counter()

    for update in range(start_update + 1, int(config["updates"]) + 1):
        episode_successes = 0
        episode_count = 0
        rollout_initial_hidden = hidden.detach().clone()
        episode_starts.zero_()
        for step in range(steps):
            image = renderer.render(environment, randomize=True)
            proprio = student_proprioception(environment)
            oracle = normalizer.normalize(environment.observe())
            with torch.no_grad():
                teacher_action = teacher.actor.distribution(
                    oracle.gap_points, oracle.state_action
                ).mean
                termination = environment.termination_label()
                student_output, hidden = student(image, proprio, hidden)
                action = student_output[:, :4]
            images[step].copy_(image.to(torch.uint8))
            proprioception[step].copy_(proprio)
            targets[step, :, :4].copy_(teacher_action)
            targets[step, :, 4].copy_(termination)
            _, _, done, info = environment.step(action)
            if done.any():
                hidden[:, done] = 0.0
                if step + 1 < steps:
                    episode_starts[step + 1].copy_(done)
                episode_successes += int(info["success"].sum().item())
                episode_count += int(done.sum().item())

        losses: list[float] = []
        action_losses: list[float] = []
        termination_losses: list[float] = []
        environment_batch = int(config["environment_minibatch_size"])
        for _ in range(int(config["learning_epochs"])):
            permutation = torch.randperm(count, device=device)
            for start in range(0, count, environment_batch):
                indices = permutation[start : start + environment_batch]
                output, _ = student.forward_sequence(
                    images[:, indices].float(),
                    proprioception[:, indices],
                    rollout_initial_hidden[:, indices],
                    episode_starts[:, indices],
                )
                action_loss = functional.mse_loss(output[..., :4], targets[:, indices, :4])
                termination_loss = functional.mse_loss(
                    torch.tanh(output[..., 4]), targets[:, indices, 4]
                )
                loss = (
                    float(config["action_loss_weight"]) * action_loss
                    + float(config["termination_loss_weight"]) * termination_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(student.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                losses.append(loss.item())
                action_losses.append(action_loss.item())
                termination_losses.append(termination_loss.item())

        total_samples += steps * count
        mean_loss = sum(losses) / len(losses)
        writer.add_scalar("train/loss", mean_loss, total_samples)
        writer.add_scalar("train/action_loss", sum(action_losses) / len(action_losses), total_samples)
        writer.add_scalar(
            "train/termination_loss",
            sum(termination_losses) / len(termination_losses),
            total_samples,
        )
        if episode_count:
            writer.add_scalar(
                "train/on_policy_success_rate", episode_successes / episode_count, total_samples
            )
        print(
            json.dumps(
                {
                    "update": update,
                    "samples": total_samples,
                    "loss": mean_loss,
                    "success_rate": episode_successes / max(episode_count, 1),
                    "elapsed_seconds": round(time.perf_counter() - start_time, 1),
                }
            ),
            flush=True,
        )
        if update % int(config["checkpoint_interval"]) == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"student_{update}.pt",
                student,
                optimizer,
                config,
                update,
            )

    final_path = run_dir / "checkpoints" / "student_final.pt"
    save_checkpoint(final_path, student, optimizer, config, int(config["updates"]))
    writer.close()
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
