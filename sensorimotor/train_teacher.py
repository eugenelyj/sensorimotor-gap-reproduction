from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sensorimotor.config import ExperimentConfig, load_config, save_config
from sensorimotor.evaluation import evaluate_policy
from sensorimotor.models import ObservationNormalizer, TeacherActorCritic
from sensorimotor.ppo import PPOOptimizer, RolloutStorage
from sensorimotor.simulator import Observation, VectorizedGapEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the privileged point-observation teacher.")
    parser.add_argument("--config", default="configs/single_gap_roll.yaml")
    parser.add_argument("--total-samples", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--start-x-range", type=float, nargs=2)
    parser.add_argument("--gap-roll-range", type=float, nargs=2)
    parser.add_argument("--start-lateral-offset", type=float)
    parser.add_argument("--start-vertical-offset", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--initial-action-log-std", type=float)
    parser.add_argument("--minimum-action-log-std", type=float)
    parser.add_argument("--normalizer-freeze-samples", type=int)
    parser.add_argument(
        "--speed-reward-mode",
        choices=("overspeed_penalty", "below_limit_reward"),
    )
    parser.add_argument(
        "--action-penalty-mode",
        choices=("paper", "physical", "normalized"),
    )
    parser.add_argument(
        "--attitude-penalty-mode",
        choices=("time_scaled", "paper"),
    )
    parser.add_argument(
        "--failure-termination-mode",
        choices=("immediate", "traversal_or_horizon"),
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--informed-trajectory")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.total_samples is not None:
        config.ppo.total_samples = args.total_samples
    if args.num_envs is not None:
        config.environment.num_envs = args.num_envs
    if args.rollout_steps is not None:
        config.ppo.rollout_steps = args.rollout_steps
    if args.start_x_range is not None:
        config.environment.start_x_range = list(args.start_x_range)
    if args.gap_roll_range is not None:
        config.environment.gap_roll_min_deg = args.gap_roll_range[0]
        config.environment.gap_roll_max_deg = args.gap_roll_range[1]
    if args.start_lateral_offset is not None:
        config.environment.start_lateral_offset = args.start_lateral_offset
    if args.start_vertical_offset is not None:
        config.environment.start_vertical_offset = args.start_vertical_offset
    if args.entropy_coefficient is not None:
        config.ppo.entropy_coefficient = args.entropy_coefficient
    if args.actor_learning_rate is not None:
        config.ppo.actor_learning_rate = args.actor_learning_rate
    if args.critic_learning_rate is not None:
        config.ppo.critic_learning_rate = args.critic_learning_rate
    if args.initial_action_log_std is not None:
        config.ppo.initial_action_log_std = args.initial_action_log_std
    if args.minimum_action_log_std is not None:
        config.ppo.minimum_action_log_std = args.minimum_action_log_std
    if args.normalizer_freeze_samples is not None:
        config.ppo.normalizer_freeze_samples = args.normalizer_freeze_samples
    if args.speed_reward_mode is not None:
        config.environment.speed_reward_mode = args.speed_reward_mode
    if args.action_penalty_mode is not None:
        config.environment.action_penalty_mode = args.action_penalty_mode
    if args.attitude_penalty_mode is not None:
        config.environment.attitude_penalty_mode = args.attitude_penalty_mode
    if args.failure_termination_mode is not None:
        config.environment.failure_termination_mode = args.failure_termination_mode
    if args.run_dir is not None:
        config.logging.run_dir = args.run_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.informed_trajectory is not None:
        config.environment.informed_trajectory_path = args.informed_trajectory
    if args.smoke:
        config.environment.num_envs = 128
        config.ppo.rollout_steps = 64
        config.ppo.total_samples = 128 * 64 * 2
        config.ppo.minibatch_size = 4096
        config.ppo.update_epochs = 2
        config.ppo.eval_episodes = 256
        config.ppo.eval_interval_samples = config.ppo.total_samples
        config.ppo.checkpoint_interval_samples = config.ppo.total_samples
        config.logging.run_dir = "runs/smoke_teacher"


def save_checkpoint(
    path: Path,
    config: ExperimentConfig,
    model: TeacherActorCritic,
    normalizer: ObservationNormalizer,
    optimizer: PPOOptimizer,
    samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "samples": samples,
        },
        path,
    )


def train(
    config: ExperimentConfig,
    resume: str | None = None,
    model_seed: int | None = None,
) -> Path:
    set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_dir = Path(config.logging.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    if model_seed is not None:
        (run_dir / "model_seed.txt").write_text(f"{model_seed}\n", encoding="utf-8")
    writer = SummaryWriter(log_dir=run_dir)
    environment = VectorizedGapEnv(config.environment, device=device, seed=config.seed)
    if model_seed is not None:
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
    model = TeacherActorCritic().to(device)
    if model_seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    model.actor.log_standard_deviation.data.fill_(config.ppo.initial_action_log_std)
    normalizer = ObservationNormalizer().to(device)
    optimizer = PPOOptimizer(model, config.ppo)
    samples = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        samples = int(checkpoint["samples"])

    storage = RolloutStorage(
        config.ppo.rollout_steps,
        config.environment.num_envs,
        device,
    )
    observation = environment.observe()
    last_evaluation = samples
    last_checkpoint = samples
    training_start = time.perf_counter()
    episode_successes = 0
    episode_count = 0
    interval_successes = 0
    interval_episode_count = 0
    interval_standard_successes = 0
    interval_standard_episode_count = 0

    while samples < config.ppo.total_samples:
        rollout_start = time.perf_counter()
        reward_info: dict[str, torch.Tensor] = {}
        for step in range(config.ppo.rollout_steps):
            if (
                config.ppo.normalizer_freeze_samples is None
                or samples < config.ppo.normalizer_freeze_samples
            ):
                normalizer.update(observation)
            normalized = normalizer.normalize(observation)
            with torch.no_grad():
                output = model.act(normalized)
            storage.points[step].copy_(normalized.gap_points)
            storage.state_action[step].copy_(normalized.state_action)
            storage.actions[step].copy_(output.action)
            storage.log_probabilities[step].copy_(output.log_probability)
            storage.values[step].copy_(output.value)
            observation, reward, done, info = environment.step(output.action)
            storage.rewards[step].copy_(reward)
            storage.dones[step].copy_(done)
            reward_info = {key: value for key, value in info.items() if key.startswith("reward/")}
            if done.any():
                completed_episodes = int(done.sum().item())
                successful_episodes = int(info["success"].sum().item())
                episode_count += completed_episodes
                episode_successes += successful_episodes
                interval_episode_count += completed_episodes
                interval_successes += successful_episodes
                standard_done = done & ~info["informed_reset"]
                interval_standard_episode_count += int(standard_done.sum().item())
                interval_standard_successes += int(
                    (info["success"] & standard_done).sum().item()
                )

        with torch.no_grad():
            normalized_next = normalizer.normalize(observation)
            next_value = model.critic(
                normalized_next.gap_points, normalized_next.state_action
            )
        storage.compute_returns_and_advantages(
            next_value,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )
        rollout_seconds = time.perf_counter() - rollout_start
        update_start = time.perf_counter()
        statistics = optimizer.update(storage)
        update_seconds = time.perf_counter() - update_start
        new_samples = config.environment.num_envs * config.ppo.rollout_steps
        samples += new_samples

        writer.add_scalar("train/actor_loss", statistics.actor_loss, samples)
        writer.add_scalar("train/critic_loss", statistics.critic_loss, samples)
        writer.add_scalar("train/entropy", statistics.entropy, samples)
        writer.add_scalar("train/approximate_kl", statistics.approximate_kl, samples)
        writer.add_scalar("train/clip_fraction", statistics.clip_fraction, samples)
        writer.add_scalar("train/epochs", statistics.epochs, samples)
        writer.add_scalar(
            "train/action_log_std_mean",
            model.actor.log_standard_deviation.mean().item(),
            samples,
        )
        writer.add_scalar(
            "train/normalizer_points_count", normalizer.points.count.item(), samples
        )
        writer.add_scalar("train/rollout_samples_per_second", new_samples / rollout_seconds, samples)
        writer.add_scalar("train/update_samples_per_second", new_samples / update_seconds, samples)
        if episode_count:
            writer.add_scalar("train/online_success_rate", episode_successes / episode_count, samples)
        for key, value in reward_info.items():
            writer.add_scalar(key, value.item(), samples)

        elapsed = time.perf_counter() - training_start
        print(
            json.dumps(
                {
                    "samples": samples,
                    "success_rate": episode_successes / max(episode_count, 1),
                    "actor_loss": statistics.actor_loss,
                    "critic_loss": statistics.critic_loss,
                    "kl": statistics.approximate_kl,
                    "rollout_sps": round(new_samples / rollout_seconds),
                    "update_sps": round(new_samples / update_seconds),
                    "elapsed_seconds": round(elapsed, 1),
                }
            ),
            flush=True,
        )
        episode_successes = 0
        episode_count = 0

        if samples - last_evaluation >= config.ppo.eval_interval_samples:
            on_policy_success_rate = interval_successes / max(interval_episode_count, 1)
            standard_start_success_rate = interval_standard_successes / max(
                interval_standard_episode_count, 1
            )
            result = evaluate_policy(
                model,
                normalizer,
                config.environment,
                device,
                episodes=config.ppo.eval_episodes,
                seed=config.seed + samples,
            )
            writer.add_scalar("evaluation/success_rate", result.success_rate, samples)
            writer.add_scalar(
                "evaluation/on_policy_success_rate", on_policy_success_rate, samples
            )
            writer.add_scalar(
                "evaluation/on_policy_standard_start_success_rate",
                standard_start_success_rate,
                samples,
            )
            writer.add_scalar("evaluation/mean_return", result.mean_return, samples)
            print(
                json.dumps(
                    {
                        "evaluation_samples": samples,
                        "episodes": result.episodes,
                        "success_rate": result.success_rate,
                        "on_policy_success_rate": on_policy_success_rate,
                        "on_policy_standard_start_success_rate": standard_start_success_rate,
                        "collisions": result.collisions,
                        "timeouts": result.timeouts,
                        "out_of_bounds": result.out_of_bounds,
                    }
                ),
                flush=True,
            )
            last_evaluation = samples
            interval_successes = 0
            interval_episode_count = 0
            interval_standard_successes = 0
            interval_standard_episode_count = 0

        if samples - last_checkpoint >= config.ppo.checkpoint_interval_samples:
            save_checkpoint(
                run_dir / "checkpoints" / f"teacher_{samples}.pt",
                config,
                model,
                normalizer,
                optimizer,
                samples,
            )
            last_checkpoint = samples

    final_path = run_dir / "checkpoints" / "teacher_final.pt"
    save_checkpoint(final_path, config, model, normalizer, optimizer, samples)
    writer.close()
    return final_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    checkpoint = train(config, resume=args.resume, model_seed=args.model_seed)
    print(f"Saved final checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
