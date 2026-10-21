from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    num_envs: int = 2048
    control_dt: float = 1.0 / 60.0
    episode_seconds: float = 5.0
    gap_width: float = 0.60
    gap_height: float = 0.20
    gap_roll_min_deg: float = 0.0
    gap_roll_max_deg: float = 90.0
    start_x_range: list[float] = field(default_factory=lambda: [-4.5, -2.7])
    start_lateral_offset: float = 1.0
    start_vertical_offset: float = 0.45
    collider_size: list[float] = field(default_factory=lambda: [0.34, 0.34, 0.11])
    thrust_min: float = 6.0
    thrust_max: float = 20.0
    bodyrate_max: float = 6.0
    speed_limit: float = 4.0
    speed_reward_mode: str = "overspeed_penalty"
    action_penalty_mode: str = "normalized"
    attitude_penalty_mode: str = "paper"
    failure_termination_mode: str = "immediate"
    informed_reset_probability: float = 0.5
    informed_trajectory_path: str | None = None
    perturbation_probability: float = 0.1
    perturbation_steps: int = 20
    perturbation_max_accel: list[float] = field(default_factory=lambda: [2.0, 1.0, 1.0])
    response_randomization: bool = True
    perturbation_force: bool = True
    air_drag_linear: list[float] = field(default_factory=lambda: [0.08, 0.08, 0.12])
    air_drag_quadratic: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.03])
    action_delay_steps: list[int] = field(default_factory=lambda: [3, 1, 1, 1])
    action_average_window: int = 2
    response_scale_randomization: list[float] = field(
        default_factory=lambda: [0.10, 0.10, 0.10, 0.10]
    )
    integration_substeps: int = 2

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_seconds / self.control_dt)


@dataclass
class PPOConfig:
    rollout_steps: int = 200
    total_samples: int = 1_000_000_000
    minibatch_size: int = 100_000
    update_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    max_grad_norm: float = 1.5
    target_kl: float = 0.03
    initial_action_log_std: float = -0.5
    minimum_action_log_std: float | None = None
    normalizer_freeze_samples: int | None = None
    eval_interval_samples: int = 8_192_000
    checkpoint_interval_samples: int = 50_000_000
    eval_episodes: int = 4096


@dataclass
class LoggingConfig:
    run_dir: str = "runs/single_gap_roll_ir"


@dataclass
class ExperimentConfig:
    seed: int = 1
    device: str = "cuda:0"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    return ExperimentConfig(
        seed=raw.get("seed", 1),
        device=raw.get("device", "cuda:0"),
        environment=EnvironmentConfig(**raw.get("environment", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config.to_dict(), config_file, sort_keys=False)
