from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from sensorimotor.config import EnvironmentConfig
from sensorimotor.models import ObservationNormalizer, TeacherActorCritic
from sensorimotor.simulator import VectorizedGapEnv


@dataclass
class EvaluationResult:
    episodes: int
    successes: int
    collisions: int
    timeouts: int
    out_of_bounds: int
    mean_return: float

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.episodes, 1)


@torch.no_grad()
def evaluate_policy(
    model: TeacherActorCritic,
    normalizer: ObservationNormalizer,
    environment_config: EnvironmentConfig,
    device: torch.device,
    episodes: int,
    seed: int = 10_001,
    gap_roll_degrees: float | None = None,
    deterministic: bool = True,
) -> EvaluationResult:
    config = deepcopy(environment_config)
    config.num_envs = min(episodes, config.num_envs)
    config.informed_reset_probability = 0.0
    config.perturbation_force = False
    config.response_randomization = False
    if gap_roll_degrees is not None:
        config.gap_roll_min_deg = gap_roll_degrees
        config.gap_roll_max_deg = gap_roll_degrees
    environment = VectorizedGapEnv(config, device=device, seed=seed)
    observation = environment.observe()
    completed = 0
    successes = 0
    collisions = 0
    timeouts = 0
    out_of_bounds = 0
    returns: list[torch.Tensor] = []
    while completed < episodes:
        normalized = normalizer.normalize(observation)
        output = model.act(normalized, deterministic=deterministic)
        observation, _, done, info = environment.step(output.action)
        if done.any():
            remaining = episodes - completed
            done_indices = torch.nonzero(done, as_tuple=False).squeeze(-1)[:remaining]
            completed += done_indices.numel()
            successes += int(info["success"][done_indices].sum().item())
            collisions += int(info["collision"][done_indices].sum().item())
            timeouts += int(info["timeout"][done_indices].sum().item())
            out_of_bounds += int(info["out_of_bounds"][done_indices].sum().item())
            returns.append(info["episode_return"][done_indices].detach().cpu())
    all_returns = torch.cat(returns)
    return EvaluationResult(
        episodes=completed,
        successes=successes,
        collisions=collisions,
        timeouts=timeouts,
        out_of_bounds=out_of_bounds,
        mean_return=float(all_returns.mean().item()),
    )
