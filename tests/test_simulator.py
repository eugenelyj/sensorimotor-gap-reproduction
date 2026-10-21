import torch

from scripts.pretrain_teacher_from_ir import policy_action_parameters
from sensorimotor.config import EnvironmentConfig
from sensorimotor.models import StudentPolicy, TeacherActorCritic
from sensorimotor.rendering import BinaryGapRenderer, SyntheticRGBRenderer
from sensorimotor.simulator import VectorizedGapEnv


def small_config() -> EnvironmentConfig:
    return EnvironmentConfig(num_envs=16, informed_reset_probability=0.5)


def test_observation_and_teacher_shapes() -> None:
    environment = VectorizedGapEnv(small_config(), device="cpu", seed=3)
    observation = environment.observe()
    assert observation.gap_points.shape == (16, 32, 3)
    assert observation.state_action.shape == (16, 10)
    assert torch.isfinite(observation.gap_points).all()
    model = TeacherActorCritic()
    output = model.act(observation)
    assert output.action.shape == (16, 4)
    assert output.value.shape == (16,)


def test_environment_step_is_finite() -> None:
    environment = VectorizedGapEnv(small_config(), device="cpu", seed=4)
    action = torch.zeros((16, 4))
    observation, reward, done, _ = environment.step(action)
    assert torch.isfinite(observation.gap_points).all()
    assert torch.isfinite(observation.state_action).all()
    assert torch.isfinite(reward).all()
    assert done.shape == (16,)


def test_traversal_or_horizon_mode_does_not_end_on_bounds_failure() -> None:
    immediate_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=1, failure_termination_mode="immediate"),
        device="cpu",
    )
    horizon_environment = VectorizedGapEnv(
        EnvironmentConfig(
            num_envs=1,
            failure_termination_mode="traversal_or_horizon",
        ),
        device="cpu",
    )
    immediate_environment.position[:, 2] = 0.0
    horizon_environment.position[:, 2] = 0.0
    hover_normalized = torch.tensor([[(9.81 - 13.0) / 7.0, 0.0, 0.0, 0.0]])
    hover_action = torch.atanh(hover_normalized)

    _, _, immediate_done, _ = immediate_environment.step(hover_action)
    _, _, horizon_done, horizon_info = horizon_environment.step(hover_action)

    assert immediate_done.item()
    assert not horizon_done.item()
    assert horizon_info["out_of_bounds"].item()


def test_speed_reward_modes_match_both_paper_interpretations() -> None:
    speed = torch.tensor([0.0, 4.0, 5.0])
    penalty_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=3, speed_reward_mode="overspeed_penalty"),
        device="cpu",
    )
    literal_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=3, speed_reward_mode="below_limit_reward"),
        device="cpu",
    )

    assert torch.allclose(
        penalty_environment._speed_reward(speed),
        torch.tensor([0.0, 0.0, 0.05 * (1.0 - torch.exp(torch.tensor(1.0)))]),
    )
    assert torch.allclose(
        literal_environment._speed_reward(speed),
        torch.tensor([0.05 * (1.0 - torch.exp(torch.tensor(-4.0))), 0.0, 0.0]),
    )


def test_normalized_action_penalties_scale_each_control_range() -> None:
    paper_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=1, action_penalty_mode="paper"),
        device="cpu",
    )
    physical_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=1, action_penalty_mode="physical"),
        device="cpu",
    )
    normalized_environment = VectorizedGapEnv(
        EnvironmentConfig(num_envs=1, action_penalty_mode="normalized"),
        device="cpu",
    )
    previous_action = torch.tensor([[9.81, 0.0, 0.0, 0.0]])
    command = torch.tensor([[16.81, 6.0, -6.0, 6.0]])

    paper_magnitude, paper_variation = paper_environment._action_rewards(
        command, previous_action
    )
    physical_magnitude, physical_variation = physical_environment._action_rewards(
        command, previous_action
    )
    normalized_magnitude, normalized_variation = normalized_environment._action_rewards(
        command, previous_action
    )

    torch.testing.assert_close(paper_magnitude, torch.tensor([-1.3324]))
    torch.testing.assert_close(paper_variation, torch.tensor([-0.57]))
    torch.testing.assert_close(physical_magnitude, torch.tensor([-0.94]))
    torch.testing.assert_close(physical_variation, torch.tensor([-0.57]))
    torch.testing.assert_close(normalized_magnitude, torch.tensor([-0.15]))
    torch.testing.assert_close(normalized_variation, torch.tensor([-0.085]))


def test_behavior_cloning_action_targets_round_trip_through_environment() -> None:
    environment = VectorizedGapEnv(EnvironmentConfig(num_envs=2), device="cpu")
    physical_action = torch.tensor(
        [[9.81, 0.0, 0.0, 0.0], [16.0, 3.0, -2.0, 1.0]]
    )

    reconstructed = environment.action_to_physical(
        policy_action_parameters(physical_action)
    )

    torch.testing.assert_close(reconstructed, physical_action)


def test_student_architecture_matches_paper_dimensions() -> None:
    model = StudentPolicy()
    image = torch.zeros((2, 1, 256, 320))
    proprioception = torch.zeros((2, 6))
    output, hidden = model(image, proprioception)
    assert output.shape == (2, 5)
    assert hidden.shape == (1, 2, 96)


def test_student_sequence_resets_hidden_state_at_episode_boundaries() -> None:
    model = StudentPolicy().eval()
    images = torch.zeros((3, 2, 1, 256, 320))
    proprioception = torch.zeros((3, 2, 6))
    initial_hidden = torch.ones((1, 2, 96))
    episode_starts = torch.zeros((3, 2), dtype=torch.bool)
    episode_starts[0, 0] = True

    output, _ = model.forward_sequence(
        images, proprioception, initial_hidden, episode_starts
    )
    reset_output, _ = model.forward_sequence(
        images[:, :1], proprioception[:, :1], torch.zeros((1, 1, 96))
    )
    continued_output, _ = model.forward_sequence(
        images[:, 1:], proprioception[:, 1:], initial_hidden[:, 1:]
    )

    assert torch.allclose(output[:, :1], reset_output, atol=1e-6)
    assert torch.allclose(output[:, 1:], continued_output, atol=1e-6)


def test_binary_renderer_shape_and_values() -> None:
    environment = VectorizedGapEnv(EnvironmentConfig(num_envs=2), device="cpu", seed=5)
    renderer = BinaryGapRenderer(height=64, width=80, edge_randomization=False)
    mask = renderer.render(environment, randomize=False)
    assert mask.shape == (2, 1, 64, 80)
    assert ((mask == 0) | (mask == 1)).all()


def test_synthetic_rgb_renderer_shape_and_values() -> None:
    environment = VectorizedGapEnv(EnvironmentConfig(num_envs=2), device="cpu", seed=5)
    renderer = SyntheticRGBRenderer(height=64, width=80, edge_randomization=False)
    image = renderer.render(environment)
    assert image.shape == (2, 3, 64, 80)
    assert torch.isfinite(image).all()
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    assert image.std() > 0.05
