import torch

from scripts.generate_ir_trajectory_library import validate


def test_validation_stops_after_successful_traversal() -> None:
    positions = torch.tensor(
        [
            [[-1.0, 0.0, 0.0]],
            [[0.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.2]],
        ]
    )
    quaternions = torch.zeros((3, 1, 4))
    quaternions[..., 0] = 1.0
    collisions, successes = validate(positions, quaternions, torch.zeros(1))

    assert successes.tolist() == [True]
    assert collisions.tolist() == [False]
