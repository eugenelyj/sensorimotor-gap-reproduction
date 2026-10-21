import torch

from sensorimotor.math import quaternion_from_euler, quaternion_integrate, quaternion_to_matrix


def test_identity_quaternion_rotation() -> None:
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rotation = quaternion_to_matrix(quaternion)
    torch.testing.assert_close(rotation, torch.eye(3).unsqueeze(0))


def test_body_rate_integration_rotates_about_x() -> None:
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rate = torch.tensor([[torch.pi / 2, 0.0, 0.0]])
    integrated = quaternion_integrate(quaternion, rate, 1.0)
    expected = quaternion_from_euler(
        torch.tensor([torch.pi / 2]), torch.tensor([0.0]), torch.tensor([0.0])
    )
    torch.testing.assert_close(integrated, expected, atol=1e-6, rtol=1e-6)
