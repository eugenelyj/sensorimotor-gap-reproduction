import torch

from sensorimotor.geometry import gap_intersection_status, gap_rotation
from sensorimotor.math import quaternion_from_euler, quaternion_to_matrix


def collision_for(body_roll: float, gap_roll_angle: float) -> bool:
    position = torch.zeros((1, 3))
    center = torch.zeros((1, 3))
    body_quaternion = quaternion_from_euler(
        torch.tensor([body_roll]), torch.tensor([0.0]), torch.tensor([0.0])
    )
    body_rotation = quaternion_to_matrix(body_quaternion)
    collider_half_size = torch.tensor([0.17, 0.17, 0.055])
    collision, _, _ = gap_intersection_status(
        position,
        body_rotation,
        center,
        gap_rotation(torch.tensor([gap_roll_angle])),
        collider_half_size,
        gap_half_width=0.30,
        gap_half_height=0.10,
    )
    return bool(collision.item())


def test_aligned_hovering_body_fits_level_gap() -> None:
    assert not collision_for(0.0, 0.0)


def test_misaligned_body_collides_with_level_gap() -> None:
    assert collision_for(torch.pi / 2, 0.0)


def test_body_aligned_with_rolled_gap_fits() -> None:
    assert not collision_for(torch.pi / 2, torch.pi / 2)
