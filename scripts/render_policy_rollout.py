from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sensorimotor.config import EnvironmentConfig
from sensorimotor.math import quaternion_to_matrix
from sensorimotor.models import ObservationNormalizer, StudentPolicy, TeacherActorCritic
from sensorimotor.rendering import BinaryGapRenderer, SyntheticRGBRenderer
from sensorimotor.simulator import VectorizedGapEnv
from sensorimotor.train_student import student_proprioception


@dataclass
class Snapshot:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    rgb: np.ndarray
    mask: np.ndarray
    command: np.ndarray
    status: str


@dataclass
class Episode:
    snapshots: list[Snapshot]
    trajectory: np.ndarray
    gap_center: np.ndarray
    gap_rotation: np.ndarray
    gap_width: float
    gap_height: float
    collider_size: np.ndarray
    seed: int
    success: bool
    terminal_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a successful teacher or student narrow-gap rollout."
    )
    parser.add_argument("policy", choices=("teacher", "student"))
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--angle", type=float, default=80.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-start", type=int, default=20_080)
    parser.add_argument("--max-attempts", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    return parser.parse_args()


def resolve_path(path: str, checkpoint_path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    project_candidate = Path(__file__).resolve().parents[1] / candidate
    if project_candidate.exists():
        return project_candidate
    relative_candidate = checkpoint_path.parent / candidate
    if relative_candidate.exists():
        return relative_candidate
    raise FileNotFoundError(f"Could not find teacher checkpoint: {path}")


def load_policy(
    policy_name: str,
    checkpoint_path: Path,
    device: torch.device,
    teacher_checkpoint_override: str | None,
) -> tuple[EnvironmentConfig, object, ObservationNormalizer | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if policy_name == "teacher":
        config = EnvironmentConfig(**checkpoint["config"]["environment"])
        model = TeacherActorCritic().to(device)
        model.load_state_dict(checkpoint["model"])
        normalizer = ObservationNormalizer().to(device)
        normalizer.load_state_dict(checkpoint["normalizer"])
        model.eval()
        normalizer.eval()
        return config, model, normalizer

    model = StudentPolicy().to(device)
    model.load_state_dict(checkpoint["student"])
    model.eval()
    teacher_path = teacher_checkpoint_override or checkpoint["config"]["teacher_checkpoint"]
    teacher_checkpoint = torch.load(
        resolve_path(str(teacher_path), checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    config = EnvironmentConfig(**teacher_checkpoint["config"]["environment"])
    return config, model, None


def configured_environment(
    source: EnvironmentConfig,
    angle: float,
) -> EnvironmentConfig:
    config = deepcopy(source)
    config.num_envs = 1
    config.informed_reset_probability = 0.0
    config.perturbation_force = False
    config.response_randomization = False
    config.gap_roll_min_deg = angle
    config.gap_roll_max_deg = angle
    return config


def tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def terminal_status(info: dict[str, object]) -> str:
    for key, label in (
        ("success", "SUCCESS"),
        ("collision", "COLLISION"),
        ("out_of_bounds", "OUT OF BOUNDS"),
        ("timeout", "TIMEOUT"),
    ):
        value = info[key]
        if isinstance(value, torch.Tensor) and bool(value[0].item()):
            return label
    return "TERMINATED"


def capture_snapshot(
    environment: VectorizedGapEnv,
    renderer: BinaryGapRenderer,
    rgb_renderer: SyntheticRGBRenderer,
    time: float,
    command: torch.Tensor,
    status: str,
    mask: torch.Tensor | None = None,
) -> Snapshot:
    if mask is None:
        mask = renderer.render(environment, randomize=False)
    rgb = rgb_renderer.render(environment)
    rotation = quaternion_to_matrix(environment.quaternion)
    return Snapshot(
        time=time,
        position=tensor_numpy(environment.position[0]),
        velocity=tensor_numpy(environment.velocity[0]),
        rotation=tensor_numpy(rotation[0]),
        rgb=(tensor_numpy(rgb[0].permute(1, 2, 0)) * 255.0).astype(np.uint8),
        mask=(tensor_numpy(mask[0, 0]) * 255.0).astype(np.uint8),
        command=tensor_numpy(command[0]),
        status=status,
    )


@torch.no_grad()
def rollout(
    policy_name: str,
    model: object,
    normalizer: ObservationNormalizer | None,
    environment_config: EnvironmentConfig,
    renderer: BinaryGapRenderer,
    rgb_renderer: SyntheticRGBRenderer,
    device: torch.device,
    seed: int,
    frame_stride: int,
) -> Episode:
    environment = VectorizedGapEnv(
        environment_config,
        device=device,
        seed=seed,
        auto_reset=False,
    )
    hidden = torch.zeros((1, 1, 96), device=device)
    snapshots: list[Snapshot] = []
    trajectory = [tensor_numpy(environment.position[0])]
    final_info: dict[str, object] | None = None
    final_command = environment.action_response.clone()

    for step in range(environment_config.max_episode_steps + 1):
        if policy_name == "teacher":
            assert isinstance(model, TeacherActorCritic)
            assert normalizer is not None
            image = renderer.render(environment, randomize=False)
            normalized = normalizer.normalize(environment.observe())
            raw_action = model.act(normalized, deterministic=True).action
        else:
            assert isinstance(model, StudentPolicy)
            image = renderer.render(environment, randomize=False)
            proprioception = student_proprioception(environment)
            output, hidden = model(image, proprioception, hidden)
            raw_action = output[:, :4]

        command = environment.action_to_physical(raw_action)
        final_command = command
        if step % frame_stride == 0:
            snapshots.append(
                capture_snapshot(
                    environment,
                    renderer,
                    rgb_renderer,
                    step * environment_config.control_dt,
                    command,
                    "FLYING",
                    image,
                )
            )
        _, _, done, info = environment.step(raw_action)
        trajectory.append(tensor_numpy(environment.position[0]))
        if bool(done[0].item()):
            final_info = info
            break

    if final_info is None:
        raise RuntimeError("Episode exceeded the configured horizon without terminating")
    status = terminal_status(final_info)
    snapshots.append(
        capture_snapshot(
            environment,
            renderer,
            rgb_renderer,
            int(environment.episode_step[0].item()) * environment_config.control_dt,
            final_command,
            status,
        )
    )
    return Episode(
        snapshots=snapshots,
        trajectory=np.stack(trajectory),
        gap_center=tensor_numpy(environment.gap_center[0]),
        gap_rotation=tensor_numpy(environment.gap_rotation_matrix[0]),
        gap_width=environment_config.gap_width,
        gap_height=environment_config.gap_height,
        collider_size=np.asarray(environment_config.collider_size),
        seed=seed,
        success=status == "SUCCESS",
        terminal_status=status,
    )


def gap_corners(episode: Episode) -> np.ndarray:
    half_width = episode.gap_width * 0.5
    half_height = episode.gap_height * 0.5
    local = np.array(
        [
            [0.0, -half_width, -half_height],
            [0.0, half_width, -half_height],
            [0.0, half_width, half_height],
            [0.0, -half_width, half_height],
            [0.0, -half_width, -half_height],
        ]
    )
    return local @ episode.gap_rotation.T + episode.gap_center


def drone_front_corners(snapshot: Snapshot, collider_size: np.ndarray) -> np.ndarray:
    half_y = collider_size[1] * 0.5
    half_z = collider_size[2] * 0.5
    local = np.array(
        [
            [0.0, -half_y, -half_z],
            [0.0, half_y, -half_z],
            [0.0, half_y, half_z],
            [0.0, -half_y, half_z],
            [0.0, -half_y, -half_z],
        ]
    )
    return local @ snapshot.rotation.T + snapshot.position


def render_frame(
    policy_name: str,
    angle: float,
    episode: Episode,
    snapshot_index: int,
) -> np.ndarray:
    snapshot = episode.snapshots[snapshot_index]
    gap = gap_corners(episode)
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#10151c")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.75, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.04,
        right=0.98,
        bottom=0.08,
        top=0.89,
        wspace=0.12,
        hspace=0.27,
    )
    world = figure.add_subplot(grid[:, 0], projection="3d", facecolor="#10151c")
    camera = figure.add_subplot(grid[0, 1], facecolor="black")
    front = figure.add_subplot(grid[1, 1], facecolor="#10151c")

    completed_count = round(
        snapshot_index * (len(episode.trajectory) - 1) / max(len(episode.snapshots) - 1, 1)
    )
    world.plot(
        episode.trajectory[:, 0],
        episode.trajectory[:, 1],
        episode.trajectory[:, 2],
        color="#7f8c99",
        linestyle="--",
        linewidth=1.2,
        alpha=0.45,
    )
    world.plot(
        episode.trajectory[: completed_count + 1, 0],
        episode.trajectory[: completed_count + 1, 1],
        episode.trajectory[: completed_count + 1, 2],
        color="#36a9e1",
        linewidth=3.0,
    )
    world.plot(gap[:, 0], gap[:, 1], gap[:, 2], color="#ffb000", linewidth=4.0)
    world.scatter(*snapshot.position, color="white", s=35, depthshade=False)
    axis_colors = ("#ef476f", "#06d6a0", "#118ab2")
    for axis, color in enumerate(axis_colors):
        endpoint = snapshot.position + 0.34 * snapshot.rotation[:, axis]
        world.plot(
            [snapshot.position[0], endpoint[0]],
            [snapshot.position[1], endpoint[1]],
            [snapshot.position[2], endpoint[2]],
            color=color,
            linewidth=3.0,
        )
    world.set_xlim(-4.8, 0.8)
    world.set_ylim(-1.35, 1.35)
    world.set_zlim(0.45, 2.65)
    world.set_box_aspect((5.6, 2.7, 2.2))
    world.view_init(elev=22, azim=-63)
    world.set_xlabel("x [m]", color="white")
    world.set_ylabel("y [m]", color="white")
    world.set_zlabel("z [m]", color="white")
    world.set_title("World view", color="white", pad=12)
    world.tick_params(colors="#b8c1cc", labelsize=8)
    for axis in (world.xaxis, world.yaxis, world.zaxis):
        axis.pane.set_facecolor((0.06, 0.08, 0.11, 1.0))
        axis.pane.set_edgecolor((0.3, 0.34, 0.4, 1.0))

    camera.imshow(snapshot.rgb, interpolation="bilinear")
    camera.set_title("Synthetic onboard RGB - visualization only", color="white", pad=8)
    camera.set_xticks([])
    camera.set_yticks([])
    for spine in camera.spines.values():
        spine.set_edgecolor("#68717c")
    mask_view = camera.inset_axes([0.69, 0.04, 0.28, 0.35])
    mask_view.imshow(
        snapshot.mask,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    mask_view.set_title(
        "POLICY MASK" if policy_name == "student" else "MASK REFERENCE",
        color="white",
        fontsize=6,
        pad=2,
    )
    mask_view.set_xticks([])
    mask_view.set_yticks([])
    for spine in mask_view.spines.values():
        spine.set_edgecolor("white")

    front.plot(gap[:, 1], gap[:, 2], color="#ffb000", linewidth=4.0, label="gap")
    drone = drone_front_corners(snapshot, episode.collider_size)
    front.fill(drone[:, 1], drone[:, 2], color="#36a9e1", alpha=0.5)
    front.plot(drone[:, 1], drone[:, 2], color="white", linewidth=2.0, label="quadrotor")
    body_y = snapshot.position + 0.28 * snapshot.rotation[:, 1]
    body_z = snapshot.position + 0.18 * snapshot.rotation[:, 2]
    front.plot(
        [snapshot.position[1], body_y[1]],
        [snapshot.position[2], body_y[2]],
        color=axis_colors[1],
        linewidth=3.0,
    )
    front.plot(
        [snapshot.position[1], body_z[1]],
        [snapshot.position[2], body_z[2]],
        color=axis_colors[2],
        linewidth=3.0,
    )
    front.set_xlim(episode.gap_center[1] - 0.72, episode.gap_center[1] + 0.72)
    front.set_ylim(episode.gap_center[2] - 0.62, episode.gap_center[2] + 0.62)
    front.set_aspect("equal")
    front.set_xlabel("y [m]", color="white")
    front.set_ylabel("z [m]", color="white")
    front.set_title("Front view: attitude alignment", color="white", pad=8)
    front.tick_params(colors="#b8c1cc", labelsize=8)
    front.grid(color="#39424e", alpha=0.45)
    for spine in front.spines.values():
        spine.set_edgecolor("#68717c")

    roll_degrees = np.degrees(
        np.arctan2(snapshot.rotation[2, 1], snapshot.rotation[2, 2])
    )
    speed = np.linalg.norm(snapshot.velocity)
    policy_label = "PRIVILEGED TEACHER" if policy_name == "teacher" else "VISION STUDENT"
    status_color = "#5ee173" if snapshot.status == "SUCCESS" else "#f1f5f9"
    figure.suptitle(
        f"{policy_label}  |  {angle:.0f} deg rolling gap  |  seed {episode.seed}",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.04,
        0.025,
        (
            f"t = {snapshot.time:4.2f} s     speed = {speed:4.2f} m/s     "
            f"body roll = {roll_degrees:5.1f} deg     "
            f"command = [{snapshot.command[0]:4.1f} m/s2, "
            f"{snapshot.command[1]:4.1f}, {snapshot.command[2]:4.1f}, "
            f"{snapshot.command[3]:4.1f} rad/s]"
        ),
        color="#d8dee7",
        fontsize=10,
    )
    figure.text(
        0.975,
        0.025,
        snapshot.status,
        color=status_color,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="right",
    )
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return frame


def write_outputs(
    policy_name: str,
    angle: float,
    episode: Episode,
    output_path: Path,
    fps: int,
    hold_seconds: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keyframe_indices = sorted(
        set(
            np.linspace(0, len(episode.snapshots) - 1, 4)
            .round()
            .astype(int)
            .tolist()
        )
    )
    keyframes: list[np.ndarray] = []
    final_frame: np.ndarray | None = None
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    ) as writer:
        for index in range(len(episode.snapshots)):
            frame = render_frame(policy_name, angle, episode, index)
            writer.append_data(frame)
            if index in keyframe_indices:
                keyframes.append(frame)
            final_frame = frame
        assert final_frame is not None
        for _ in range(round(hold_seconds * fps)):
            writer.append_data(final_frame)

    while len(keyframes) < 4:
        keyframes.append(keyframes[-1])
    montage_rows = []
    for row_start in (0, 2):
        montage_rows.append(
            np.concatenate(
                [keyframes[column][::2, ::2] for column in range(row_start, row_start + 2)],
                axis=1,
            )
        )
    montage = np.concatenate(montage_rows, axis=0)
    montage_path = output_path.with_name(f"{output_path.stem}_keyframes.png")
    imageio.imwrite(montage_path, montage)
    return montage_path


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    device = torch.device(args.device)
    source_config, model, normalizer = load_policy(
        args.policy,
        checkpoint_path,
        device,
        args.teacher_checkpoint,
    )
    environment_config = configured_environment(source_config, args.angle)
    renderer = BinaryGapRenderer(edge_randomization=False)
    rgb_renderer = SyntheticRGBRenderer(edge_randomization=False)

    successful_episode: Episode | None = None
    for attempt in range(args.max_attempts):
        seed = args.seed_start + attempt
        episode = rollout(
            args.policy,
            model,
            normalizer,
            environment_config,
            renderer,
            rgb_renderer,
            device,
            seed,
            args.frame_stride,
        )
        print(
            f"attempt={attempt + 1}/{args.max_attempts} seed={seed} "
            f"status={episode.terminal_status} steps={len(episode.trajectory) - 1}"
        )
        if episode.success:
            successful_episode = episode
            break
    if successful_episode is None:
        raise RuntimeError(
            f"No successful episode found in {args.max_attempts} attempts from seed "
            f"{args.seed_start}"
        )

    montage_path = write_outputs(
        args.policy,
        args.angle,
        successful_episode,
        output_path,
        args.fps,
        args.hold_seconds,
    )
    duration = len(successful_episode.snapshots) / args.fps + args.hold_seconds
    print(f"video={output_path} duration={duration:.2f}s")
    print(f"keyframes={montage_path}")


if __name__ == "__main__":
    main()
