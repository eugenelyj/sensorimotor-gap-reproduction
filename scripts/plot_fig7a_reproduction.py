from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three-seed IR and no-IR reproductions with Fig. 7A."
    )
    parser.add_argument("--ir-runs", nargs="+", required=True)
    parser.add_argument("--no-ir-runs", nargs="+", required=True)
    parser.add_argument(
        "--reference", default="reference_data/fig7a_training_curves.csv"
    )
    parser.add_argument(
        "--metric", default="evaluation/success_rate"
    )
    parser.add_argument("--max-samples", type=int, default=1_000_000_000)
    parser.add_argument("--interval-samples", type=int, default=8_192_000)
    parser.add_argument(
        "--output", default="artifacts/training/fig7a_reproduction.png"
    )
    parser.add_argument(
        "--csv-output", default="artifacts/training/fig7a_reproduction.csv"
    )
    parser.epilog = (
        "A run may be a comma-separated chain of directories, for example "
        "runs/seed1,runs/seed1_1b. Later directories override overlapping steps."
    )
    return parser.parse_args()


def load_run_directory(
    run_dir: str, metric: str, max_samples: int
) -> dict[int, float]:
    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    values: dict[int, float] = {}
    fallback = "evaluation/success_rate"
    if fallback in tags:
        values.update(
            (item.step, item.value)
            for item in accumulator.Scalars(fallback)
            if item.step <= max_samples
        )
    if metric in tags:
        values.update(
            (item.step, item.value)
            for item in accumulator.Scalars(metric)
            if item.step <= max_samples
        )
    if not values:
        raise RuntimeError(f"No {metric!r} or {fallback!r} scalar in {run_dir}")
    return values


def load_run(run_spec: str, metric: str, max_samples: int) -> dict[int, float]:
    values: dict[int, float] = {}
    for run_dir in run_spec.split(","):
        continuation = load_run_directory(run_dir, metric, max_samples)
        if values and continuation:
            continuation_start = min(continuation)
            values = {
                step: value for step, value in values.items() if step < continuation_start
            }
        values.update(continuation)
    return values


def aggregate(
    run_dirs: list[str], metric: str, max_samples: int, interval_samples: int
) -> list[tuple[int, float, float, int]]:
    runs = [load_run(run_dir, metric, max_samples) for run_dir in run_dirs]
    steps = range(interval_samples, max_samples + 1, interval_samples)
    result = []
    for step in steps:
        values = []
        for run in runs:
            run_steps = np.array(sorted(run))
            if run_steps[0] <= step <= run_steps[-1]:
                run_values = np.array([run[int(run_step)] for run_step in run_steps])
                values.append(float(np.interp(step, run_steps, run_values)))
        if not values:
            continue
        result.append((step, float(np.mean(values)), float(np.std(values)), len(values)))
    return result


def load_reference(
    path: str, max_samples: int
) -> dict[str, list[tuple[int, float, float]]]:
    conditions: dict[str, list[tuple[int, float, float]]] = {}
    with Path(path).open(encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            step = int(row["step"])
            if step <= max_samples:
                conditions.setdefault(row["condition"], []).append(
                    (step, float(row["mean"]), float(row["std"]))
                )
    return conditions


def main() -> None:
    args = parse_args()
    reproduced = {
        "with_ir": aggregate(
            args.ir_runs, args.metric, args.max_samples, args.interval_samples
        ),
        "without_ir": aggregate(
            args.no_ir_runs, args.metric, args.max_samples, args.interval_samples
        ),
    }
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("condition", "step", "mean", "std", "num_runs"))
        for condition, values in reproduced.items():
            for value in values:
                writer.writerow((condition, *value))

    colors = {"with_ir": "#d936d9", "without_ir": "#23a8d8"}
    labels = {"with_ir": "IR", "without_ir": "no IR"}
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for condition, values in load_reference(args.reference, args.max_samples).items():
        values.sort()
        steps = [value[0] for value in values]
        means = np.array([value[1] for value in values])
        standard_deviations = np.array([value[2] for value in values])
        axis.plot(steps, means, color=colors[condition], linestyle="--", alpha=0.75)
        axis.fill_between(
            steps,
            np.maximum(0.0, means - standard_deviations),
            np.minimum(1.0, means + standard_deviations),
            color=colors[condition],
            alpha=0.10,
        )
    for condition, values in reproduced.items():
        steps = [value[0] for value in values]
        means = np.array([value[1] for value in values])
        standard_deviations = np.array([value[2] for value in values])
        axis.plot(steps, means, color=colors[condition], label=f"reproduction {labels[condition]}")
        axis.fill_between(
            steps,
            np.maximum(0.0, means - standard_deviations),
            np.minimum(1.0, means + standard_deviations),
            color=colors[condition],
            alpha=0.22,
        )
    axis.plot([], [], color="black", linestyle="--", label="paper")
    axis.set(xlabel="Samples", ylabel="Success rate", xlim=(0, args.max_samples), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"Wrote {csv_output}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
