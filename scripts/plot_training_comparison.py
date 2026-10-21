from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a run with the paper's Fig. 7A data.")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--reference", default="reference_data/fig7a_training_curves.csv"
    )
    parser.add_argument("--output", default="artifacts/training/fig7a_comparison.png")
    parser.add_argument("--max-samples", type=int, default=1_000_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions: dict[str, list[tuple[int, float, float]]] = {}
    with Path(args.reference).open(encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["step"]) > args.max_samples:
                continue
            conditions.setdefault(row["condition"], []).append(
                (int(row["step"]), float(row["mean"]), float(row["std"]))
            )
    accumulator = EventAccumulator(args.run_dir)
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    comparison_tag = next(
        tag
        for tag in (
            "evaluation/success_rate",
            "evaluation/on_policy_standard_start_success_rate",
            "evaluation/on_policy_success_rate",
            "train/online_success_rate",
        )
        if tag in tags
    )
    reproduced = [
        value
        for value in accumulator.Scalars(comparison_tag)
        if value.step <= args.max_samples
    ]
    reproduced_steps = [value.step for value in reproduced]
    reproduced_values = [value.value for value in reproduced]
    if comparison_tag == "train/online_success_rate" and len(reproduced_values) >= 20:
        window = 20
        reproduced_steps = reproduced_steps[window - 1 :]
        reproduced_values = [
            sum(reproduced_values[index - window + 1 : index + 1]) / window
            for index in range(window - 1, len(reproduced_values))
        ]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = {"with_ir": "#e83ee8", "without_ir": "#31b5e8"}
    for condition, values in conditions.items():
        values.sort()
        steps = [value[0] for value in values]
        means = [value[1] for value in values]
        lower = [max(0.0, value[1] - value[2]) for value in values]
        upper = [min(1.0, value[1] + value[2]) for value in values]
        axis.plot(steps, means, color=colors[condition], label=f"paper {condition}")
        axis.fill_between(steps, lower, upper, color=colors[condition], alpha=0.2)
    axis.plot(
        reproduced_steps,
        reproduced_values,
        color="black",
        linewidth=2,
        label=f"reproduction ({comparison_tag.rsplit('/', 1)[-1]})",
    )
    axis.set(
        xlabel="Samples",
        ylabel="Success rate",
        xlim=(0, args.max_samples),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
