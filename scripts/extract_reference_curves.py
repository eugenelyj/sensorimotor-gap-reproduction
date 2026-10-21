from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Zenodo Fig. 7A pickles to CSV.")
    parser.add_argument(
        "--input",
        default="reference_data/zenodo/data_plots/training_evolution",
    )
    parser.add_argument("--output", default="reference_data/fig7a_training_curves.csv")
    return parser.parse_args()


def load_group(directory: Path) -> list[dict[int, float]]:
    runs: list[dict[int, float]] = []
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as data_file:
            data = pickle.load(data_file)
        runs.append(
            {int(step): float(value) for step, value in zip(data["steps"], data["values"])}
        )
    return runs


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        "with_ir": load_group(source / "results_w_IR"),
        "without_ir": load_group(source / "results_wo_IR"),
    }
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("condition", "step", "mean", "std", "num_runs"))
        for condition, runs in groups.items():
            steps = sorted(set().union(*(run.keys() for run in runs)))
            for step in steps:
                values = [run[step] for run in runs if step in run]
                if not values:
                    continue
                writer.writerow(
                    (
                        condition,
                        step,
                        np.mean(values),
                        np.std(values),
                        len(values),
                    )
                )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
