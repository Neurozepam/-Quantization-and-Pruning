"""DAY 4: график фактически измеренных ошибок и clipping для девяти вариантов."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "artefacts"
GROUPS = ("representative", "unrelated", "tiny")
POLICIES = ("max", "p99_9", "p99")


def main():
    with (ROOT / "threshold_policy_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    fig, (ax_error, ax_clip) = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.24

    for group_index, group in enumerate(GROUPS):
        selected = [
            next(row for row in rows if row["group"] == group and row["policy"] == policy)
            for policy in POLICIES
        ]
        errors = []
        clipping_percent = []
        for row in selected:
            errors.append(float(row["evaluation_output_mse"]))
            clipping_percent.append(float(row["evaluation_clipping_fraction"]) * 100)

        positions = [index + (group_index - 1) * width for index in range(len(POLICIES))]
        ax_error.bar(positions, errors, width=width, label=group)
        ax_clip.bar(positions, clipping_percent, width=width, label=group)

    for axis in (ax_error, ax_clip):
        axis.set_xticks(range(len(POLICIES)), ("max-abs", "p99.9", "p99"))
        axis.set_xlabel("Calibration threshold rule")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)

    ax_error.set_ylabel("Evaluation output MSE (lower is better)")
    ax_error.set_title("Local Linear output error")
    ax_error.legend()
    ax_clip.set_ylabel("Evaluation values outside range (%)")
    ax_clip.set_title("Clipping fraction")
    fig.suptitle("SmolLM2-135M | first down_proj input | symmetric INT4 | 8 evaluation texts")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = ROOT / "calibration_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    main()
