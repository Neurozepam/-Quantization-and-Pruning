"""DAY 4: сравнение трёх заранее заданных правил порога на трёх calibration sets.

Exploratory follow-up: evaluation уже наблюдалась в max-abs эксперименте.
Выбор итогового правила по этой таблице потребует новой проверки на свежих текстах.
"""

import csv
from pathlib import Path

import torch

from day4_experiment import fake_quantize, measure_errors

ROOT = Path(__file__).resolve().parent / "artefacts"
POLICIES = ("max", "p99_9", "p99")


def select_alpha(x_cal, policy):
    match policy:
        case "max":
            return x_cal.abs().max().item()
        case "p99":
            return torch.quantile(x_cal.abs().flatten(), 0.99).item()
        case "p99_9":
            return torch.quantile(x_cal.abs().flatten(), 0.999).item()


def test_select_alpha():
    x = torch.tensor([-20.0, 0.0, 10.0])
    assert abs(select_alpha(x, "max") - 20.0) < 1e-6
    assert abs(select_alpha(x, "p99") - 19.8) < 1e-4
    assert abs(select_alpha(x, "p99_9") - 19.98) < 1e-4
    print("Threshold selection tests: PASS")


def main():
    test_select_alpha()
    snapshot = torch.load(ROOT / "calibration_snapshot.pt", weights_only=True)
    input_sets = snapshot["inputs"]
    weight = snapshot["weight"]
    x_eval = input_sets["evaluation"]
    rows = []

    with torch.inference_mode():
        for group in ("representative", "unrelated", "tiny"):
            x_cal = input_sets[group]
            for policy in POLICIES:
                alpha = select_alpha(x_cal, policy)
                cal = measure_errors(x_cal, fake_quantize(x_cal, alpha), weight, alpha)
                evaluation = measure_errors(x_eval, fake_quantize(x_eval, alpha), weight, alpha)
                row = {
                    "group": group,
                    "policy": policy,
                    "calibration_tokens": x_cal.shape[0],
                    "alpha": alpha,
                    "scale": alpha / 7,
                    "calibration_activation_mse": cal["activation_mse"],
                    "calibration_output_mse": cal["output_mse"],
                    "calibration_clipping_fraction": cal["clipping_fraction"],
                    "evaluation_activation_mse": evaluation["activation_mse"],
                    "evaluation_output_mse": evaluation["output_mse"],
                    "evaluation_clipping_fraction": evaluation["clipping_fraction"],
                }
                rows.append(row)
                print(
                    f"{group:14s} {policy:6s} alpha={alpha:.6f} "
                    f"cal_output_mse={cal['output_mse']:.6f} "
                    f"eval_output_mse={evaluation['output_mse']:.6f} "
                    f"eval_clip={evaluation['clipping_fraction']:.6%}"
                )

    path = ROOT / "threshold_policy_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


if __name__ == "__main__":
    main()
