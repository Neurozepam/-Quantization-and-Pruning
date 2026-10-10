"""DAY 4: локальное статическое INT4 fake-квантование входа down_proj."""

import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent / "artefacts"


def fake_quantize(x, alpha, bits=4):
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if alpha == 0:
        return torch.zeros_like(x)

    Q = 2 ** (bits - 1) - 1
    scale = alpha / Q
    q = (x / scale).round()
    q = q.clamp(-Q, Q)
    x_hat = q * scale

    return x_hat


def measure_errors(x, x_hat, weight, alpha):
    activation_mse = ((x - x_hat) ** 2).mean().item()
    output_mse = ((x @ weight.T - x_hat @ weight.T) ** 2).mean().item()
    clipping_fraction = (x.abs() > alpha).float().mean().item()

    return {
        "activation_mse": activation_mse,
        "output_mse": output_mse,
        "clipping_fraction": clipping_fraction,
    }


def test_measure_errors():
    x = torch.tensor([[0.26]])
    x_hat = torch.tensor([[0.30]])
    weight = torch.tensor([[2.0]])
    metrics = measure_errors(x, x_hat, weight, alpha=0.7)
    assert abs(metrics["activation_mse"] - 0.0016) < 1e-7
    assert abs(metrics["output_mse"] - 0.0064) < 1e-7
    assert metrics["clipping_fraction"] == 0.0

    x = torch.tensor([[2.0, -2.0, 0.5]])
    x_hat = fake_quantize(x, alpha=1.0)
    weight = torch.ones((1, 3))
    metrics = measure_errors(x, x_hat, weight, alpha=1.0)
    assert abs(metrics["clipping_fraction"] - 2 / 3) < 1e-6
    print("Manual metric tests: PASS")


def test_fake_quantize():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    original = x.clone()
    expected = torch.tensor([-1.4, -1.0, 0.0, 1.0, 1.4])
    assert torch.allclose(fake_quantize(x, alpha=1.4), expected, atol=1e-6)
    assert torch.equal(x, original), "Input was modified"
    assert torch.equal(fake_quantize(x, alpha=0.0), torch.zeros_like(x))
    assert torch.equal(fake_quantize(x, alpha=3.0, bits=3), x)
    assert torch.allclose(
        fake_quantize(torch.tensor([-0.26, 0.26]), alpha=0.7),
        torch.tensor([-0.3, 0.3]),
        atol=1e-6,
    )
    print("Manual fake-quantization tests: PASS")


def main():
    test_fake_quantize()
    test_measure_errors()
    snapshot = torch.load(ROOT / "calibration_snapshot.pt", weights_only=True)
    input_sets = snapshot["inputs"]
    x_eval = input_sets["evaluation"]
    weight = snapshot["weight"]
    rows = []

    with torch.inference_mode():
        for group in ("representative", "unrelated", "tiny"):
            alpha = input_sets[group].abs().max().item()
            x_hat = fake_quantize(x_eval, alpha=alpha, bits=4)
            assert x_hat.shape == x_eval.shape
            metrics = measure_errors(x_eval, x_hat, weight, alpha)
            row = {
                "group": group,
                "calibration_tokens": input_sets[group].shape[0],
                "alpha": alpha,
                "scale": alpha / 7,
                **metrics,
            }
            rows.append(row)
            print(row)

    path = ROOT / "max_abs_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


if __name__ == "__main__":
    main()
