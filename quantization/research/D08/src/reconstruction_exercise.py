"""CPU learning check: weight error versus local output error."""
import torch


def objectives(x, ew):
    """Return (sum of squared weight errors, sum of squared output errors)."""

    weight_errors = (ew).square().sum()
    squared_output_errors = (x @ ew.T).square().sum()

    return weight_errors, squared_output_errors


def main():
    x = torch.tensor([[10.0, 1.0]], dtype=torch.float64)
    errors = {
        "A": torch.tensor([[0.1, 0.0]], dtype=torch.float64),
        "B": torch.tensor([[0.0, 0.2]], dtype=torch.float64),
    }
    expected = {"A": (0.01, 1.0), "B": (0.04, 0.04)}
    for name, ew in errors.items():
        weight_loss, output_loss = objectives(x, ew)
        for value, target in zip((weight_loss, output_loss), expected[name]):
            torch.testing.assert_close(
                value, torch.tensor(target, dtype=x.dtype), rtol=1e-10, atol=1e-12
            )
        print(f"{name}: weight={weight_loss.item():.4f}, output={output_loss.item():.4f}")
    ew = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=x.dtype)
    weight_loss, output_loss = objectives(x, ew)
    torch.testing.assert_close(weight_loss, torch.tensor(2.0, dtype=x.dtype))
    torch.testing.assert_close(output_loss, torch.tensor(200.0, dtype=x.dtype))
    print("PASS (including multiple output rows)")


if __name__ == "__main__":
    main()
