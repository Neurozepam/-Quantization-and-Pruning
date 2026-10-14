"""D08 learning check: exact local Linear error, CPU only, no model/data."""
import torch


def predicted_error(x, w, ex, ew, mode):
    """Return the derived output error, not a difference of two forwards."""
    
    match mode:
        case "W-only":
            return x @ ew.T
        case "A-only":
            return ex @ w.T
        case "W+A":
            return ex @ w.T + x @ ew.T + ex @ ew.T

    raise ValueError(f"Unknown mode {mode}")


def main():
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    w = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    ex = 0.1 * torch.randn(x.shape, generator=generator, dtype=x.dtype)
    ew = 0.1 * torch.randn(w.shape, generator=generator, dtype=w.dtype)
    y = x @ w.T
    for mode, actual in (
        ("W-only", x @ (w + ew).T - y),
        ("A-only", (x + ex) @ w.T - y),
        ("W+A", (x + ex) @ (w + ew).T - y),
    ):
        predicted = predicted_error(x, w, ex, ew, mode)
        assert predicted.shape == y.shape
        torch.testing.assert_close(predicted, actual, rtol=1e-10, atol=1e-12)
        print(f"PASS {mode}")


if __name__ == "__main__":
    main()
