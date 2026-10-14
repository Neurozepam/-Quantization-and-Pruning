"""Motivating compensation example, NOT a GPTQ implementation.

Two identical input channels: only the sum of weights matters.
Unit-step rounding is used solely to expose the mechanism.
No Hessian inverse, GPTQ damping, grouping or model data here.
"""
import torch


def compensate(w, q_first):
    """Return adjusted second weight preserving the original weight sum.

    w has shape (2,); q_first is the already rounded first weight.
    """
    return w[1] + (w[0] - q_first)


def main():
    w = torch.tensor([0.4, 0.4], dtype=torch.float64)
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]], dtype=w.dtype)
    q_first = w[0].round()
    original = w.clone()
    adjusted = compensate(w, q_first)
    torch.testing.assert_close(w, original, rtol=0, atol=0)
    torch.testing.assert_close(q_first + adjusted, w.sum())
    q_rtn = w.round()
    q_compensated = torch.stack((q_first, adjusted.round()))
    reference = x @ w
    loss_rtn = (x @ q_rtn - reference).square().sum()
    loss_compensated = (x @ q_compensated - reference).square().sum()
    assert loss_compensated < loss_rtn
    print(f"RTN codes: {q_rtn.tolist()}, loss={loss_rtn.item():.4f}")
    print(f"Compensated codes: {q_compensated.tolist()}, loss={loss_compensated.item():.4f}")
    print("PASS: improvement in this constructed example only, not a general guarantee")


if __name__ == "__main__":
    main()
