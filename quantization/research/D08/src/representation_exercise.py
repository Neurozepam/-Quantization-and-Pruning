"""CPU learning check: equivalent channel scaling of a single Linear."""
import torch


def transform(x, w, d):
    """d: positive vector of input-channel scales. Return x_prime, w_prime."""
    x_prime = x / d
    w_prime = w * d
    return x_prime, w_prime


def main():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(5, 7, generator=g, dtype=torch.float64)
    w = torch.randn(3, 7, generator=g, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 2.0, 4.0, 0.25, 3.0, 1.5], dtype=x.dtype)
    xp, wp = transform(x, w, d)
    torch.testing.assert_close(xp, x / d)
    torch.testing.assert_close(wp, w * d)
    torch.testing.assert_close(xp @ wp.T, x @ w.T, rtol=1e-10, atol=1e-12)
    print("PASS: operands changed; FP Linear output preserved")
    print("No quantization quality claim: this check contains no quantizer.")


if __name__ == "__main__":
    main()
