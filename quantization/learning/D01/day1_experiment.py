"""DAY 1: сравнение симметричного и асимметричного квантования."""

import numpy as np

from day1_quantization import quantize_tensor, dequantize_tensor


def main():
    rng = np.random.default_rng(42)
    n = 10_000
    bits = 4

    distributions = {
        "gaussian": rng.normal(0, 1, n),
        "uniform": rng.uniform(-1, 1, n),
        "laplace": rng.laplace(0, 1, n),
        "skewed": rng.exponential(1, n),
    }

    for name, x in distributions.items():
        print(name, "min=", x.min(), "max=", x.max())

        for symmetric in (True, False):
            q, scale, zero_point = quantize_tensor(x, bits, symmetric=symmetric)
            x_hat = dequantize_tensor(q, scale, zero_point)
            mse = np.mean((x - x_hat) ** 2)
            print("  symmetric=", symmetric, "MSE=", mse)


if __name__ == "__main__":
    main()
