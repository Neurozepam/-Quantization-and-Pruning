import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quantization.learning.D02.day2_quantization import threshold_errors

rng = np.random.default_rng(42)
x = rng.normal(size=10_000)
x[0] = 14.0
bits = 4

alphas = np.linspace(0.5, np.max(np.abs(x)), 200)
errors = []

for alpha in alphas:
    mse_total, mse_clipping, mse_rounding = threshold_errors(x, bits, alpha)
    errors.append((mse_total, mse_clipping, mse_rounding))

errors = np.array(errors)

best_idx = np.argmin(errors[:, 0])
best_alpha = alphas[best_idx]

print(f"Best alpha: {best_alpha:.4f} | MSE_total: {errors[best_idx, 0]:.6f} | \
      MSE_clipping: {errors[best_idx, 1]:.6f} | MSE_rounding: {errors[best_idx, 2]:.6f}")

print(f"Last alpha: {alphas[-1]:.4f} | MSE_total: {errors[-1, 0]:.6f} | \
      MSE_clipping: {errors[-1, 1]:.6f} | MSE_rounding: {errors[-1, 2]:.6f}")


plt.figure(figsize=(10, 6))
plt.plot(alphas, errors[:, 0], label="MSE total", color="blue")
plt.plot(alphas, errors[:, 1], label="MSE clipping", color="orange")
plt.plot(alphas, errors[:, 2], label="MSE rounding", color="green")
plt.xlabel("Clipping threshold alpha")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("clipping_mse.png"), dpi=150)
plt.close()

for percentile in (99, 99.9):
    alpha = np.percentile(np.abs(x), percentile)
    total, clipping, rounding = threshold_errors(x, bits, alpha)
    print(
        f"Percentile {percentile}: alpha={alpha:.4f}, "
        f"total={total:.6f}, clipping={clipping:.6f}, "
        f"rounding={rounding:.6f}"
    )