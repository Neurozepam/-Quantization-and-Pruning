"""DAY 3: сравнение ошибки весов и ошибки выхода линейного слоя."""

import numpy as np

from quantization.learning.D02.day2_quantization import quantize_with_threshold

def measure_errors(x, w, w_hat):
    """
    Формы:
        x:     [batch, in]
        w:     [out, in]
        w_hat: [out, in] — восстановленные после квантования веса.
    """
    y = x @ w.T
    y_hat = x @ w_hat.T

    weight_mse = np.mean((w - w_hat) ** 2)
    output_mse = np.mean((y - y_hat) ** 2)

    return weight_mse, output_mse


if __name__ == "__main__":
    x = np.array([[10.0, 1.0]])
    w = np.array([[1.0, 1.0]])
    w_hat_a = np.array([[0.9, 1.0]])

    weight_mse, output_mse = measure_errors(x, w, w_hat_a)
    print("Test case A:")
    print(f"weight MSE: {weight_mse:.6f}")
    print(f"output MSE: {output_mse:.6f}")

    w_hat_b = np.array([[1, 0.8]])

    weight_mse, output_mse = measure_errors(x, w, w_hat_b)
    print("Test case B:")
    print(f"weight MSE: {weight_mse:.6f}")
    print(f"output MSE: {output_mse:.6f}")  

    print("=== Quantization with different thresholds ===")

    w = np.array([[1.0, 0.2]])
    q_a, scale_a = quantize_with_threshold(w, bits=4, alpha=1.0)
    q_b, scale_b = quantize_with_threshold(w, bits=4, alpha=0.7)
    w_hat_a = q_a * scale_a
    w_hat_b = q_b * scale_b
    print(f"Weight vector (case A): {w_hat_a}")
    print(f"Weight vector (case B): {w_hat_b}")

    print("=== Measure errors with different inputs ===")

    x = np.array([[0.1, 10.0]])
    weight_mse_a, output_mse_a = measure_errors(x, w, w_hat_a)
    weight_mse_b, output_mse_b = measure_errors(x, w, w_hat_b)
    print("X = [[0.1, 10.0]]")
    print("Test case A:")
    print(f"weight MSE: {weight_mse_a:.6f}")
    print(f"output MSE: {output_mse_a:.6f}")
    print("Test case B:")
    print(f"weight MSE: {weight_mse_b:.6f}")
    print(f"output MSE: {output_mse_b:.6f}")

    x = np.array([[10.0, 0.1]])
    weight_mse_a, output_mse_a = measure_errors(x, w, w_hat_a)
    weight_mse_b, output_mse_b = measure_errors(x, w, w_hat_b)
    print("X = [[10.0, 0.1]]")
    print("Test case A:")
    print(f"weight MSE: {weight_mse_a:.6f}")
    print(f"output MSE: {output_mse_a:.6f}")
    print("Test case B:")
    print(f"weight MSE: {weight_mse_b:.6f}")
    print(f"output MSE: {output_mse_b:.6f}")