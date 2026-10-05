import numpy as np

from quantization.learning.D01.day1_quantization import quantize_tensor


def split_into_groups(w, group_size):
    """Возвращает группы с осями: строка, группа, элемент."""
    if w.ndim != 2:
        raise ValueError("Ожидается матрица")

    rows, cols = w.shape

    if group_size <= 0:
        raise ValueError("group_size должен быть положительным числом")

    if cols % group_size != 0:
        raise ValueError("cols должен делиться на group_size без остатка")

    return w.reshape(rows, cols // group_size, group_size)


def quantize_groupwise(w, bits, group_size):
    groups = split_into_groups(w, group_size)
    q_groups = np.empty(groups.shape, dtype=np.int32)
    scales = np.empty(groups.shape[:2], dtype=np.float64)

    for r in range(groups.shape[0]):
        for k in range(groups.shape[1]):
            q_groups[r, k], scales[r, k], zero_point = quantize_tensor(groups[r, k], bits, symmetric=True)

    return q_groups, scales


def dequantize_groupwise(q_groups, scales):
    w_hat_groups = q_groups.astype(np.float64)
    for r in range(q_groups.shape[0]):
        for k in range(q_groups.shape[1]):
            w_hat_groups[r, k] = q_groups[r, k].astype(np.float64) * scales[r, k]

    W = w_hat_groups.reshape(w_hat_groups.shape[0], -1)

    return W

def quantize_with_threshold(x, bits, alpha):
    if alpha <= 0:
        raise ValueError("alpha должен быть положительным числом")

    Q = 2**(bits - 1) - 1
    scale = alpha / Q
    x_scaled = np.round(x / scale)
    x_clipped = np.clip(x_scaled, -Q, Q).astype(np.int32)

    return x_clipped, scale

def threshold_errors(x, bits, alpha):
    q, scale = quantize_with_threshold(x, bits, alpha)
    x_hat = q * scale
    x_c = np.clip(x, -alpha, alpha)

    mse_rounding = np.mean((x_c - x_hat)**2)
    mse_clipping = np.mean((x - x_c)**2)
    mse_total = np.mean((x - x_hat)**2)

    return mse_total, mse_clipping, mse_rounding