"""DAY 1: равномерное квантование массивов NumPy."""

import numpy as np


def quantize_tensor(x, bits, symmetric=True):

    if bits not in [2,3,4,8]:

        raise ValueError

    if np.all(x == 0):

        return np.zeros_like(x, dtype=np.int32), 1.0, 0

    if symmetric:

        q_min, q_max = (-1) * (2**(bits - 1) - 1), 2**(bits - 1) - 1
        scale = max(abs(min(x)), abs(max(x)))/q_max
        zero_point = 0
        q = np.clip(np.round((x/scale) + zero_point), q_min, q_max).astype(np.int32)

    else:

        q_min, q_max = 0, 2**(bits) - 1
        a = min(min(x), 0)
        c = max(max(x), 0)
        scale = (c - a)/(q_max - q_min)
        zero_point = np.clip(np.round(q_min - a/scale), q_min, q_max).astype(np.int32)
        q = np.clip(np.round((x/scale) + zero_point), q_min, q_max).astype(np.int32)

    return (q, scale, zero_point)


def dequantize_tensor(q, scale, zero_point):

    x_hat = scale * (q - zero_point)

    return x_hat


if __name__ == "__main__":
    x = np.array([-1.0, 0.0, 0.5, 1.0])

    q, scale, zero_point = quantize_tensor(x, 4)

    print(q)

    x_hat = dequantize_tensor(q, scale, zero_point)

    print(x_hat)
