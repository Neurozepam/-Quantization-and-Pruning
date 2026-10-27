import numpy as np
from common import grid, decode as affine, fp16, regression, mse


def decode(p):
    U = np.zeros(p['shape'])
    for i, (parents, coefficients, residual) in enumerate(p['records']):
        prediction = coefficients@U[parents] if len(parents) else 0
        U[i] = prediction+affine(residual)[0]
    return U+p['shift']


def quantize(W, parents):
    shift = fp16(W.mean(0))
    U, reconstructed, records = W-shift, np.zeros_like(W), []
    for i, family in enumerate(parents):
        family = np.asarray(family, dtype=int)
        if len(family) and np.max(family) >= i:
            raise ValueError('Parents must already be reconstructed')
        coefficients = fp16(np.linalg.lstsq(reconstructed[family].T, U[i], rcond=None)[0]) if len(family) else np.array([])
        prediction = coefficients@reconstructed[family] if len(family) else 0
        residual = grid((U[i]-prediction)[None])
        reconstructed[i] = prediction+affine(residual)[0]
        records.append((family, coefficients, residual))
    p = dict(shape=W.shape, shift=shift, records=records)
    return p, decode(p)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    W = rng.normal(size=(8, 16))
    parents = [[], [], *[[i-2, i-1] for i in range(2, 8)]]
    print(mse(quantize(W, parents)[1], W))
