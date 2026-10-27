import numpy as np
from common import grid, decode as affine, fp16, layer_loss, example


def decode(p):
    Q = np.zeros(p['shape'])
    for a, b, base, residual, alpha in p['pairs']:
        Q[a] = affine(base)[0]
        Q[b] = alpha * Q[a] + affine(residual)[0]
    return Q


def quantize(W, X):
    remaining = list(range(len(W)))
    pairs = []
    while remaining:
        a = remaining.pop(0)
        b = max(remaining, key=lambda j: abs(W[a] @ W[j]) / max(np.linalg.norm(W[j]), 1e-12))
        remaining.remove(b)
        base = grid(W[a:a+1], bits=5)
        vector = affine(base)[0]
        alpha = float(fp16(W[b] @ vector / max(vector @ vector, 1e-12)))
        residual = grid((W[b]-alpha*vector)[None], bits=3)
        pairs.append((a, b, base, residual, alpha))
    p = dict(shape=W.shape, pairs=pairs)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
