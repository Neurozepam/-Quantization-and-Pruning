import numpy as np
from common import grid, decode as affine, softmax, fp16, example, mse


def decode(p):
    Q = np.zeros(p['shape'])
    for row, anchor, alpha, residual in p['rows']:
        if row == anchor:
            Q[row] = affine(residual)[0]
        else:
            Q[row] = alpha*Q[anchor]+affine(residual)[0]
    return Q


def quantize(W, X):
    P = softmax(X@W.T)
    affinity = P.T@P
    remaining, records = list(range(len(W))), []
    reconstructed = {}
    while remaining:
        anchor = max(remaining, key=lambda i: P[:, i].sum())
        remaining.remove(anchor)
        base = grid(W[anchor:anchor+1])
        reconstructed[anchor] = affine(base)[0]
        records.append((anchor, anchor, 1., base))
        if remaining:
            row = max(remaining, key=lambda i: affinity[anchor, i])
            remaining.remove(row)
            vector = reconstructed[anchor]
            alpha = float(fp16(W[row]@vector/max(vector@vector, 1e-12)))
            residual = grid((W[row]-alpha*vector)[None])
            records.append((row, anchor, alpha, residual))
    p = dict(shape=W.shape, rows=records)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(mse(softmax(X@quantize(W, X)[1].T), softmax(X@W.T)))
