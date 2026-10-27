import numpy as np
from common import grid, decode as affine, fields, fp16, example, layer_loss


def decode(p):
    Q = affine(p)
    s, _ = fields(p)
    c = p['q'].reshape(p['shape'])
    Q[:, 1::2] += s[:, 1::2] * p['T'][c[:, ::2]]
    return Q


def quantize(W, X):
    p = grid(W)
    p['T'] = np.zeros(16)
    s, z = fields(p)
    c = p['q'].reshape(W.shape)
    for _ in range(3):
        for row in range(len(W)):
            for j in range(0, W.shape[1], 2):
                best, pair = np.inf, (int(c[row, j]), int(c[row, j + 1]))
                for a in range(16):
                    for b in range(16):
                        c[row, j:j+2] = a, b
                        error = layer_loss(W, decode(p), X)
                        if error < best:
                            best, pair = error, (a, b)
                c[row, j:j+2] = pair
        target = (W[:, 1::2] - affine(p)[:, 1::2]) / s[:, 1::2]
        before = layer_loss(W, decode(p), X)
        old = p['T'].copy()
        for a in range(16):
            mask = c[:, ::2] == a
            if mask.any():
                p['T'][a] = target[mask].mean()
        p['T'] = fp16(p['T'])
        if layer_loss(W, decode(p), X) > before:
            p['T'] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
