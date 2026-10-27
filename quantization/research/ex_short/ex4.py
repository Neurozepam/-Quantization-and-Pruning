import itertools
import numpy as np
from common import grid, decode, fields, layer_loss, example


def quantize(W, X, width=8):
    p = grid(W)
    s, z = fields(p)
    lo = np.clip(np.floor(W / s) + z, 0, 15)
    hi = np.clip(np.ceil(W / s) + z, 0, 15)
    H = X.T @ X / len(X)
    a = np.linalg.eigh(H)[1][:, -1]
    c = p['q'].reshape(W.shape)
    for row in range(len(W)):
        for start in range(0, W.shape[1], width):
            idx = np.arange(start, min(start + width, W.shape[1]))
            bits = np.array(list(itertools.product((0, 1), repeat=len(idx))))
            choices = lo[row, idx] + bits * (hi[row, idx] - lo[row, idx])
            candidates = np.repeat(decode(p)[row:row+1], len(choices), 0)
            candidates[:, idx] = s[row, idx] * (choices - z[row, idx])
            score = ((candidates - W[row]) @ a) ** 2
            proposal = choices[score.argmin()]
            old, before = c[row, idx].copy(), layer_loss(W, decode(p), X)
            c[row, idx] = proposal
            if layer_loss(W, decode(p), X) > before:
                c[row, idx] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
