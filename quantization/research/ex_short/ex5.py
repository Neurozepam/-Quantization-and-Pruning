import itertools
import numpy as np
from common import grid, decode, fields, layer_loss, example


def quantize(W, X):
    p = grid(W)
    s, z = fields(p)
    low, high = np.clip(np.floor(W / s) + z, 0, 15), np.clip(np.ceil(W / s) + z, 0, 15)
    a = np.linalg.eigh(X.T @ X)[1][:, -1]
    bits = np.array(list(itertools.product((0, 1), repeat=4)))
    c = p['q'].reshape(W.shape)
    for start in range(0, W.shape[1] - 15, 16):
        def coverage(indices):
            rest = np.setdiff1d(np.arange(start, start + 16), indices)
            band = max(.5 * np.abs(a[rest]).sum(), 1e-12)
            sums = (bits - .5) @ a[indices]
            targets = np.linspace(-band, band, 65)
            return np.min(np.abs(targets[:, None] - sums), 1).max() / band
        choices = [np.array(v) + start for v in itertools.combinations(range(8, 16), 4)]
        idx = min(choices, key=coverage)
        for row in range(len(W)):
            best, selected = layer_loss(W, decode(p), X), c[row, idx].copy()
            for pattern in bits:
                c[row, idx] = low[row, idx] + pattern * (high[row, idx] - low[row, idx])
                error = layer_loss(W, decode(p), X)
                if error < best:
                    best, selected = error, c[row, idx].copy()
            c[row, idx] = selected
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
