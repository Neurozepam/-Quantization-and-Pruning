import numpy as np
from common import grid, decode as affine, layer_loss, example


def permutation(width, tag):
    if not 0 <= tag < 16:
        raise ValueError('tag must be in 0..15')
    parity = [(j & tag).bit_count() % 2 for j in range(width)]
    return np.argsort(parity, kind='stable')


def decode(p):
    order = permutation(p['base']['shape'][1], p['tag'])
    return affine(p['base'])[:, np.argsort(order)]


def quantize(W, X, group=4):
    best, selected = np.inf, None
    for tag in range(16):
        order = permutation(W.shape[1], tag)
        p = dict(base=grid(W[:, order], group=group), tag=tag)
        error = layer_loss(W, decode(p), X)
        if error < best:
            best, selected = error, p
    return selected, decode(selected)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
