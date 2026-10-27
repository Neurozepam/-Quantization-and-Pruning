import numpy as np
from common import grid, search, fp16, regression, layer_loss, example


def decode(p):
    c = p['q']
    levels = p['L'][c] - p['z'] + p['D'][np.roll(c, 1, -1)] - p['D'][np.roll(c, -1, -1)]
    return (p['s']*levels).reshape(p['shape'])


def quantize(W, X):
    p = grid(W)
    p['L'], p['D'] = np.arange(16.), .05*np.sin(np.arange(16.))
    for _ in range(3):
        search(p, lambda Q: layer_loss(W, Q, X), decoder=decode)
        c = p['q'].ravel()
        F = np.zeros((len(c), 32))
        ids = np.arange(len(c))
        F[ids, c] = 1
        np.add.at(F, (ids, 16+np.roll(p['q'], 1, -1).ravel()), 1)
        np.add.at(F, (ids, 16+np.roll(p['q'], -1, -1).ravel()), -1)
        target = (W.reshape(p['q'].shape)/p['s']+p['z']).ravel()
        h = np.broadcast_to(np.mean(X*X, 0), W.shape).reshape(p['q'].shape)
        weight = np.sqrt(h*p['s']**2).ravel()
        theta = regression(F*weight[:, None], target*weight)
        old, before = (p['L'].copy(), p['D'].copy()), layer_loss(W, decode(p), X)
        p['L'], p['D'] = fp16(np.maximum.accumulate(theta[:16])), fp16(theta[16:])
        if layer_loss(W, decode(p), X) > before:
            p['L'], p['D'] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
