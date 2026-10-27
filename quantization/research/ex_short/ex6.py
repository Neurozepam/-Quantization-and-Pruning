import numpy as np
from common import grid, fp16, fit_mixture, layer_loss, example


def rank01(x):
    if x.size == 1:
        return np.full(x.shape, .5)
    _, inverse, counts = np.unique(np.abs(x), return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - (counts + 1) / 2
    return (average[inverse] / max(x.size - 1, 1)).reshape(x.shape)


def decode(p):
    t = np.broadcast_to(rank01(p['gamma']), p['shape']).reshape(p['q'].shape)
    return (p['s'] * ((1-t) * p['A'][p['q']] + t * p['B'][p['q']] - p['z'])).reshape(p['shape'])


def quantize(W, X, gamma):
    p = grid(W)
    p['gamma'] = np.asarray(gamma)
    base = np.arange(16.)
    p['A'], p['B'] = fp16(base + .15*np.sin(base)), fp16(base - .15*np.sin(base))
    t = np.broadcast_to(rank01(gamma), W.shape).reshape(p['q'].shape)
    w = W.reshape(p['q'].shape)
    h = np.broadcast_to(np.mean(X*X, 0), W.shape).reshape(w.shape)
    for _ in range(6):
        levels = (1-t[..., None])*p['A'] + t[..., None]*p['B']
        p['q'] = np.argmin((w[..., None]/p['s'][..., None] + p['z'][..., None]-levels)**2, -1).astype(np.uint8)
        p['A'], p['B'] = fit_mixture(w, p['q'], t, p['s'], p['z'], p['A'], p['B'], h)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X, np.linspace(.5, 2, W.shape[1]))[1], X))
