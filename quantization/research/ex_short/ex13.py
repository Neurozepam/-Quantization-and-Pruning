import numpy as np
from common import grid, search, fp16, regression, layer_loss, example


def normalized(p):
    c, L = p['q'], p['L']
    mu = c.mean(-1, keepdims=True)
    ml = L[c].mean(-1, keepdims=True)
    a = np.sqrt(c.var(-1, keepdims=True)/np.maximum(L[c].var(-1, keepdims=True), 1e-12)) if p['variance'] else 1
    return a*(L[c]-ml)+mu


def decode(p):
    return (p['s']*(normalized(p)-p['z'])).reshape(p['shape'])


def quantize(W, X, variance=True):
    p = grid(W)
    p['L'], p['variance'] = np.arange(16.), variance
    for _ in range(3):
        search(p, lambda Q: layer_loss(W, Q, X), decoder=decode)
        c = p['q']
        eye = np.eye(16)[c]
        F = eye-eye.mean(-2, keepdims=True)
        if variance:
            a = np.sqrt(c.var(-1, keepdims=True)/np.maximum(p['L'][c].var(-1, keepdims=True), 1e-12))
            F *= a[..., None]
        target = W.reshape(c.shape)/p['s']+p['z']-c.mean(-1, keepdims=True)
        candidate = fp16(np.maximum.accumulate(regression(F.reshape(-1, 16), target.ravel())))
        old, before = p['L'].copy(), layer_loss(W, decode(p), X)
        p['L'] = candidate
        if layer_loss(W, decode(p), X) > before:
            p['L'] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
