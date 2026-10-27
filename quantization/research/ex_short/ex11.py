import numpy as np
from common import grid, fp16, layer_loss, example


def decode(p):
    rho = np.where(p['s'] < 0, -1, 1)
    idx = (15 + rho*(p['q'].astype(float)-p['z'])).astype(int)
    return (np.abs(p['s'])*rho*p['T'][idx]).reshape(p['shape'])


def quantize(W, X):
    p = grid(W)
    s, w = np.abs(p['s']).astype(float), W.reshape(p['q'].shape)
    p['T'] = np.arange(-15., 16.).astype(np.float16)
    h = np.broadcast_to(np.mean(X*X, 0), W.shape).reshape(w.shape)
    for _ in range(6):
        best = np.full(s.shape, np.inf)
        for rho in (1, -1):
            for z in range(16):
                levels = rho*p['T'].astype(float)[(15+rho*(np.arange(16)-z)).astype(int)]
                c = np.argmin((w[..., None]-s[..., None]*levels)**2, -1)
                cost = np.sum(h*(s*levels[c]-w)**2, -1, keepdims=True)
                use = cost < best
                p['q'] = np.where(use, c, p['q']).astype(np.uint8)
                p['z'] = np.where(use, z, p['z'])
                p['s'] = fp16(np.where(use, rho*s, p['s']))
                best = np.minimum(best, cost)
        old, before = p['T'].copy(), layer_loss(W, decode(p), X)
        rho = np.where(p['s'] < 0, -1, 1)
        idx = (15+rho*(p['q'].astype(float)-p['z'])).astype(int)
        target = w/(s*rho)
        for k in range(31):
            mask = idx == k
            if mask.any():
                weight = np.broadcast_to(h*s*s, w.shape)[mask]
                p['T'][k] = np.sum(weight*target[mask])/max(weight.sum(), 1e-30)
        p['T'] = np.maximum.accumulate(p['T'])
        p['T'] = fp16(p['T']-p['T'][15])
        if layer_loss(W, decode(p), X) > before:
            p['T'] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
