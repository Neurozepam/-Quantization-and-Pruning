import numpy as np
from common import grid, decode as affine, fields, example, mse


def encode(W):
    base = grid(W)
    mean = W.mean(1, keepdims=True).astype(np.float32)
    radius = np.nextafter(np.sum(W*W, 1, keepdims=True).astype(np.float32), np.float32(np.inf))
    return dict(base=base, mean=mean, radius=radius)


def decode(p):
    base = p['base']
    q = affine(base)
    s, _ = fields(base)
    c = base['q'].reshape(base['shape'])
    lower, upper = q-s/2, q+s/2
    lower[c == 0], upper[c == 15] = -np.inf, np.inf
    x, pb, pm, pn = q.copy(), np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
    for _ in range(128):
        v = x+pb
        x = np.clip(v, lower, upper)
        pb = v-x
        v = x+pm
        mean = v.mean(1, keepdims=True)
        low = np.nextafter(p['mean'], np.float32(-np.inf))
        high = np.nextafter(p['mean'], np.float32(np.inf))
        x = v+np.clip(mean, low, high)-mean
        pm = v-x
        v = x+pn
        factor = np.minimum(1, np.sqrt(p['radius']/np.maximum(np.sum(v*v, 1, keepdims=True), 1e-30)))
        x = v*factor
        pn = v-x
    return x.astype(np.float32)


if __name__ == '__main__':
    W, X = example()
    print(mse(decode(encode(W)), W))
