import numpy as np
from common import grid, decode, fields, search, example, mse


def layernorm(Y):
    C = Y-Y.mean(-1, keepdims=True)
    return C/np.sqrt(np.mean(C*C, -1, keepdims=True)+1e-5)


def quantize(W, X):
    p = grid(W)
    s, z = fields(p)
    target = layernorm(X@W.T)
    loss = lambda Q: mse(layernorm(X@Q.T), target)
    direction = np.ones_like(W)
    events = [0.]
    for k in range(15):
        values = (s*(k+.5-z)-W)/direction
        events.extend(values[(values > -.5) & (values < .5)].tolist())
    events = np.unique(events)
    best, codes = loss(decode(p)), p['q'].copy()
    for shift in np.r_[events, (events[:-1]+events[1:])/2]:
        p['q'] = np.clip(np.rint((W+shift*direction)/s)+z, 0, 15).astype(np.uint8).reshape(p['q'].shape)
        error = loss(decode(p))
        if error < best:
            best, codes = error, p['q'].copy()
    p['q'] = codes
    search(p, loss)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(mse(layernorm(X@quantize(W, X)[1].T), layernorm(X@W.T)))
