import numpy as np
from common import grid, decode, attention, mse


def rope(Y):
    pairs = Y.reshape(len(Y), -1, 2)
    angles = np.arange(len(Y))[:, None]*np.exp(-np.arange(pairs.shape[1]))[None]
    a, b = pairs[..., 0], pairs[..., 1]
    return np.stack((a*np.cos(angles)-b*np.sin(angles), a*np.sin(angles)+b*np.cos(angles)), -1).reshape(Y.shape)


def quantize(WQ, WK, X, V):
    target = attention(rope(X@WQ.T), rope(X@WK.T), V)
    best, selected = np.inf, None
    for angle in (0., -.02, .02):
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(len(WQ))
        R[:4, :4] = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
        p = [grid(R@WQ), grid(R@WK)]
        Y = attention(rope(X@decode(p[0]).T), rope(X@decode(p[1]).T), V)
        error = mse(Y, target)
        if error < best:
            best, selected = error, (p, Y)
    return selected


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    X, WQ, WK, V = [rng.normal(size=s) for s in [(12, 8), (4, 8), (4, 8), (12, 4)]]
    print(mse(quantize(WQ, WK, X, V)[1], attention(rope(X@WQ.T), rope(X@WK.T), V)))
