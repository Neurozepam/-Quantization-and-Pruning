import numpy as np
from common import grid, decode, search, rms, example, mse


def quantize(W, X, direct=False):
    Y = X@W.T
    r = np.sqrt(np.mean(Y*Y, -1, keepdims=True)+1e-5)
    def loss(Q):
        if direct:
            return mse(rms(X@Q.T), rms(Y))
        delta = X@(Q-W).T
        tangent = delta/r-Y*np.mean(Y*delta, -1, keepdims=True)/(r**3)
        return float(np.mean(tangent*tangent))
    p = search(grid(W), loss)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(mse(rms(X@quantize(W, X)[1].T), rms(X@W.T)))
