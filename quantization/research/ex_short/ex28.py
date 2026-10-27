import numpy as np
from common import grid, decode, joint_search, attention, mse


def quantize(WQ, WK, X, V):
    p = [grid(WQ), grid(WK)]
    target = attention(X@WQ.T, X@WK.T, V)
    def output(p):
        return attention(X@decode(p[0]).T, X@decode(p[1]).T, V)
    sites = [[(0, (i, 0, 0)), (0, (i, 0, 1)), (1, (i, 0, 0)), (1, (i, 0, 1))] for i in range(len(WQ))]
    joint_search(p, output, lambda y: mse(y, target), sites)
    return p, output(p)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    X, WQ, WK, V = [rng.normal(size=s) for s in [(12, 8), (4, 8), (4, 8), (12, 4)]]
    print(mse(quantize(WQ, WK, X, V)[1], attention(X@WQ.T, X@WK.T, V)))
