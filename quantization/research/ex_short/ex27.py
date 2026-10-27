import numpy as np
from common import grid, decode, search, softmax, mse


def response(X, V, maps, outputs):
    return sum(A@X@V.T@O.T for A, O in zip(maps, outputs))


def quantize(V, X, maps, outputs):
    target = response(X, V, maps, outputs)
    p = search(grid(V), lambda Q: mse(response(X, Q, maps, outputs), target))
    return p, decode(p)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    X, V = rng.normal(size=(12, 8)), rng.normal(size=(4, 8))
    maps = [softmax(rng.normal(size=(12, 12))) for _ in range(2)]
    outputs = [rng.normal(size=(8, 4)) for _ in maps]
    print(mse(response(X, quantize(V, X, maps, outputs)[1], maps, outputs), response(X, V, maps, outputs)))
