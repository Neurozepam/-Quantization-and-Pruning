import numpy as np
from common import grid, decode, joint_search, softmax, mse


def quantize(V, O, X, A):
    packets = [grid(V), grid(O)]
    target = A @ X @ V.T @ O.T
    def output(p):
        return A @ X @ decode(p[0]).T @ decode(p[1]).T
    sites = []
    for channel in range(len(V)):
        sites.append([(0, (channel, 0, 0)), (1, (0, 0, channel))])
    joint_search(packets, output, lambda y: mse(y, target), sites)
    return packets, output(packets)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    X, V, O = rng.normal(size=(12, 8)), rng.normal(size=(4, 8)), rng.normal(size=(8, 4))
    A = softmax(rng.normal(size=(12, 12)))
    print(mse(quantize(V, O, X, A)[1], A @ X @ V.T @ O.T))
