import numpy as np
from common import grid, decode, joint_search, example, mse


def quantize(W, X, strength=.1):
    p = grid(W)
    def loss(Q):
        error = X@(Q-W).T
        return float(np.mean(error**2)+strength*np.mean(error**4))
    sites = []
    for row in range(len(W)):
        for j in range(0, min(W.shape[1]-1, 8), 2):
            sites.append([(0, (row, 0, j)), (0, (row, 0, j+1))])
    joint_search([p], lambda pp: decode(pp[0]), loss, sites)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(np.mean((X@(quantize(W, X)[1]-W).T)**2))
