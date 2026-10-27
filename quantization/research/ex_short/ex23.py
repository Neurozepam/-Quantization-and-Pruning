import itertools
import numpy as np
from common import grid, decode, fields, softmax, joint_search, example


def quantize(W, X):
    p = grid(W)
    target = softmax(X@W.T)
    def loss(Q):
        prob = softmax(X@Q.T)
        return float(np.mean(np.sum(target*np.log(np.maximum(target, 1e-30)/np.maximum(prob, 1e-30)), -1)))
    s, _ = fields(p)
    sites = list(np.ndindex(p['q'].shape))[:8]
    deltas = []
    for row, group, j in sites:
        Z = np.zeros_like(target)
        col = group*p['group']+j
        Z[:, row] = s[row, col]*X[:, col]
        deltas.append(Z)
    def interaction(pair):
        a, b = [deltas[i] for i in pair]
        return np.mean(np.sum(target*a*b, -1)-np.sum(target*a, -1)*np.sum(target*b, -1))
    pairs = sorted(itertools.combinations(range(len(sites)), 2), key=interaction)
    proposed = [[(0, sites[i]), (0, sites[j])] for i, j in pairs[:4]]
    joint_search([p], lambda pp: decode(pp[0]), loss, proposed)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(np.mean((X@(quantize(W, X)[1]-W).T)**2))
