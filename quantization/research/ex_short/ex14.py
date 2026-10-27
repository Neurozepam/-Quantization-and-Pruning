import numpy as np
from common import grid, decode as affine, fields, fp16, search, example, layer_loss


def decode(p):
    Q = affine(p)
    c = p['q'].reshape(p['shape'])
    s, _ = fields(p)
    for idx, direction in zip(p['triples'], p['directions']):
        key = 256*c[:, idx[0]].astype(int)+16*c[:, idx[1]]+c[:, idx[2]]
        Q[:, idx] += s[:, idx[0], None]*p['F'][key, None]*direction
    return Q


def quantize(W, X):
    p = grid(W)
    H = X.T@X/len(X)
    unused, triples = list(range(W.shape[1])), []
    corr = np.abs(H)/np.sqrt(np.maximum(np.outer(np.diag(H), np.diag(H)), 1e-30))
    while len(unused) >= 3:
        i = unused.pop(0)
        j = max(unused, key=lambda j: corr[i, j])
        unused.remove(j)
        k = max(unused, key=lambda k: corr[i, k]+corr[j, k])
        unused.remove(k)
        triples.append([i, j, k])
    p['triples'] = triples
    p['directions'] = fp16(np.array([np.linalg.eigh(H[np.ix_(idx, idx)])[1][:, 0] for idx in triples]))
    p['F'] = np.zeros(4096)
    for _ in range(3):
        search(p, lambda Q: layer_loss(W, Q, X), decoder=decode)
        c, s = p['q'].reshape(W.shape), fields(p)[0]
        num, den = np.zeros(4096), np.zeros(4096)
        for idx, b in zip(triples, p['directions']):
            key = 256*c[:, idx[0]].astype(int)+16*c[:, idx[1]]+c[:, idx[2]]
            v = s[:, idx[0], None]*b
            residual = W[:, idx]-affine(p)[:, idx]
            h = np.diag(H)[idx]
            np.add.at(num, key, np.sum(h*v*residual, 1))
            np.add.at(den, key, np.sum(h*v*v, 1))
        old, cost = p['F'].copy(), layer_loss(W, decode(p), X)
        p['F'] = fp16(num/np.maximum(den, 1e-12))
        if layer_loss(W, decode(p), X) > cost:
            p['F'] = old
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
