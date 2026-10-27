import numpy as np
from common import grid, decode, attention, mse


def encode(K, V):
    pk, pv = grid(K), grid(V)
    k, v = decode(pk), decode(pv)
    mk, mv, mqk, mqv = K.mean(0), V.mean(0), k.mean(0), v.mean(0)
    defect = (K-mk).T@(V-mv)-(k-mqk).T@(v-mqv)
    return dict(K=pk, V=pv, meanK=mk, meanV=mv, meanKq=mqk, meanVq=mqv, defect=defect)


def restore(p, Q):
    K, V = decode(p['K']), decode(p['V'])
    root = np.sqrt(K.shape[1])
    denominator = np.exp(Q@(K-p['meanKq']).T/root).sum(-1, keepdims=True)
    return attention(Q, K, V)+p['meanV']-p['meanVq']+Q@p['defect']/(root*denominator)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    Q, K, V = [rng.normal(size=s) for s in [(16, 4), (12, 4), (12, 4)]]
    p = encode(K, V)
    print(mse(restore(p, Q), attention(Q, K, V)))
