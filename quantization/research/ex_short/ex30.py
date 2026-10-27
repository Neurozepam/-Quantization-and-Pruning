import numpy as np
from common import grid, decode, softmax, attention, mse


def correction(logits, V, covariance):
    p = softmax(logits)
    diagonal = np.diag(covariance)
    Cp = p@covariance.T
    bias = .5*p*(diagonal-2*Cp+2*np.sum(p*Cp, -1, keepdims=True)-np.sum(p*diagonal, -1, keepdims=True))
    return bias@V


def quantize(K, V, queries):
    pk, pv = grid(K), grid(V)
    logits = queries@decode(pk).T/np.sqrt(K.shape[1])
    error = queries@(decode(pk)-K).T/np.sqrt(K.shape[1])
    centered = error-error.mean(0)
    covariance = centered.T@centered/len(error)
    p = dict(K=pk, V=pv, covariance=covariance)
    return p, attention(queries, decode(pk), decode(pv))-correction(logits, decode(pv), covariance)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    Q, K, V = [rng.normal(size=s) for s in [(16, 4), (12, 4), (12, 4)]]
    print(mse(quantize(K, V, Q)[1], attention(Q, K, V)))
