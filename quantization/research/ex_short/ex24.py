import numpy as np
from common import grid, decode as affine, fp16, example, layer_loss


def decode(p, role='output'):
    return affine(p['base'])*p[role]


def quantize(W, X):
    base = grid(W)
    Q = affine(base)
    H = X.T@X/len(X)
    ain = np.sum(W*Q, 1, keepdims=True)/np.maximum(np.sum(Q*Q, 1, keepdims=True), 1e-12)
    aout = np.sum((W@H)*Q, 1, keepdims=True)/np.maximum(np.sum((Q@H)*Q, 1, keepdims=True), 1e-12)
    p = dict(base=base, input=fp16(ain), output=fp16(aout))
    return p, decode(p, 'input'), decode(p, 'output')


if __name__ == '__main__':
    W, X = example()
    p, Ein, Eout = quantize(W, X)
    print(layer_loss(W, Eout, X), np.mean((Ein-W)**2))
