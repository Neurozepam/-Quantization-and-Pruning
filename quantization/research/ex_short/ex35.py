import numpy as np
from common import grid, decode as affine, sqrt_psd, polar, example, mse


def encode(W, whiten=True):
    gram = W@W.T
    root = sqrt_psd(gram)
    transformed = sqrt_psd(gram, inverse=True)@W if whiten else W
    return dict(base=grid(transformed), root=root, whiten=whiten)


def decode(p):
    Q = affine(p['base'])
    return p['root']@polar(Q if p['whiten'] else p['root']@Q)


if __name__ == '__main__':
    W, X = example()
    print(mse(decode(encode(W)), W))
