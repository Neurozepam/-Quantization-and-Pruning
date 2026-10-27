import numpy as np
from common import grid, decode as affine, polar, fp16, example, mse


def encode(W):
    eigenvalues, U = np.linalg.eigh(W@W.T)
    return dict(base=grid(U.T@W), U=U, eigenvalues=fp16(np.maximum(eigenvalues, 1e-8)))


def decode(p, noise_metric=False):
    Q = affine(p['base'])
    root = np.sqrt(np.maximum(p['eigenvalues'], 1e-10))[:, None]
    restored = root*polar(Q/root if noise_metric else root*Q)
    return p['U']@restored


if __name__ == '__main__':
    W, X = example()
    print(mse(decode(encode(W)), W))
