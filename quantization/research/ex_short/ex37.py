import numpy as np
from common import grid, decode, fields, fp16, example, mse


def encode(W, Xs, targets):
    p = grid(W)
    Q = decode(p)
    s, z = fields(p)
    direction = s*(1-2*(p['q'].reshape(W.shape).astype(int)%2))
    residual = np.concatenate([(X@Q.T-Y).ravel() for X, Y in zip(Xs, targets)])
    alternating = np.concatenate([((-1)**i*X@direction.T).ravel() for i, X in enumerate(Xs)])
    coefficient = -float(residual@alternating)/max(float(alternating@alternating), 1e-12)
    return dict(base=p, coefficient=float(fp16(coefficient)))


def restore(p, step):
    base = p['base']
    s, _ = fields(base)
    direction = s*(1-2*(base['q'].reshape(base['shape']).astype(int)%2))
    return decode(base)+(-1)**step*p['coefficient']*direction


if __name__ == '__main__':
    W, X = example()
    Xs = [X*(1+.1*i) for i in range(4)]
    p = encode(W, Xs, [A@W.T for A in Xs])
    print(np.mean([mse(A@restore(p, i).T, A@W.T) for i, A in enumerate(Xs)]))
