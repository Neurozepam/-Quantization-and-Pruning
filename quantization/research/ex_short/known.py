import sys
import numpy as np
from common import grid, decode, fields, search, layer_loss, fp16, example


def gptq(W, X):
    p = grid(W)
    s, z = fields(p)
    H = X.T@X/len(X)
    H += max(.01*np.mean(np.diag(H)), 1e-8)*np.eye(len(H))
    R = np.linalg.cholesky(np.linalg.inv(H)).T
    work, Q = W.copy(), np.zeros_like(W)
    for j in range(W.shape[1]):
        codes = np.clip(np.rint(work[:, j]/s[:, j])+z[:, j], 0, 15)
        Q[:, j] = s[:, j]*(codes-z[:, j])
        error = (work[:, j]-Q[:, j])/R[j, j]
        work[:, j:] -= error[:, None]*R[j, j:]
    p['q'] = np.clip(np.rint(Q/s)+z, 0, 15).astype(np.uint8).reshape(p['q'].shape)
    return p, decode(p)


def awq(W, X):
    statistic = np.maximum(np.mean(np.abs(X), 0), 1e-6)
    best, selected = np.inf, None
    for alpha in np.linspace(0, 1, 11):
        scale = statistic**alpha
        scale /= np.sqrt(scale.min()*scale.max())
        p = grid(W*scale)
        Q = decode(p)/scale
        error = layer_loss(W, Q, X)
        if error < best:
            best, selected = error, (p, scale, Q)
    return selected


def smoothquant(W, X, alpha=.5):
    a = np.maximum(np.max(np.abs(X), 0), 1e-6)
    w = np.maximum(np.max(np.abs(W), 0), 1e-6)
    scale = a**alpha/w**(1-alpha)
    pw, px = grid(W*scale, bits=8), grid(X/scale, bits=8)
    return dict(W=pw, X=px, scale=scale), decode(px)@decode(pw).T


def correction(X, residual, rank=4):
    xm, ym = X.mean(0), residual.mean(0)
    A, B = X-xm, residual-ym
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    ridge = max(1e-4*np.sum(S*S)/len(X), 1e-8)
    Z = (S/np.sqrt(S*S+ridge))[:, None]*(U.T@B)
    V = np.linalg.svd(Z, full_matrices=False)[2][:rank].T
    left = Vh.T@((S/(S*S+ridge))[:, None]*((U.T@B)@V))
    left, right = fp16(left), fp16(V.T)
    bias = fp16(ym-xm@left@right)
    return left, right, bias


def clipping(A, tail):
    target = tail(A)
    best, chosen = np.inf, None
    for ratio in np.linspace(.2, 1, 17):
        threshold = ratio*np.max(np.abs(A))
        scale = threshold/7
        Q = scale*np.clip(np.rint(A/scale), -7, 7)
        error = np.mean((tail(Q)-target)**2)
        if error < best:
            best, chosen = error, (threshold, Q)
    return chosen


if __name__ == '__main__':
    W, X = example()
    mode = sys.argv[1] if len(sys.argv)>1 else 'rtn'
    if mode == 'rtn':
        print(layer_loss(W, decode(grid(W)), X))
    elif mode == 'gptq':
        print(layer_loss(W, gptq(W, X)[1], X))
    elif mode == 'awq':
        print(layer_loss(W, awq(W, X)[2], X))
    elif mode == 'smoothquant':
        print(np.mean((smoothquant(W, X)[1]-X@W.T)**2))
    elif mode == 'full_h':
        p = search(grid(W), lambda Q: layer_loss(W, Q, X))
        print(layer_loss(W, decode(p), X))
    elif mode == 'groups':
        print([(g, layer_loss(W, decode(grid(W, group=g)), X)) for g in (16, 8, 4)])
    elif mode == 'correction':
        Q = decode(grid(W))
        a, b, bias = correction(X, X@(W-Q).T, rank=2)
        print(np.mean((X@Q.T+X@a@b+bias-X@W.T)**2))
    elif mode == 'clipping':
        threshold, Q = clipping(X, lambda A: A@W.T)
        print(threshold, np.mean(((Q-X)@W.T)**2))
    else:
        raise ValueError(mode)
