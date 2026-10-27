import math
import numpy as np
from common import fp16, example, mse, softmax


def encode(W, group=16):
    w = W.reshape(-1, group)
    winner = np.argmax(np.abs(w), 1)
    scale = fp16(-w[np.arange(len(w)), winner])
    scale = np.where(scale == 0, 1e-6, scale)
    levels = np.linspace(-1, 1, 16)
    c = np.argmin((w[..., None]/scale[:, None, None]-levels)**2, -1).astype(np.uint8)
    return dict(q=c, scale=scale, levels=levels, shape=W.shape)


def decode(p):
    c, L = p['q'], p['levels']
    hist = np.array([np.bincount(row, minlength=16) for row in c])
    if np.any(hist[:, 0] == 0):
        raise ValueError('Signed-absmax winner must have code zero')
    nodes, weights = np.polynomial.legendre.leggauss(96)
    t, weights = (nodes+1)*8, weights*8
    edges = np.r_[-1., (L[:-1]+L[1:])/2, 1.]
    y, yw = np.polynomial.legendre.leggauss(48)
    points = (edges[:-1, None]+edges[1:, None])/2+(edges[1:]-edges[:-1])[:, None]*y/2
    density = np.exp(-.5*(t[:, None, None]*points)**2)
    mass = np.sum(density*yw, -1)*(edges[1:]-edges[:-1])/2
    probability = np.maximum(t[:, None]*mass/np.sqrt(2*np.pi), 1e-300)
    mean = np.sum(density*yw*points, -1)/np.maximum(np.sum(density*yw, -1), 1e-300)
    count = hist.copy()
    count[:, 0] -= 1
    posterior = softmax(count@np.log(probability).T-.5*t*t+np.log(weights))
    centers = posterior@mean
    n0 = hist[:, 0]
    centers[:, 0] = -1/n0+(1-1/n0)*centers[:, 0]
    return (p['scale'][:, None]*np.take_along_axis(centers, c, 1)).reshape(p['shape'])


if __name__ == '__main__':
    W, X = example()
    p = encode(W)
    print(mse(decode(p), W))
