import numpy as np
from common import grid, fp16, fit_mixture, layer_loss, example


def unpack(s):
    word = s.astype(np.float16).view(np.uint16)
    scale = (word & 0x7ffc).view(np.float16).astype(float)
    tag = (word & 3) + (word >> 15) * 4
    return scale, tag


def decode(p):
    s, tag = unpack(p['s'])
    t = tag / 7
    return (s*((1-t)*p['A'][p['q']]+t*p['B'][p['q']]-p['z'])).reshape(p['shape'])


def quantize(W, X):
    p = grid(W)
    word = p['s'].astype(np.float16).view(np.uint16)
    word = np.clip((word.astype(int)+2) & ~3, 4, 0x7bfc).astype(np.uint16)
    s = word.view(np.float16).astype(float)
    w, base = W.reshape(p['q'].shape), np.arange(16.)
    p['A'], p['B'] = fp16(base+.15*np.sin(base)), fp16(base-.15*np.sin(base))
    h = np.broadcast_to(np.mean(X*X, 0), W.shape).reshape(w.shape)
    for _ in range(6):
        best = np.full(s.shape, np.inf)
        tags = np.zeros(s.shape, dtype=np.uint16)
        c = p['q'].copy()
        for tag in range(8):
            levels = (1-tag/7)*p['A'] + tag/7*p['B']
            proposal = np.argmin((w[..., None]/s[..., None]+p['z'][..., None]-levels)**2, -1)
            error = np.sum(h*(s*(levels[proposal]-p['z'])-w)**2, -1, keepdims=True)
            use = error < best
            c = np.where(use, proposal, c)
            tags, best = np.where(use, tag, tags), np.minimum(best, error)
        p['q'] = c.astype(np.uint8)
        p['s'] = (word | (tags & 3) | ((tags >> 2) << 15)).view(np.float16)
        p['A'], p['B'] = fit_mixture(w, c, tags/7, s, p['z'], p['A'], p['B'], h)
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
