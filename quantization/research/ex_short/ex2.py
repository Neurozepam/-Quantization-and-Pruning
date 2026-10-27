import numpy as np
from common import grid, fields, fit_mixture, fp16, example, layer_loss


def decode(p):
    c = p['q'].reshape(-1, p['group'])
    t = (c.sum(1, keepdims=True) % 2)
    return (p['s'].reshape(-1, 1) * (np.where(t, p['B'][c], p['A'][c])
            - p['z'].reshape(-1, 1))).reshape(p['shape'])


def quantize(W, X):
    p = grid(W)
    w = W.reshape(-1, p['group'])
    s, z = p['s'].reshape(-1, 1), p['z'].reshape(-1, 1)
    h = np.broadcast_to(np.mean(X * X, 0), W.shape).reshape(w.shape)
    base = np.arange(16.)
    p['A'], p['B'] = fp16(base + .15 * np.sin(base)), fp16(base - .15 * np.sin(base))
    for _ in range(6):
        options, costs = [], []
        for bit, table in enumerate((p['A'], p['B'])):
            cost = (w[:, :, None] - s[:, :, None] * (table - z[:, :, None])) ** 2 * h[:, :, None]
            c = cost.argmin(-1)
            for row in range(len(c)):
                if c[row].sum() % 2 != bit:
                    delta = cost[row] - cost[row, np.arange(c.shape[1]), c[row]][:, None]
                    delta[:, :] = np.where((np.arange(16) % 2) != (c[row, :, None] % 2), delta, np.inf)
                    j, k = np.unravel_index(delta.argmin(), delta.shape)
                    c[row, j] = k
            options.append(c)
            costs.append(np.sum(h * (s * (table[c] - z) - w) ** 2, 1))
        t = (costs[1] < costs[0])[:, None]
        c = np.where(t, options[1], options[0])
        p['q'] = c.astype(np.uint8).reshape(p['q'].shape)
        a, b = fit_mixture(w, c, t, s, z, p['A'], p['B'], h)
        old = layer_loss(W, decode(p), X)
        aa, bb = p['A'], p['B']
        p['A'], p['B'] = a, b
        if layer_loss(W, decode(p), X) > old:
            p['A'], p['B'] = aa, bb
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    p, Q = quantize(W, X)
    print(layer_loss(W, Q, X))
