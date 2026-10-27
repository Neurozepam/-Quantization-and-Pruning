import numpy as np
from common import grid, layer_loss, example


def decode(p):
    phase = np.repeat(np.stack((p['phase'], -p['phase']), -1), 1, axis=0)
    phase = phase.reshape(p['shape'][0], -1, 1)
    return (p['s'] * (p['q'].astype(float) + phase - p['z'])).reshape(p['shape'])


def quantize(W, X, group=8):
    p = grid(W, group=group)
    if p['q'].shape[1] % 2:
        raise ValueError('Need pairs of groups')
    p['phase'] = np.zeros((len(W), p['q'].shape[1]//2))
    w = W.reshape(p['q'].shape)
    for row in range(len(W)):
        for pair in range(p['phase'].shape[1]):
            best, chosen = np.inf, None
            for phase in (0., .25, -.25):
                p['phase'][row, pair] = phase
                sl = slice(2*pair, 2*pair+2)
                shift = np.array([phase, -phase])[:, None]
                p['q'][row, sl] = np.clip(np.rint(w[row, sl]/p['s'][row, sl]+p['z'][row, sl]-shift), 0, 15)
                loss = layer_loss(W, decode(p), X)
                if loss < best:
                    best, chosen = loss, (phase, p['q'][row, sl].copy())
            p['phase'][row, pair], p['q'][row, sl] = chosen
    return p, decode(p)


if __name__ == '__main__':
    W, X = example()
    print(layer_loss(W, quantize(W, X)[1], X))
