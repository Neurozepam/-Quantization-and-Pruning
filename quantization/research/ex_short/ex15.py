import numpy as np
from common import grid, decode, silu, mlp, mlp_example, mse


def quantize(X, G, U, D, max_distortion=.02):
    gate = silu(X@G.T)
    distance = np.mean((gate[:, :, None]-gate[:, None, :])**2, 0)
    np.fill_diagonal(distance, np.inf)
    i, j = np.unravel_index(distance.argmin(), distance.shape)
    target = mlp(X, G, U, D)
    baseline = [grid(A) for A in (G, U, D)]
    best, score = baseline, mse(mlp(X, *map(decode, baseline)), target)
    for angle in (-.2, -.1, .1, .2):
        R = np.eye(len(U))
        c, s = np.cos(angle), np.sin(angle)
        R[np.ix_([i, j], [i, j])] = [[c, -s], [s, c]]
        UU, DD = R@U, D@R.T
        if mse(mlp(X, G, UU, DD), target) > max_distortion:
            continue
        packets = [grid(A) for A in (G, UU, DD)]
        error = mse(mlp(X, *map(decode, packets)), target)
        if error < score:
            best, score = packets, error
    return best, mlp(X, *map(decode, best))


if __name__ == '__main__':
    X, G, U, D = mlp_example()
    print(mse(quantize(X, G, U, D)[1], mlp(X, G, U, D)))
