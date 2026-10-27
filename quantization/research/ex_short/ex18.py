import numpy as np
from common import grid, decode, fields, mlp, mlp_example, mse


def quantize(X, G, U, D):
    pg, pu, pd = grid(G), grid(U), grid(D)
    su, zu = fields(pu)
    sd, zd = fields(pd)
    target = mlp(X, G, U, D)
    best = mse(mlp(X, decode(pg), decode(pu), decode(pd)), target)
    for channel in range(len(U)):
        boundaries = []
        for level in range(15):
            boundaries.extend((su[channel]*(level+.5-zu[channel])/np.where(U[channel] == 0, np.inf, U[channel])).tolist())
            boundaries.extend((D[:, channel]/np.where(sd[:, channel]*(level+.5-zd[:, channel]) == 0, np.inf, sd[:, channel]*(level+.5-zd[:, channel]))).tolist())
        events = np.unique([.5, 2., *[x for x in boundaries if .5 < x < 2]])
        candidates = np.r_[1., events, (events[:-1]+events[1:])/2]
        old_u, old_d = pu['q'].copy(), pd['q'].copy()
        chosen = old_u.copy(), old_d.copy()
        for alpha in candidates:
            u, d = U.copy(), D.copy()
            u[channel] *= alpha
            d[:, channel] /= alpha
            pu['q'] = np.clip(np.rint(u/su)+zu, 0, 15).astype(np.uint8).reshape(pu['q'].shape)
            pd['q'] = np.clip(np.rint(d/sd)+zd, 0, 15).astype(np.uint8).reshape(pd['q'].shape)
            error = mse(mlp(X, decode(pg), decode(pu), decode(pd)), target)
            if error < best:
                best, chosen = error, (pu['q'].copy(), pd['q'].copy())
        pu['q'], pd['q'] = chosen
    return [pg, pu, pd], mlp(X, decode(pg), decode(pu), decode(pd))


if __name__ == '__main__':
    X, G, U, D = mlp_example()
    print(mse(quantize(X, G, U, D)[1], mlp(X, G, U, D)))
