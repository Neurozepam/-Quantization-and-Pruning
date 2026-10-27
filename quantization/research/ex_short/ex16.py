import numpy as np
from common import grid, decode as affine, fields, fp16, mlp, mlp_example, mse


def decode(p):
    pg, pu = p['G'], p['U']
    a, b = pg['q'].reshape(pg['shape']), pu['q'].reshape(pu['shape'])
    G, U = affine(pg), affine(pu)
    G += fields(pg)[0] * p['RG'][a, b]
    U += fields(pu)[0] * p['RU'][a, b]
    return G, U, affine(p['D'])


def fit_corrections(p, G, U, cross):
    a = p['G']['q'].reshape(G.shape)
    b = p['U']['q'].reshape(U.shape)
    error_g = (G - affine(p['G'])) / fields(p['G'])[0]
    error_u = (U - affine(p['U'])) / fields(p['U'])[0]
    rg, ru = np.zeros((16, 16)), np.zeros((16, 16))
    for i in range(16):
        for j in range(16 if cross else 1):
            mask = (a == i) & ((b == j) if cross else True)
            if not mask.any():
                continue
            if cross:
                rg[i, j] = error_g[mask].mean()
                ru[i, j] = error_u[mask].mean()
            else:
                ru[i, :] = error_u[mask].mean()
    return fp16(rg), fp16(ru)


def quantize(X, G, U, D, cross=True):
    p = dict(G=grid(G), U=grid(U), D=grid(D),
             RG=np.zeros((16, 16), dtype=np.float16),
             RU=np.zeros((16, 16), dtype=np.float16))
    target = mlp(X, G, U, D)
    before = mse(mlp(X, *decode(p)), target)
    rg, ru = fit_corrections(p, G, U, cross)
    p['RG'], p['RU'] = rg, ru
    if mse(mlp(X, *decode(p)), target) >= before:
        p['RG'], p['RU'] = np.zeros_like(rg), np.zeros_like(ru)
    return p, mlp(X, *decode(p))


if __name__ == '__main__':
    X, G, U, D = mlp_example()
    print(mse(quantize(X, G, U, D)[1], mlp(X, G, U, D)))
