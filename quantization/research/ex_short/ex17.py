import numpy as np
from common import grid, decode, joint_search, silu, mlp, mlp_example, mse


def quantize(X, G, U, D, full_output=True):
    p = [grid(G), grid(U)]
    pd = grid(D)
    def output(p):
        z = silu(X@decode(p[0]).T)*(X@decode(p[1]).T)
        return z@decode(pd).T if full_output else z
    target = mlp(X, G, U, D) if full_output else silu(X@G.T)*(X@U.T)
    sites = [[(0, idx), (1, idx)] for idx in np.ndindex(p[0]['q'].shape)]
    joint_search(p, output, lambda y: mse(y, target), sites)
    return [*p, pd], output(p)


if __name__ == '__main__':
    X, G, U, D = mlp_example()
    print(mse(quantize(X, G, U, D)[1], mlp(X, G, U, D)))
