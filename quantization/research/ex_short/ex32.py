import numpy as np
from common import grid, decode, silu, silu_prime, fp16, mlp, mlp_example, mse


def encode(X, G, U, D):
    pg, pu, pd = grid(G), grid(U), grid(D)
    a, b = X@decode(pg).T, X@decode(pu).T
    ar, br = X@G.T, X@U.T
    coefficients, tau, covariance = [], [], []
    for j in range(len(G)):
        F = np.column_stack((np.ones(len(X)), a[:, j], b[:, j]))
        C = fp16(np.linalg.lstsq(F, np.column_stack((ar[:, j], br[:, j])), rcond=None)[0])
        residual = np.column_stack((ar[:, j], br[:, j]))-F@C
        coefficients.append(C)
        tau.append(np.std(residual[:, 0]))
        covariance.append(np.mean(residual[:, 0]*residual[:, 1]))
    da, db = silu(ar)-silu(a), br-b
    mixed = np.stack((da*b, silu(a)*db, da*db), -1)
    target = silu(ar)*br-silu(a)*b
    beta = fp16(np.array([np.linalg.lstsq(mixed[:, j], target[:, j], rcond=None)[0] for j in range(len(G))]))
    p = dict(G=pg, U=pu, D=pd, C=fp16(coefficients), tau=fp16(tau), covariance=fp16(covariance), beta=beta,
             biasG=np.zeros(len(D)), biasM=np.zeros(len(D)))
    target_output = mlp(X, G, U, D)
    p['biasG'] = fp16((target_output-restore(p, X)).mean(0))
    p['biasM'] = fp16((target_output-restore(p, X, False, G, U)).mean(0))
    return p


def restore(p, X, gaussian=True, raw_G=None, raw_U=None):
    a, b = X@decode(p['G']).T, X@decode(p['U']).T
    if gaussian:
        F = np.stack((np.ones_like(a), a, b), -1)
        means = np.einsum('nhi,hij->nhj', F, p['C'])
        ma, mb = means[..., 0], means[..., 1]
        nodes, weights = np.polynomial.hermite.hermgauss(5)
        points = ma[..., None]+np.sqrt(2)*p['tau'][None, :, None]*nodes
        z = mb*np.sum(silu(points)*weights, -1)/np.sqrt(np.pi)
        z += p['covariance']*np.sum(silu_prime(points)*weights, -1)/np.sqrt(np.pi)
    else:
        if raw_G is None or raw_U is None:
            raise ValueError('Mixed-error diagnostic requires raw reference projections')
        da, db = silu(X@raw_G.T)-silu(a), X@raw_U.T-b
        features = np.stack((da*b, silu(a)*db, da*db), -1)
        z = silu(a)*b+np.sum(features*p['beta'], -1)
    return z@decode(p['D']).T+p['biasG' if gaussian else 'biasM']


if __name__ == '__main__':
    X, G, U, D = mlp_example()
    p = encode(X, G, U, D)
    print(mse(restore(p, X), mlp(X, G, U, D)))
