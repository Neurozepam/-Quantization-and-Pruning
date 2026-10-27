import numpy as np


def fraction(c):
    return ((c <= 3) | (c >= 12)).mean(axis=1, keepdims=True, dtype=np.float32)


def restore(c, s, z, a, b):
    t = fraction(c)
    return s * ((1 - t) * a[c] + t * b[c] - z)


def decode(p):
    return restore(p['codes'], p['scales'].astype(np.float32),
                   p['zeros'].astype(np.float32), p['A'].astype(np.float32),
                   p['B'].astype(np.float32)).reshape(p['shape'])


def fit_tables(w, c, s, z, importance, a, b):
    t = np.broadcast_to(fraction(c), w.shape).ravel()
    weight = (importance * s ** 2).astype(np.float64).ravel()
    target = (w / s + z).astype(np.float64).ravel()
    design = np.zeros((w.size, 32))
    i, code = np.arange(w.size), c.ravel()
    design[i, code], design[i, code + 16] = 1 - t, t
    C = np.kron(np.eye(2), np.tril(np.ones((16, 16))))
    weighted = design * np.sqrt(weight[:, None])
    mass = max(weight.sum(), 1e-30)
    M = weighted.T @ weighted / mass + 1e-9 * np.eye(32)
    rhs = design.T @ (weight * target) / mass + 1e-9 * np.r_[a, b]
    M, rhs = C.T @ M @ C, C.T @ rhs
    u = np.r_[a[0], np.diff(a), b[0], np.diff(b)].astype(np.float64)
    for _ in range(2000):
        previous = u.copy()
        for j in range(32):
            value = u[j] + (rhs[j] - M[j] @ u) / M[j, j]
            u[j] = value if j in (0, 16) else max(0, value)
        if np.max(np.abs(u - previous)) < 1e-10:
            break
    tables = (C @ u).astype(np.float16).astype(np.float32)
    aa, bb = tables[:16], tables[16:]
    before = np.sum(importance * (restore(c, s, z, a, b) - w) ** 2)
    after = np.sum(importance * (restore(c, s, z, aa, bb) - w) ** 2)
    if np.isfinite(tables).all() and after <= before * (1 + 1e-7):
        return aa, bb
    return a, b


def quantize(W, X, steps=6):
    W, X = np.asarray(W, dtype=np.float32), np.asarray(X, dtype=np.float32)
    if W.ndim != 2 or X.ndim != 2 or X.shape[1] != W.shape[1]:
        raise ValueError('W: outputs x inputs; X: samples x inputs')
    if min(W.shape) == 0 or len(X) == 0 or W.shape[1] % 128:
        raise ValueError('Nonempty inputs; row width must be divisible by 128')
    if not np.isfinite(W).all() or not np.isfinite(X).all():
        raise ValueError('Finite inputs required')
    w = W.reshape(-1, 128)
    low, high = w.min(axis=1, keepdims=True), w.max(axis=1, keepdims=True)
    native_scale = np.maximum(high - low, 1e-5) / 15
    s = native_scale.astype(np.float16).astype(np.float32)
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('Scale is not representable in FP16')
    z = np.clip(-np.rint(low / native_scale), 0, 15)
    c = np.clip(np.rint(w / s) + z, 0, 15).astype(np.uint8)
    H = X.T @ X / len(X)
    importance = np.broadcast_to(np.diag(H), W.shape).reshape(w.shape)
    base = np.arange(16, dtype=np.float32)

    def packet(a, b):
        return dict(shape=W.shape, codes=c.copy(), scales=s.astype(np.float16),
                    zeros=z.astype(np.uint8), A=a.astype(np.float16), B=b.astype(np.float16))

    def error(p):
        e = decode(p) - W
        return np.sum((e @ H) * e) / W.shape[0]

    best = packet(base, base)
    best_error = error(best)
    curve = .15 * np.sin(base * np.pi / 15)
    a, b = [(base + sign * curve).astype(np.float16).astype(np.float32) for sign in (1, -1)]
    for _ in range(steps):
        t = fraction(c)
        levels = (1 - t) * a + t * b
        proposal = np.argmin((w[:, :, None] / s[:, :, None] + z[:, :, None]
                              - levels[:, None, :]) ** 2, axis=2).astype(np.uint8)
        before = np.sum(importance * (restore(c, s, z, a, b) - w) ** 2, axis=1)
        after = np.sum(importance * (restore(proposal, s, z, a, b) - w) ** 2, axis=1)
        c = np.where((after <= before)[:, None], proposal, c)
        a, b = fit_tables(w, c, s, z, importance, a, b)
        candidate = packet(a, b)
        candidate_error = error(candidate)
        if candidate_error < best_error:
            best, best_error = candidate, candidate_error
    return best, decode(best)


if __name__ == '__main__':
    rng = np.random.default_rng(17)
    W = rng.normal(size=(8, 256)).astype(np.float32)
    X = rng.normal(size=(96, 256)).astype(np.float32)
    baseline, W0 = quantize(W, X, steps=0)
    packet, Wq = quantize(W, X)
    print('W:', W.shape, 'X:', X.shape)
    print('codes:', packet['codes'].shape, packet['codes'].dtype)
    print('A:', packet['A'])
    print('B:', packet['B'])
    print('t:', fraction(packet['codes']).ravel())
    print('affine calibration output MSE:', np.mean((X @ (W0 - W).T) ** 2))
    print('histogram calibration output MSE:', np.mean((X @ (Wq - W).T) ** 2))
