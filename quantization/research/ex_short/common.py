import itertools
import numpy as np


def fp16(x):
    return np.asarray(x).astype(np.float16)


def grid(W, bits=4, group=128):
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    g = min(group, W.shape[1])
    if W.shape[1] % g:
        raise ValueError('Use full row-local groups')
    w = W.reshape(W.shape[0], -1, g)
    low, high = w.min(-1, keepdims=True), w.max(-1, keepdims=True)
    native = np.maximum(high - low, 1e-5) / (2 ** bits - 1)
    s = fp16(native)
    z = np.clip(-np.rint(low / native), 0, 2 ** bits - 1).astype(np.uint8)
    q = np.clip(np.rint(w / s) + z, 0, 2 ** bits - 1).astype(np.uint8)
    return dict(q=q, s=s, z=z, shape=W.shape, bits=bits, group=g)


def decode(p):
    return (p['s'] * (p['q'].astype(float) - p['z'])).reshape(p['shape'])


def fields(p):
    return (np.broadcast_to(p['s'], p['q'].shape).reshape(p['shape']),
            np.broadcast_to(p['z'], p['q'].shape).reshape(p['shape']))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def layer_loss(W, Q, X):
    return mse(X @ W.T, X @ Q.T)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def silu(x):
    return x * sigmoid(x)


def silu_prime(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def rms(x, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def mlp(X, G, U, D):
    return (silu(X @ G.T) * (X @ U.T)) @ D.T


def attention(Q, K, V):
    return softmax(Q @ K.T / np.sqrt(K.shape[1])) @ V


def search(p, loss, passes=1, decoder=decode):
    best = loss(decoder(p))
    for _ in range(passes):
        for idx in np.ndindex(p['q'].shape):
            old = int(p['q'][idx])
            chosen = old
            for c in (old - 1, old + 1):
                if not 0 <= c < 2 ** p['bits']:
                    continue
                p['q'][idx] = c
                value = loss(decoder(p))
                if value < best:
                    chosen, best = c, value
            p['q'][idx] = chosen
    return p


def joint_search(packets, decoder, loss, sites):
    best = loss(decoder(packets))
    for positions in sites:
        previous = [int(packets[k]['q'][idx]) for k, idx in positions]
        chosen = previous.copy()
        choices = [[max(0, c - 1), c, min(15, c + 1)] for c in previous]
        for values in itertools.product(*choices):
            for (k, idx), value in zip(positions, values):
                packets[k]['q'][idx] = value
            cost = loss(decoder(packets))
            if cost < best:
                best, chosen = cost, values
        for (k, idx), value in zip(positions, chosen):
            packets[k]['q'][idx] = value
    return packets


def polar(A):
    U, _, V = np.linalg.svd(A, full_matrices=False)
    return U @ V


def sqrt_psd(A, inverse=False):
    d, U = np.linalg.eigh((A + A.T) / 2)
    d = np.maximum(d, 1e-10)
    return (U * (1 / np.sqrt(d) if inverse else np.sqrt(d))) @ U.T


def regression(X, Y, ridge=1e-6):
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def nearest(W, levels):
    return np.argmin((W[..., None] - levels) ** 2, axis=-1).astype(np.uint8)


def fit_mixture(w, c, t, s, z, A, B, importance):
    s, A, B = s.astype(float), A.astype(float), B.astype(float)
    n = w.size
    design = np.zeros((n, 32))
    ids, code = np.arange(n), c.ravel()
    tt = np.broadcast_to(t, w.shape).ravel()
    design[ids, code], design[ids, code + 16] = 1 - tt, tt
    weight = (importance * s ** 2).ravel()
    target = (w / s + z).ravel()
    C = np.kron(np.eye(2), np.tril(np.ones((16, 16))))
    M = design.T @ (weight[:, None] * design) + 1e-7 * np.eye(32)
    b = design.T @ (weight * target) + 1e-7 * np.r_[A, B]
    M, b = C.T @ M @ C, C.T @ b
    u = np.r_[A[0], np.diff(A), B[0], np.diff(B)].astype(np.float64)
    for _ in range(200):
        old = u.copy()
        for j in range(32):
            value = u[j] + (b[j] - M[j] @ u) / M[j, j]
            u[j] = value if j in (0, 16) else max(0, value)
        if np.max(np.abs(u - old)) < 1e-9:
            break
    tables = fp16(C @ u)
    return tables[:16], tables[16:]


def example(seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 16)), rng.normal(size=(24, 16))


def mlp_example(seed=7):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(16, 8)), rng.normal(size=(6, 8)),
            rng.normal(size=(6, 8)), rng.normal(size=(8, 6)))
