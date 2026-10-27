import numpy as np
from common import grid, decode, fields, softmax, example, mse


def logits(W, ids, R):
    return np.tanh(W[ids]@R)@W.T


def quantize(W, ids, R, input_budget=.02):
    p = grid(W)
    reference = softmax(logits(W, ids, R))
    def loss(Q):
        value = softmax(logits(Q, ids, R))
        return float(np.mean(np.sum(reference*np.log(np.maximum(reference, 1e-30)/np.maximum(value, 1e-30)), -1)))
    base = decode(p)
    hidden = np.tanh(base[ids]@R)
    base_probability = softmax(logits(base, ids, R))
    gradient = (base_probability-reference)/len(ids)
    current, winner, predicted = loss(base), None, 0.
    for index in np.ndindex(p['q'].shape):
        old = int(p['q'][index])
        for code in (old-1, old+1):
            if not 0 <= code < 16:
                continue
            p['q'][index] = code
            proposal = decode(p)
            delta = proposal-base
            change = hidden@delta.T + ((1-hidden*hidden)*(delta[ids]@R))@base.T
            score = float(np.sum(gradient*change))
            if score < predicted and mse(proposal[ids], W[ids]) <= input_budget:
                predicted, winner = score, p['q'].copy()
        p['q'][index] = old
    if winner is not None:
        old = p['q'].copy()
        p['q'] = winner
        if loss(decode(p)) >= current:
            p['q'] = old
    return p, decode(p)


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    W = rng.normal(size=(8, 8))
    ids, R = np.arange(16)%8, rng.normal(size=(8, 8))*.1
    print(mse(logits(quantize(W, ids, R)[1], ids, R), logits(W, ids, R)))
