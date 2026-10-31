"""Blind structure round 7a. Uses only numpy and the standard library."""
import numpy as np

N, DIM, D, T0 = 500, 32, 1.0, 200


def _vec(rng, length):
    return rng.normal(scale=length / np.sqrt(DIM), size=DIM)


def generate(rng):
    c0 = rng.normal(scale=10.0, size=DIM)
    ack_offset = _vec(rng, 2.0 * D)  # fixed content of a content-free acknowledgement
    s = np.zeros((N, DIM)); E = np.zeros((N, DIM)); H = np.zeros((N, DIM))
    m = np.zeros(N, bool); elic = np.zeros(N, bool)
    walk = np.zeros(DIM)
    s[0] = c0 + _vec(rng, D)
    h = c0.copy()
    for t in range(N):
        walk = walk + _vec(rng, 0.3 * D)
        u = rng.random()
        if t < 40:
            arrive, kind = True, "content"
            m[t] = True
        elif 460 <= t < 480:
            arrive, kind = True, "content"
            elic[t] = True
        elif t < T0:
            arrive, kind = u < 0.4, "content"
        else:
            if u < 0.30:
                arrive, kind = True, "content"
                elic[t] = True
            elif u < 0.40:
                arrive, kind = True, "ack"
            else:
                arrive, kind = False, None
        if arrive:
            m[t] = True
            if kind == "content":
                rn = 2.0 * D if t < 40 else D
                e = c0 + walk + _vec(rng, 3.0 * D) + _vec(rng, rn)
            else:
                e = c0 + ack_offset + _vec(rng, 0.4 * D)
            E[t] = e
            h = e
        H[t] = h
        if t + 1 < N:
            s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (h - c0) + _vec(rng, D)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    c = E[np.flatnonzero(m)[:10]].mean(axis=0)
    return {"s": s, "E": E, "m": m, "elicited": elic, "E_in": E_in, "H": H,
            "c": c, "t0": T0, "kind": "blind"}
