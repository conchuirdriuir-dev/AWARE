"""Blind structure, round 6. Uses only numpy and the standard library."""
import numpy as np

N, K, D, T0 = 500, 32, 1.0, 200


def _v(rng, length):
    return rng.normal(scale=length / np.sqrt(K), size=K)


def generate(rng):
    c0 = rng.normal(scale=10.0, size=K)
    s = np.zeros((N, K)); E = np.zeros((N, K)); E_in = np.zeros((N, K))
    H = np.zeros((N, K)); m = np.zeros(N, dtype=bool)
    walk = np.zeros(K)
    s[0] = c0 + _v(rng, D)
    h = None
    library = []          # contents of arrivals in the reference stretch
    idx = 0
    for t in range(N):
        u = rng.random()
        if t < 40 or 460 <= t < 480:
            arrive = True
        else:
            arrive = u < 0.4
        walk = walk + _v(rng, 0.3 * D)
        jump = _v(rng, 3.0 * D)
        noise = _v(rng, (2.0 if t < 40 else 1.0) * D)
        if arrive:
            fresh = c0 + walk + jump
            if t < T0:
                content = fresh
                if t >= 60:
                    library.append(content.copy())
            else:
                content = library[idx % len(library)]
                idx += 1
            x = content + noise
            E[t] = x; m[t] = True; h = x
            E_in[t] = 0.5 * x + 0.5 * s[t]
        H[t] = h
        step = _v(rng, D)
        if t + 1 < N:
            s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (h - c0) + step
    c = E[np.flatnonzero(m)[:10]].mean(axis=0)
    return {"s": s, "E": E, "m": m, "E_in": E_in, "H": H, "c": c,
            "t0": T0, "kind": "blind"}
