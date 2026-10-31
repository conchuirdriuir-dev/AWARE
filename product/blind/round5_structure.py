"""Blind structure, round 5. numpy + stdlib only."""
import numpy as np

T, N, D = 500, 32, 1.0


def _vec(rng, length):
    return rng.normal(scale=length / np.sqrt(N), size=N)


def generate(rng, _probe_scale=1.0):
    c0 = _vec(rng, 10.0)
    s = np.zeros((T, N)); E = np.zeros((T, N)); H = np.zeros((T, N))
    m = np.zeros(T, dtype=bool)
    walk = np.zeros(N)
    s[0] = c0 + _vec(rng, D)
    h = None
    cyc = None
    for t in range(T):
        # partner side
        if t < 40 or t >= 460:
            arrive = True
        else:
            arrive = rng.random() < 0.4
        if arrive:
            walk = walk + _vec(rng, 0.3 * D)
            read = 2.0 * D if t < 40 else D
            scale = _probe_scale if t >= 460 else 1.0
            e = c0 + scale * (walk + _vec(rng, 3.0 * D)) + _vec(rng, read)
            E[t] = e; m[t] = True; h = e
        H[t] = h
        if t == T - 1:
            break
        # entity side
        if t < 200:
            target = h
        else:
            if cyc is None:
                cyc = s[194:200].copy()
            target = cyc[(t - 200) % len(cyc)]
        s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (target - c0) + _vec(rng, D)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    c = E[m][:10].mean(axis=0)
    return {"s": s, "E": E, "m": m, "E_in": E_in, "H": H, "c": c,
            "t0": 200, "kind": "blind"}
