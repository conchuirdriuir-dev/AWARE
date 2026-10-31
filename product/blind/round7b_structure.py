"""Blind structure, round 7b. numpy + stdlib only."""
import numpy as np

T, d, D = 500, 32, 1.0


def _v(rng, L):
    return rng.normal(scale=L / np.sqrt(d), size=d)


def _unit(rng):
    u = rng.normal(size=d)
    return u / np.linalg.norm(u)


def generate(rng):
    c0 = _unit(rng) * 10.0
    s = np.zeros((T, d)); E = np.zeros((T, d)); H = np.zeros((T, d))
    m = np.zeros(T, bool); el = np.zeros(T, bool)
    w = np.zeros(d)
    s[0] = c0 + _v(rng, D)
    h = c0.copy()

    # ---- steps 0..199: healthy baseline
    for t in range(200):
        w = w + _v(rng, 0.3 * D)
        if t < 40:
            arrive, noise = True, 2 * D
        else:
            arrive, noise = rng.random() < 0.4, D
        if arrive:
            h = c0 + w + _unit(rng) * 3 * D + _v(rng, noise)
            E[t] = h; m[t] = True
        H[t] = h
        if t + 1 < T:
            s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (h - c0) + _v(rng, D)

    # ---- from 200: fixed cycle of K actions, each answered by the same result
    K = 9
    gaps = [2, 3, 2, 3, 2, 3, 2, 3, 3]
    base = c0 + w
    R = np.array([base + _unit(rng) * 3 * D for _ in range(K)])
    sched = {}
    t, i = 200, 0
    while t < T:
        sched[t] = i % K
        t += gaps[i % K]; i += 1

    loop_h = h.copy()
    for t in range(200, 460):
        if t in sched:
            loop_h = R[sched[t]] + _v(rng, D)
            E[t] = loop_h; m[t] = True; el[t] = True
        h = loop_h
        H[t] = h
        s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (h - c0) + _v(rng, D)

    # ---- from 460: everything below is drawn after all pre-460 values
    probe = _unit(rng)
    for t in range(460, T):
        if 460 <= t < 480:
            w = w + _v(rng, 0.3 * D)
            E[t] = c0 + w + _unit(rng) * 3 * D + _v(rng, D)
            m[t] = True; el[t] = True
            h = E[t]
            if t in sched:
                loop_h = R[sched[t]] + _v(rng, D)
        elif t in sched:
            loop_h = R[sched[t]] + _v(rng, D)
            E[t] = loop_h; m[t] = True; el[t] = True
            h = loop_h
        H[t] = h
        if t + 1 < T:
            s[t + 1] = c0 + 0.5 * (s[t] - c0) + 0.5 * (loop_h - c0) + _v(rng, D)
            if t == 460:
                s[t + 1] = s[t + 1] + 3 * D * probe

    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    c = E[np.where(m)[0][:10]].mean(axis=0)
    return {"s": s, "E": E, "m": m, "elicited": el, "probe": probe,
            "E_in": E_in, "H": H, "c": c, "t0": 200, "kind": "blind"}
