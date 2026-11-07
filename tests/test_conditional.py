"""Rate, conditional (body/conditional.py): it sees a dependence of the action on the reply, and only that."""
import numpy as np

from body.conditional import conditional, pairs


def _episode(contingent: bool, n_turns: int = 80, d: int = 40, seed: int = 0):
    """Alternating action (self) and reply (world) chunks. A contingent action copies a direction of the reply
    before it; a blind one plays the same actions against other replies."""
    rng = np.random.default_rng(seed)
    replies = rng.normal(size=(n_turns + 1, d))
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    actions = np.array([replies[t] @ A + 0.3 * rng.normal(size=d) for t in range(n_turns)])
    if not contingent:                                   # same actions, other replies
        replies = np.random.default_rng(seed + 1000).normal(size=(n_turns + 1, d))
    rows, slots, turns = [replies[0]], ["world"], [-1]
    for t in range(n_turns):
        rows += [actions[t], replies[t + 1]]
        slots += ["self", "world"]
        turns += [t, t]
    return np.array(rows), slots, turns


def test_pairs_take_the_world_between_two_actions():
    assert pairs(["world", "self", "world", "world", "self"], [-1, 0, 0, 0, 1]) == [(4, 1, [2, 3])]


def test_the_reply_saves_bits_only_when_the_action_follows_it():
    xs = {}
    for contingent in (True, False):
        X, slots, turns = _episode(contingent)
        xs[contingent] = np.nanmean(conditional(X, slots, turns, floor_var=1e-3)["X_c"])
    assert xs[True] > 1.0 and xs[True] > xs[False] + 1.0
    assert xs[False] < 0.5                                 # a useless predictor costs, never pays


def test_same_actions_same_replies_same_reading():
    X, slots, turns = _episode(True, seed=3)
    a = conditional(X, slots, turns, floor_var=1e-3)["X_c"]
    b = conditional(X.copy(), list(slots), list(turns), floor_var=1e-3)["X_c"]
    assert np.allclose(a, b, equal_nan=True)


def test_nothing_scored_before_the_burn_in_or_enough_pairs():
    X, slots, turns = _episode(True, n_turns=30)
    out = conditional(X, slots, turns, floor_var=1e-3, burn_turns=10, min_pairs=20)["X_c"]
    scored = [turns[i] for i in np.flatnonzero(~np.isnan(out))]
    assert scored and min(scored) >= 21


def test_the_conditional_form_reads_fresh_seeds_and_the_registered_one_is_unchanged():
    from body import rate_calibration as rc
    assert rc.seeds() == list(range(7000, 7040))                           # v1 as registered
    assert rc.seeds(form="conditional") == list(range(7100, 7140))
    assert rc.seeds(rehearsal=True, form="conditional")[0] == 907100
    assert not set(rc.seeds(form="conditional")) & set(rc.seeds())
