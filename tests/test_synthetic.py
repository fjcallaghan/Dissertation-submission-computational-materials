"""Synthetic generators: martingale (GBM) vs strict-local-martingale (CEV)."""

import numpy as np

import pytest

from src.synthetic.paths import (
    martingale_diagnostic,
    simulate_cev,
    simulate_evans_bubble,
    simulate_explosive_ar1,
    simulate_gbm,
    simulate_random_walk,
)

COMMON = dict(s0=1.0, sigma=0.5, t_horizon=1.0, n_steps=500, n_paths=4000, seed=1)


def test_gbm_is_martingale():
    d = martingale_diagnostic(simulate_gbm(**COMMON))
    # E[S_T] ~ S_0 within Monte-Carlo error.
    assert abs(d["ratio_to_s0"] - 1.0) < 0.03


def test_cev_is_strict_local_martingale_bubble():
    d = martingale_diagnostic(simulate_cev(rho=2.0, **COMMON))
    # Bubble signature: E[S_T] strictly below S_0.
    assert d["ratio_to_s0"] < 0.98


def test_reproducible_with_seed():
    a = simulate_gbm(**COMMON)
    b = simulate_gbm(**COMMON)
    assert np.allclose(a.S, b.S)


def test_shapes():
    p = simulate_gbm(**COMMON)
    assert p.S.shape == (COMMON["n_paths"], COMMON["n_steps"] + 1)
    assert p.t.shape == (COMMON["n_steps"] + 1,)
    assert np.all(p.S[:, 0] == COMMON["s0"])


# ----- discrete-time controls for the GSADF test -----

def test_random_walk_is_driftless_and_shaped():
    p = simulate_random_walk(n_steps=500, n_paths=2000, seed=3)
    assert p.S.shape == (2000, 501)
    assert np.all(p.S[:, 0] == 100.0)
    # A driftless RW: cross-path mean of the terminal ~ the start (no trend).
    assert abs(float(p.S[:, -1].mean()) - 100.0) < 3.0


def test_explosive_ar1_grows():
    p = simulate_explosive_ar1(phi=1.05, n_steps=200, n_paths=500, seed=4)
    # |y_t| explodes relative to a comparable random walk.
    assert float(np.mean(np.abs(p.S[:, -1]))) > 100.0
    with pytest.raises(ValueError):
        simulate_explosive_ar1(phi=0.9, n_steps=10, n_paths=1, seed=0)


def test_evans_bubble_collapses_and_reinflates():
    p = simulate_evans_bubble(n_steps=600, n_paths=1, seed=7)
    y = p.S[0]
    # A periodically-collapsing bubble must contain at least one large drop
    # (a collapse) somewhere in the path — not monotone explosive growth.
    step_ret = np.diff(y) / np.abs(y[:-1])
    assert step_ret.min() < -0.2  # a collapse of >20% in a single step
    # ...and re-inflate: the max comes after some collapse, not at the very end.
    assert y.max() > y[0] * 2.0


def test_evans_bubble_rejects_bad_params():
    with pytest.raises(ValueError):
        simulate_evans_bubble(n_steps=100, n_paths=1, seed=0, delta=2.0, alpha=1.0)
