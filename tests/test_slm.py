"""Strict-local-martingale diagnostic: recovers the volatility elasticity ρ.

GBM has σ(x) ∝ x (ρ = 1, true martingale); the CEV bubble has σ(x) ∝ x² (ρ = 2).
The tail log-log slope of σ̂²(x) should recover 2ρ and separate the two.
"""

import numpy as np

from src.detect.slm import local_volatility, slm_diagnostic, volatility_elasticity
from src.synthetic.paths import simulate_cev, simulate_gbm

COMMON = dict(s0=1.0, sigma=0.5, t_horizon=1.0, n_steps=500, n_paths=3000, seed=7)


def test_gbm_elasticity_near_one():
    gbm = simulate_gbm(**COMMON)
    res = volatility_elasticity(gbm.S, dt=gbm.t[1] - gbm.t[0])
    assert abs(res["rho_hat"] - 1.0) < 0.4
    assert "bubble" not in res
    assert "rho_grid_se" in res


def test_cev_elasticity_above_one():
    cev = simulate_cev(rho=2.0, **COMMON)
    res = volatility_elasticity(cev.S, dt=cev.t[1] - cev.t[0])
    # ρ clearly above the martingale boundary of 1.
    assert res["rho_hat"] > 1.3
    assert "bubble" not in res


def test_cev_rho_exceeds_gbm_rho():
    dt = 1.0 / 500
    g = volatility_elasticity(simulate_gbm(**COMMON).S, dt=dt)["rho_hat"]
    c = volatility_elasticity(simulate_cev(rho=2.0, **COMMON).S, dt=dt)["rho_hat"]
    assert c > g + 0.4


def test_local_volatility_grid_shapes():
    gbm = simulate_gbm(**COMMON)
    grid, s2 = local_volatility(gbm.S, dt=gbm.t[1] - gbm.t[0], n_grid=40)
    assert grid.shape == (40,) and s2.shape == (40,)
    assert np.all(np.diff(grid) > 0)


def test_diagnostic_verdict_string():
    cev = simulate_cev(rho=2.0, **COMMON)
    d = slm_diagnostic(cev.S, dt=cev.t[1] - cev.t[0], label="cev")
    assert d["label"] == "cev"
    assert "martingale" in d["verdict"]
