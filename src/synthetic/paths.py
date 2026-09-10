"""Synthetic price-path generators for later Phase IV benchmarking.

Two models with known martingale behaviour, so a diagnostic can be tested where
the answer is known by construction:

  * GBM (null):  dS = sigma * S dW  — a true martingale; E[S_t] = S_0.
    Simulated exactly via S_t = S_0 * exp(sigma W_t - 0.5 sigma^2 t).

  * CEV (bubble): dS = sigma * S^rho dW with rho > 1 — the discounted price is a
    strict *local* martingale, so E[S_t] < S_0 (the bubble signature).
    Simulated by Euler–Maruyama with absorption at 0 and a high numerical
    barrier (the superlinear diffusion can otherwise overflow float64). The
    barrier is a numerical approximation. Mean estimates require scheme, step
    size and clipping sensitivity rather than an assumed robustness guarantee.

Infrastructure, deliberately simple and well-labelled — not a primary contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_common(s0: float, sigma: float, t_horizon: float,
                     n_steps: int, n_paths: int) -> None:
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if t_horizon <= 0:
        raise ValueError(f"t_horizon must be positive, got {t_horizon}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError(f"n_steps and n_paths must be >= 1, got {n_steps}, {n_paths}")


@dataclass
class Paths:
    t: np.ndarray            # (n_steps + 1,) time grid
    S: np.ndarray            # (n_paths, n_steps + 1) simulated paths
    model: str
    params: dict

    @property
    def terminal(self) -> np.ndarray:
        return self.S[:, -1]


def simulate_gbm(
    *, s0: float, sigma: float, t_horizon: float, n_steps: int,
    n_paths: int, seed: int,
) -> Paths:
    """Driftless GBM (true martingale), simulated exactly."""
    _validate_common(s0, sigma, t_horizon, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    dt = t_horizon / n_steps
    t = np.linspace(0.0, t_horizon, n_steps + 1)
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    S = s0 * np.exp(sigma * W - 0.5 * sigma**2 * t[None, :])
    return Paths(t=t, S=S, model="gbm",
                 params=dict(s0=s0, sigma=sigma, t_horizon=t_horizon,
                             n_steps=n_steps, n_paths=n_paths, seed=seed))


def simulate_cev(
    *, s0: float, sigma: float, rho: float, t_horizon: float, n_steps: int,
    n_paths: int, seed: int, barrier: float = 1e8,
) -> Paths:
    """CEV with rho > 1 (strict local martingale) via Euler–Maruyama.

    Absorbs at zero and clips at the upper numerical barrier. The upper
    barrier is not absorbing: later steps can move down from it.
    """
    _validate_common(s0, sigma, t_horizon, n_steps, n_paths)
    if rho <= 1:
        raise ValueError(f"CEV rho must be > 1 for the bubble regime, got {rho}")
    rng = np.random.default_rng(seed)
    dt = t_horizon / n_steps
    sqrt_dt = np.sqrt(dt)
    t = np.linspace(0.0, t_horizon, n_steps + 1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = s0
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n_steps):
            s = S[:, i]
            incr = sigma * np.power(s, rho) * sqrt_dt * rng.standard_normal(n_paths)
            nxt = s + incr
            nxt = np.where(np.isfinite(nxt), nxt, barrier)
            S[:, i + 1] = np.clip(nxt, 0.0, barrier)
    return Paths(t=t, S=S, model="cev",
                 params=dict(s0=s0, sigma=sigma, rho=rho, t_horizon=t_horizon,
                             n_steps=n_steps, n_paths=n_paths, seed=seed,
                             barrier=barrier))


# =====================================================================
# Discrete-time matched controls for the GSADF explosive-root test.
#
# The GBM/CEV pair checks numerical behaviour against specified diffusion models.
# The GSADF/PSY diagnostic (src/detect) is a *discrete-time* explosive-AR test,
# so it needs discrete ground truth: a unit-root null it must NOT flag, a mildly
# explosive series it MUST flag, and a periodically-collapsing bubble (Evans
# 1991) — the canonical PSY case where a single SADF misses later episodes but
# the GSADF date-stamps them.
# =====================================================================


def _validate_discrete(n_steps: int, n_paths: int, sigma: float) -> None:
    if n_steps < 2 or n_paths < 1:
        raise ValueError(f"n_steps>=2 and n_paths>=1 required, got {n_steps}, {n_paths}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")


def simulate_random_walk(
    *, n_steps: int, n_paths: int, seed: int, y0: float = 100.0, sigma: float = 1.0,
) -> Paths:
    """Unit-root null: y_t = y_{t-1} + e_t, e_t ~ N(0, sigma^2).

    The GSADF must **not** flag this as explosive (beyond its nominal size).
    """
    _validate_discrete(n_steps, n_paths, sigma)
    rng = np.random.default_rng(seed)
    e = sigma * rng.standard_normal((n_paths, n_steps))
    y = np.empty((n_paths, n_steps + 1))
    y[:, 0] = y0
    y[:, 1:] = y0 + np.cumsum(e, axis=1)
    t = np.arange(n_steps + 1, dtype=float)
    return Paths(t=t, S=y, model="random_walk",
                 params=dict(y0=y0, sigma=sigma, n_steps=n_steps,
                             n_paths=n_paths, seed=seed))


def simulate_explosive_ar1(
    *, phi: float, n_steps: int, n_paths: int, seed: int,
    y0: float = 1.0, sigma: float = 1.0,
) -> Paths:
    """Mildly explosive AR(1): y_t = phi*y_{t-1} + e_t with phi > 1.

    A single sustained explosive regime — the GSADF **must** flag it.
    """
    if phi <= 1:
        raise ValueError(f"explosive AR requires phi > 1, got {phi}")
    _validate_discrete(n_steps, n_paths, sigma)
    rng = np.random.default_rng(seed)
    e = sigma * rng.standard_normal((n_paths, n_steps))
    y = np.empty((n_paths, n_steps + 1))
    y[:, 0] = y0
    for i in range(n_steps):
        y[:, i + 1] = phi * y[:, i] + e[:, i]
    t = np.arange(n_steps + 1, dtype=float)
    return Paths(t=t, S=y, model="explosive_ar1",
                 params=dict(phi=phi, y0=y0, sigma=sigma, n_steps=n_steps,
                             n_paths=n_paths, seed=seed))


def simulate_evans_bubble(
    *, n_steps: int, n_paths: int, seed: int,
    r: float = 0.05, delta: float = 0.5, pi: float = 0.85, alpha: float = 1.0,
    tau: float = 0.05, b0: float | None = None, kappa: float = 20.0,
    fundamental_drift: float = 0.02, fundamental_sigma: float = 0.5,
    f0: float = 100.0,
) -> Paths:
    """Evans (1991) periodically-collapsing bubble on a random-walk fundamental.

    Bubble component (E[u]=1, theta ~ Bernoulli(pi) is the survival indicator)::

        B_{t+1} = (1+r) B_t u_{t+1}                                if B_t <= alpha
        B_{t+1} = [delta + pi^-1 (1+r) theta_{t+1}
                   (B_t - (1+r)^-1 delta)] u_{t+1}                  if B_t >  alpha

    While surviving, the bubble grows explosively at gross rate (1+r); once it
    exceeds ``alpha`` it collapses back toward ``delta`` with probability
    ``1-pi`` each period, then re-inflates — so the series contains *several*
    explosive episodes. The observed price is ``P_t = F_t + kappa * B_t`` with a
    random-walk fundamental ``F_t``. This is the standard PSY date-stamping
    stress test: a plain SADF tends to detect only the first/largest episode,
    while the GSADF recovers the others.
    """
    _validate_discrete(n_steps, n_paths, fundamental_sigma)
    if not (0.0 < pi <= 1.0):
        raise ValueError(f"pi must be in (0, 1], got {pi}")
    if not (0.0 < delta < (1.0 + r) * alpha):
        raise ValueError(
            f"require 0 < delta < (1+r)*alpha; got delta={delta}, "
            f"(1+r)*alpha={(1.0 + r) * alpha}"
        )
    rng = np.random.default_rng(seed)
    b0 = delta if b0 is None else b0

    B = np.empty((n_paths, n_steps + 1))
    B[:, 0] = b0
    one_r = 1.0 + r
    for i in range(n_steps):
        u = np.exp(rng.normal(-0.5 * tau**2, tau, size=n_paths))  # E[u] = 1
        theta = (rng.random(n_paths) < pi).astype(float)
        b = B[:, i]
        grow = one_r * b * u
        collapse = (delta + (one_r / pi) * theta * (b - delta / one_r)) * u
        B[:, i + 1] = np.where(b <= alpha, grow, collapse)

    # Random-walk fundamental (kept modest so the bubble drives the explosivity).
    eps = fundamental_sigma * rng.standard_normal((n_paths, n_steps))
    F = np.empty((n_paths, n_steps + 1))
    F[:, 0] = f0
    F[:, 1:] = f0 + np.cumsum(fundamental_drift + eps, axis=1)

    P = F + kappa * B
    t = np.arange(n_steps + 1, dtype=float)
    return Paths(t=t, S=P, model="evans_bubble",
                 params=dict(r=r, delta=delta, pi=pi, alpha=alpha, tau=tau,
                             b0=b0, kappa=kappa, fundamental_drift=fundamental_drift,
                             fundamental_sigma=fundamental_sigma, f0=f0,
                             n_steps=n_steps, n_paths=n_paths, seed=seed))


def martingale_diagnostic(paths: Paths) -> dict:
    """Sample check of E[S_T] vs S_0.

    For GBM this should be ~S_0; for a CEV bubble it should be strictly below.
    """
    term = paths.terminal
    s0 = paths.params["s0"]
    mean = float(np.mean(term))
    se = float(np.std(term, ddof=1) / np.sqrt(len(term)))
    return {
        "model": paths.model,
        "s0": s0,
        "E[S_T]_hat": mean,
        "std_error": se,
        "ratio_to_s0": mean / s0,
        "frac_absorbed_zero": float(np.mean(term <= 0.0)),
        "frac_at_barrier": float(np.mean(term >= paths.params.get("barrier", np.inf))),
    }
