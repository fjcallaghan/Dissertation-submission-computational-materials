"""Exploratory observed-range volatility slope, without a martingale verdict.

The diffusion integral criterion concerns asymptotic volatility under maintained
model assumptions. A finite-range kernel fit cannot identify that tail law.
Its grid-regression standard error is descriptive, not sampling uncertainty for
the asymptotic exponent. Pooled-path recovery is not single-path validation.
"""

from __future__ import annotations

import numpy as np


def _pool_increments(prices: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled (level x, scaled squared increment y) pairs.

    Accepts a 1-D series or a 2-D ``(n_paths, n_steps+1)`` array (increments are
    pooled across paths). Non-positive levels are dropped so ``log x`` is defined.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    dS = np.diff(p, axis=1)
    x = p[:, :-1].ravel()
    y = (dS.ravel() ** 2) / dt
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[ok], y[ok]


def local_volatility(
    prices: np.ndarray, *, dt: float = 1.0, n_grid: int = 60,
    bandwidth: float | None = None, q_lo: float = 0.02, q_hi: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Nadaraya–Watson estimate of ``σ²(x)`` on a grid of price levels."""
    x, y = _pool_increments(prices, dt)
    if x.shape[0] < 10:
        raise ValueError("too few increments for a volatility estimate")
    lo, hi = np.quantile(x, [q_lo, q_hi])
    grid = np.linspace(lo, hi, n_grid)
    if bandwidth is None:
        # Silverman rule of thumb on the level distribution.
        bandwidth = 0.9 * min(x.std(), (np.subtract(*np.quantile(x, [0.75, 0.25]))) / 1.34)
        bandwidth *= x.shape[0] ** (-1 / 5)
        bandwidth = max(bandwidth, 1e-8)
    sigma2 = np.empty(n_grid)
    for i, xg in enumerate(grid):
        w = np.exp(-0.5 * ((x - xg) / bandwidth) ** 2)
        sw = w.sum()
        sigma2[i] = (w @ y) / sw if sw > 0 else np.nan
    return grid, sigma2


def volatility_elasticity(
    prices: np.ndarray, *, dt: float = 1.0, n_grid: int = 60,
    bandwidth: float | None = None, tail_q: float = 0.6,
) -> dict:
    """Estimate the volatility elasticity ρ from the tail log-log slope.

    Returns the observed-range slope and descriptive grid-regression error.
    No test of the martingale property is performed.
    """
    grid, sigma2 = local_volatility(prices, dt=dt, n_grid=n_grid, bandwidth=bandwidth)
    ok = np.isfinite(sigma2) & (sigma2 > 0) & (grid > 0)
    lx, ly = np.log(grid[ok]), np.log(sigma2[ok])
    # Restrict to the upper tail, where the bubble condition lives.
    if lx.shape[0] >= 6:
        cut = np.quantile(lx, tail_q)
        tail = lx >= cut
        if tail.sum() >= 4:
            lx, ly = lx[tail], ly[tail]
    n = lx.shape[0]
    if n < 3:
        raise ValueError("not enough valid tail points to estimate elasticity")
    X = np.column_stack([np.ones(n), lx])
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ beta
    dof = max(n - 2, 1)
    sigma2_reg = float(resid @ resid) / dof
    cov = sigma2_reg * np.linalg.inv(X.T @ X)
    slope = float(beta[1])
    slope_se = float(np.sqrt(cov[1, 1]))
    rho = slope / 2.0
    rho_se = slope_se / 2.0
    return {
        "rho_hat": rho,
        "rho_grid_se": rho_se,
        "slope": slope,          # = 2ρ
        "n_tail": n,
        "inference": "descriptive grid fit only; no martingale test",
    }


def slm_diagnostic(prices: np.ndarray, *, dt: float = 1.0, label: str = "", tail_q: float = 0.6) -> dict:
    """Convenience wrapper returning the elasticity result plus a verdict string."""
    res = volatility_elasticity(prices, dt=dt, tail_q=tail_q)
    return {"label": label, **res,
            "verdict": "exploratory slope only; martingale status not identified"}
