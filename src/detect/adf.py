"""Augmented Dickey-Fuller regression core (right-tailed, for explosive-root tests).

The ADF regression used for bubble testing includes a constant and **no** time
trend — the specification Phillips, Shi & Yu (2015) adopt so that a mildly
explosive root shows up cleanly:

    Δy_t = α + β·y_{t-1} + Σ_{i=1}^{p} γ_i·Δy_{t-i} + ε_t

The unit-root null is β = 0; the *right-tailed* alternative β > 0 is the mildly
explosive regime a bubble produces, so the test statistic is the **t-ratio on
β** (large positive ⇒ explosive). Large negative values (stationary mean
reversion) are irrelevant here and are not what we test.

The OLS is hand-rolled in numpy for one specific reason: the recursive PSY
statistics in :mod:`psy` re-estimate this same regression on O(T²) nested
sub-windows. Because the design matrix rows are *identical* across every window
(a window just selects a contiguous block of rows), :func:`build_adf_design`
exposes the full design once and ``psy`` assembles each window's normal
equations by differencing cumulative cross-products — which is what makes the
double recursion tractable. :func:`adf_tstat` is the plain single-window
reference used in tests and for sanity checks.
"""

from __future__ import annotations

import numpy as np


def build_adf_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ADF design for a series ``y`` with ``p`` lagged differences.

    Returns ``(X, Y, t_index)`` where

      * ``X`` has columns ``[const, y_{t-1}, Δy_{t-1}, …, Δy_{t-p}]`` (shape
        ``(m, p + 2)``),
      * ``Y`` is the response ``Δy_t`` (shape ``(m,)``),
      * ``t_index`` gives, for each design row, the original-series index ``t``
        of its response (so callers can date-stamp a row).

    A regression window over original indices ``[a, b]`` (inclusive) is exactly
    the design rows with ``t_index`` in ``[a + p + 1, b]`` — i.e. the ordinary
    sub-sample ADF on ``y[a…b]`` — so ``psy`` can select windows by slicing.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if p < 0:
        raise ValueError(f"lag order p must be >= 0, got {p}")
    k = p + 2
    if n < k + 2:
        raise ValueError(f"series too short for ADF(p={p}): need >= {k + 2}, got {n}")

    dy = np.empty(n)
    dy[0] = np.nan
    dy[1:] = np.diff(y)  # dy[t] = y[t] - y[t-1] for t >= 1

    t_index = np.arange(p + 1, n)  # response times with all lags available
    Y = dy[t_index]
    cols = [np.ones(t_index.shape[0]), y[t_index - 1]]
    for i in range(1, p + 1):
        cols.append(dy[t_index - i])
    X = np.column_stack(cols)
    return X, Y, t_index


def adf_tstat(y: np.ndarray, p: int = 1) -> float:
    """Right-tailed ADF t-statistic on β (the ``y_{t-1}`` coefficient).

    Plain single-window OLS reference implementation. ``psy`` does not call this
    in its hot loop (it uses the cumulative-cross-product path for speed), but
    the two agree by construction and a test pins them together.
    """
    X, Y, _ = build_adf_design(y, p)
    n_obs, k = X.shape
    dof = n_obs - k
    if dof <= 0:
        raise ValueError(f"not enough observations ({n_obs}) for ADF(p={p})")
    xtx = X.T @ X
    xty = X.T @ Y
    beta = np.linalg.solve(xtx, xty)
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / dof
    var_beta = sigma2 * np.linalg.inv(xtx)
    se_beta1 = float(np.sqrt(var_beta[1, 1]))
    return float(beta[1]) / se_beta1
