"""Recursive right-tailed ADF statistics: SADF, GSADF and BSADF date-stamping.

Implements the Phillips–Shi–Yu (2015) explosive-root machinery:

  * **SADF** — sup of the ADF t-stat over an *expanding* window anchored at the
    sample start. Detects a single bubble but can miss later episodes once an
    earlier collapse contaminates the anchored window.
  * **GSADF** — sup over *all* start/end sub-windows (the double recursion).
    Detects multiple, periodically-collapsing bubbles.
  * **BSADF** — the backward-SADF *sequence*: at each date ``b`` the sup over
    start points of the ADF t-stat ending at ``b``. Comparing this sequence to a
    critical-value sequence date-stamps the origin and collapse of each episode.

Performance. The double recursion is O(T²) ADF regressions. Every window is a
contiguous block of the *same* design-matrix rows (see :func:`adf.build_adf_design`),
so we precompute cumulative cross-products ``Σxxᵀ``, ``Σxy``, ``Σy²`` once and
assemble each window's normal equations by subtraction. For a fixed endpoint the
sup over all start points is then a single *batched* linear solve, so the whole
BSADF sequence costs O(T) numpy calls rather than O(T²) Python-level regressions.
This makes GSADF on the full ~4,300-day BTC series run in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .adf import build_adf_design


def default_r0(n: int) -> float:
    """PSY minimum-window rule ``r0 = 0.01 + 1.8/√T`` (their recommendation)."""
    return 0.01 + 1.8 / math.sqrt(n)


def resolve_min_obs(n: int, *, r0: float | None, min_obs: int | None, k: int) -> int:
    """Minimum window length in observations, from ``min_obs`` or ``r0``."""
    if min_obs is not None:
        mo = int(min_obs)
    else:
        r0 = default_r0(n) if r0 is None else float(r0)
        mo = int(math.floor(r0 * n))
    # Need enough degrees of freedom in the smallest window.
    return max(mo, k + 3)


@dataclass
class RecursiveADF:
    """Result of the recursive ADF sweep, indexed by original series position."""

    index: np.ndarray        # (n,) original positions 0..n-1
    bsadf: np.ndarray        # (n,) backward-SADF sequence (NaN before min window)
    adf_expanding: np.ndarray  # (n,) ADF of the start-anchored expanding window
    sadf_stat: float         # sup of adf_expanding
    gsadf_stat: float        # sup of bsadf
    min_obs: int
    p: int


def _batched_tstats(
    A: np.ndarray, v: np.ndarray, yty: np.ndarray, nobs: np.ndarray, k: int
) -> np.ndarray:
    """Right-tailed ADF t-stat on β for a batch of windows sharing an endpoint.

    ``A`` = XᵀX (b,k,k), ``v`` = Xᵀy (b,k), ``yty`` = yᵀy (b,), ``nobs`` (b,).
    Returns t-stats with ``-inf`` where a window is singular or has no d.o.f.
    """
    dof = nobs - k
    sign, logdet = np.linalg.slogdet(A)
    ok = np.isfinite(logdet) & (sign > 0) & (dof > 0)
    eye = np.eye(k)
    A_safe = np.where(ok[:, None, None], A, eye)
    # Column-vector rhs so numpy always takes the stacked matrix-solve path
    # (a bare (b,k) rhs is ambiguous vs a single 2-D system).
    beta = np.linalg.solve(A_safe, v[:, :, None])[:, :, 0]   # (b,k)
    Ainv = np.linalg.inv(A_safe)                             # (b,k,k)
    ss = yty - np.einsum("bk,bk->b", beta, v)        # residual sum of squares
    dof_safe = np.where(dof > 0, dof, 1)
    sigma2 = ss / dof_safe
    var_beta1 = sigma2 * Ainv[:, 1, 1]
    good = ok & (var_beta1 > 0) & (ss >= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta[:, 1] / np.sqrt(var_beta1)
    return np.where(good, t, -np.inf)


def recursive_adf(
    y: np.ndarray, *, p: int = 1, r0: float | None = None, min_obs: int | None = None
) -> RecursiveADF:
    """Compute the BSADF sequence, the expanding-window ADF, and SADF/GSADF."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    k = p + 2
    mo = resolve_min_obs(n, r0=r0, min_obs=min_obs, k=k)
    if n < mo + 1:
        raise ValueError(f"series length {n} too short for min window {mo}")

    X, Y, t_index = build_adf_design(y, p)  # design row j ↔ original time p+1+j
    m = X.shape[0]
    XX = np.einsum("ij,ik->ijk", X, X)
    CXX = np.zeros((m + 1, k, k)); CXX[1:] = np.cumsum(XX, axis=0)
    CXY = np.zeros((m + 1, k)); CXY[1:] = np.cumsum(X * Y[:, None], axis=0)
    CYY = np.zeros(m + 1); CYY[1:] = np.cumsum(Y * Y)

    bsadf = np.full(n, np.nan)
    adf_exp = np.full(n, np.nan)

    # Endpoint b (original index): needs a window [0,b] of >= mo observations.
    for b in range(mo - 1, n):
        j1 = b - p                       # exclusive end of design-row block
        if j1 < 1:
            continue
        # start a in [0, b - mo + 1]  (window length b-a+1 >= mo)
        a_max = b - mo + 1
        j0 = np.arange(0, a_max + 1)      # design-row start = original start a
        A = CXX[j1] - CXX[j0]
        v = CXY[j1] - CXY[j0]
        yty = CYY[j1] - CYY[j0]
        nobs = (j1 - j0).astype(float)
        t = _batched_tstats(A, v, yty, nobs, k)
        bsadf[b] = float(np.max(t))
        adf_exp[b] = float(t[0])          # a = 0 → the expanding window

    sadf_stat = float(np.nanmax(adf_exp))
    gsadf_stat = float(np.nanmax(bsadf))
    return RecursiveADF(
        index=np.arange(n), bsadf=bsadf, adf_expanding=adf_exp,
        sadf_stat=sadf_stat, gsadf_stat=gsadf_stat, min_obs=mo, p=p,
    )


def sadf(y: np.ndarray, *, p: int = 1, r0: float | None = None,
         min_obs: int | None = None) -> float:
    """Scalar SADF statistic."""
    return recursive_adf(y, p=p, r0=r0, min_obs=min_obs).sadf_stat


def gsadf(y: np.ndarray, *, p: int = 1, r0: float | None = None,
          min_obs: int | None = None) -> float:
    """Scalar GSADF statistic."""
    return recursive_adf(y, p=p, r0=r0, min_obs=min_obs).gsadf_stat


def bsadf_sequence(y: np.ndarray, *, p: int = 1, r0: float | None = None,
                   min_obs: int | None = None) -> np.ndarray:
    """Backward-SADF sequence (length ``len(y)``; NaN before the first window)."""
    return recursive_adf(y, p=p, r0=r0, min_obs=min_obs).bsadf


def default_min_duration(n: int, multiplier: float = 1.0) -> int:
    """PSY minimum episode duration ≈ ``multiplier·log(T)`` observations."""
    return max(1, int(round(multiplier * math.log(n))))


def date_stamp(
    bsadf: np.ndarray, cv: np.ndarray, *, min_duration: int
) -> list[tuple[int, int]]:
    """Date-stamp episodes where BSADF exceeds its critical value.

    Returns ``[(start, end), …]`` in original-index positions for each run of
    ``bsadf > cv`` lasting at least ``min_duration`` consecutive observations
    (the PSY rule that filters transient single-day exceedances).
    """
    bsadf = np.asarray(bsadf, dtype=float)
    cv = np.asarray(cv, dtype=float)
    if bsadf.shape != cv.shape:
        raise ValueError(f"bsadf {bsadf.shape} and cv {cv.shape} must match")
    exceed = np.isfinite(bsadf) & np.isfinite(cv) & (bsadf > cv)
    episodes: list[tuple[int, int]] = []
    n = exceed.shape[0]
    i = 0
    while i < n:
        if not exceed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and exceed[j + 1]:
            j += 1
        if (j - i + 1) >= min_duration:
            episodes.append((i, j))
        i = j + 1
    return episodes
