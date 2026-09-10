"""Finite-sample critical values for the SADF / GSADF / BSADF statistics.

The explosive-root statistics have non-standard limit distributions, so critical
values are obtained by simulation under the unit-root null. Two methods:

  * ``"mc"`` — i.i.d. Gaussian random walks. Simulated at a bounded sample size
    ``sim_size`` with the **same** minimum-window fraction ``r0`` as the analysis
    (the sup range, not the length, is what the null distribution mainly depends
    on), then the BSADF critical-value *sequence* is interpolated by sample
    fraction onto the real series. This is a bounded-length approximation, not exact target-length calibration.
  * ``"wild"`` — wild-bootstrap sensitivity motivated by Harvey et al. (2016):
    resample centred first differences with Rademacher signs, which
    preserves its heteroskedasticity (volatile crypto returns). Runs at the real
    length, so it is slower; use a smaller ``n_sim``.

Both the scalar test critical values (for the GSADF/SADF decision) and the
per-fraction BSADF critical-value curve (for date-stamping) come from one
simulation pass and are cached to disk keyed by every parameter that affects
them, so the expensive run happens once.
"""

from __future__ import annotations

import warnings
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .psy import recursive_adf, resolve_min_obs

SIGNIFICANCE = (0.90, 0.95, 0.99)


@dataclass
class CriticalValues:
    stat: str                    # "gsadf" or "sadf"
    method: str                  # "mc" or "wild"
    scalar: dict[float, float]   # {level: cv} for the scalar test decision
    frac_grid: np.ndarray        # sample fractions r in [~r0, 1] on the sim grid
    cv_curve: dict[float, np.ndarray]   # {level: BSADF cv over frac_grid}
    meta: dict

    def scalar_cv(self, level: float = 0.95) -> float:
        return self.scalar[level]

    def bsadf_cv_sequence(self, n_target: int, level: float = 0.95) -> np.ndarray:
        """Interpolate the BSADF critical-value curve onto a length-``n_target``
        series by sample fraction. Positions below the minimum window are NaN."""
        frac_target = (np.arange(n_target) + 1) / n_target
        curve = self.cv_curve[level]
        good = np.isfinite(self.frac_grid) & np.isfinite(curve)
        cv = np.interp(frac_target, self.frac_grid[good], curve[good],
                       left=np.nan, right=curve[good][-1])
        cv[frac_target < self.frac_grid[good][0]] = np.nan
        return cv


def _null_series(method: str, rng: np.random.Generator, *,
                 sim_size: int, real_diffs: np.ndarray | None) -> np.ndarray:
    if method == "mc":
        return np.concatenate([[0.0], np.cumsum(rng.standard_normal(sim_size - 1))])
    if method == "wild":
        w = rng.choice((-1.0, 1.0), size=real_diffs.shape[0])
        return np.concatenate([[0.0], np.cumsum(w * real_diffs)])
    raise ValueError(f"unknown critical-value method {method!r}")


def cache_metadata(stat, method, sim_size, r0, p, n_sim, seed, significance, real_series):
    """Exact configuration and input identity, including the bootstrap scheme."""
    data_hash = None
    if method == "wild":
        data = np.asarray(real_series, dtype="<f8")
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
    return dict(cache_version=2, stat=stat, method=method, sim_size=int(sim_size),
                r0=float(r0), p=int(p), n_sim=int(n_sim), seed=int(seed),
                significance=list(map(float, significance)), real_series_sha256=data_hash,
                scheme="centred-rademacher-v1" if method == "wild" else "gaussian-random-walk-v1")


def _cache_path(cache_dir: Path, metadata: dict) -> Path:
    identity = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return cache_dir / f"cv_v2_{metadata['stat']}_{metadata['method']}_{digest}.npz"


def simulate_critical_values(
    *,
    stat: str = "gsadf",
    method: str = "mc",
    r0: float,
    p: int = 1,
    n_sim: int = 499,
    seed: int = 12345,
    sim_size: int = 1000,
    real_series: np.ndarray | None = None,
    cache_dir: Path | None = None,
    significance: tuple[float, ...] = SIGNIFICANCE,
) -> CriticalValues:
    """Simulate (or load cached) critical values for ``stat`` under the null."""
    if stat not in ("gsadf", "sadf"):
        raise ValueError(f"stat must be 'gsadf' or 'sadf', got {stat!r}")

    real_diffs = None
    if method == "wild":
        if real_series is None:
            raise ValueError("wild bootstrap needs the real series")
        real_diffs = np.diff(np.asarray(real_series, dtype=float))
        real_diffs = real_diffs - real_diffs.mean()
        sim_size = real_diffs.shape[0] + 1

    identity = cache_metadata(stat, method, sim_size, r0, p, n_sim, seed, significance, real_series)
    if cache_dir is not None:
        cache = _cache_path(cache_dir, identity)
        if cache.exists():
            loaded = _load(cache)
            if all(loaded.meta.get(k) == v for k, v in identity.items()):
                return loaded
            raise ValueError("Critical-value cache metadata does not match its key")

    k = p + 2
    min_obs = resolve_min_obs(sim_size, r0=r0, min_obs=None, k=k)
    rng = np.random.default_rng(seed)

    scalar_stats = np.empty(n_sim)
    bsadf_mat = np.full((n_sim, sim_size), np.nan)
    for i in range(n_sim):
        y = _null_series(method, rng, sim_size=sim_size, real_diffs=real_diffs)
        r = recursive_adf(y, p=p, r0=r0)
        scalar_stats[i] = r.gsadf_stat if stat == "gsadf" else r.sadf_stat
        bsadf_mat[i] = r.bsadf if stat == "gsadf" else r.adf_expanding

    frac_grid = (np.arange(sim_size) + 1) / sim_size
    scalar = {lvl: float(np.quantile(scalar_stats, lvl)) for lvl in significance}
    cv_curve = {}
    # Columns before the minimum window are all-NaN by construction; the
    # resulting "All-NaN slice" notice is expected, not a problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lvl in significance:
            cv_curve[lvl] = np.nanquantile(bsadf_mat, lvl, axis=0)

    meta = {**identity, "min_obs": min_obs}
    cvs = CriticalValues(stat=stat, method=method, scalar=scalar,
                         frac_grid=frac_grid, cv_curve=cv_curve, meta=meta)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _save(cache, cvs)
    return cvs


def _save(path: Path, cvs: CriticalValues) -> None:
    levels = np.array(list(cvs.cv_curve.keys()))
    payload = dict(
        stat=cvs.stat, method=cvs.method, frac_grid=cvs.frac_grid,
        levels=levels,
        scalar=np.array([cvs.scalar[l] for l in levels]),
        cv_curve=np.stack([cvs.cv_curve[l] for l in levels]),
        meta_keys=np.array(list(cvs.meta.keys()), dtype=object),
        meta_vals=np.array(list(cvs.meta.values()), dtype=object),
    )
    np.savez(path, **payload)


def _load(path: Path) -> CriticalValues:
    d = np.load(path, allow_pickle=True)
    levels = d["levels"]
    scalar = {float(l): float(s) for l, s in zip(levels, d["scalar"])}
    cv_curve = {float(l): row for l, row in zip(levels, d["cv_curve"])}
    meta = dict(zip(d["meta_keys"], d["meta_vals"]))
    return CriticalValues(stat=str(d["stat"]), method=str(d["method"]),
                          scalar=scalar, frac_grid=d["frac_grid"],
                          cv_curve=cv_curve, meta=meta)
