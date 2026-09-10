"""Critical-value simulation: ordering, caching round-trip, wild bootstrap."""

import numpy as np

from src.detect.critvals import simulate_critical_values
from src.detect.psy import default_r0


def test_scalar_cv_ordering_and_positive():
    cv = simulate_critical_values(stat="gsadf", method="mc", r0=default_r0(200),
                                  p=1, n_sim=60, seed=1, sim_size=200)
    assert cv.scalar[0.90] < cv.scalar[0.95] < cv.scalar[0.99]
    assert cv.scalar[0.90] > 0.0


def test_bsadf_cv_sequence_matches_length_and_nan_prefix():
    cv = simulate_critical_values(stat="gsadf", method="mc", r0=default_r0(200),
                                  p=1, n_sim=60, seed=2, sim_size=200)
    seq = cv.bsadf_cv_sequence(300, level=0.95)
    assert seq.shape == (300,)
    assert np.isnan(seq[0])          # below the minimum window
    assert np.isfinite(seq[-1])


def test_caching_round_trip(tmp_path):
    kw = dict(stat="gsadf", method="mc", r0=default_r0(200), p=1,
              n_sim=40, seed=3, sim_size=200, cache_dir=tmp_path)
    a = simulate_critical_values(**kw)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    b = simulate_critical_values(**kw)  # served from cache
    assert a.scalar[0.95] == b.scalar[0.95]
    assert np.allclose(a.frac_grid, b.frac_grid)


def test_wild_bootstrap_runs_on_real_series():
    rng = np.random.default_rng(0)
    y = 100.0 + np.cumsum(rng.standard_normal(220))
    cv = simulate_critical_values(stat="gsadf", method="wild", r0=default_r0(220),
                                  p=1, n_sim=40, seed=4, real_series=y)
    assert cv.method == "wild"
    assert cv.scalar[0.95] > cv.scalar[0.90]


def test_cache_distinguishes_data_levels_and_exact_window(tmp_path):
    y = np.cumsum(np.random.default_rng(7).normal(size=80))
    kw = dict(method="wild", r0=.20001, n_sim=5, real_series=y, cache_dir=tmp_path)
    a = simulate_critical_values(**kw)
    changed = y.copy()
    changed[40] += 3
    b = simulate_critical_values(**{**kw, "real_series": changed})
    c = simulate_critical_values(**{**kw, "significance": (.9, .95)})
    d = simulate_critical_values(**{**kw, "r0": .20002})
    assert len(list(tmp_path.glob("*.npz"))) == 4
    assert a.meta["real_series_sha256"] != b.meta["real_series_sha256"]
    assert set(c.scalar) == {.9, .95}
    assert a.meta["r0"] != d.meta["r0"]
