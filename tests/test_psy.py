"""Recursive PSY statistics: known-answer separation and date-stamping.

The decisive property: the GSADF must be large for explosive / periodically-
collapsing series and small for the unit-root null, and the expanding-window
recursion must agree with the plain single-window ADF at the full sample.
"""

import numpy as np

from src.detect.adf import adf_tstat
from src.detect.psy import (
    bsadf_sequence,
    date_stamp,
    default_min_duration,
    gsadf,
    recursive_adf,
    sadf,
)
from src.synthetic.paths import (
    simulate_evans_bubble,
    simulate_explosive_ar1,
    simulate_random_walk,
)


def test_expanding_recursion_matches_single_window_adf():
    rng = np.random.default_rng(2)
    y = 50.0 + np.cumsum(rng.standard_normal(250))
    r = recursive_adf(y, p=1)
    # adf_expanding at the last date is the ADF on the whole series.
    assert abs(r.adf_expanding[-1] - adf_tstat(y, p=1)) < 1e-8


def test_gsadf_at_least_sadf():
    y = simulate_random_walk(n_steps=400, n_paths=1, seed=5).S[0]
    r = recursive_adf(y, p=1)
    assert r.gsadf_stat >= r.sadf_stat - 1e-9


def test_gsadf_separates_explosive_from_random_walk():
    rw = simulate_random_walk(n_steps=400, n_paths=1, seed=6).S[0]
    ex = simulate_explosive_ar1(phi=1.04, n_steps=400, n_paths=1, seed=6).S[0]
    g_rw = gsadf(rw, p=1)
    g_ex = gsadf(ex, p=1)
    # The explosive series produces a far larger GSADF than the null.
    assert g_ex > g_rw + 3.0
    assert g_rw < 2.5  # null stays small (well below typical 95% CV territory)


def test_gsadf_flags_evans_bubble_that_sadf_can_miss():
    ev = simulate_evans_bubble(n_steps=700, n_paths=1, seed=11).S[0]
    s = sadf(ev, p=1)
    g = gsadf(ev, p=1)
    # A periodically-collapsing bubble: the flexible-window GSADF is at least as
    # strong as the anchored SADF, and clearly explosive.
    assert g >= s - 1e-9
    assert g > 2.0


def test_bsadf_sequence_shape_and_nan_prefix():
    y = simulate_random_walk(n_steps=200, n_paths=1, seed=8).S[0]
    r = recursive_adf(y, p=1)
    seq = bsadf_sequence(y, p=1)
    assert seq.shape == y.shape
    # No statistic before the minimum window is filled.
    assert np.isnan(seq[: r.min_obs - 1]).all()
    assert np.isfinite(seq[-1])


def test_date_stamp_min_duration_filter():
    bsadf = np.array([np.nan, 0.0, 3.0, 3.0, 3.0, 0.0, 3.0, 0.0])
    cv = np.full_like(bsadf, 1.0)
    # runs of exceedance: [2,4] (len 3) and [6,6] (len 1)
    eps = date_stamp(bsadf, cv, min_duration=2)
    assert eps == [(2, 4)]


def test_default_min_duration_scales_with_log_t():
    assert default_min_duration(4325) == round(np.log(4325))
