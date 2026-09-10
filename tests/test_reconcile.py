"""Reconciliation: agreement stats behave on known inputs."""

import numpy as np
import pandas as pd

from src.reconcile.compare import compare_to_reference

IDX = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")


def test_identical_series_perfect_agreement():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1e-4, size=100), index=IDX)
    a = compare_to_reference(s, s.copy(), venue="X", reference="Ref")
    assert a.overlap_days == 100
    assert np.isclose(a.pearson, 1.0)
    assert np.isclose(a.mean_abs_diff, 0.0)
    assert np.isclose(a.sign_agreement, 1.0)


def test_overlap_restricted_to_common_dates():
    s1 = pd.Series(np.linspace(0, 1, 100), index=IDX)
    s2 = pd.Series(np.linspace(0, 1, 60), index=IDX[:60])
    a = compare_to_reference(s1, s2, venue="X", reference="Ref")
    assert a.overlap_days == 60


def test_anticorrelated_series():
    base = pd.Series(np.linspace(-1e-3, 1e-3, 100), index=IDX)
    a = compare_to_reference(base, -base, venue="X", reference="Ref")
    assert a.pearson < -0.99
