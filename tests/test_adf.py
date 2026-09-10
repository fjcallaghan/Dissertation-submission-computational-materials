"""ADF core: agreement with statsmodels and internal consistency."""

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from src.detect.adf import adf_tstat, build_adf_design


def _series(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.standard_normal(n))  # a random walk


@pytest.mark.parametrize("p", [0, 1, 3])
def test_matches_statsmodels_adfuller(p):
    y = _series(seed=1, n=400)
    ours = adf_tstat(y, p=p)
    # statsmodels with a constant, fixed lag, no autolag: the reported stat is
    # the t-ratio on the lagged level — the same quantity we compute.
    theirs = adfuller(y, maxlag=p, regression="c", autolag=None)[0]
    assert abs(ours - theirs) < 1e-8


def test_design_shapes_and_alignment():
    y = np.arange(10, dtype=float) ** 1.5
    X, Y, t_index = build_adf_design(y, p=2)
    # rows start at t = p + 1 = 3, run to n - 1 = 9  → 7 rows, k = p + 2 = 4 cols
    assert X.shape == (7, 4)
    assert Y.shape == (7,)
    assert t_index[0] == 3 and t_index[-1] == 9
    assert np.allclose(X[:, 0], 1.0)                 # constant column
    assert np.allclose(X[:, 1], y[t_index - 1])      # y_{t-1} column
    assert np.allclose(Y, np.diff(y)[t_index - 1])   # response Δy_t


def test_rejects_too_short():
    with pytest.raises(ValueError):
        adf_tstat(np.arange(4, dtype=float), p=3)
