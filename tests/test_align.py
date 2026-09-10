"""Alignment stage: returns and daily funding aggregation."""

import numpy as np
import pandas as pd

from src.align.aggregate import funding_to_daily
from src.align.returns import compute_returns

START = pd.Timestamp("2019-09-10 00:00:00", tz="UTC")


def test_log_returns():
    p = pd.Series([100.0, 110.0, 121.0])
    r = compute_returns(p, method="log")
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], np.log(1.1))
    assert np.isclose(r.iloc[2], np.log(1.1))


def test_simple_returns():
    p = pd.Series([100.0, 110.0])
    r = compute_returns(p, method="simple")
    assert np.isclose(r.iloc[1], 0.10)


def test_funding_daily_sum_and_mean():
    # Two full days of 8-hourly funding (3 obs/day).
    times = pd.date_range(START, periods=6, freq="8h", tz="UTC")
    df = pd.DataFrame({"funding_time": times,
                       "funding_rate": [0.001, 0.002, 0.003, 0.0, 0.0, 0.0]})
    daily = funding_to_daily(df, rate_col="funding_rate")
    assert len(daily) == 2
    day0 = daily.iloc[0]
    assert np.isclose(day0["funding_sum"], 0.006)
    assert np.isclose(day0["funding_mean"], 0.002)
    assert int(day0["n_obs"]) == 3
