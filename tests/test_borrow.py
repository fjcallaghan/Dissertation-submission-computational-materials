"""BTC margin borrow-rate proxy: candle parsing, daily aggregation, spike flags,
and validation — all pure, no network."""

import numpy as np
import pandas as pd
import pytest

from src.acquire.bitfinex_borrow import parse_candles
from src.align.aggregate import funding_to_daily
from src.clean.spikes import flag_funding_spikes
from src.validate import checks

# 2021-01-01 / 01-02 in ms since epoch (UTC midnight).
MS_JAN1 = 1609459200000
MS_JAN2 = 1609545600000
DAY_MS = 86_400_000


def test_parse_candles_maps_close_to_rate():
    # [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]; CLOSE (index 2) is the day's rate.
    rows = [
        [MS_JAN1, 4e-8, 5e-8, 6e-8, 3e-8, 100.0],
        [MS_JAN2, 5e-8, 2e-3, 3e-3, 4e-8, 50.0],
    ]
    df = parse_candles(rows)
    assert list(df.columns) == ["borrow_time", "borrow_rate"]
    assert len(df) == 2
    assert df["borrow_time"].iloc[0] == pd.Timestamp("2021-01-01", tz="UTC")
    assert np.isclose(df["borrow_rate"].iloc[0], 5e-8)
    assert np.isclose(df["borrow_rate"].iloc[1], 2e-3)


def test_parse_candles_sorts_and_dedupes():
    rows = [
        [MS_JAN2, 0, 2e-3, 0, 0, 0],
        [MS_JAN1, 0, 1e-3, 0, 0, 0],
        [MS_JAN2, 0, 2e-3, 0, 0, 0],  # duplicate timestamp
    ]
    df = parse_candles(rows)
    assert len(df) == 2
    assert df["borrow_time"].is_monotonic_increasing


def test_parse_candles_empty():
    df = parse_candles([])
    assert df.empty
    assert list(df.columns) == ["borrow_time", "borrow_rate"]


def test_parse_candles_malformed_row_raises():
    # A short row (missing OHLC fields) must raise a clear error, not a cryptic
    # pandas "columns passed" error.
    with pytest.raises(RuntimeError, match="malformed candle"):
        parse_candles([[MS_JAN1, 1e-4, 2e-4, 3e-4, 4e-5, 10.0], [MS_JAN2, 1e-4]])


def test_parse_candles_tolerates_extra_fields():
    # Extra trailing fields are sliced off, not an error.
    df = parse_candles([[MS_JAN1, 1e-4, 2e-4, 3e-4, 4e-5, 10.0, 99, 100]])
    assert len(df) == 1
    assert np.isclose(df["borrow_rate"].iloc[0], 2e-4)


def test_borrow_daily_aggregation_prefix():
    # One record per day (borrow is already daily): sum == mean == the value.
    times = pd.to_datetime([MS_JAN1, MS_JAN2], unit="ms", utc=True)
    df = pd.DataFrame({"borrow_time": times, "borrow_rate": [1e-4, 3e-4]})
    daily = funding_to_daily(df, rate_col="borrow_rate", time_col="borrow_time",
                             prefix="borrow")
    assert set(["date", "borrow_sum", "borrow_mean", "n_obs"]).issubset(daily.columns)
    assert np.isclose(daily["borrow_mean"].iloc[0], 1e-4)
    assert np.isclose(daily["borrow_sum"].iloc[1], 3e-4)
    assert int(daily["n_obs"].iloc[0]) == 1


def test_borrow_spike_flag_daily_cadence():
    # A flat daily series with one squeeze spike; obs_per_day=1.
    n = 40
    times = pd.to_datetime([MS_JAN1 + i * DAY_MS for i in range(n)], unit="ms", utc=True)
    rate = [1e-4] * n
    rate[35] = 1e-2  # ~100x the baseline -> unambiguous spike
    df = pd.DataFrame({"borrow_time": times, "borrow_rate": rate})
    out = flag_funding_spikes(
        df, sigma=5.0, baseline_window_days=30,
        rate_col="borrow_rate", time_col="borrow_time", obs_per_day=1,
    )
    assert bool(out["is_spike"].iloc[35]) is True
    assert int(out["is_spike"].sum()) == 1
    # Winsorising pulls the spike down toward the baseline, never above it.
    assert out["borrow_rate_clean"].iloc[35] < df["borrow_rate"].iloc[35]


def test_check_borrow_flags_negative_and_out_of_band():
    times = pd.to_datetime([MS_JAN1 + i * DAY_MS for i in range(5)], unit="ms", utc=True)
    df = pd.DataFrame({
        "borrow_time": times,
        "borrow_rate": [1e-4, -1e-4, 5e-4, 0.05, 2e-4],  # one negative, one > sanity
    })
    res = checks.check_borrow(df)
    joined = " ".join(res.findings)
    assert "negative" in joined
    assert "beyond" in joined


def test_check_borrow_clean_series_ok():
    times = pd.to_datetime([MS_JAN1 + i * DAY_MS for i in range(10)], unit="ms", utc=True)
    df = pd.DataFrame({"borrow_time": times, "borrow_rate": [1e-4] * 10})
    res = checks.check_borrow(df)
    assert res.ok
    assert res.stats["missing_days"] == 0
