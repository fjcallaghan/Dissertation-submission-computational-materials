"""Cleaning stage: spike flagging and price gap-filling."""

import numpy as np
import pandas as pd

from src.clean.gapfill import gapfill_price
from src.clean.spikes import flag_funding_spikes

START = pd.Timestamp("2019-09-10 00:00:00", tz="UTC")


def _funding(n=300, rate=0.0001):
    times = pd.date_range(START, periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_time": times, "funding_rate": [rate] * n})


def test_spike_flagged_and_winsorised():
    df = _funding()
    df.loc[200, "funding_rate"] = 0.02  # a clear liquidation-cascade spike
    out = flag_funding_spikes(df, sigma=5.0, baseline_window_days=30)
    assert bool(out.loc[200, "is_spike"]) is True
    # Raw kept, cleaned value pulled back toward the baseline.
    assert out.loc[200, "funding_rate"] == 0.02
    assert out.loc[200, "funding_rate_clean"] < 0.02
    # A normal point is not flagged.
    assert bool(out.loc[100, "is_spike"]) is False


def test_no_false_positives_on_flat_series():
    out = flag_funding_spikes(_funding(), sigma=5.0, baseline_window_days=30)
    assert int(out["is_spike"].sum()) == 0


def test_gapfill_fills_missing_day_and_flags():
    dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": dates, "open": range(10), "high": range(10),
        "low": range(10), "close": range(1, 11), "volume": [1] * 10,
    })
    df = df.drop(index=[5]).reset_index(drop=True)  # remove one calendar day
    out = gapfill_price(df, method="ffill")
    assert len(out) == 10  # calendar restored
    assert int(out["was_filled"].sum()) == 1
    filled = out[out["was_filled"]].iloc[0]
    assert filled["close"] == 5  # forward-filled from the prior day (close=5)
