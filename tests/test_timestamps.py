"""Timestamp helper round-trips."""

import datetime as dt

import pandas as pd

from src.acquire import base


def test_ms_utc_round_trip():
    # 1568102400000 ms — Binance BTCUSDT first funding settlement, 08:00 UTC.
    ms = 1568102400000
    ts = base.ms_to_utc(ms)
    assert isinstance(ts, pd.Timestamp)
    assert str(ts.tz) == "UTC"
    assert ts == pd.Timestamp("2019-09-10 08:00:00", tz="UTC")
    assert base.utc_to_ms(ts.to_pydatetime()) == ms


def test_utc_to_ms_from_date():
    d = dt.date(2019, 9, 1)
    assert base.utc_to_ms(d) == 1567296000000


def test_naive_datetime_treated_as_utc():
    naive = dt.datetime(2020, 1, 1, 0, 0, 0)
    aware = dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    assert base.utc_to_ms(naive) == base.utc_to_ms(aware)
