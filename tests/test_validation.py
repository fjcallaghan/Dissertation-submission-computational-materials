"""Validation flags fire on crafted-bad frames and stay quiet on clean ones."""

import pandas as pd

from src.validate import checks

EIGHT_H = pd.Timedelta(hours=8)
START = pd.Timestamp("2019-09-10 00:00:00", tz="UTC")


def _clean_funding(n=30):
    times = pd.date_range(START, periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_time": times, "funding_rate": [0.0001] * n})


def _clean_spot(n=30):
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": range(1, n + 1),
            "high": range(1, n + 1),
            "low": range(1, n + 1),
            "close": range(1, n + 1),
            "volume": [100] * n,
        }
    )


def test_clean_funding_passes():
    res = checks.check_funding(_clean_funding())
    assert res.ok, res.findings
    assert res.stats["completeness_%"] == 100.0


def test_funding_duplicate_flagged():
    df = _clean_funding()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    res = checks.check_funding(df)
    assert any("duplicate" in f for f in res.findings)


def test_funding_cadence_gap_flagged():
    df = _clean_funding()
    df = df.drop(index=[10, 11, 12]).reset_index(drop=True)  # 32h hole
    res = checks.check_funding(df)
    assert any("cadence gap" in f for f in res.findings)


def test_funding_out_of_band_flagged():
    df = _clean_funding()
    df.loc[7, "funding_rate"] = 0.05  # far beyond the +/-0.75% sanity band
    res = checks.check_funding(df)
    assert any("beyond" in f for f in res.findings)


def test_clean_spot_passes():
    res = checks.check_spot(_clean_spot())
    assert res.ok, res.findings


def test_spot_missing_day_flagged():
    df = _clean_spot()
    df = df.drop(index=[15]).reset_index(drop=True)
    res = checks.check_spot(df)
    assert any("missing calendar day" in f for f in res.findings)
    assert res.stats["missing_days"] == 1


def test_spot_nonpositive_flagged():
    df = _clean_spot()
    df.loc[3, "close"] = 0
    res = checks.check_spot(df)
    assert any("non-positive" in f for f in res.findings)
