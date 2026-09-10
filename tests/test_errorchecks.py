"""Guard clauses raise clear errors on bad inputs."""

import numpy as np
import pandas as pd
import pytest

from src.align.aggregate import funding_to_daily
from src.clean.gapfill import gapfill_price
from src.clean.spikes import flag_funding_spikes
from src.config_loader import _validate_config_choices
from src.datasets import ensure_columns, ensure_nonempty
from src.synthetic.paths import simulate_cev, simulate_gbm


def test_ensure_columns_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing expected column"):
        ensure_columns(df, ["a", "b"], source="test")


def test_ensure_nonempty_raises():
    with pytest.raises(ValueError, match="empty"):
        ensure_nonempty(pd.DataFrame(), source="test")


def test_spikes_bad_params():
    df = pd.DataFrame({"funding_time": pd.date_range("2020", periods=3, freq="8h", tz="UTC"),
                       "funding_rate": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="sigma must be positive"):
        flag_funding_spikes(df, sigma=0, baseline_window_days=30)
    with pytest.raises(ValueError, match="missing column"):
        flag_funding_spikes(df.rename(columns={"funding_rate": "x"}),
                            sigma=5, baseline_window_days=30)


def test_gapfill_bad_method():
    df = pd.DataFrame({"date": pd.date_range("2020", periods=2, tz="UTC"), "close": [1, 2]})
    with pytest.raises(ValueError, match="unknown price_gap_fill"):
        gapfill_price(df, method="bogus")


def test_funding_to_daily_missing_col():
    df = pd.DataFrame({"funding_time": pd.date_range("2020", periods=2, freq="8h", tz="UTC")})
    with pytest.raises(ValueError, match="missing column"):
        funding_to_daily(df, rate_col="funding_rate")


def test_synthetic_bad_params():
    with pytest.raises(ValueError, match="sigma must be positive"):
        simulate_gbm(s0=1, sigma=-1, t_horizon=1, n_steps=10, n_paths=10, seed=0)
    with pytest.raises(ValueError, match="rho must be > 1"):
        simulate_cev(s0=1, sigma=0.5, rho=1.0, t_horizon=1, n_steps=10, n_paths=10, seed=0)


def test_config_choices_valid_passes():
    # A fully valid subset should not raise.
    _validate_config_choices({
        "funding": {"daily_aggregation": "sum", "rolling_window_days": 30},
        "returns": {"method": "log"},
        "cleaning": {"price_gap_fill": "ffill", "spike_sigma": 5.0,
                     "spike_baseline_window_days": 30},
        "borrow": {"daily_aggregation": "mean"},
    })


def test_config_bad_enum_raises():
    with pytest.raises(ValueError, match="returns.method"):
        _validate_config_choices({"returns": {"method": "bogus"}})
    with pytest.raises(ValueError, match="daily_aggregation"):
        _validate_config_choices({"funding": {"daily_aggregation": "median"}})


def test_config_nonpositive_raises():
    with pytest.raises(ValueError, match="spike_sigma"):
        _validate_config_choices({"cleaning": {"spike_sigma": 0}})
    with pytest.raises(ValueError, match="must be a number"):
        _validate_config_choices({"funding": {"rolling_window_days": "thirty"}})
