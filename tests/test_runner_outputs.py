"""Regression checks for unsupported settings and stale episode outputs."""
import copy
import importlib
import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config_loader import DEFAULT_CONFIG_PATH, load_config
from src.detect.critvals import CriticalValues


@pytest.mark.parametrize("section,key,value,match", [
    ("psy", "stat", "sadf", "supports only 'gsadf'"),
    ("funding", "daily_aggregation", "none", "funding.daily_aggregation"),
])
def test_configuration_rejects_unsupported_runner_modes(tmp_path, section, key, value, match):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    target = raw["detect"][section] if section == "psy" else raw[section]
    target[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_zero_episode_run_replaces_previous_episode_table(tmp_path, monkeypatch):
    runner = importlib.import_module("src.detect.__main__")
    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)
    raw["cache"]["processed_dir"] = str(tmp_path)
    raw["plots"]["output_dir"] = str(tmp_path)
    raw["detect"]["slm"]["enabled"] = False
    cfg = replace(cfg, raw=raw)
    n = 90
    close = np.exp(np.cumsum(np.random.default_rng(11).normal(0, .01, n)))
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
                          "close": close, "return": np.r_[np.nan, np.diff(np.log(close))],
                          "borrow_daily": np.nan, "funding_daily": np.nan})
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    monkeypatch.setattr(runner, "synthetic_validation", lambda *a, **kw: (["test calibration"], True))
    monkeypatch.setattr(runner.datasets, "load_processed",
                        lambda cfg, name, **kw: frame if name == "aligned_daily_full.parquet" else None)
    monkeypatch.setattr(runner.datasets, "load_bitfinex_borrow", lambda *a, **kw: None)
    monkeypatch.setattr(runner.datasets, "load_binance_funding", lambda *a, **kw: None)
    monkeypatch.setattr(runner.figures, "figure_gsadf_datestamp", lambda *a, **kw: Mock())
    for threshold, expected_empty in [(-1e6, False), (1e6, True)]:
        cv = CriticalValues("gsadf", "mc", {.95: threshold},
                            (np.arange(n) + 1) / n, {.95: np.full(n, threshold)}, {})
        monkeypatch.setattr(runner, "simulate_critical_values", lambda **kw: cv)
        assert runner.main([]) == 0
        episodes = pd.read_parquet(tmp_path / "bubble_episodes.parquet")
        assert episodes.empty == expected_empty
        assert list(episodes.columns) == ["start_date", "end_date", "n_days", "peak_date", "peak_price"]
    sequence = pd.read_parquet(tmp_path / "gsadf_sequence.parquet")
    assert sequence.in_episode.sum() == 0
    assert json.loads((tmp_path / "detect_meta.json").read_text())["n_episodes"] == 0
