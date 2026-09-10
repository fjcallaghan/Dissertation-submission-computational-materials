"""Shared helpers for locating and loading the pipeline's datasets.

Centralises the raw/processed file naming so the clean, align, reconcile and
plot stages all agree on where data lives.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config_loader import Config


# ----- lightweight, reusable data-quality guards -----
def ensure_columns(df: pd.DataFrame, cols: Iterable[str], *, source: str) -> None:
    """Raise a clear error if ``df`` is missing any expected column."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing expected column(s) {missing}; got {list(df.columns)}"
        )


def ensure_nonempty(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Raise if ``df`` is empty, otherwise return it unchanged."""
    if df is None or df.empty:
        raise ValueError(f"{source}: dataset is empty — nothing to process.")
    return df


# ----- raw (produced by src.acquire) -----
def raw_spot_path(cfg: Config) -> Path:
    return cfg.raw_dir / f"yfinance_{cfg.spot_ticker}.parquet"


def raw_binance_path(cfg: Config) -> Path:
    return cfg.raw_dir / f"binance_{cfg.perp_symbol}_funding.parquet"


def raw_bybit_path(cfg: Config) -> Path:
    return cfg.raw_dir / f"bybit_{cfg.perp_symbol}_funding.parquet"


def raw_coinalyze_path(cfg: Config) -> Path:
    return cfg.raw_dir / "coinalyze_BTC_funding.parquet"


def raw_bitfinex_borrow_path(cfg: Config) -> Path:
    return cfg.raw_dir / "bitfinex_BTC_borrow.parquet"


def _load(path: Path, required: bool) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    if required:
        raise FileNotFoundError(
            f"{path.name} not found — run `python -m src.acquire` first."
        )
    return None


def load_spot(cfg: Config, required: bool = True) -> pd.DataFrame | None:
    return _load(raw_spot_path(cfg), required)


def load_binance_funding(cfg: Config, required: bool = True) -> pd.DataFrame | None:
    return _load(raw_binance_path(cfg), required)


def load_bybit_funding(cfg: Config, required: bool = False) -> pd.DataFrame | None:
    return _load(raw_bybit_path(cfg), required)


def load_coinalyze_funding(cfg: Config, required: bool = False) -> pd.DataFrame | None:
    return _load(raw_coinalyze_path(cfg), required)


def load_bitfinex_borrow(cfg: Config, required: bool = False) -> pd.DataFrame | None:
    return _load(raw_bitfinex_borrow_path(cfg), required)


# ----- processed (produced by src.clean / src.align) -----
def processed_path(cfg: Config, name: str) -> Path:
    return cfg.processed_dir / name


def save_processed(df: pd.DataFrame, cfg: Config, name: str) -> Path:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_path(cfg, name)
    df.to_parquet(path, index=False)
    return path


def load_processed(cfg: Config, name: str, required: bool = True) -> pd.DataFrame | None:
    return _load(processed_path(cfg, name), required)
