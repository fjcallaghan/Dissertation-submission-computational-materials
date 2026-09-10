"""BTC-USD daily spot price via ``yfinance``.

Normalises yfinance's output (which uses MultiIndex columns and adjusts prices
by default in recent versions) into a tidy frame:
    date (datetime64[ns, UTC]), open, high, low, close, volume
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from ..config_loader import Config
from . import base

logger = logging.getLogger(__name__)

SOURCE = "yfinance"


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Recent yfinance returns MultiIndex columns like (Close, BTC-USD)."""
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level, keeping the price field level.
        level0 = df.columns.get_level_values(0)
        df = df.copy()
        df.columns = level0
    return df


def fetch_spot(cfg: Config) -> pd.DataFrame:
    """Fetch and cache the daily BTC-USD spot series."""
    ticker = cfg.spot_ticker
    interval = cfg.spot_frequency
    start = cfg.window.price_start
    # yfinance ``end`` is exclusive, so add a day to include today.
    end = cfg.window.end + dt.timedelta(days=1)

    parquet_path = cfg.raw_dir / f"{SOURCE}_{ticker}.parquet"

    def _fetch() -> pd.DataFrame:
        import yfinance as yf  # imported lazily so config checks don't need it

        logger.info("yfinance download %s %s %s..%s", ticker, interval, start, end)
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")

        raw = _flatten_columns(raw, ticker)
        raw = raw.reset_index()

        # The date column is 'Date' (daily) or 'Datetime' (intraday).
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        rename = {
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        raw = raw.rename(columns=rename)
        keep = ["date", "open", "high", "low", "close", "volume"]
        df = raw[[c for c in keep if c in raw.columns]].copy()
        if "close" not in df.columns:
            raise RuntimeError(
                f"yfinance {ticker}: no 'close' column after normalising "
                f"(got {list(raw.columns)}); the API schema may have changed"
            )

        # Normalise the date to tz-aware UTC midnight.
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    return base.get_or_fetch(
        parquet_path,
        _fetch,
        source=SOURCE,
        symbol=ticker,
        time_col="date",
        use_cache=cfg.use_cache,
        refresh=cfg.refresh,
        params={"interval": interval, "start": str(start), "end": str(end)},
    )
