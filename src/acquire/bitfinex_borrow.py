"""Executed BTC lending-rate proxy from Bitfinex's fBTC market.

The series records completed lending transactions across different loan terms.
It is not a borrowing offer available to every investor on identical terms.
Positive perpetual funding, by contrast, is paid to a short position.

Endpoint: GET https://api-pub.bitfinex.com/v2/candles/{candle_key}/hist
The key trade:1D:fBTC:a30:p2:p30 selects an aggregate lending-trade candle
across loan periods p2 through p30. It is distinct from Bitfinex's separately
defined Flash Return Rate. Rows are [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME];
CLOSE is the last executed rate in the daily candle, expressed per day.
The saved history begins in mid-2016.

At acquisition, this endpoint rejected start/end parameters with HTTP 500.
The fetch therefore requests the most recent limit candles and filters the
configured window locally. A full response at the 10,000-row cap raises a
truncation warning. This documents the observed behaviour, not a permanent
API guarantee.

Output: borrow_time (datetime64[ns, UTC]), borrow_rate (float, per day).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config_loader import Config
from . import base

logger = logging.getLogger(__name__)

SOURCE = "bitfinex"
BASE_URL = "https://api-pub.bitfinex.com/v2/candles"
PAGE_LIMIT = 10_000  # Bitfinex candle history hard cap; fBTC has far fewer rows


def parse_candles(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert raw Bitfinex candle arrays to the borrow-rate schema.

    Each row is ``[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]``; CLOSE is the day's
    last executed rate. Pure (no network / no config) so it can be unit-tested directly.
    """
    if not rows:
        return pd.DataFrame(columns=["borrow_time", "borrow_rate"])
    # Bitfinex candles are [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]; validate the
    # shape so a malformed payload raises a clear error, not a cryptic pandas one.
    bad = [i for i, r in enumerate(rows) if not isinstance(r, (list, tuple)) or len(r) < 6]
    if bad:
        raise RuntimeError(
            f"bitfinex borrow: {len(bad)} malformed candle row(s) "
            f"(e.g. index {bad[0]}: {rows[bad[0]]!r}); expected >=6 fields each"
        )
    rows = [r[:6] for r in rows]  # tolerate extra trailing fields
    df = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
    out = pd.DataFrame(
        {
            "borrow_time": df["mts"].astype("int64").map(base.ms_to_utc),
            "borrow_rate": df["close"].astype(float),
        }
    )
    return (
        out.sort_values("borrow_time")
        .drop_duplicates("borrow_time")
        .reset_index(drop=True)
    )


def fetch_bitfinex_borrow(cfg: Config, client: base.HttpClient | None = None) -> pd.DataFrame:
    """Fetch and cache the Bitfinex BTC borrow-rate series (filtered to the configured window)."""
    client = client or base.HttpClient()
    candle_key = cfg.borrow_candle
    start_ms = base.utc_to_ms(cfg.borrow_start)
    end_ms = base.utc_to_ms(cfg.window.end) + 86_400_000
    url = f"{BASE_URL}/{candle_key}/hist"

    parquet_path = cfg.raw_dir / "bitfinex_BTC_borrow.parquet"

    def _fetch() -> pd.DataFrame:
        # Pull the most-recent `limit` candles (no start/end — see module note),
        # then filter to the configured window client-side.
        data = client.get_json(url, params={"sort": -1, "limit": PAGE_LIMIT})
        if not isinstance(data, list):
            raise RuntimeError(f"bitfinex borrow: unexpected response {str(data)[:120]!r}")
        if len(data) >= PAGE_LIMIT:
            logger.warning(
                "bitfinex borrow: hit limit=%d; history may be truncated at the far end",
                PAGE_LIMIT,
            )
        df = parse_candles(data)  # sorted ascending, de-duplicated
        lo, hi = base.ms_to_utc(start_ms), base.ms_to_utc(end_ms)
        df = df[(df["borrow_time"] >= lo) & (df["borrow_time"] <= hi)].reset_index(drop=True)
        logger.info("bitfinex borrow: %d candles in window", len(df))
        return df

    return base.get_or_fetch(
        parquet_path,
        _fetch,
        source=SOURCE,
        symbol="fBTC",
        time_col="borrow_time",
        use_cache=cfg.use_cache,
        refresh=cfg.refresh,
        params={"candle": candle_key, "start_ms": start_ms, "end_ms": end_ms},
    )
