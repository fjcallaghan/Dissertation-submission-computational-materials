"""Bitfinex BTC margin borrow (lending) rate — the DIRECT short-selling-cost proxy.

The perpetual funding rate (``binance_funding`` / ``bybit_funding``) is only an
*indirect* proxy for short-selling frictions: it is the premium longs pay, and a
short is *paid* on a positive rate. The margin borrow rate is the cost of the
friction itself — the rate paid to borrow BTC in order to sell it short. It is
the crypto analogue of the equity short-borrow fee used by Ofek & Richardson
(2003) to link short-sale constraints to the dot-com bubble.

Source: the Bitfinex public funding/lending market for BTC (``fBTC``), via the
v2 candles endpoint. Unlike the 8-hourly perpetual funding series this is a
**daily** rate, and its history begins in mid-2016 — so it also covers the 2017
cycle that the Sep-2019 funding window misses.

Endpoint: ``GET https://api-pub.bitfinex.com/v2/candles/{candle_key}/hist``
  ``candle_key`` e.g. ``trade:1D:fBTC:a30:p2:p30`` (30-day-aggregated Flash
  Return Rate). Response: a JSON array of candles
  ``[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]``; CLOSE is the day's rate, a per-day
  fraction.

  Note: this aggregated-FRR candle endpoint returns HTTP 500 when given
  ``start``/``end`` time-range params (a server-side quirk), but serves the full
  history reliably with ``sort=-1`` + ``limit``. We therefore pull the most
  recent ``limit`` candles in one request and filter to the configured window
  client-side. fBTC has ~3.6k daily rows since 2016, well under the 10k cap.

Output schema: borrow_time (datetime64[ns, UTC]), borrow_rate (float, per day).
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
    rate. Pure (no network / no config) so it can be unit-tested directly.
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
    """Fetch and cache the Bitfinex BTC borrow-rate series (paged forward)."""
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
