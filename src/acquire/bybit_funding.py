"""Bybit BTCUSDT perpetual funding history (independent-venue cross-check).

Endpoint: ``GET https://api.bybit.com/v5/market/funding/history``
Params:   category=linear, symbol, endTime, limit (max 200).
Response: {retCode, result: {list: [{symbol, fundingRate (str),
          fundingRateTimestamp (ms)}]}}, sorted *descending* by timestamp.

Bybit is FCA-registered and a large, independent derivatives venue, so its
agreement with Binance is the core robustness check that the funding signal is
not a Binance-specific artefact. Bybit's USDT perp launched later than Binance
and its public history is shorter, so the overlap window (not the full 2019
span) is what matters here.

Output schema: funding_time (datetime64[ns, UTC]), funding_rate (float).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config_loader import Config
from . import base

logger = logging.getLogger(__name__)

SOURCE = "bybit"
BASE_URL = "https://api.bybit.com/v5/market/funding/history"
PAGE_LIMIT = 200
TS_KEY = "fundingRateTimestamp"


def fetch_bybit_funding(cfg: Config, client: base.HttpClient | None = None) -> pd.DataFrame:
    """Fetch and cache Bybit BTCUSDT funding history (paged backward)."""
    symbol = cfg.perp_symbol
    client = client or base.HttpClient()
    start_ms = base.utc_to_ms(cfg.window.funding_start)
    end_ms = base.utc_to_ms(cfg.window.end) + 86_400_000

    parquet_path = cfg.raw_dir / f"{SOURCE}_{symbol}_funding.parquet"

    def _fetch_page(cursor_ms: int) -> list[dict[str, Any]]:
        params = {
            "category": "linear",
            "symbol": symbol,
            "endTime": cursor_ms,
            "limit": PAGE_LIMIT,
        }
        data = client.get_json(BASE_URL, params=params)
        if not isinstance(data, dict) or data.get("retCode") != 0:
            msg = data.get("retMsg") if isinstance(data, dict) else data
            raise RuntimeError(f"bybit funding error: {msg}")
        rows = data.get("result", {}).get("list", []) or []
        # Coerce the timestamp key to int for the paginator.
        for r in rows:
            r[TS_KEY] = int(r[TS_KEY])
        return rows

    def _fetch() -> pd.DataFrame:
        records = base.paginate_backward(
            _fetch_page,
            start_ms=start_ms,
            end_ms=end_ms,
            timestamp_key=TS_KEY,
            page_limit=PAGE_LIMIT,
        )
        logger.info("bybit funding: %d records", len(records))
        if not records:
            return pd.DataFrame(columns=["funding_time", "funding_rate"])
        df = pd.DataFrame.from_records(records)
        out = pd.DataFrame(
            {
                "funding_time": df[TS_KEY].map(base.ms_to_utc),
                "funding_rate": df["fundingRate"].astype(float),
            }
        )
        return out.sort_values("funding_time").reset_index(drop=True)

    return base.get_or_fetch(
        parquet_path,
        _fetch,
        source=SOURCE,
        symbol=symbol,
        time_col="funding_time",
        use_cache=cfg.use_cache,
        refresh=cfg.refresh,
        params={"category": "linear", "start_ms": start_ms, "end_ms": end_ms},
    )
