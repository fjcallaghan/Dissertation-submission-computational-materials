"""Binance BTCUSDT perpetual funding-rate history (the primary novel series).

Endpoint: ``GET https://fapi.binance.com/fapi/v1/fundingRate``
Params:   symbol, startTime, endTime, limit (max 1000).
Response: list of {symbol, fundingTime (ms), fundingRate (string), markPrice}.

Public market-data endpoint — read-only, no authentication, accessible despite
the FCA restriction on Binance *trading* for UK retail customers. Funding
settles 8-hourly; the contract's first settlement was 2019-09-10, so requests
before then simply return nothing.

Output schema: funding_time (datetime64[ns, UTC]), funding_rate (float).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config_loader import Config
from . import base

logger = logging.getLogger(__name__)

SOURCE = "binance"
BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
PAGE_LIMIT = 1000
TS_KEY = "fundingTime"


def fetch_binance_funding(cfg: Config, client: base.HttpClient | None = None) -> pd.DataFrame:
    """Fetch and cache the full Binance BTCUSDT funding history."""
    symbol = cfg.perp_symbol
    client = client or base.HttpClient()
    start_ms = base.utc_to_ms(cfg.window.funding_start)
    end_ms = base.utc_to_ms(cfg.window.end) + 86_400_000  # inclusive of today

    parquet_path = cfg.raw_dir / f"{SOURCE}_{symbol}_funding.parquet"

    def _fetch_page(cursor_ms: int) -> list[dict[str, Any]]:
        params = {
            "symbol": symbol,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": PAGE_LIMIT,
        }
        data = client.get_json(BASE_URL, params=params)
        # Binance signals errors with a JSON object ({"code": ..., "msg": ...}).
        # Surface it rather than silently returning an empty series.
        if isinstance(data, dict):
            raise RuntimeError(f"binance funding error: {data.get('msg', data)}")
        if not isinstance(data, list):
            raise RuntimeError(f"binance funding: unexpected response {str(data)[:120]!r}")
        return data

    def _fetch() -> pd.DataFrame:
        records = base.paginate_forward(
            _fetch_page,
            start_ms=start_ms,
            end_ms=end_ms,
            timestamp_key=TS_KEY,
            page_limit=PAGE_LIMIT,
        )
        logger.info("binance funding: %d records", len(records))
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
        params={"interval": "8h", "start_ms": start_ms, "end_ms": end_ms},
    )
