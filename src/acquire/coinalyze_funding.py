"""Coinalyze aggregated funding-rate history (market-wide cross-check).

Endpoint: ``GET https://api.coinalyze.com/v1/funding-rate-history``
Params:   symbols (comma-separated, exchange-coded), interval, from, to (unix
          seconds). Auth via the ``api_key`` query parameter.
Response: list of {symbol, history: [{t (unix s), o, h, l, c}]} where o/h/l/c are
          the funding-rate open/high/low/close within each interval.

Coinalyze is an aggregator: pulling several exchange-coded perp symbols lets us
compare Binance against the broader market, strengthening the "not an
exchange-specific artefact" argument beyond the single Bybit venue.

Requires a *free* API key from https://coinalyze.net, read from the environment
variable ``COINALYZE_API_KEY``. If the key is absent this fetcher **skips
gracefully** (logs a warning, returns None) so the rest of the run still
completes. Symbols can be overridden via ``COINALYZE_SYMBOLS`` (comma-separated).

Output schema: funding_time (datetime64[ns, UTC]), funding_rate (float), symbol.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from ..config_loader import Config
from . import base

logger = logging.getLogger(__name__)

SOURCE = "coinalyze"
BASE_URL = "https://api.coinalyze.com/v1/funding-rate-history"

API_KEY_ENV = "COINALYZE_API_KEY"
SYMBOLS_ENV = "COINALYZE_SYMBOLS"
# Exchange-coded BTC perpetuals (suffix denotes the venue in Coinalyze's scheme:
# .A=Binance, .6=Bybit, .3=OKX). Override with COINALYZE_SYMBOLS if these drift.
DEFAULT_SYMBOLS = "BTCUSDT_PERP.A,BTCUSDT_PERP.6,BTCUSDT_PERP.3"
INTERVAL = "daily"


def fetch_coinalyze_funding(
    cfg: Config, client: base.HttpClient | None = None
) -> pd.DataFrame | None:
    """Fetch and cache Coinalyze aggregated funding, or skip if no API key."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        logger.warning(
            "%s not set — skipping Coinalyze cross-check. Get a free key at "
            "https://coinalyze.net and `export %s=...` to enable it.",
            API_KEY_ENV, API_KEY_ENV,
        )
        return None

    symbols = os.environ.get(SYMBOLS_ENV, DEFAULT_SYMBOLS)
    client = client or base.HttpClient()
    from_s = base.utc_to_ms(cfg.window.funding_start) // 1000
    to_s = base.utc_to_ms(cfg.window.end) // 1000 + 86_400

    parquet_path = cfg.raw_dir / f"{SOURCE}_BTC_funding.parquet"

    def _fetch() -> pd.DataFrame:
        params = {
            "symbols": symbols,
            "interval": INTERVAL,
            "from": from_s,
            "to": to_s,
            "api_key": api_key,
        }
        data = client.get_json(BASE_URL, params=params)
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected Coinalyze response: {type(data)}")

        frames = []
        for entry in data:
            sym = entry.get("symbol")
            hist: list[dict[str, Any]] = entry.get("history", []) or []
            if not hist:
                continue
            h = pd.DataFrame.from_records(hist)
            missing = [c for c in ("t", "c") if c not in h.columns]
            if missing:
                logger.warning(
                    "coinalyze %s: history missing column(s) %s — skipping symbol",
                    sym, missing,
                )
                continue
            frames.append(
                pd.DataFrame(
                    {
                        # 't' is unix *seconds*; use the interval close 'c'.
                        "funding_time": (h["t"].astype("int64") * 1000).map(base.ms_to_utc),
                        "funding_rate": h["c"].astype(float),
                        "symbol": sym,
                    }
                )
            )
        logger.info("coinalyze funding: %d symbols, %d rows",
                    len(frames), sum(len(f) for f in frames))
        if not frames:
            return pd.DataFrame(columns=["funding_time", "funding_rate", "symbol"])
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "funding_time"]).reset_index(drop=True)

    return base.get_or_fetch(
        parquet_path,
        _fetch,
        source=SOURCE,
        symbol=symbols,
        time_col="funding_time",
        use_cache=cfg.use_cache,
        refresh=cfg.refresh,
        params={"symbols": symbols, "interval": INTERVAL, "from": from_s, "to": to_s},
    )
