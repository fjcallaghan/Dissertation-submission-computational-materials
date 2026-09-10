"""Acquisition orchestrator — ``python -m src.acquire``.

Runs every source fetcher, writes raw parquet + manifests into ``data/raw/``,
and prints a one-line coverage summary per source. Individual source failures
are caught and reported so one flaky endpoint doesn't abort the whole run.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from ..config_loader import load_config
from . import base
from .binance_funding import fetch_binance_funding
from .bitfinex_borrow import fetch_bitfinex_borrow
from .bybit_funding import fetch_bybit_funding
from .coinalyze_funding import fetch_coinalyze_funding
from .spot import fetch_spot


def _summarise(name: str, df: pd.DataFrame | None, time_col: str) -> str:
    if df is None:
        return f"  {name:<20} skipped"
    if df.empty:
        return f"  {name:<20} 0 rows"
    lo, hi = df[time_col].min(), df[time_col].max()
    return f"  {name:<20} {len(df):>6} rows   {lo}  ->  {hi}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    client = base.HttpClient()
    print(f"Acquisition — window {cfg.window.funding_start} .. {cfg.window.end}")
    print(f"Cache dir: {cfg.raw_dir}  (use_cache={cfg.use_cache}, refresh={cfg.refresh})")

    lines: list[str] = []
    failures = 0

    tasks = [
        ("BTC-USD spot", lambda: fetch_spot(cfg), "date"),
        ("Binance funding", lambda: fetch_binance_funding(cfg, client), "funding_time"),
        ("Bybit funding", lambda: fetch_bybit_funding(cfg, client), "funding_time"),
        ("Coinalyze funding", lambda: fetch_coinalyze_funding(cfg, client), "funding_time"),
    ]
    if cfg.borrow_enabled:
        tasks.append(
            ("BTC borrow rate", lambda: fetch_bitfinex_borrow(cfg, client), "borrow_time")
        )
    for name, fn, time_col in tasks:
        try:
            df = fn()
            lines.append(_summarise(name, df, time_col))
        except Exception as exc:  # noqa: BLE001 — report, don't abort the run
            failures += 1
            logging.exception("%s failed", name)
            lines.append(f"  {name:<20} FAILED: {exc}")

    print("\nAcquisition summary:")
    print("\n".join(lines))
    if failures:
        print(f"\n{failures} source(s) failed — see logs above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
