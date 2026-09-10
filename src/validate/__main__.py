"""Validation runner — ``python -m src.validate``.

Reads the cached raw parquet files, runs the coverage/anomaly checks, and writes
a human-readable report to ``logs/validation_<date>.txt`` (and stdout). Reports
only; it does not modify the data.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from ..config_loader import load_config
from . import checks


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    raw = cfg.raw_dir
    symbol = cfg.perp_symbol
    ticker = cfg.spot_ticker

    # (label, path, check-fn) for each raw series produced by acquisition.
    datasets = [
        ("BTC-USD spot", raw / f"yfinance_{ticker}.parquet", checks.check_spot),
        ("Binance funding", raw / f"binance_{symbol}_funding.parquet", checks.check_funding),
        ("Bybit funding", raw / f"bybit_{symbol}_funding.parquet", checks.check_funding),
    ]
    if cfg.borrow_enabled:
        datasets.append(
            ("BTC borrow rate", raw / "bitfinex_BTC_borrow.parquet", checks.check_borrow)
        )

    results: list[checks.CheckResult] = []
    missing: list[str] = []
    for label, path, fn in datasets:
        df = _load(path)
        if df is None:
            missing.append(f"{label}: {path.name} not found — run `python -m src.acquire` first")
            continue
        results.append(fn(df, name=label))

    stamp = dt.datetime.now(dt.timezone.utc)
    header = [
        "BTC bubble-detection — validation report",
        f"generated: {stamp.isoformat()}",
        f"analysis window: {cfg.window.funding_start} .. {cfg.window.end}",
        "=" * 64,
    ]
    body = [r.report() for r in results]
    if missing:
        body += ["", "Missing datasets:"] + [f"  - {m}" for m in missing]

    n_flagged = sum(1 for r in results if not r.ok)
    footer = [
        "=" * 64,
        f"{len(results)} series checked, {n_flagged} with flagged findings.",
        "Note: findings are reported, not cleaned. Cleaning is a later phase.",
    ]
    report = "\n\n".join(["\n".join(header), *body, "\n".join(footer)])

    log_dir = cfg._resolve_dir("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"validation_{stamp.date().isoformat()}.txt"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
