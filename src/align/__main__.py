"""Alignment runner — ``python -m src.align``.

Joins the cleaned daily price and the daily-aggregated funding onto a common
UTC date index, computes returns, and adds a rolling-mean funding indicator.
Writes the analysis-ready dataset to ``data/processed/aligned_daily.parquet``.

Requires the cleaned datasets from ``python -m src.clean``.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from .. import datasets
from ..config_loader import load_config
from .aggregate import funding_to_daily
from .returns import compute_returns


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    cfg = load_config()

    agg = cfg.funding_daily_aggregation
    win = cfg.rolling_window_days
    thr = cfg.elevated_threshold
    print(f"Alignment — daily_aggregation={agg}, returns={cfg.returns_method}, "
          f"rolling={win}d, elevated>{thr}")

    # Cleaned inputs (fall back to raw if cleaning wasn't run).
    spot = datasets.load_processed(cfg, "spot_clean.parquet", required=False)
    if spot is None:
        logging.warning("spot_clean not found — using raw spot. Run `python -m src.clean`.")
        spot = datasets.load_spot(cfg)
    funding = datasets.load_processed(cfg, "binance_funding_clean.parquet", required=False)
    rate_col = "funding_rate_clean"
    if funding is None:
        logging.warning("binance_funding_clean not found — using raw funding.")
        funding = datasets.load_binance_funding(cfg)
        rate_col = "funding_rate"

    # Fail clearly if an upstream stage produced an unexpected shape.
    datasets.ensure_columns(datasets.ensure_nonempty(spot, source="align spot"),
                            ["date", "close"], source="align spot")
    datasets.ensure_columns(datasets.ensure_nonempty(funding, source="align funding"),
                            ["funding_time", rate_col], source="align funding")

    # Daily funding (both scales); pick the configured one as funding_daily.
    fdaily = funding_to_daily(funding, rate_col=rate_col)
    if agg == "mean":
        fdaily["funding_daily"] = fdaily["funding_mean"]
    else:  # "sum"; configuration validation rejects unsupported modes
        fdaily["funding_daily"] = fdaily["funding_sum"]

    # Rolling mean is computed on the per-8h (mean) scale so it is comparable to
    # the exchange-scale ``elevated_threshold``.
    fdaily[f"funding_roll{win}d"] = (
        fdaily["funding_mean"].rolling(window=win, min_periods=max(win // 3, 5)).mean()
    )
    fdaily["funding_elevated"] = fdaily[f"funding_roll{win}d"] > thr

    # Executed BTC lending-rate proxy (2016+, so it also
    # covers the 2017 cycle in the full series). Prefer the cleaned series.
    borrow_daily = None
    if cfg.borrow_enabled:
        borrow = datasets.load_processed(cfg, "bitfinex_borrow_clean.parquet", required=False)
        b_rate_col = "borrow_rate_clean"
        if borrow is None:
            borrow = datasets.load_bitfinex_borrow(cfg, required=False)
            b_rate_col = "borrow_rate"
        if borrow is not None:
            bdaily = funding_to_daily(
                borrow, rate_col=b_rate_col, time_col="borrow_time", prefix="borrow"
            )
            # Honour the configured collapse rule (borrow is daily, so sum==mean
            # unless a day ever carries multiple records).
            b_col = "borrow_sum" if cfg.borrow_daily_aggregation == "sum" else "borrow_mean"
            borrow_daily = bdaily[["date", b_col]].rename(
                columns={b_col: "borrow_daily"}
            )
            borrow_daily[f"borrow_roll{win}d"] = (
                borrow_daily["borrow_daily"]
                .rolling(window=win, min_periods=max(win // 3, 5))
                .mean()
            )

    # Daily price + returns.
    price = spot[["date", "close"]].copy()
    price["return"] = compute_returns(price["close"], method=cfg.returns_method)
    if "was_filled" in spot.columns:
        price["price_filled"] = spot["was_filled"].to_numpy()

    # Join on the common daily index. Price spans 2014+; funding 2019-09+;
    # borrow 2016+.
    merged = price.merge(fdaily, on="date", how="left")
    if borrow_daily is not None:
        merged = merged.merge(borrow_daily, on="date", how="left")
    # Restrict the joint dataset to the analysis window (funding available).
    start = pd.Timestamp(cfg.window.analysis_start, tz="UTC")
    joint = merged[merged["date"] >= start].reset_index(drop=True)

    out_all = datasets.save_processed(merged, cfg, "aligned_daily_full.parquet")
    out_joint = datasets.save_processed(joint, cfg, "aligned_daily.parquet")

    n_elev = int((joint["funding_elevated"] == True).sum())  # noqa: E712 — NaN-safe count
    print(f"  full series (price 2014+):     {len(merged)} rows -> {out_all.name}")
    print(f"  joint window ({cfg.window.analysis_start}+): {len(joint)} rows -> {out_joint.name}")
    print(f"  days with elevated rolling funding (>{thr}): {n_elev}")
    if borrow_daily is not None and "borrow_daily" in merged.columns:
        n_borrow = int(merged["borrow_daily"].notna().sum())
        b_lo = merged.loc[merged["borrow_daily"].notna(), "date"].min()
        print(f"  borrow-rate coverage (full series): {n_borrow} days from {b_lo}")
    print(f"\nAligned dataset written to {cfg.processed_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
