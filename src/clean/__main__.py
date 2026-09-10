"""Cleaning runner — ``python -m src.clean``.

Reads the raw datasets, flags funding spikes, gap-fills the price calendar, and
writes cleaned parquet files to ``data/processed/``. Reports what it changed;
nothing is dropped silently.
"""

from __future__ import annotations

import logging
import sys

from .. import datasets
from ..config_loader import load_config
from .gapfill import gapfill_price
from .spikes import flag_funding_spikes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    cfg = load_config()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cleaning — spike_sigma={cfg.spike_sigma}, "
          f"baseline={cfg.spike_baseline_window_days}d, gap_fill={cfg.price_gap_fill}")

    # --- funding spikes (Binance primary; Bybit too if present) ---
    binance = datasets.load_binance_funding(cfg)
    binance_clean = flag_funding_spikes(
        binance, sigma=cfg.spike_sigma,
        baseline_window_days=cfg.spike_baseline_window_days,
    )
    n_spikes = int(binance_clean["is_spike"].sum())
    datasets.save_processed(binance_clean, cfg, "binance_funding_clean.parquet")
    print(f"  Binance funding: {n_spikes} spikes flagged / {len(binance_clean)} obs "
          f"({100*n_spikes/len(binance_clean):.2f}%)")

    bybit = datasets.load_bybit_funding(cfg)
    if bybit is not None:
        bybit_clean = flag_funding_spikes(
            bybit, sigma=cfg.spike_sigma,
            baseline_window_days=cfg.spike_baseline_window_days,
        )
        nb = int(bybit_clean["is_spike"].sum())
        datasets.save_processed(bybit_clean, cfg, "bybit_funding_clean.parquet")
        print(f"  Bybit funding:   {nb} spikes flagged / {len(bybit_clean)} obs")

    # --- borrow-rate squeeze spikes (daily cadence -> obs_per_day=1) ---
    if cfg.borrow_enabled:
        borrow = datasets.load_bitfinex_borrow(cfg)
        if borrow is not None:
            borrow_clean = flag_funding_spikes(
                borrow, sigma=cfg.spike_sigma,
                baseline_window_days=cfg.spike_baseline_window_days,
                rate_col="borrow_rate", time_col="borrow_time", obs_per_day=1,
            )
            nbr = int(borrow_clean["is_spike"].sum())
            datasets.save_processed(borrow_clean, cfg, "bitfinex_borrow_clean.parquet")
            print(f"  BTC borrow rate: {nbr} squeeze spikes flagged / {len(borrow_clean)} obs")

    # --- price gap-filling ---
    spot = datasets.load_spot(cfg)
    spot_clean = gapfill_price(spot, method=cfg.price_gap_fill)
    n_filled = int(spot_clean["was_filled"].sum())
    datasets.save_processed(spot_clean, cfg, "spot_clean.parquet")
    print(f"  BTC-USD spot:    {n_filled} calendar days gap-filled / {len(spot_clean)} days")

    print(f"\nCleaned datasets written to {cfg.processed_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
