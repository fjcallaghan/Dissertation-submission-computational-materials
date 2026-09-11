"""Figure runner — ``python -m src.plots``.

Regenerates the two exploratory figures from the pipeline data into the
configured figures directory.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .. import datasets
from ..config_loader import load_config
from . import figures


def _episodes_from_indicator(in_episode) -> list[tuple[int, int]]:
    """Reconstruct (start, end) index runs from a 0/1 in-episode column."""
    mask = np.asarray(in_episode, dtype=bool)
    episodes, i, n = [], 0, mask.shape[0]
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        episodes.append((i, j))
        i = j + 1
    return episodes


def _render_detect_figures(cfg, fig_dir, dpi) -> list[str]:
    """Redraw the Phase IV figures (4, 5) from the saved GSADF sequence, if present."""
    seq = datasets.load_processed(cfg, "gsadf_sequence.parquet", required=False)
    if seq is None:
        return []
    meta_path = cfg.processed_dir / "detect_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    episodes = _episodes_from_indicator(seq["in_episode"])
    written = []

    f4 = figures.figure_gsadf_datestamp(
        seq, episodes,
        scalar_cv=float(meta.get("scalar_cv", np.nan)),
        gsadf_stat=float(meta.get("gsadf_stat", np.nanmax(seq["bsadf"]))),
        level=float(meta.get("level", cfg.detect_significance)),
    )
    f4.savefig(fig_dir / "fig4_gsadf_datestamp.png", dpi=dpi)
    written.append("fig4_gsadf_datestamp.png")

    borrow = datasets.load_processed(cfg, "bitfinex_borrow_clean.parquet", required=False)
    b_col = "borrow_rate_clean"
    if borrow is None:
        borrow = datasets.load_bitfinex_borrow(cfg, required=False)
        b_col = "borrow_rate"
    funding = datasets.load_processed(cfg, "binance_funding_clean.parquet", required=False)
    f_col = "funding_rate_clean"
    if funding is None:
        funding = datasets.load_binance_funding(cfg, required=False)
        f_col = "funding_rate"
    if borrow is not None and funding is not None:
        f5 = figures.figure_bubbles_vs_proxies(
            seq, episodes, borrow, funding,
            rate_col_borrow=b_col, rate_col_funding=f_col,
            rolling_days=cfg.rolling_window_days,
            borrow_start=pd.Timestamp(cfg.borrow_start, tz="UTC"),
            funding_start=pd.Timestamp(cfg.window.funding_start, tz="UTC"),
        )
        f5.savefig(fig_dir / "fig5_bubbles_vs_proxies.png", dpi=dpi)
        written.append("fig5_bubbles_vs_proxies.png")
    return written


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.raw.get("plots", {}).get("dpi", 200))

    # Prefer cleaned/aligned data; fall back to raw so plots work after acquire.
    spot = datasets.load_processed(cfg, "spot_clean.parquet", required=False)
    if spot is None:
        spot = datasets.load_spot(cfg)
    funding = datasets.load_processed(cfg, "binance_funding_clean.parquet", required=False)
    rate_col = "funding_rate_clean"
    if funding is None:
        funding = datasets.load_binance_funding(cfg)
        rate_col = "funding_rate"

    # Figure 1
    f1 = figures.figure_price_log(spot)
    p1 = fig_dir / "fig1_btc_price_log.png"
    f1.savefig(p1, dpi=dpi)

    # Figure 2 (rolling window in 8h observations = days * 3)
    f2 = figures.figure_price_funding_overlay(
        spot, funding,
        rate_col=rate_col,
        rolling_obs=cfg.rolling_window_days * 3,
        threshold=cfg.elevated_threshold,
        analysis_start=pd.Timestamp(cfg.window.analysis_start, tz="UTC"),
    )
    p2 = fig_dir / "fig2_price_funding_overlay.png"
    f2.savefig(p2, dpi=dpi)

    written = [p1.name, p2.name]

    # Figure 3 — executed lending-rate proxy, 2016+ (covers 2017 too).
    if cfg.borrow_enabled:
        borrow = datasets.load_processed(cfg, "bitfinex_borrow_clean.parquet", required=False)
        b_rate_col = "borrow_rate_clean"
        if borrow is None:
            borrow = datasets.load_bitfinex_borrow(cfg, required=False)
            b_rate_col = "borrow_rate"
        if borrow is not None:
            f3 = figures.figure_price_borrow_overlay(
                spot, borrow,
                rate_col=b_rate_col,
                rolling_days=cfg.rolling_window_days,
                start=pd.Timestamp(cfg.borrow_start, tz="UTC"),
            )
            p3 = fig_dir / "fig3_price_borrow_overlay.png"
            f3.savefig(p3, dpi=dpi)
            written.append(p3.name)

    # Phase IV figures (4, 5) — only if `python -m src.detect` has produced its
    # GSADF sequence; otherwise the three Phase III figures above are all.
    written += _render_detect_figures(cfg, fig_dir, dpi)

    print(f"Figures written to {fig_dir}:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
