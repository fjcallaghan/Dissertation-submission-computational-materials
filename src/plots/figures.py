"""Reproduce the two exploratory thesis figures.

Figure 1: BTC-USD close on a log scale, 2014 to present, with the 2017 and 2021
          cycle peaks annotated.
Figure 2: BTC-USD close (log, left axis) overlaid with the Binance 8-hourly
          funding rate (right axis) from Sep 2019, its rolling mean, and shaded
          bands where the rolling mean exceeds the elevated threshold.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — save files, never open a window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _annotate_peak(ax, dates, prices, lo, hi, label, *, dx: float = 0.0,
                   dy: float = 30.0) -> None:
    """Annotate the maximum close within [lo, hi].

    Text is offset by (dx, dy) points from the peak, drawn in a semi-opaque box
    so it stays legible over gridlines, and never clipped by the axes frame
    (the caller adds headroom so it clears the top spine).
    """
    mask = (dates >= pd.Timestamp(lo, tz="UTC")) & (dates <= pd.Timestamp(hi, tz="UTC"))
    if not mask.any():
        return
    sub_p = prices[mask]
    sub_d = dates[mask]
    i = int(np.argmax(sub_p.to_numpy()))
    peak_price, peak_date = float(sub_p.iloc[i]), sub_d.iloc[i]
    ax.annotate(
        f"{label}\n${peak_price:,.0f}",
        xy=(peak_date, peak_price), xytext=(dx, dy), textcoords="offset points",
        ha="center", va="bottom", fontsize=9, annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"),
    )


def _require(df: pd.DataFrame, cols, *, name: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{name}: no data to plot — run the pipeline first.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}")


def figure_price_log(spot: pd.DataFrame) -> plt.Figure:
    """Figure 1 — long-run BTC price on a log scale with cycle peaks marked."""
    _require(spot, ["date", "close"], name="figure_price_log")
    d = spot.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(d["date"], d["close"], lw=1.0, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_title("BTC-USD spot price (log scale), 2014 to present")
    ax.set_ylabel("Price (USD, log)")
    ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Headroom above the highest point so annotation boxes clear the top frame.
    ax.set_ylim(top=float(d["close"].max()) * 2.6)
    _annotate_peak(ax, d["date"], d["close"], "2017-06-01", "2018-02-01",
                   "2017 cycle peak", dy=30)
    _annotate_peak(ax, d["date"], d["close"], "2021-01-01", "2022-01-01",
                   "2021 cycle peak", dy=30)
    fig.tight_layout()
    return fig


def _contiguous_true_spans(dates: pd.Series, mask: np.ndarray):
    """Yield (start_date, end_date) for each run of True in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    idx = np.flatnonzero(
        np.diff(np.concatenate(([0], mask.view(np.int8), [0]))) != 0
    )
    for s, e in zip(idx[0::2], idx[1::2]):
        yield dates.iloc[s], dates.iloc[min(e, len(dates) - 1)]


def figure_price_funding_overlay(
    spot: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    rate_col: str,
    rolling_obs: int,
    threshold: float,
    analysis_start: pd.Timestamp,
) -> plt.Figure:
    """Figure 2 — price (log) overlaid with funding, rolling mean and bands."""
    _require(spot, ["date", "close"], name="overlay figure (price)")
    _require(funding, ["funding_time", rate_col], name="overlay figure (funding)")
    if rolling_obs < 1:
        raise ValueError(f"rolling_obs must be >= 1, got {rolling_obs}")
    d = spot[spot["date"] >= analysis_start].sort_values("date")
    f = funding[funding["funding_time"] >= analysis_start].sort_values("funding_time").copy()
    f["roll"] = f[rate_col].rolling(window=rolling_obs, min_periods=max(rolling_obs // 3, 5)).mean()
    elevated = (f["roll"] > threshold).to_numpy()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(d["date"], d["close"], lw=1.0, color="#1f77b4", label="BTC-USD (log)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price (USD, log)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = ax1.twinx()
    ax2.plot(f["funding_time"], f[rate_col], lw=0.4, color="#d62728", alpha=0.35,
             label="funding (8h)")
    ax2.plot(f["funding_time"], f["roll"], lw=1.3, color="#7f0000",
             label=f"funding {rolling_obs//3}d rolling mean")
    ax2.axhline(threshold, color="#7f0000", ls="--", lw=0.8, alpha=0.7)
    ax2.set_ylabel("Funding rate (per 8h)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Shade contiguous elevated stretches across the full height.
    for lo, hi in _contiguous_true_spans(f["funding_time"], elevated):
        ax1.axvspan(lo, hi, color="#d62728", alpha=0.08, lw=0)

    ax1.set_title(
        "BTC-USD price (log) and Binance funding rate, with "
        f"{rolling_obs//3}-day rolling mean; shaded where rolling mean > {threshold:g}"
    )
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def figure_price_borrow_overlay(
    spot: pd.DataFrame,
    borrow: pd.DataFrame,
    *,
    rate_col: str,
    rolling_days: int,
    start: pd.Timestamp,
    annualise: int = 365,
) -> plt.Figure:
    """Figure 3 — price (log) overlaid with the executed BTC lending-rate proxy.

    The daily BTC margin borrow rate (Bitfinex fBTC) is annualised for
    readability and drawn with its rolling mean. Because borrow history reaches
    back to 2016 this spans both the 2017 and 2021 cycles — unlike the funding
    overlay (Fig 2), which starts at the Sep-2019 contract launch.
    """
    _require(spot, ["date", "close"], name="borrow figure (price)")
    _require(borrow, ["borrow_time", rate_col], name="borrow figure (borrow)")
    if rolling_days < 1:
        raise ValueError(f"rolling_days must be >= 1, got {rolling_days}")
    d = spot[spot["date"] >= start].sort_values("date")
    b = borrow[borrow["borrow_time"] >= start].sort_values("borrow_time").copy()
    b["ann"] = b[rate_col].astype(float) * annualise
    b["roll"] = b["ann"].rolling(window=rolling_days, min_periods=max(rolling_days // 3, 5)).mean()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(d["date"], d["close"], lw=1.0, color="#1f77b4", label="BTC-USD (log)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price (USD, log)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Date")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = ax1.twinx()
    ax2.plot(b["borrow_time"], b["ann"], lw=0.4, color="#2ca02c", alpha=0.35,
             label="borrow rate (annualised)")
    ax2.plot(b["borrow_time"], b["roll"], lw=1.3, color="#1b5e20",
             label=f"borrow {rolling_days}d rolling mean")
    ax2.set_ylabel("BTC lending rate (annualised)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title(
        "BTC-USD price (log) and the executed BTC lending-rate proxy "
        f"(Bitfinex fBTC borrow rate), {rolling_days}-day rolling mean"
    )
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# Phase IV figures — formal detection and its link to the proxies.
# =====================================================================

def _shade_episodes(ax, dates: pd.Series, episodes, *, color="#d62728", alpha=0.14):
    """Shade each detected bubble episode (list of (start_idx, end_idx))."""
    d = dates.reset_index(drop=True)
    for s, e in episodes:
        ax.axvspan(d.iloc[s], d.iloc[min(e, len(d) - 1)], color=color, alpha=alpha, lw=0)


def figure_gsadf_datestamp(
    seq: pd.DataFrame, episodes, *, scalar_cv: float, gsadf_stat: float,
    level: float,
) -> plt.Figure:
    """Figure 4 — GSADF date-stamping.

    Top: log price with detected episodes shaded. Bottom: the BSADF statistic
    against its critical-value sequence, with the same episodes shaded — the
    explosive-root evidence that date-stamps each bubble.
    """
    _require(seq, ["date", "close", "bsadf", "bsadf_cv"], name="figure_gsadf_datestamp")
    d = seq.sort_values("date").reset_index(drop=True)
    pct = int(round(level * 100))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3])
    )
    ax1.plot(d["date"], d["close"], lw=1.0, color="#1f77b4")
    ax1.set_yscale("log")
    ax1.set_ylabel("Price (USD, log)")
    ax1.set_title(
        f"GSADF explosive-root date-stamping  (GSADF={gsadf_stat:.2f} vs "
        f"{pct}% critical value {scalar_cv:.2f})"
    )
    _shade_episodes(ax1, d["date"], episodes)

    ax2.plot(d["date"], d["bsadf"], lw=1.0, color="#7f0000", label="BSADF statistic")
    ax2.plot(d["date"], d["bsadf_cv"], lw=1.0, ls="--", color="0.4",
             label=f"{pct}% critical-value sequence")
    ax2.set_ylabel("Backward SADF")
    ax2.set_xlabel("Date")
    _shade_episodes(ax2, d["date"], episodes)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def figure_bubbles_vs_proxies(
    seq: pd.DataFrame, episodes, borrow: pd.DataFrame, funding: pd.DataFrame,
    *, rate_col_borrow: str, rate_col_funding: str, rolling_days: int,
    borrow_start: pd.Timestamp, funding_start: pd.Timestamp,
) -> plt.Figure:
    """Figure 5 — detected episodes against both short-friction proxies.

    Two stacked panels sharing the price axis make the *era handoff* explicit:
    the executed lending rate (2016+, top) provides observations around the
    2017 episode, while perpetual funding (2019+, bottom) is the live signal
    around 2021. Detected GSADF episodes are shaded on both.
    """
    _require(seq, ["date", "close"], name="figure_bubbles_vs_proxies (price)")
    d = seq.sort_values("date").reset_index(drop=True)

    fig, (axb, axf) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    # --- top: executed lending-rate proxy (2016+) ---
    axb.plot(d["date"], d["close"], lw=1.0, color="#1f77b4", label="BTC-USD (log)")
    axb.set_yscale("log")
    axb.set_ylabel("Price (USD, log)", color="#1f77b4")
    axb.tick_params(axis="y", labelcolor="#1f77b4")
    _shade_episodes(axb, d["date"], episodes)
    b = borrow[borrow["borrow_time"] >= borrow_start].sort_values("borrow_time").copy()
    b["roll"] = (b[rate_col_borrow].astype(float) * 365).rolling(
        rolling_days, min_periods=max(rolling_days // 3, 5)).mean()
    axb2 = axb.twinx()
    axb2.plot(b["borrow_time"], b["roll"], lw=1.2, color="#1b5e20",
              label=f"borrow {rolling_days}d mean (ann.)")
    axb2.set_ylabel("Borrow cost (ann.)", color="#2ca02c")
    axb2.tick_params(axis="y", labelcolor="#2ca02c")
    axb.set_title("Explosive episodes and BTC borrowing rates (Bitfinex, 2016 onwards)")

    # --- bottom: funding rate (indirect proxy, 2019+) ---
    axf.plot(d["date"], d["close"], lw=1.0, color="#1f77b4", label="BTC-USD (log)")
    axf.set_yscale("log")
    axf.set_ylabel("Price (USD, log)", color="#1f77b4")
    axf.tick_params(axis="y", labelcolor="#1f77b4")
    axf.set_xlabel("Date")
    _shade_episodes(axf, d["date"], episodes)
    f = funding[funding["funding_time"] >= funding_start].sort_values("funding_time").copy()
    f["roll"] = f[rate_col_funding].rolling(
        rolling_days * 3, min_periods=max(rolling_days, 5)).mean()
    axf2 = axf.twinx()
    axf2.plot(f["funding_time"], f["roll"], lw=1.2, color="#7f0000",
              label=f"funding {rolling_days}d mean")
    axf2.set_ylabel("Funding (per 8h)", color="#d62728")
    axf2.tick_params(axis="y", labelcolor="#d62728")
    axf.set_title("Explosive episodes and perpetual funding rates (Binance, 2019 onwards)")
    axf.xaxis.set_major_locator(mdates.YearLocator())
    axf.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    return fig
