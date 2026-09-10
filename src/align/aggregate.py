"""Collapse the 8-hourly funding series to a daily frequency.

The aggregation choice matters and is configurable (``funding.daily_aggregation``):
  * "sum"  — total funding paid over the day (3 settlements); the natural
             "daily cost of a short" quantity used for the analysis dataset.
  * "mean" — average per-8h rate; keeps the series on the native per-interval
             scale that the exchange caps and that the overlay figure uses.
  * "none" — keep the 8-hourly granularity (handled by the caller, which aligns
             price forward instead).

We always return both sum and mean so downstream code can pick the correct
scale without re-aggregating; ``funding_daily`` mirrors the configured choice.
"""

from __future__ import annotations

import pandas as pd


def funding_to_daily(
    df: pd.DataFrame,
    *,
    rate_col: str,
    time_col: str = "funding_time",
    prefix: str = "funding",
) -> pd.DataFrame:
    """Group a sub-daily rate frame to daily sums and means (UTC calendar).

    Series-generic: ``prefix`` names the output value columns, so the same
    routine collapses the 8-hourly funding series (``prefix="funding"``) and the
    already-daily borrow-rate series (``prefix="borrow"``, where sum == mean).

    Returns columns: date, ``<prefix>_sum``, ``<prefix>_mean``, n_obs.
    """
    for col in (time_col, rate_col):
        if col not in df.columns:
            raise ValueError(f"funding_to_daily: missing column {col!r}")
    if df.empty:
        raise ValueError("funding_to_daily: empty funding frame")
    g = df[[time_col, rate_col]].copy()
    g["date"] = g[time_col].dt.normalize()
    daily = g.groupby("date")[rate_col].agg(["sum", "mean", "count"])
    daily = daily.rename(
        columns={"sum": f"{prefix}_sum", "mean": f"{prefix}_mean", "count": "n_obs"}
    )
    return daily.reset_index()
