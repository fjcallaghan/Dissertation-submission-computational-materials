"""Price-series gap-filling across weekends and holidays.

BTC trades every day, but the spot feed can still miss the odd calendar day.
This reindexes the daily price onto a gap-free calendar and fills according to
``cleaning.price_gap_fill`` ("ffill" | "interpolate" | "none"), flagging every
filled row so the treatment is transparent and never silent.
"""

from __future__ import annotations

import pandas as pd

_PRICE_COLS = ["open", "high", "low", "close", "volume"]


def gapfill_price(
    df: pd.DataFrame,
    *,
    method: str = "ffill",
    time_col: str = "date",
) -> pd.DataFrame:
    """Reindex to a full daily UTC calendar and fill gaps by ``method``.

    Adds a boolean ``was_filled`` column marking synthetic rows.
    """
    if method not in ("ffill", "interpolate", "none"):
        raise ValueError(f"unknown price_gap_fill method: {method!r}")
    for col in (time_col, "close"):
        if col not in df.columns:
            raise ValueError(f"gapfill_price: missing column {col!r}")
    out = df.sort_values(time_col).reset_index(drop=True).copy()
    if out.empty:
        out["was_filled"] = pd.Series(dtype=bool)
        return out

    full = pd.date_range(out[time_col].min(), out[time_col].max(), freq="D", tz="UTC")
    out = out.set_index(time_col).reindex(full)
    out.index.name = time_col
    was_filled = out["close"].isna()

    cols = [c for c in _PRICE_COLS if c in out.columns]
    if method == "ffill":
        out[cols] = out[cols].ffill()
    elif method == "interpolate":
        out[cols] = out[cols].interpolate(method="time")
    elif method == "none":
        pass
    else:
        raise ValueError(f"unknown price_gap_fill method: {method!r}")

    out["was_filled"] = was_filled.to_numpy()
    return out.reset_index()
