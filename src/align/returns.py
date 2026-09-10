"""Returns calculation (documented preprocessing step)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(price: pd.Series, method: str = "log") -> pd.Series:
    """Period-over-period returns of a price series.

    "log": ln(P_t) - ln(P_{t-1});  "simple": P_t / P_{t-1} - 1.
    The first observation is NaN by construction.
    """
    price = price.astype(float)
    if method == "log":
        return np.log(price) - np.log(price.shift(1))
    if method == "simple":
        return price.pct_change()
    raise ValueError(f"unknown returns method: {method!r}")
