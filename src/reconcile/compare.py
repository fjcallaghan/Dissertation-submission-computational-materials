"""Pairwise agreement statistics between venue funding series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Agreement:
    """Agreement of one venue against the reference (Binance)."""

    venue: str
    reference: str
    overlap_days: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    pearson: float
    spearman: float
    mean_abs_diff: float
    rmse: float
    sign_agreement: float  # fraction of days both series share the funding sign

    def as_row(self) -> dict[str, object]:
        return {
            "venue": self.venue,
            "overlap_days": self.overlap_days,
            "pearson": round(self.pearson, 4),
            "spearman": round(self.spearman, 4),
            "mean_abs_diff": f"{self.mean_abs_diff:.3e}",
            "rmse": f"{self.rmse:.3e}",
            "sign_agreement_%": round(100 * self.sign_agreement, 1),
        }


def compare_to_reference(
    ref: pd.Series, other: pd.Series, *, venue: str, reference: str
) -> Agreement:
    """Compute agreement stats on the inner-joined overlap of two daily series.

    Both inputs are date-indexed daily funding (per-8h scale, i.e. daily mean).
    """
    joined = pd.concat({"ref": ref, "other": other}, axis=1).dropna()
    n = len(joined)
    if n < 3:
        return Agreement(venue, reference, n, None, None,
                         np.nan, np.nan, np.nan, np.nan, np.nan)
    a, b = joined["ref"], joined["other"]
    diff = a - b
    sign_agree = float((np.sign(a) == np.sign(b)).mean())
    return Agreement(
        venue=venue,
        reference=reference,
        overlap_days=n,
        first=joined.index.min(),
        last=joined.index.max(),
        pearson=float(a.corr(b, method="pearson")),
        spearman=float(a.corr(b, method="spearman")),
        mean_abs_diff=float(diff.abs().mean()),
        rmse=float(np.sqrt((diff ** 2).mean())),
        sign_agreement=sign_agree,
    )
