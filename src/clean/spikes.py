"""Statistical outlier handling for funding and lending-rate series.

We flag unusually large deviations from a rolling baseline and keep both raw
and winsorised values. These flags do not establish data errors, liquidations,
squeezes or any other economic cause.

Rule (documented and configurable in ``config.yaml``):
    An observation is flagged as a spike when it lies more than
    ``spike_sigma`` standard deviations from a trailing rolling baseline of
    ``spike_baseline_window_days`` days. The cleaned series winsorises flagged
    points to that boundary (mean +/- sigma*std) rather than deleting them, so
    the series keeps its length and cadence for downstream alignment.
"""

from __future__ import annotations

import pandas as pd

# Funding settles every 8 hours -> 3 observations per day.
OBS_PER_DAY = 3


def flag_funding_spikes(
    df: pd.DataFrame,
    *,
    sigma: float,
    baseline_window_days: int,
    rate_col: str = "funding_rate",
    time_col: str = "funding_time",
    obs_per_day: int = OBS_PER_DAY,
) -> pd.DataFrame:
    """Return ``df`` with ``is_spike`` and ``<rate_col>_clean`` columns added.

    The raw ``rate_col`` is left untouched (keep_raw_alongside_cleaned). The rule
    is series-generic: ``obs_per_day`` sets the observation cadence so the same
    logic serves the 8-hourly funding series (3/day) and the daily borrow-rate
    series (1/day).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if baseline_window_days <= 0:
        raise ValueError(f"baseline_window_days must be positive, got {baseline_window_days}")
    if obs_per_day <= 0:
        raise ValueError(f"obs_per_day must be positive, got {obs_per_day}")
    missing = [c for c in (rate_col, time_col) if c not in df.columns]
    if missing:
        raise ValueError(f"flag_funding_spikes: missing column(s) {missing}")
    if df.empty:
        raise ValueError("flag_funding_spikes: empty funding frame")
    out = df.sort_values(time_col).reset_index(drop=True).copy()
    s = out[rate_col].astype(float)

    window = max(baseline_window_days * obs_per_day, 3)
    min_periods = max(window // 3, 5)
    roll = s.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std()

    lower = mu - sigma * sd
    upper = mu + sigma * sd
    # Only flag where a baseline scale exists (sd defined and > 0).
    valid = sd.notna() & (sd > 0)
    is_spike = valid & ((s < lower) | (s > upper))

    clean = s.copy()
    clean[is_spike] = s[is_spike].clip(lower=lower[is_spike], upper=upper[is_spike])

    out["is_spike"] = is_spike.to_numpy()
    out[f"{rate_col}_clean"] = clean.to_numpy()
    return out
