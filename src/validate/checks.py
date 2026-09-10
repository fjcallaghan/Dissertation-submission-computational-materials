"""Coverage and anomaly checks over the raw datasets.

These functions *report only* — they never modify or drop data. Findings feed a
human-readable report; cleaning decisions are a later phase and must be made
with documented justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Funding rates are clamped by the exchange. Binance's BTCUSDT cap is well
# inside +/-0.75% per 8h; anything beyond that is almost certainly bad data.
FUNDING_ABS_SANITY = 0.0075
EIGHT_HOURS = pd.Timedelta(hours=8)

# The margin borrow rate is a per-day fraction. Historically it has stayed well
# under 1%/day even during squeezes (largest observed ~0.7%/day); anything past
# this bound is almost certainly bad data. Borrow rates should never be negative
# (you are not paid to borrow), so negatives are flagged too.
BORROW_ABS_SANITY = 0.02
ONE_DAY = pd.Timedelta(days=1)


@dataclass
class CheckResult:
    """Outcome of validating one series."""

    name: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    findings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def report(self) -> str:
        head = f"[{self.name}] {self.rows} rows  {self.first} .. {self.last}"
        stat_lines = [f"    - {k}: {v}" for k, v in self.stats.items()]
        if self.ok:
            body = ["    OK — no anomalies flagged."]
        else:
            body = [f"    ! {f}" for f in self.findings]
        return "\n".join([head, *stat_lines, *body])


def _basic_ts_checks(df: pd.DataFrame, time_col: str, findings: list[str]) -> None:
    ts = df[time_col]
    n_dupe = int(ts.duplicated().sum())
    if n_dupe:
        findings.append(f"{n_dupe} duplicate timestamps")
    if not ts.is_monotonic_increasing:
        findings.append("timestamps not monotonically increasing")


def check_spot(df: pd.DataFrame, name: str = "spot") -> CheckResult:
    """Daily price series: calendar gaps, duplicates, non-positive prices, NaNs."""
    time_col = "date"
    df = df.sort_values(time_col).reset_index(drop=True)
    res = CheckResult(
        name=name,
        rows=len(df),
        first=df[time_col].min() if len(df) else None,
        last=df[time_col].max() if len(df) else None,
    )
    if df.empty:
        res.findings.append("empty series")
        return res

    _basic_ts_checks(df, time_col, res.findings)

    # Expected daily calendar coverage (crypto trades every day).
    full = pd.date_range(res.first, res.last, freq="D", tz="UTC")
    present = pd.DatetimeIndex(df[time_col].dt.normalize().unique())
    missing = full.difference(present)
    res.stats["expected_days"] = len(full)
    res.stats["missing_days"] = len(missing)
    res.stats["completeness_%"] = round(100 * (1 - len(missing) / len(full)), 3)
    if len(missing):
        sample = ", ".join(d.date().isoformat() for d in missing[:5])
        res.findings.append(f"{len(missing)} missing calendar days (e.g. {sample})")

    n_nan = int(df["close"].isna().sum())
    if n_nan:
        res.findings.append(f"{n_nan} NaN close prices")
    n_nonpos = int((df["close"] <= 0).sum())
    if n_nonpos:
        res.findings.append(f"{n_nonpos} non-positive close prices")

    return res


def check_funding(df: pd.DataFrame, name: str = "funding") -> CheckResult:
    """8-hourly funding series: cadence gaps, duplicates, out-of-band values."""
    time_col = "funding_time"
    df = df.sort_values(time_col).reset_index(drop=True)
    res = CheckResult(
        name=name,
        rows=len(df),
        first=df[time_col].min() if len(df) else None,
        last=df[time_col].max() if len(df) else None,
    )
    if df.empty:
        res.findings.append("empty series")
        return res

    _basic_ts_checks(df, time_col, res.findings)

    # Cadence: consecutive gaps should be ~8h. Flag any gap > one interval.
    deltas = df[time_col].diff().dropna()
    expected = int(round((res.last - res.first) / EIGHT_HOURS)) + 1
    res.stats["expected_8h_obs"] = expected
    res.stats["completeness_%"] = round(100 * len(df) / expected, 3) if expected else None

    big_gaps = deltas[deltas > EIGHT_HOURS + pd.Timedelta(minutes=5)]
    if len(big_gaps):
        worst = big_gaps.max()
        res.stats["max_gap"] = str(worst)
        res.findings.append(
            f"{len(big_gaps)} cadence gaps larger than 8h (worst {worst})"
        )

    n_nan = int(df["funding_rate"].isna().sum())
    if n_nan:
        res.findings.append(f"{n_nan} NaN funding rates")
    out_of_band = df[np.abs(df["funding_rate"]) > FUNDING_ABS_SANITY]
    if len(out_of_band):
        res.stats["max_abs_rate"] = float(df["funding_rate"].abs().max())
        res.findings.append(
            f"{len(out_of_band)} funding rates beyond +/-{FUNDING_ABS_SANITY} "
            "(flagged, not dropped)"
        )

    return res


def check_borrow(df: pd.DataFrame, name: str = "borrow") -> CheckResult:
    """Daily BTC margin borrow rate: calendar gaps, duplicates, negatives, spikes."""
    time_col = "borrow_time"
    df = df.sort_values(time_col).reset_index(drop=True)
    res = CheckResult(
        name=name,
        rows=len(df),
        first=df[time_col].min() if len(df) else None,
        last=df[time_col].max() if len(df) else None,
    )
    if df.empty:
        res.findings.append("empty series")
        return res

    _basic_ts_checks(df, time_col, res.findings)

    # Expected daily calendar coverage.
    full = pd.date_range(res.first.normalize(), res.last.normalize(), freq="D", tz="UTC")
    present = pd.DatetimeIndex(df[time_col].dt.normalize().unique())
    missing = full.difference(present)
    res.stats["expected_days"] = len(full)
    res.stats["missing_days"] = len(missing)
    res.stats["completeness_%"] = round(100 * (1 - len(missing) / len(full)), 3)
    if len(missing):
        sample = ", ".join(d.date().isoformat() for d in missing[:5])
        res.findings.append(f"{len(missing)} missing calendar days (e.g. {sample})")

    n_nan = int(df["borrow_rate"].isna().sum())
    if n_nan:
        res.findings.append(f"{n_nan} NaN borrow rates")
    n_neg = int((df["borrow_rate"] < 0).sum())
    if n_neg:
        res.findings.append(f"{n_neg} negative borrow rates (should be >= 0)")
    out_of_band = df[np.abs(df["borrow_rate"]) > BORROW_ABS_SANITY]
    if len(out_of_band):
        res.stats["max_rate_per_day"] = float(df["borrow_rate"].max())
        res.findings.append(
            f"{len(out_of_band)} borrow rates beyond {BORROW_ABS_SANITY}/day "
            "(flagged, not dropped)"
        )

    return res
