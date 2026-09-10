"""Reconciliation runner — ``python -m src.reconcile``.

Compares Binance funding against Bybit (and Coinalyze's aggregate when the raw
pull exists) on their overlapping days, and writes a short markdown note on the
degree of agreement to ``logs/reconciliation_<date>.md``.
"""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from .. import datasets
from ..align.aggregate import funding_to_daily
from ..config_loader import load_config
from .compare import Agreement, compare_to_reference


def _daily_mean_series(df: pd.DataFrame) -> pd.Series:
    """Date-indexed daily-mean funding (per-8h scale) from an 8-hourly frame."""
    rate_col = "funding_rate_clean" if "funding_rate_clean" in df.columns else "funding_rate"
    daily = funding_to_daily(df, rate_col=rate_col)
    return daily.set_index("date")["funding_mean"]


def _coinalyze_series(df: pd.DataFrame) -> pd.Series:
    """Market-wide daily funding = mean across Coinalyze's exchange-coded symbols."""
    g = df.copy()
    g["date"] = pd.to_datetime(g["funding_time"]).dt.normalize()
    return g.groupby("date")["funding_rate"].mean()


def _render(agreements: list[Agreement], notes: list[str], stamp: dt.datetime) -> str:
    lines = [
        "# Cross-exchange funding reconciliation",
        "",
        f"_Generated {stamp.isoformat()}_",
        "",
        "Robustness check that the Binance BTCUSDT funding signal is a "
        "market-wide phenomenon rather than a single-venue artefact. "
        "All series are compared on the per-8h daily-mean scale over their "
        "overlapping days.",
        "",
        "| venue (vs Binance) | overlap days | pearson | spearman | mean abs diff | RMSE | sign agree % |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in agreements:
        r = a.as_row()
        lines.append(
            f"| {r['venue']} | {r['overlap_days']} | {r['pearson']} | "
            f"{r['spearman']} | {r['mean_abs_diff']} | {r['rmse']} | "
            f"{r['sign_agreement_%']} |"
        )
    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    binance = datasets.load_processed(cfg, "binance_funding_clean.parquet", required=False)
    if binance is None:
        binance = datasets.load_binance_funding(cfg)
    ref = _daily_mean_series(binance)

    agreements: list[Agreement] = []
    notes: list[str] = []

    bybit = datasets.load_processed(cfg, "bybit_funding_clean.parquet", required=False)
    if bybit is None:
        bybit = datasets.load_bybit_funding(cfg)
    if bybit is not None:
        agreements.append(compare_to_reference(
            ref, _daily_mean_series(bybit), venue="Bybit", reference="Binance"))
        notes.append("Bybit is an independent, FCA-registered venue; strong agreement "
                     "is the core evidence the signal is not Binance-specific.")
    else:
        notes.append("Bybit series not found — run `python -m src.acquire`.")

    coinalyze = datasets.load_coinalyze_funding(cfg)
    if coinalyze is not None and not coinalyze.empty:
        agreements.append(compare_to_reference(
            ref, _coinalyze_series(coinalyze), venue="Coinalyze (aggregate)",
            reference="Binance"))
        notes.append("Coinalyze provides a market-wide aggregate across venues.")
    else:
        notes.append("Coinalyze aggregate unavailable (no cached pull — the host was "
                     "unreachable from this network). The fetcher is wired and will "
                     "populate this row once run from a network where "
                     "api.coinalyze.com is reachable.")

    stamp = dt.datetime.now(dt.timezone.utc)
    report = _render(agreements, notes, stamp)

    log_dir = cfg._resolve_dir("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"reconciliation_{stamp.date().isoformat()}.md"
    out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
