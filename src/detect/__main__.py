"""Phase IV detection runner — ``python -m src.detect``.

Pipeline:
  1. Synthetic validation gate — GSADF must flag the explosive / Evans controls
     and not the random-walk null as an implementation smoke check, not a size/power certificate.
  2. GSADF explosive-root test + finite-sample critical values on log BTC price
     (2014+), then PSY date-stamping of bubble episodes.
  3. Strict-local-martingale volatility diagnostic (secondary/exploratory).
  4. Linking: relate detected episodes to the funding (2019+) and borrow (2016+)
     short-friction proxies — regime comparison, lead-lag, and retrospective
     probit that controls for momentum/volatility.
  5. Writes a markdown report to logs/, the BSADF sequence and episode table to
     data/processed/, and Figures 4 and 5.

``--quick`` shrinks the critical-value simulation for a fast smoke test (the
statistics are unchanged; only the CV precision drops).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys

import numpy as np
import pandas as pd

from .. import datasets
from ..config_loader import PROJECT_ROOT, load_config
from ..plots import figures
from ..synthetic.paths import (
    simulate_evans_bubble,
    simulate_explosive_ar1,
    simulate_random_walk,
)
from . import link as linkmod
from .critvals import simulate_critical_values
from .psy import default_min_duration, default_r0, recursive_adf, date_stamp
from .slm import slm_diagnostic


def _fmt_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional ``tabulate`` dependency."""
    try:
        return df.to_markdown(index=False)
    except Exception:  # tabulate missing → simple pipe table
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in df.itertuples(index=False)]
        return "\n".join([head, sep, *body])


def synthetic_validation(cache_dir, *, p: int, n_sim: int, seed: int) -> tuple[list[str], bool]:
    """The gate: GSADF must separate explosive/Evans controls from the RW null."""
    n = 601  # 600 increments include the initial observation
    r0 = default_r0(n)
    rw = simulate_random_walk(n_steps=n-1, n_paths=1, seed=seed).S[0]
    ex = simulate_explosive_ar1(phi=1.03, n_steps=n-1, n_paths=1, seed=seed).S[0]
    ev = simulate_evans_bubble(n_steps=n-1, n_paths=1, seed=seed).S[0]
    cv = simulate_critical_values(stat="gsadf", method="mc", r0=r0, p=p,
                                  n_sim=n_sim, seed=seed + 1, sim_size=n,
                                  cache_dir=cache_dir)
    cv95 = cv.scalar_cv(0.95)
    lines = [f"Synthetic validation gate (GSADF, 95% CV = {cv95:.2f} at T={n}):"]
    results = {}
    for name, y in (("random_walk (null)", rw), ("explosive_ar1", ex),
                    ("evans_bubble", ev)):
        g = recursive_adf(y, p=p, r0=r0).gsadf_stat
        flag = g > cv95
        results[name] = flag
        lines.append(f"  {name:<20} GSADF={g:6.2f}  {'FLAGGED' if flag else 'not flagged'}")
    ok = (not results["random_walk (null)"]) and results["explosive_ar1"] and results["evans_bubble"]
    lines.append(f"  => gate {'PASSED' if ok else 'FAILED'} "
                 "(null must be unflagged; explosive + Evans must be flagged)")
    return lines, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.detect")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test: small critical-value simulation")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    cfg = load_config()
    cache_dir = cfg.processed_dir / "detect_cache"

    p = cfg.detect_adf_lag
    level = cfg.detect_significance
    n_sim = 99 if args.quick else cfg.cv_n_sim
    sim_size = 400 if args.quick else cfg.cv_sim_size
    method = cfg.cv_method
    seed = cfg.cv_seed

    report: list[str] = [f"# Phase IV — Bubble Detection Report ({dt.date.today()})", ""]

    # ---- 1. synthetic validation gate ----
    gate_lines, gate_ok = synthetic_validation(cache_dir, p=p, n_sim=min(n_sim, 199), seed=seed)
    print("\n".join(gate_lines))
    report += ["## 1. Synthetic validation gate", "", "```", *gate_lines, "```", ""]
    if not gate_ok:
        print("\nABORT: synthetic validation gate failed — not running on BTC.")
        return 1

    # ---- 2. GSADF on log BTC price ----
    full = datasets.load_processed(cfg, "aligned_daily_full.parquet")
    datasets.ensure_columns(datasets.ensure_nonempty(full, source="detect"),
                            ["date", "close", "return"], source="detect")
    full = full.sort_values("date").reset_index(drop=True)
    price = full["close"].to_numpy(dtype=float)
    if (price <= 0).any():
        raise ValueError("non-positive prices; cannot take log")
    y = np.log(price) if cfg.detect_use_log_price else price
    n = y.shape[0]
    r0 = cfg.detect_min_window_frac or default_r0(n)

    print(f"\nGSADF on {'log ' if cfg.detect_use_log_price else ''}price: "
          f"n={n}, p={p}, r0={r0:.4f}, level={level}")
    r = recursive_adf(y, p=p, r0=r0)
    cvs = simulate_critical_values(
        stat="gsadf", method=method, r0=r0, p=p, n_sim=n_sim, seed=seed,
        sim_size=sim_size, real_series=(y if method == "wild" else None),
        cache_dir=cache_dir,
    )
    scalar_cv = cvs.scalar_cv(level)
    reject = r.gsadf_stat > scalar_cv
    cv_seq = cvs.bsadf_cv_sequence(n, level)
    min_dur = cfg.date_stamp_min_duration or default_min_duration(n)
    episodes = date_stamp(r.bsadf, cv_seq, min_duration=min_dur)

    dates = full["date"]
    ep_rows = []
    for s, e in episodes:
        seg = full.iloc[s:e + 1]
        i_pk = int(seg["close"].to_numpy().argmax())
        ep_rows.append({
            "start_date": _fmt_date(dates.iloc[s]),
            "end_date": _fmt_date(dates.iloc[e]),
            "n_days": e - s + 1,
            "peak_date": _fmt_date(seg["date"].iloc[i_pk]),
            "peak_price": float(seg["close"].iloc[i_pk]),
        })
    ep_df = pd.DataFrame(ep_rows, columns=[
        "start_date", "end_date", "n_days", "peak_date", "peak_price",
    ])

    stat_lines = [
        f"GSADF statistic     : {r.gsadf_stat:.3f}",
        f"SADF statistic      : {r.sadf_stat:.3f}",
        f"{int(level*100)}% critical value : {scalar_cv:.3f}  ({method}, "
        f"n_sim={n_sim}, sim_size={sim_size if method=='mc' else n})",
        f"Decision            : {'REJECT unit root — explosive dynamics present' if reject else 'no rejection'}",
        f"Episodes (>= {min_dur} days): {len(episodes)}",
    ]
    print("\n".join(["", *stat_lines]))
    report += ["## 2. GSADF explosive-root test (log BTC price, 2014+)", "",
               "```", *stat_lines, "```", ""]
    if not ep_df.empty:
        report += ["Date-stamped episodes:", "",
                   _df_to_markdown(ep_df), ""]
        print(ep_df.to_string(index=False))

    # in-episode indicator
    in_ep = np.zeros(n, dtype=int)
    for s, e in episodes:
        in_ep[s:e + 1] = 1

    # ---- 3. strict-local-martingale diagnostic ----
    report += ["## 3. Strict-local-martingale volatility diagnostic (exploratory)", ""]
    if cfg.slm_enabled:
        slm = slm_diagnostic(price, dt=1.0, label="BTC", tail_q=cfg.slm_tail_quantile)
        slm_lines = [
            f"volatility elasticity rho_hat = {slm['rho_hat']:.3f} "
            f"(descriptive grid SE {slm['rho_grid_se']:.3f})",
            f"verdict: {slm['verdict']}",
            "No martingale test: the fitted range does not establish an asymptotic tail law.",
        ]
        print("\nSLM: " + slm_lines[0] + " | " + slm['verdict'])
        report += ["```", *slm_lines, "```", ""]
    else:
        report += ["_disabled in config._", ""]

    # ---- 4. linking to the short-friction proxies ----
    report += ["## 4. Linking detected episodes to the short-friction proxies", ""]
    link_df = full[["date", "return", "funding_daily", "borrow_daily"]].copy()
    link_df["bsadf"] = r.bsadf
    link_df["in_episode"] = in_ep
    link_df["realized_vol"] = linkmod.realized_volatility(
        link_df["return"], window=cfg.link_rvol_window)
    link_df = link_df.set_index("date")

    controls = ["return", "realized_vol"]
    for proxy, label, start_col in (("borrow_daily", "Executed BTC lending rate (2016+)", "borrow_daily"),
                                    ("funding_daily", "Funding rate (indirect, 2019+)", "funding_daily")):
        sub = linkmod.daily_frame(link_df.copy())
        sub[proxy] *= 10000  # coefficients and regime means per basis point/day
        report += [f"### {label}", ""]
        if sub[proxy].notna().sum() < 100 or sub["in_episode"].nunique() < 2:
            report += ["_insufficient data or no regime variation in this window._", ""]
            continue
        reg = linkmod.regime_comparison(sub, proxy_col=proxy, regime_col="in_episode")
        ll = linkmod.lead_lag_correlation(sub, proxy_col=proxy, stat_col="bsadf",
                                          max_lag=cfg.link_max_lag)
        peak = ll.loc[ll["corr"].idxmax()]
        pred = linkmod.predictive_regression(
            sub, regime_col="in_episode", proxy_col=proxy, control_cols=controls,
            lag=cfg.link_predict_lag, model=cfg.link_model,
            hac_lags=int(cfg.link_cfg.get("hac_lags", 30)))

        lines = [
            f"Regime means: inside={reg['mean_inside']:.4g}  outside={reg['mean_outside']:.4g}  "
            f"diff={reg['mean_diff']:.4g} basis points/day "
            f"(n_in={reg['n_inside']}, n_out={reg['n_outside']})",
            f"Lead-lag: peak Spearman corr {peak['corr']:.3f} at lag {int(peak['lag'])} "
            "(lag>0 => proxy leads the BSADF statistic)",
        ]
        if "error" in pred:
            lines.append(f"Retrospective {cfg.link_model}: {pred['error']}")
        else:
            lines += [
                f"Retrospective {pred['model']} of in-episode on lagged proxy + {controls}:",
                f"  proxy coef per basis point={pred['coef_proxy']:.4f} "
                f"(HAC {pred['hac_lags']} calendar lags, p={pred['p_proxy']:.3g})",
                f"  pseudo-R2 {pred['pseudo_r2_restricted']:.3f} -> {pred['pseudo_r2_full']:.3f} "
                f"(descriptive fit only), n={pred['n']}",
            ]
        print("\n[" + label + "]\n  " + "\n  ".join(lines))
        report += ["```", *lines, "```", ""]

    # ---- 5. persist outputs + figures ----
    seq_df = pd.DataFrame({
        "date": dates, "close": price, "bsadf": r.bsadf,
        "bsadf_cv": cv_seq, "in_episode": in_ep,
    })
    datasets.save_processed(seq_df, cfg, "gsadf_sequence.parquet")
    # Replace a previous run's episodes even when this run finds none.
    datasets.save_processed(ep_df, cfg, "bubble_episodes.parquet")
    # Small sidecar so `python -m src.plots` can redraw fig4 with the right title.
    (cfg.processed_dir / "detect_meta.json").write_text(json.dumps({
        "gsadf_stat": r.gsadf_stat, "sadf_stat": r.sadf_stat,
        "scalar_cv": scalar_cv, "level": level, "reject": bool(reject),
        "n_episodes": len(episodes), "method": method, "calibration": cvs.meta,
        "quick": args.quick, "min_duration": min_dur,
    }, indent=2), encoding="utf-8")

    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.raw.get("plots", {}).get("dpi", 200))

    f4 = figures.figure_gsadf_datestamp(
        seq_df, episodes, scalar_cv=scalar_cv, gsadf_stat=r.gsadf_stat, level=level)
    f4.savefig(fig_dir / "fig4_gsadf_datestamp.png", dpi=dpi)

    # Figure 5 needs the raw proxy time series (with their own time columns).
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
    wrote_f5 = False
    if borrow is not None and funding is not None:
        f5 = figures.figure_bubbles_vs_proxies(
            seq_df, episodes, borrow, funding,
            rate_col_borrow=b_col, rate_col_funding=f_col,
            rolling_days=cfg.rolling_window_days,
            borrow_start=pd.Timestamp(cfg.borrow_start, tz="UTC"),
            funding_start=pd.Timestamp(cfg.window.funding_start, tz="UTC"),
        )
        f5.savefig(fig_dir / "fig5_bubbles_vs_proxies.png", dpi=dpi)
        wrote_f5 = True

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / f"detect_{dt.date.today()}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"\nWrote: {report_path}")
    print(f"       {cfg.processed_dir/'gsadf_sequence.parquet'}")
    print(f"       {fig_dir/'fig4_gsadf_datestamp.png'}"
          + (f"\n       {fig_dir/'fig5_bubbles_vs_proxies.png'}" if wrote_f5 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
