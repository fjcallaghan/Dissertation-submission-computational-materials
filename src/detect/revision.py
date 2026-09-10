"""Reproduce the corrected chapter from cached observations and audited draws.

Run ``python -m src.detect.revision``. The saved 199-draw bootstrap is imported
only after input/configuration and sample-replay checks. No live data is fetched.
"""
from pathlib import Path
import hashlib
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .critvals import CriticalValues, SIGNIFICANCE, cache_metadata, _cache_path, _save
from .psy import recursive_adf, default_r0, date_stamp, resolve_min_obs
from .link import daily_frame, lead_lag_correlation, predictive_regression, regime_comparison, realized_volatility

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/reviews/verification"
OUT = ROOT / "docs/implementation/ch5"
THESIS = ROOT / "thesis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_verified_bootstrap(y):
    baseline = np.load(AUDIT / "baseline.npz")
    draws = np.load(AUDIT / "fresh_wild_199.npz")
    summary = json.loads((AUDIT / "wild_summary.json").read_text())
    assert np.array_equal(y, baseline["y"]), "Audited bootstrap is for different price data"
    assert summary["n_sim"] == 199 and summary["seed"] == 12345
    assert summary["n"] == len(y) and summary["r0"] == default_r0(len(y))
    assert draws["bsadf"].shape == (199, len(y)) and draws["scalars"].shape == (199,)
    # Replay the first and final draws through the current statistic code.
    dif = np.diff(y)
    dif -= dif.mean()
    rng = np.random.default_rng(12345)
    for i in range(199):
        simulated = np.r_[0., np.cumsum(rng.choice((-1., 1.), len(dif))*dif)]
        if i in (0, 198):
            result = recursive_adf(simulated, p=1)
            assert np.allclose(result.bsadf, draws["bsadf"][i], equal_nan=True, atol=1e-10, rtol=0)
            assert abs(result.gsadf_stat-draws["scalars"][i]) < 1e-10
    identity = cache_metadata("gsadf", "wild", len(y), default_r0(len(y)), 1, 199, 12345, SIGNIFICANCE, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        curves = {level: np.nanquantile(draws["bsadf"], level, axis=0) for level in SIGNIFICANCE}
    scalars = {level: float(np.quantile(draws["scalars"], level)) for level in SIGNIFICANCE}
    assert abs(scalars[.95]-summary["cv95"]) < 1e-12
    metadata = {**identity, "min_obs": resolve_min_obs(len(y), r0=default_r0(len(y)), min_obs=None, k=3),
                "imported_draws_sha256": sha(AUDIT / "fresh_wild_199.npz"),
                "replayed_draw_indices": [0, 198]}
    cv = CriticalValues("gsadf", "wild", scalars, (np.arange(len(y))+1)/len(y), curves, metadata)
    folder = ROOT / "data/processed/detect_cache"
    folder.mkdir(exist_ok=True)
    _save(_cache_path(folder, identity), cv)
    return baseline, cv


def sensitivity_analysis(df, bsadf, schemes):
    df = daily_frame(df[["close", "return", "borrow_daily", "funding_daily"]])
    df["bsadf"] = bsadf
    df["realized_vol"] = realized_volatility(df["return"])
    common = max(df[p].first_valid_index() for p in ("borrow_daily", "funding_daily"))
    results, curves = [], []
    for scheme, cv in schemes.items():
        df["in_episode"] = 0
        episodes = date_stamp(bsadf, cv, min_duration=8)
        for a, b in episodes:
            df.iloc[a:b+1, df.columns.get_loc("in_episode")] = 1
        for window in ("all", "common_2019", "exclude_2017"):
            sub = df.copy()
            if window == "common_2019":
                sub = sub.loc[common:]
            elif window == "exclude_2017":
                sub.loc[sub.index.year == 2017, :] = np.nan
            for proxy in ("borrow_daily", "funding_daily"):
                scaled = sub.copy()
                scaled[proxy] *= 10000
                ll = lead_lag_correlation(scaled, proxy_col=proxy, stat_col="bsadf")
                if scheme == "wild":
                    curves.append(ll.assign(proxy=proxy, window=window))
                peak = ll.loc[ll["corr"].idxmax()]
                record = {"scheme": scheme, "window": window, "proxy": proxy,
                          "means_basis_points_day": regime_comparison(scaled, proxy_col=proxy, regime_col="in_episode"),
                          "peak_lag": int(peak["lag"]), "peak_corr": float(peak["corr"]),
                          "zero_lag_corr": float(ll.loc[ll.lag == 0, "corr"].iloc[0]), "fits": {}}
                for covariance, lag in (("HC1", 0), ("HAC", 7), ("HAC", 30), ("HAC", 60)):
                    fit = predictive_regression(scaled, regime_col="in_episode", proxy_col=proxy,
                                                control_cols=["return", "realized_vol"],
                                                covariance=covariance, hac_lags=lag)
                    assert "error" not in fit, fit
                    record["fits"]["HC1" if covariance == "HC1" else str(lag)] = fit
                results.append(record)
    return results, pd.concat(curves, ignore_index=True)


def write_lag_figure(curves):
    """Render the saved lag comparisons, identifying coincident funding curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    labels = {"all": "Full window", "common_2019": "Common window (from September 2019)", "exclude_2017": "Excluding 2017"}
    styles = {"all": "-", "common_2019": "--", "exclude_2017": ":"}
    for ax, proxy in zip(axes, ("borrow_daily", "funding_daily")):
        for window, label in labels.items():
            sub = curves[(curves.proxy == proxy) & (curves.window == window)]
            ax.plot(sub.lag, sub["corr"], label=label, linewidth=1.6,
                    linestyle=styles[window])
        ax.axvline(0, color="grey", linewidth=.7)
        ax.axhline(0, color="grey", linewidth=.7)
        ax.set_title("BTC borrowing rate" if proxy == "borrow_daily" else "Perpetual funding rate")
        ax.set_xlabel("Calendar lag (days, positive = proxy leads)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Spearman correlation with BSADF")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=9)
    axes[1].text(.5, .08, "Full = excluding 2017\nCommon window nearly overlaps",
                 transform=axes[1].transAxes, ha="center", va="bottom", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    fig.tight_layout(rect=(0, .1, 1, 1), w_pad=1.2)
    fig.savefig(THESIS / "fig_ch5_lags.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_exhibits(results, curves):
    write_lag_figure(curves)
    labels = {"all": "Own full window", "common_2019": "Common window from September 2019", "exclude_2017": "Excluding calendar year 2017"}

    def select(scheme, window, proxy):
        return next(r for r in results if (r["scheme"], r["window"], r["proxy"]) == (scheme, window, proxy))

    rows = []
    for proxy, label in (("borrow_daily", "Borrow"), ("funding_daily", "Funding")):
        r = select("wild", "all", proxy)
        m, f = r["means_basis_points_day"], r["fits"]["30"]
        pvalue = f"{f['p_proxy']:.4f}" if f["p_proxy"] >= .0001 else r"$<0.0001$"
        rows.append(f"{label} & {m['mean_inside']:.3f} & {m['mean_outside']:.3f} & {f['coef_proxy']:.4f} & [{f['ci95_proxy'][0]:.4f}, {f['ci95_proxy'][1]:.4f}] & {pvalue} & {f['n']:,} " + r"\\")
    (THESIS / "ch5_association_rows.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for scheme in ("mc", "wild"):
        for window in labels:
            r = select(scheme, window, "borrow_daily")
            name = {"all": "Full", "common_2019": "Common from 2019", "exclude_2017": "Excluding 2017"}[window]
            f = r["fits"]
            rows.append(f"{scheme.upper()} & {name} & {r['peak_lag']:+d} & {r['peak_corr']:.3f} & {f['HC1']['p_proxy']:.4f} & {f['30']['p_proxy']:.4f} & {f['60']['p_proxy']:.4f} " + r"\\")
    (THESIS / "ch5_sensitivity_rows.tex").write_text("\n".join(rows) + "\n")
    eps = pd.read_parquet(ROOT / "data/processed/bubble_episodes.parquet")
    rows = [f"{r.start_date} & {r.end_date} & {r.n_days} & {r.peak_date} & {r.peak_price:,.0f} " + r"\\" for r in eps.itertuples()]
    (THESIS / "ch5_episode_rows.tex").write_text("\n".join(rows) + "\n")


def write_option_exhibit():
    """Readable thesis exhibit of the saved laws and inner/outer ranges."""
    folder = ROOT / "pilot/option_reliability/outputs/initial_20260907T041046748784Z/2026-09-25"
    laws = [json.loads((folder / name).read_text()) for name in ("witness_A_0.json", "witness_A_4.json")]
    result = json.loads((folder / "analysis.json").read_text())
    fig, (ax, bound_ax) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    maximum = max(max(w["support"]) for w in laws) * 1.03
    for w, color, style in zip(laws, ("#17648b", "#cf7831"), ("-", "--")):
        order = np.argsort(w["support"])
        x, mass = np.array(w["support"])[order], np.array(w["probabilities"])[order]
        ax.step(np.r_[0., x, maximum], np.r_[0., np.cumsum(mass), 1.], where="post",
                label=f"d = {100*w['target_defect']:.4f}%", color=color, linestyle=style, linewidth=2)
    ax.set(xlabel="Normalised terminal index X", ylabel="Cumulative probability", title="Two compatible put-pricing laws", xlim=(0, maximum), ylim=(0, 1.02))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.2)
    positive, upper = 100*laws[1]["target_defect"], 100*result["bounds"]["U_put"]
    bound_ax.barh([1, 0], [positive, upper], color=["#17648b", "#d8e0e5"], height=.32)
    bound_ax.scatter([0, positive], [1, 1], color="#cf7831", s=35, zorder=3)
    bound_ax.text(positive, 1.25, f"{positive:.4f}%", ha="center", fontsize=10)
    bound_ax.text(upper, .25, f"{upper:.4f}%", ha="center", fontsize=10)
    bound_ax.set(yticks=[1, 0], yticklabels=["Constructed\nfeasible range", "Necessary\nouter range"],
                 xlabel="Defect (% of reference index)", title="Feasibility is only partly resolved",
                 xlim=(-.008, upper*1.2), ylim=(-.45, 1.6))
    bound_ax.set_xticks([0, .1, .2])
    bound_ax.grid(axis="x", alpha=.2)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(THESIS / "fig_ch5_option_witness.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/aligned_daily_full.parquet"
    df = pd.read_parquet(source).set_index("date").sort_index()
    baseline, cv = import_verified_bootstrap(np.log(df.close.to_numpy()))
    from .__main__ import main as detect_main
    assert detect_main([]) == 0
    sequence = pd.read_parquet(ROOT / "data/processed/gsadf_sequence.parquet")
    assert np.allclose(sequence.bsadf, baseline["bsadf"], equal_nan=True, rtol=0, atol=1e-10)
    results, curves = sensitivity_analysis(df, sequence.bsadf.to_numpy(), {"mc": baseline["cv"], "wild": cv.bsadf_cv_sequence(len(df))})
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    curves.to_csv(OUT / "lag_curves.csv", index=False)
    write_exhibits(results, curves)
    write_option_exhibit()
    for name in ("fig4_gsadf_datestamp.png", "fig5_bubbles_vs_proxies.png"):
        shutil.copy2(ROOT / "figures" / name, THESIS / name)
    (OUT / "provenance.json").write_text(json.dumps({"aligned_data_sha256": sha(source),
        "wild_calibration": cv.meta, "historical_mc_comparison_sha256": sha(AUDIT / "baseline.npz"),
        "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "interpretation": "Retrospective estimates; HAC sensitivity does not establish prediction"}, indent=2) + "\n")
    print("Corrected chapter exhibits and analysis:", OUT)


if __name__ == "__main__":
    main()
