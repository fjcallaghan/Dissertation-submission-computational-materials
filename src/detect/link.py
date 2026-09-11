"""Retrospective associations with explosive episodes, not onset forecasting.

Lags retain calendar positions. HAC covariance measures a limited dependence
sensitivity, not protection against selected lags, endogenous regressors,
generated episode labels or a small effective number of cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import S_hac_simple


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Insert missing calendar days before shifting, without filling values."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Linking requires a daily DatetimeIndex")
    if df.index.has_duplicates or not df.index.equals(df.index.normalize()):
        raise ValueError("Linking requires unique midnight observations")
    return df.sort_index().asfreq("D")


def realized_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Trailing realized volatility (rolling std of returns)."""
    return returns.rolling(window=window, min_periods=max(window // 3, 5)).std()


def regime_comparison(
    df: pd.DataFrame, *, proxy_col: str, regime_col: str
) -> dict:
    """Compare a proxy inside vs outside detected episodes.

    Reports descriptive counts, means and medians. Ordinary daily-observation
    Welch and rank tests are omitted because they ignore serial dependence.
    """
    sub = df[[proxy_col, regime_col]].dropna()
    inside = sub.loc[sub[regime_col].astype(bool), proxy_col].to_numpy()
    outside = sub.loc[~sub[regime_col].astype(bool), proxy_col].to_numpy()
    out: dict = {
        "proxy": proxy_col,
        "n_inside": int(inside.size),
        "n_outside": int(outside.size),
        "mean_inside": float(np.mean(inside)) if inside.size else np.nan,
        "mean_outside": float(np.mean(outside)) if outside.size else np.nan,
        "median_inside": float(np.median(inside)) if inside.size else np.nan,
        "median_outside": float(np.median(outside)) if outside.size else np.nan,
    }
    out["mean_diff"] = out["mean_inside"] - out["mean_outside"]
    return out


def lead_lag_correlation(
    df: pd.DataFrame, *, proxy_col: str, stat_col: str, max_lag: int = 30,
    method: str = "spearman",
) -> pd.DataFrame:
    """Cross-correlation of the proxy against an explosiveness statistic.

    Lag ``L > 0`` correlates ``proxy_{t-L}`` with ``stat_t`` (proxy **leads**);
    ``L < 0`` is the proxy lagging. Returns a tidy frame ``[lag, corr, n]``.
    """
    sub = daily_frame(df[[proxy_col, stat_col]])
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = sub[proxy_col].shift(lag)  # lag>0 → past proxy aligned to now
        pair = pd.concat([shifted, sub[stat_col]], axis=1).dropna()
        if pair.shape[0] > 5:
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
        else:
            c = np.nan
        rows.append({"lag": lag, "corr": c, "n": pair.shape[0]})
    return pd.DataFrame(rows)


def predictive_regression(
    df: pd.DataFrame, *, regime_col: str, proxy_col: str,
    control_cols: list[str], lag: int = 1, model: str = "probit",
    covariance: str = "HAC", hac_lags: int = 30,
) -> dict:
    """Fit a retrospective episode association using lagged covariates.

    The historical function name is retained for callers. HAC uses Bartlett
    score products at calendar lags, with zero score contributions on omitted
    dates and no finite-sample correction. HC1 is a comparison only. The ordinary
    LR p-value is not reported as dependence-robust inference.
    """
    if model not in ("probit", "logit"):
        raise ValueError(f"model must be 'probit' or 'logit', got {model!r}")
    if covariance not in ("HAC", "HC1") or hac_lags < 0 or lag < 0:
        raise ValueError("Use HAC or HC1 with nonnegative lags")
    cols = [proxy_col, *control_cols]
    data = daily_frame(df[[regime_col, *cols]])
    for c in cols:
        data[c] = data[c].shift(lag)      # use only information available at t-lag
    data = data.dropna()
    y = data[regime_col].astype(float).to_numpy()
    if np.unique(y).size < 2:
        return {"error": "regime indicator has no variation in the estimation window",
                "n": int(y.size)}

    Estimator = sm.Probit if model == "probit" else sm.Logit
    X_full = sm.add_constant(data[cols].to_numpy())
    X_rest = sm.add_constant(data[control_cols].to_numpy()) if control_cols \
        else np.ones((data.shape[0], 1))

    full = Estimator(y, X_full).fit(disp=0, maxiter=200)
    rest = Estimator(y, X_rest).fit(disp=0, maxiter=200)
    converged = bool(full.mle_retvals["converged"] and rest.mle_retvals["converged"])
    if not converged:
        return {"error": "association fit did not converge", "n": int(y.size)}

    # proxy is the first regressor after the constant.
    b_idx = 1
    lr = 2.0 * (full.llf - rest.llf)
    scores = full.model.score_obs(full.params)
    bread = np.linalg.inv(full.model.hessian(full.params))
    if covariance == "HAC":
        calendar_scores = pd.DataFrame(scores, index=data.index)
        calendar_scores = calendar_scores.reindex(
            pd.date_range(data.index.min(), data.index.max(), freq="D"), fill_value=0.)
        meat = S_hac_simple(calendar_scores.to_numpy(), nlags=hac_lags)
    else:
        nobs, nparams = X_full.shape
        if nobs <= nparams:
            raise ValueError("HC1 requires more observations than fitted parameters")
        meat = (scores.T @ scores) * (nobs / (nobs - nparams))
    cov = bread @ meat @ bread.T
    se = float(np.sqrt(cov[b_idx, b_idx]))
    z = float(full.params[b_idx] / se)
    return {
        "model": model,
        "proxy": proxy_col,
        "controls": list(control_cols),
        "lag": lag,
        "n": int(y.size),
        "coef_proxy": float(full.params[b_idx]),
        "se_proxy": se,
        "z_proxy": z,
        "p_proxy": float(2 * stats.norm.sf(abs(z))),
        "ci95_proxy": [float(full.params[b_idx]-1.96*se), float(full.params[b_idx]+1.96*se)],
        "covariance": covariance, "hac_lags": hac_lags if covariance == "HAC" else None,
        "converged": converged,
        "pseudo_r2_full": float(full.prsquared),
        "pseudo_r2_restricted": float(rest.prsquared),
        "lr_stat_proxy": float(lr),
        "interpretation": "Retrospective association; no forecast or causal claim",
    }
