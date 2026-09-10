"""Linking analysis: recovers a planted predictive relationship, and finds none
when the proxy is unrelated to the regime."""

import numpy as np
import pandas as pd

from src.detect.link import (
    lead_lag_correlation,
    predictive_regression,
    realized_volatility,
    regime_comparison,
)


def _frame(seed, planted):
    rng = np.random.default_rng(seed)
    n = 1200
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    proxy = pd.Series(rng.standard_normal(n), index=idx).rolling(3, min_periods=1).mean()
    ret = pd.Series(0.01 * rng.standard_normal(n), index=idx)
    rvol = realized_volatility(ret, window=20)
    if planted:
        latent = 2.0 * proxy.shift(1) + 0.5 * rng.standard_normal(n)
    else:
        latent = pd.Series(rng.standard_normal(n), index=idx)  # unrelated to proxy
    regime = (latent > latent.quantile(0.7)).astype(int)
    return pd.DataFrame({"proxy": proxy, "return": ret, "rvol": rvol,
                         "regime": regime, "bsadf": latent})


def test_predictive_regression_recovers_planted_signal():
    df = _frame(seed=1, planted=True)
    res = predictive_regression(df, regime_col="regime", proxy_col="proxy",
                                control_cols=["return", "rvol"], lag=1, model="probit")
    assert res["coef_proxy"] > 0
    assert res["p_proxy"] < 0.01
    assert res["covariance"] == "HAC"
    assert "lr_p_proxy" not in res
    assert res["pseudo_r2_full"] > res["pseudo_r2_restricted"]


def test_predictive_regression_null_is_insignificant():
    df = _frame(seed=2, planted=False)
    res = predictive_regression(df, regime_col="regime", proxy_col="proxy",
                                control_cols=["return", "rvol"], lag=1, model="probit")
    assert res["p_proxy"] > 0.05


def test_regime_comparison_detects_higher_proxy_inside():
    df = _frame(seed=3, planted=True)
    # Align regime to the contemporaneous proxy driver for the descriptive split.
    df = df.assign(proxy_lag=df["proxy"].shift(1))
    out = regime_comparison(df, proxy_col="proxy_lag", regime_col="regime")
    assert out["mean_inside"] > out["mean_outside"]
    assert "welch_p" not in out


def test_lead_lag_shape_and_peak():
    df = _frame(seed=4, planted=True)
    ll = lead_lag_correlation(df, proxy_col="proxy", stat_col="bsadf", max_lag=10)
    assert list(ll["lag"]) == list(range(-10, 11))
    # proxy leads the latent stat by one period → peak positive corr at lag +1.
    peak_lag = int(ll.loc[ll["corr"].idxmax(), "lag"])
    assert peak_lag == 1


def test_logit_alternative_runs():
    df = _frame(seed=5, planted=True)
    res = predictive_regression(df, regime_col="regime", proxy_col="proxy",
                                control_cols=["return"], lag=1, model="logit")
    assert res["model"] == "logit"
    assert res["p_proxy"] < 0.05


def test_calendar_gaps_do_not_compress_lags():
    df = _frame(seed=9, planted=True)
    sparse = df.drop(df.index[100:110])
    explicit = sparse.reindex(df.index)
    a = lead_lag_correlation(sparse, proxy_col="proxy", stat_col="bsadf", max_lag=12)
    b = lead_lag_correlation(explicit, proxy_col="proxy", stat_col="bsadf", max_lag=12)
    pd.testing.assert_frame_equal(a, b)
    kw = dict(regime_col="regime", proxy_col="proxy", control_cols=["return", "rvol"])
    a = predictive_regression(sparse, **kw)
    b = predictive_regression(explicit, **kw)
    assert a["n"] == b["n"] and a["coef_proxy"] == b["coef_proxy"]
    assert a["se_proxy"] == b["se_proxy"]


def test_calendar_hac_matches_statsmodels_on_complete_days():
    import statsmodels.api as sm
    df = _frame(seed=10, planted=True)
    kw = dict(regime_col="regime", proxy_col="proxy", control_cols=["return", "rvol"])
    got = predictive_regression(df, **kw, hac_lags=7)
    predictors = df[["proxy", "return", "rvol"]].shift(1)
    data = pd.concat([df.regime, predictors], axis=1).dropna()
    expected = sm.Probit(data.regime, sm.add_constant(data[["proxy", "return", "rvol"]])).fit(
        disp=0, cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": False})
    np.testing.assert_allclose(got["se_proxy"], expected.bse["proxy"], rtol=1e-8)
