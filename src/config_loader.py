"""Load and resolve ``config.yaml``.

Every methodological choice that affects the data lives in ``config.yaml``. This
module is the single place that reads it, resolves the sentinel ``end: "present"``
to today's UTC date, and exposes the result as a lightweight typed object so the
rest of the pipeline never touches raw YAML or hardcodes a parameter.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root is the parent of the ``src`` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _today_utc() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def load_dotenv(path: Path = DEFAULT_ENV_PATH) -> None:
    """Minimal ``.env`` loader (no third-party dependency).

    Parses ``KEY=VALUE`` lines and populates ``os.environ`` without overwriting
    variables already set in the real environment (which take precedence).
    Keeps secrets like ``COINALYZE_API_KEY`` out of the code and out of git.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Window:
    price_start: dt.date
    funding_start: dt.date
    analysis_start: dt.date
    end: dt.date  # "present" resolved to today (UTC) at load time


@dataclass(frozen=True)
class Config:
    """Resolved view of ``config.yaml`` plus the raw dict for anything not
    promoted to a typed field."""

    window: Window
    raw: dict[str, Any] = field(repr=False)
    path: Path = field(repr=False, default=DEFAULT_CONFIG_PATH)

    # --- convenience accessors for the sections used this pass ---
    @property
    def spot_ticker(self) -> str:
        return self.raw["data"]["spot_ticker"]

    @property
    def spot_frequency(self) -> str:
        return self.raw["data"]["spot_frequency"]

    @property
    def perp_symbol(self) -> str:
        return self.raw["data"]["perp_symbol"]

    @property
    def use_cache(self) -> bool:
        return bool(self.raw["cache"]["use_cache"])

    @property
    def refresh(self) -> bool:
        return bool(self.raw["cache"]["refresh"])

    @property
    def raw_dir(self) -> Path:
        return self._resolve_dir(self.raw["cache"]["raw_dir"])

    @property
    def processed_dir(self) -> Path:
        return self._resolve_dir(self.raw["cache"]["processed_dir"])

    def _resolve_dir(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    # --- accessors for the processing stages (clean / align / plots / synthetic) ---
    @property
    def figures_dir(self) -> Path:
        return self._resolve_dir(self.raw.get("plots", {}).get("output_dir", "figures"))

    @property
    def funding_daily_aggregation(self) -> str:
        return self.raw["funding"]["daily_aggregation"]

    @property
    def rolling_window_days(self) -> int:
        return int(self.raw["funding"]["rolling_window_days"])

    @property
    def elevated_threshold(self) -> float:
        return float(self.raw["funding"]["elevated_threshold"])

    # --- direct short-cost proxy: BTC margin borrow rate (optional section) ---
    @property
    def borrow_enabled(self) -> bool:
        return bool(self.raw.get("borrow", {}).get("enabled", False))

    @property
    def borrow_source(self) -> str:
        return str(self.raw.get("borrow", {}).get("source", "bitfinex"))

    @property
    def borrow_candle(self) -> str:
        return str(self.raw.get("borrow", {}).get(
            "bitfinex_candle", "trade:1D:fBTC:a30:p2:p30"))

    @property
    def borrow_start(self) -> dt.date:
        value = self.raw.get("borrow", {}).get("start")
        return _parse_date(value) if value else self.window.price_start

    @property
    def borrow_daily_aggregation(self) -> str:
        return str(self.raw.get("borrow", {}).get("daily_aggregation", "mean"))

    @property
    def spike_sigma(self) -> float:
        return float(self.raw["cleaning"]["spike_sigma"])

    @property
    def spike_baseline_window_days(self) -> int:
        return int(self.raw["cleaning"]["spike_baseline_window_days"])

    @property
    def price_gap_fill(self) -> str:
        return self.raw["cleaning"]["price_gap_fill"]

    @property
    def returns_method(self) -> str:
        return self.raw["returns"]["method"]

    @property
    def synthetic(self) -> dict[str, Any]:
        return self.raw["synthetic"]

    # --- Phase IV: bubble-detection diagnostics (optional section) ---
    @property
    def detect(self) -> dict[str, Any]:
        return self.raw.get("detect", {})

    @property
    def detect_psy(self) -> dict[str, Any]:
        return self.detect.get("psy", {})

    @property
    def detect_stat(self) -> str:
        return str(self.detect_psy.get("stat", "gsadf"))

    @property
    def detect_adf_lag(self) -> int:
        return int(self.detect_psy.get("adf_lag", 1))

    @property
    def detect_min_window_frac(self) -> float | None:
        v = self.detect_psy.get("min_window_frac")
        return float(v) if v is not None else None

    @property
    def detect_significance(self) -> float:
        return float(self.detect_psy.get("significance", 0.95))

    @property
    def detect_use_log_price(self) -> bool:
        return bool(self.detect_psy.get("use_log_price", True))

    @property
    def detect_cv(self) -> dict[str, Any]:
        return self.detect.get("critvals", {})

    @property
    def cv_method(self) -> str:
        return str(self.detect_cv.get("method", "mc"))

    @property
    def cv_n_sim(self) -> int:
        return int(self.detect_cv.get("n_sim", 499))

    @property
    def cv_sim_size(self) -> int:
        return int(self.detect_cv.get("sim_size", 1500))

    @property
    def cv_seed(self) -> int:
        return int(self.detect_cv.get("seed", 12345))

    @property
    def date_stamp_min_duration(self) -> int | None:
        v = self.detect.get("date_stamp", {}).get("min_duration_days")
        return int(v) if v is not None else None

    @property
    def slm_enabled(self) -> bool:
        return bool(self.detect.get("slm", {}).get("enabled", True))

    @property
    def slm_tail_quantile(self) -> float:
        return float(self.detect.get("slm", {}).get("tail_quantile", 0.6))

    @property
    def link_cfg(self) -> dict[str, Any]:
        return self.detect.get("link", {})

    @property
    def link_max_lag(self) -> int:
        return int(self.link_cfg.get("max_lag_days", 30))

    @property
    def link_predict_lag(self) -> int:
        return int(self.link_cfg.get("predict_lag_days", 1))

    @property
    def link_rvol_window(self) -> int:
        return int(self.link_cfg.get("realized_vol_window", 30))

    @property
    def link_model(self) -> str:
        return str(self.link_cfg.get("model", "probit"))


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value))


# Enumerated choices, validated at load time so a typo in config.yaml fails
# immediately with a clear message rather than deep inside a pipeline stage.
_CHOICES = {
    ("funding", "daily_aggregation"): {"sum", "mean", "none"},
    ("returns", "method"): {"log", "simple"},
    ("cleaning", "price_gap_fill"): {"ffill", "interpolate", "none"},
    ("borrow", "daily_aggregation"): {"mean", "sum"},
}
# Values that must be strictly positive if present.
_POSITIVE = [
    ("funding", "rolling_window_days"),
    ("cleaning", "spike_sigma"),
    ("cleaning", "spike_baseline_window_days"),
]


def _validate_detect(raw: dict[str, Any]) -> None:
    """Validate the optional Phase IV ``detect`` section (nested), fail fast."""
    detect = raw.get("detect")
    if not detect:
        return
    psy = detect.get("psy", {})
    stat = psy.get("stat")
    if stat is not None and stat not in {"gsadf", "sadf"}:
        raise ValueError(f"config.yaml detect.psy.stat={stat!r}; expected 'gsadf' or 'sadf'")
    sig = psy.get("significance")
    if sig is not None and not (0.0 < float(sig) < 1.0):
        raise ValueError(f"config.yaml detect.psy.significance={sig} must be in (0, 1)")
    lag = psy.get("adf_lag")
    if lag is not None and (int(lag) < 0):
        raise ValueError(f"config.yaml detect.psy.adf_lag={lag} must be >= 0")
    mwf = psy.get("min_window_frac")
    if mwf is not None and not (0.0 < float(mwf) < 1.0):
        raise ValueError(f"config.yaml detect.psy.min_window_frac={mwf} must be in (0, 1)")

    cv = detect.get("critvals", {})
    method = cv.get("method")
    if method is not None and method not in {"mc", "wild"}:
        raise ValueError(f"config.yaml detect.critvals.method={method!r}; expected 'mc' or 'wild'")
    for key in ("n_sim", "sim_size"):
        val = cv.get(key)
        if val is not None and int(val) <= 0:
            raise ValueError(f"config.yaml detect.critvals.{key}={val} must be positive")

    model = detect.get("link", {}).get("model")
    if model is not None and model not in {"probit", "logit"}:
        raise ValueError(f"config.yaml detect.link.model={model!r}; expected 'probit' or 'logit'")
    hac_lags = detect.get("link", {}).get("hac_lags", 30)
    if not isinstance(hac_lags, int) or isinstance(hac_lags, bool) or hac_lags < 0:
        raise ValueError("config.yaml detect.link.hac_lags must be a nonnegative integer")


def _validate_config_choices(raw: dict[str, Any]) -> None:
    """Fail fast on out-of-range enumerated / numeric config values."""
    for (section, key), allowed in _CHOICES.items():
        value = raw.get(section, {}).get(key)
        if value is not None and value not in allowed:
            raise ValueError(
                f"config.yaml {section}.{key}={value!r} is invalid; "
                f"expected one of {sorted(allowed)}"
            )
    for section, key in _POSITIVE:
        value = raw.get(section, {}).get(key)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"config.yaml {section}.{key}={value!r} must be a number"
                ) from None
            if numeric <= 0:
                raise ValueError(
                    f"config.yaml {section}.{key}={value} must be positive"
                )


def load_config(path: str | Path | None = None) -> Config:
    """Read ``config.yaml`` (and ``.env``) and return a resolved :class:`Config`."""
    load_dotenv()
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")

    required = {"window", "data", "cache", "funding", "cleaning", "returns",
                "plots", "synthetic"}
    missing = required - raw.keys()
    if missing:
        raise KeyError(f"config.yaml missing required section(s): {sorted(missing)}")

    w = raw["window"]
    end_raw = str(w.get("end", "present")).strip().lower()
    end = _today_utc() if end_raw == "present" else _parse_date(w["end"])

    window = Window(
        price_start=_parse_date(w["price_start"]),
        funding_start=_parse_date(w["funding_start"]),
        analysis_start=_parse_date(w["analysis_start"]),
        end=end,
    )
    for name, start in (("price_start", window.price_start),
                        ("funding_start", window.funding_start),
                        ("analysis_start", window.analysis_start)):
        if start > window.end:
            raise ValueError(f"window.{name} ({start}) is after window.end ({window.end})")

    _validate_config_choices(raw)
    _validate_detect(raw)

    # The optional borrow window must also sit inside the analysis end.
    borrow_start = raw.get("borrow", {}).get("start")
    if borrow_start is not None and _parse_date(borrow_start) > window.end:
        raise ValueError(
            f"borrow.start ({borrow_start}) is after window.end ({window.end})"
        )

    return Config(window=window, raw=raw, path=cfg_path)


if __name__ == "__main__":  # quick manual sanity check
    cfg = load_config()
    print(f"config: {cfg.path}")
    print(f"window: {cfg.window}")
    print(f"spot={cfg.spot_ticker} perp={cfg.perp_symbol}")
    print(f"raw_dir={cfg.raw_dir} use_cache={cfg.use_cache} refresh={cfg.refresh}")
