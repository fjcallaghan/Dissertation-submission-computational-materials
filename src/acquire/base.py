"""Shared acquisition infrastructure.

Keeps each source fetcher thin by centralising:
  * a ``requests`` session with retry/backoff on 429 + 5xx and a polite
    inter-request delay (exchanges rate-limit by weight);
  * millisecond-epoch <-> UTC timestamp helpers (all exchange times are ms);
  * a generic paginator that walks a time-ordered endpoint page by page;
  * a parquet cache layer honouring ``cache.use_cache`` / ``cache.refresh``,
    writing a small JSON manifest alongside each series for provenance.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Timestamp helpers — exchanges speak millisecond epoch; we keep UTC internally.
# ----------------------------------------------------------------------------
def utc_to_ms(value: dt.date | dt.datetime) -> int:
    """UTC date/datetime -> integer milliseconds since the epoch."""
    if isinstance(value, dt.datetime):
        d = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    else:  # date -> midnight UTC
        d = dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def ms_to_utc(ms: int | float) -> pd.Timestamp:
    """Integer milliseconds since the epoch -> tz-aware UTC ``pd.Timestamp``."""
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


def now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


# ----------------------------------------------------------------------------
# HTTP session with retry/backoff.
# ----------------------------------------------------------------------------
class HttpClient:
    """Thin wrapper over ``requests`` adding retry, backoff and a polite delay."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff: float = 1.5,
        min_interval: float = 0.25,
        user_agent: str = "btc-bubble-detection/0.1 (academic; read-only)",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = min_interval
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def get_json(self, url: str, params: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None) -> Any:
        """GET returning parsed JSON, retrying transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                # Retry on rate-limit / server errors; raise on other 4xx.
                if resp.status_code in (429, 418) or resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"{resp.status_code} for {resp.url}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                sleep_s = self.backoff ** attempt
                logger.warning(
                    "GET failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, self.max_retries, exc, sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts") from last_exc


# ----------------------------------------------------------------------------
# Generic time-paginator.
# ----------------------------------------------------------------------------
def paginate_forward(
    fetch_page: Callable[[int], Sequence[dict[str, Any]]],
    *,
    start_ms: int,
    end_ms: int,
    timestamp_key: str,
    page_limit: int,
    max_pages: int = 10_000,
) -> list[dict[str, Any]]:
    """Walk a time-ordered endpoint *forward* from ``start_ms`` to ``end_ms``.

    ``fetch_page(cursor_ms)`` must return records sorted ascending by
    ``timestamp_key``. After each page the cursor advances to
    ``last_timestamp + 1`` so no record is fetched twice. Stops when a page is
    shorter than ``page_limit`` (caught up) or the cursor passes ``end_ms``.
    De-duplicates on ``timestamp_key`` defensively in case of seam overlaps.
    """
    out: dict[int, dict[str, Any]] = {}
    cursor = start_ms
    for _ in range(max_pages):
        if cursor > end_ms:
            break
        page = list(fetch_page(cursor))
        if not page:
            break
        for rec in page:
            ts = int(rec[timestamp_key])
            if ts <= end_ms:
                out[ts] = rec
        last_ts = int(page[-1][timestamp_key])
        if len(page) < page_limit or last_ts <= cursor:
            break  # caught up, or endpoint refused to advance
        cursor = last_ts + 1
    else:
        logger.warning("paginate_forward hit max_pages=%d; result may be truncated",
                       max_pages)
    return [out[k] for k in sorted(out)]


def paginate_backward(
    fetch_page: Callable[[int], Sequence[dict[str, Any]]],
    *,
    start_ms: int,
    end_ms: int,
    timestamp_key: str,
    page_limit: int,
    max_pages: int = 10_000,
) -> list[dict[str, Any]]:
    """Walk a time-ordered endpoint *backward* from ``end_ms`` toward ``start_ms``.

    For venues (e.g. Bybit) whose history endpoint pages by ``endTime`` and
    returns records sorted descending. ``fetch_page(cursor_ms)`` returns the
    records ending at ``cursor_ms``; the cursor then steps back to the oldest
    timestamp seen minus one. Returns records ascending by ``timestamp_key``.
    """
    out: dict[int, dict[str, Any]] = {}
    cursor = end_ms
    for _ in range(max_pages):
        if cursor < start_ms:
            break
        page = list(fetch_page(cursor))
        if not page:
            break
        oldest = min(int(r[timestamp_key]) for r in page)
        for rec in page:
            ts = int(rec[timestamp_key])
            if start_ms <= ts <= end_ms:
                out[ts] = rec
        if len(page) < page_limit or oldest >= cursor:
            break
        cursor = oldest - 1
    else:
        logger.warning("paginate_backward hit max_pages=%d; result may be truncated",
                       max_pages)
    return [out[k] for k in sorted(out)]


# ----------------------------------------------------------------------------
# Parquet cache layer.
# ----------------------------------------------------------------------------
def _manifest_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".manifest.json")


def cache_load(parquet_path: Path) -> pd.DataFrame | None:
    """Return the cached frame if present, else ``None``."""
    if parquet_path.exists():
        logger.info("cache hit: %s", parquet_path)
        return pd.read_parquet(parquet_path)
    return None


def cache_save(
    df: pd.DataFrame,
    parquet_path: Path,
    *,
    source: str,
    symbol: str,
    time_col: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Write ``df`` to parquet plus a JSON manifest recording provenance."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    ts = df[time_col]
    manifest = {
        "source": source,
        "symbol": symbol,
        "rows": int(len(df)),
        "time_col": time_col,
        "min_time": None if df.empty else str(ts.min()),
        "max_time": None if df.empty else str(ts.max()),
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": params or {},
    }
    with open(_manifest_path(parquet_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("cached %d rows -> %s", len(df), parquet_path)


def get_or_fetch(
    parquet_path: Path,
    fetch_fn: Callable[[], pd.DataFrame],
    *,
    source: str,
    symbol: str,
    time_col: str,
    use_cache: bool,
    refresh: bool,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return cached data when allowed, otherwise fetch fresh and cache it.

    ``refresh=True`` forces a re-fetch even when a cache file exists;
    ``use_cache=False`` fetches without reading the cache (but still writes it).
    """
    if use_cache and not refresh:
        cached = cache_load(parquet_path)
        if cached is not None:
            return cached
    df = fetch_fn()
    cache_save(df, parquet_path, source=source, symbol=symbol,
               time_col=time_col, params=params)
    return df
