"""Paginator stitching — no network; pages are served from an in-memory series."""

from src.acquire import base

EIGHT_H_MS = 8 * 3600 * 1000
START = 1568102400000  # 2019-09-10 00:00 UTC


def _make_series(n: int) -> list[dict]:
    """A clean 8-hourly funding series of ``n`` records."""
    return [
        {"fundingTime": START + i * EIGHT_H_MS, "fundingRate": "0.0001"}
        for i in range(n)
    ]


def _forward_pager(series: list[dict], limit: int):
    """Emulate a forward endpoint: records with ts >= cursor, capped at limit."""
    def fetch(cursor_ms: int):
        page = [r for r in series if r["fundingTime"] >= cursor_ms]
        return page[:limit]
    return fetch


def _backward_pager(series: list[dict], limit: int, ts_key: str):
    """Emulate a backward endpoint: records with ts <= cursor, newest first."""
    def fetch(cursor_ms: int):
        page = [r for r in series if r[ts_key] <= cursor_ms]
        page.sort(key=lambda r: r[ts_key], reverse=True)
        return page[:limit]
    return fetch


def test_forward_stitches_across_pages_without_dupes():
    n, limit = 2500, 1000  # forces 3 pages
    series = _make_series(n)
    end = START + (n - 1) * EIGHT_H_MS
    out = base.paginate_forward(
        _forward_pager(series, limit),
        start_ms=START, end_ms=end,
        timestamp_key="fundingTime", page_limit=limit,
    )
    times = [r["fundingTime"] for r in out]
    assert len(out) == n
    assert times == sorted(times)
    assert len(set(times)) == n  # no duplicates at page seams


def test_forward_respects_end_bound():
    series = _make_series(100)
    end = START + 9 * EIGHT_H_MS  # only first 10 records
    out = base.paginate_forward(
        _forward_pager(series, 1000),
        start_ms=START, end_ms=end,
        timestamp_key="fundingTime", page_limit=1000,
    )
    assert len(out) == 10
    assert out[-1]["fundingTime"] == end


def test_backward_returns_ascending_and_complete():
    n, limit = 500, 200  # forces 3 pages backward
    series = [
        {"fundingRateTimestamp": START + i * EIGHT_H_MS, "fundingRate": "0.0001"}
        for i in range(n)
    ]
    end = START + (n - 1) * EIGHT_H_MS
    out = base.paginate_backward(
        _backward_pager(series, limit, "fundingRateTimestamp"),
        start_ms=START, end_ms=end,
        timestamp_key="fundingRateTimestamp", page_limit=limit,
    )
    times = [r["fundingRateTimestamp"] for r in out]
    assert len(out) == n
    assert times == sorted(times)  # returned ascending despite descending pages
