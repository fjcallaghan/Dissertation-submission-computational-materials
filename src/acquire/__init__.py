"""Acquisition stage: fetch and cache raw market data.

Each source is a thin fetcher built on the shared infrastructure in ``base.py``
(HTTP session with retry, a generic paginator, and a parquet cache layer), so
adding a new venue later (e.g. CoinGlass) is an isolated addition.
"""
