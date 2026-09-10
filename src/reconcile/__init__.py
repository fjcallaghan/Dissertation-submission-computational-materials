"""Cross-exchange reconciliation stage.

Compares the Binance funding signal against other venues (Bybit, and Coinalyze's
market-wide aggregate when available) to confirm it is a market-wide phenomenon
rather than an exchange-specific artefact — the robustness check underpinning the
thesis's use of funding as an (indirect) proxy for short-selling frictions.
"""
