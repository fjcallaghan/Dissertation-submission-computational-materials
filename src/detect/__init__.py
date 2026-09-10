"""Phase IV bubble-detection diagnostics.

Formal, discrete-time explosive-root testing (PSY / GSADF) that date-stamps
bubble episodes independently of the funding/borrow proxies, plus a secondary
strict-local-martingale volatility diagnostic and the linking analysis that
relates detected episodes to the short-selling-cost proxies.

Submodules
----------
adf       : single-window Augmented Dickey-Fuller regression (right-tailed).
psy       : recursive SADF / GSADF / BSADF statistics and PSY date-stamping.
critvals  : finite-sample critical values (Monte-Carlo + wild bootstrap).
slm       : strict-local-martingale volatility diagnostic (exploratory).
link      : regime comparison, lead-lag, and predictive probit vs the proxies.
"""
