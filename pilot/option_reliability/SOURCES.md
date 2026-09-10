# Sources and contribution boundary

Checked 7 September 2026. This is a focused pilot bibliography, not a claim to a comprehensive literature review or a new identification theorem.

## Contract and data

- [Deribit linear USDC option specifications](https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options): payoff, settlement average, premium timing, currency and multiplier. The conditional mapping and approximations are in `CONVENTIONS.md`.
- [Public book summaries](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency): screening metadata. Production requests explicitly use `www.deribit.com`; the documentation’s testnet examples are not copied as the live environment.
- [Public order books](https://docs.deribit.com/api-reference/market-data/public-get_order_book): displayed quotes, sizes, timestamps and ancillary index/futures metadata. The saved payloads are the primary evidence for this pilot’s actual observations.

## Finite-strike moments and consistency

[Bertsimas and Popescu (2002), *On the Relation Between Option and Stock Prices: A Convex Optimization Approach*](https://pubsonline.informs.org/doi/10.1287/opre.50.2.358.424), Operations Research 50(2), 358–374, studies bounds on moments and other option prices from observed option prices using optimisation, including transaction costs. This is direct prior work for the principle of finite-strike moment restrictions. The pilot uses elementary inequalities and feasible examples; it does not compute their general sharp bounds.

[Davis and Hobson (2007), *The Range of Traded Option Prices*](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9965.2007.00291.x), Mathematical Finance 17(1), 1–14, studies consistency of finite option prices with arbitrage-free models and distinguishes boundary cases from straightforward arbitrage. The [author-hosted preprint](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/hobson/publications/range.pdf) is available. Their frictionless setting is not automatically the full-defect convention maintained here.

[Gerhold and Gülüm (2020), *Consistency of Option Prices under Bid–Ask Spreads*](https://doi.org/10.1111/mafi.12230), with [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7155110/), directly addresses quote consistency with spreads, including the role of an underlying-price spread. This supports treating observed intervals as constraints rather than sampling distributions, and keeping reference uncertainty explicit. The present finite-support witness exercise is narrower than proving their general consistency conditions.

The plan’s Chapter 2 and Chapter 3 review notes govern the defect identities and the distinction between terminal laws and dynamic bubble models. The contribution of this run is its timestamped, checked empirical restrictions and documented convention failures. It does not claim that using linear programming for option-implied moments, or the general difficulty of finite-strike identification, is new.
