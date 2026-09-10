# Third-party data and public release scope

Reviewed 10 September 2026. The author indicated that public redistribution permission for the saved Yahoo Finance and Deribit observations is absent or uncertain. This release therefore withholds saved market observations and dependent material. No legal conclusion about research use is asserted.

- Yahoo Finance: the [yfinance project notice](https://github.com/ranaroussi/yfinance) says the Yahoo Finance API is intended for personal use only and directs users to Yahoo's terms. The software licence does not grant rights in downloaded observations.
- Deribit: [published Terms of Service, section 4.6](https://statics.deribit.com/files/TermsofServiceDeribit.pdf) restrict publishing market data and derived data without explicit approval. Public API access is not itself redistribution permission.
- Binance, Bybit and Bitfinex: this deposit does not assert redistribution rights; their saved series are also withheld pending review.

The public archive includes code, tests using artificial fixtures, configuration, dependency records and documentation. It excludes all saved raw/processed market data, saved bootstrap/calibration arrays, full-precision option laws and empirical output files. Thesis source and assets, which contain observed quotations and empirical exhibits, are retained locally. Historical internal review and editorial files are outside the public code overview.

`OMITTED-FILES.json` is an inventory of exclusions from the complete local supplement, with SHA-256 hashes and reasons. `MANIFEST-SHA256.json` covers only files actually delivered in the public archive. Neither a checksum nor a documented path implies that the file is publicly available. No data have been silently replaced, and new acquisition is not claimed to reproduce the original snapshot.

The author's code has no added reuse licence. Any future permission-based data publication should use a new release and explain the revised reproduction scope.
