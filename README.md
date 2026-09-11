# Asset price bubbles: dissertation code

Computational materials for *Modelling and Detection of Asset Price Bubbles: Discrete Time Theory, Short Selling Restrictions, and Bitcoin*, Freddie Callaghan, MSc Financial Mathematics, University of Bath, September 2026.

The repository contains the analysis code. The version corresponding to the submitted dissertation is [release `thesis-submission-v2`](https://github.com/fjcallaghan/Dissertation-submission-computational-materials/releases/tag/thesis-submission-v2). Download `thesis-supplement-2026-09-07.zip`, extract it into an empty directory, and read `SUPPLEMENT-README.md`. Its numerical-date filename identifies the frozen numerical work; its manifest identifies the publication revision.

**The public package does not contain the saved market observations or all inputs needed to reproduce the empirical results.** Yahoo Finance and Deribit redistribution permission has not been established. Saved market data, processed observations, observation-dependent calibration arrays, option laws and outputs, and thesis assets are withheld from this public deposit. Other providers' saved observations are also withheld pending a redistribution review. `OMITTED-FILES.json` lists every file excluded from the complete local snapshot and its checksum. The complete snapshot remains local; this release makes no promise of access to the withheld data.

Verify `MANIFEST-SHA256.json` in the extracted release before running code. The public tests use synthetic fixtures and can run without saved market observations. The historical and option replay commands require the complete original inputs, obtained with appropriate permissions, at the paths documented in `SUPPLEMENT-README.md`. Downloading new observations is not an exact replay of the frozen results.

The historical analysis concerns retrospective statistical explosiveness and financing associations. The separate option exercise constructs conditional compatible pricing laws. Neither exercise establishes a general advance trading signal or identifies a positive Bitcoin bubble defect.

## Final audit corrections

Release v2 explicitly computes the HC1 degrees-of-freedom correction, rejects unsupported runner settings, and replaces stale episode files when a run finds no episodes. It also corrects descriptions of lending candles, funding receipts and statistical outlier flags. The revised dissertation states the innovation filtrations, the provisional status of the last daily price and the exact two-sided option screen. The accompanying synthetic test suites pass all 102 tests. See [CHANGELOG.md](CHANGELOG.md).

## Code map

- `src/`, `tests/`, `config.yaml`: historical pipeline and tests.
- `pilot/option_reliability/scripts/` and `tests/`: option analysis and synthetic tests; adjacent configuration, conventions and sources document assumptions.
- `docs/implementation/final-2026-09-07/analyse.py` and `verify.py`: final exhibits and independent numerical checks, requiring the withheld inputs and supporting source snapshots.
- `requirements-final.txt`: recorded dependency versions, checked with Python 3.9.6. Root and pilot requirements give supported ranges; the historical environment record controls exact replay.

## Tests

With dependencies installed, run from the repository or extracted archive root:

```bash
python -m pytest -q tests pilot/option_reliability/tests
```

No reuse licence has been selected for the author's code. Third-party data rights are separate; see `DATA-TERMS.md`. Public access does not grant a redistribution licence for third-party observations.
