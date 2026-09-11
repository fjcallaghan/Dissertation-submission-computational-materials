# Public thesis supplement: numerical snapshot dated 7 September 2026

Publication revision: 11 September 2026 (final audit corrections). Accompanies *Modelling and Detection of Asset Price Bubbles: Discrete Time Theory, Short Selling Restrictions, and Bitcoin*, Freddie Callaghan, University of Bath, September 2026.

Release: [thesis-submission-v2](https://github.com/fjcallaghan/Dissertation-submission-computational-materials/releases/tag/thesis-submission-v2).

## Read this first: reproduction limits

This is the public code supplement. **It cannot independently reproduce the empirical thesis results.** The complete frozen snapshot remains local. Saved market observations, processed calendars, calibration arrays, option witnesses and outputs, and thesis source/assets have been withheld pending redistribution permission. `OMITTED-FILES.json` inventories all exclusions and their hashes; see `DATA-TERMS.md` for the reasons. Do not interpret the replay commands below as runnable using only this download.

Extract into an empty directory. Run commands from that directory. Verify delivered files before changing them:

```bash
python -c 'import hashlib,json,pathlib; m=json.loads(pathlib.Path("MANIFEST-SHA256.json").read_text()); bad=[p for p,h in m["files"].items() if not pathlib.Path(p).is_file() or hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("All",len(m["files"]),"payload hashes match")'
```

The manifest covers the delivered public files only, not the withheld files.

## Environment and public tests

The historical environment used Python 3.9.6 and `requirements-final.txt`; the environment record is `docs/implementation/final-2026-09-07/environment.json`. Exact pins require a compatible Python/platform. Package installation requires network access.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-final.txt
python -m pytest -q tests pilot/option_reliability/tests
```

The tests use synthetic fixtures and do not require withheld observations. They check implementation behaviour; passing them is not independent reproduction of the empirical results. The acquisition-time yfinance version was not recorded and is not inferred from the later environment record.

## Corrections in this release

HC1 now applies n/(n-k), with n counting estimation observations and k including the intercept. The principal HAC implementation is unchanged. The runner supports GSADF and daily sum/mean alignment; unsupported SADF or eight-hourly alignment settings are rejected. Zero-episode runs now replace any older episode table. All 102 synthetic tests pass.

The frozen price snapshot is retained. Its final 20 July 2026 daily bar was retrieved at 23:03:44 UTC that day and is provisional. The revised dissertation discloses this and a completed-day sensitivity that preserves the main conclusions. The completed-day inputs and outputs remain part of the withheld local materials.

## Conditional historical replay

Only after obtaining permission and the complete original inputs/supporting files at their recorded relative paths, verify the complete snapshot's own manifest before running:

```bash
python -m src.detect.revision
python docs/implementation/final-2026-09-07/analyse.py
python docs/implementation/final-2026-09-07/verify.py
```

Dependencies include `data/raw/`, `data/processed/`, `docs/reviews/verification/`, supporting records in `docs/implementation/ch5/` and `docs/implementation/final-2026-09-07/`, saved option outputs, and `thesis/`. The independent verifier uses the historical `before/main.tex` snapshot as well as the current source. These files are not delivered here.

The complete offline replay imports the saved 199-draw wild calibration, checks the price vector and bootstrap configuration, and regenerates historical/final exhibits. It does not generate a fresh full calibration or establish nominal test size. Reported comparison targets are GSADF 4.0317, fifteen wild-calibrated episodes and 538 flagged days; this public package alone cannot verify those targets.

## Conditional option replay

Requires the complete saved request/response payloads and session manifests, obtained with appropriate permissions:

```bash
python pilot/option_reliability/scripts/analyse.py \
  pilot/option_reliability/raw/initial_20260907T041046748784Z/session.json \
  --recheck pilot/option_reliability/raw/recheck_20260907T041825082569Z/session.json \
  --output-dir pilot/option_reliability/outputs/reproduced_final
```

The script validates saved hashes, quotes and timing, recalculates bounds, solves finite-support feasibility problems and reprices successful laws. Acquisition-time configurations, not the current collection configuration, govern saved sessions. `CONVENTIONS.md` and `SOURCES.md` explain the assumptions. Feasible witnesses are not proved sharp endpoints or estimated pricing probabilities.

Full-precision laws cited in the thesis were recorded in `pilot/option_reliability/outputs/reviewed_20260907/2026-09-25/` and `2026-10-30/`, including `witness_A_0.json` and `witness_A_4.json`. They are withheld from this public deposit.

## Thesis and later revisions

The submitted thesis is a separate PDF. Its full source/assets and the complete local supplement are not public release assets. The thesis appendix distinguishes this public code package from the complete local snapshot. Later corrections or permission-based additions should use a new release.
