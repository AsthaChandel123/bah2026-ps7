# `examples/` — curated exopipe deliverables

Representative, committed artifacts produced by the `exopipe` pipeline on the
offline synthetic generator. Everything here is reproducible with **core
dependencies only** (no network) using the commands at the bottom.

## Contents

| File | What it is |
| --- | --- |
| `vetting_transit.png` | One-page vetting sheet for a correctly-classified **planetary transit**. |
| `vetting_eclipsing_binary.png` | Vetting sheet for a correctly-classified **eclipsing binary** (note the secondary eclipse + odd/even panels). |
| `vetting_blend.png` | Vetting sheet for a correctly-classified **blend** (diluted depth, low CROWDSAP, centroid offset). |
| `vetting_other.png` | Vetting sheet for an **other** case (stellar variability / no transit). |
| `example_catalog.csv` | Machine-readable candidate catalog (one row per light curve) in the canonical `exopipe.catalog` column order: ephemeris ± uncertainties, depth (ppm), SNR/SDE, class + calibrated probabilities, and vetting flags. |
| `example_report.pdf` | The ≤3-page methodology + results report (`exopipe.report.generate_report`): Objective, Data, Methodology (tools named), Assumptions, Uncertainty estimation, Results, and an embedded vetting sheet. |

### The one-page vetting sheet

Each sheet mirrors the TESS SPOC Data-Validation one-page summary and packs the
full diagnostic story for one candidate onto a single A4 page:

- full detrended light curve with predicted transit epochs marked,
- global and local (zoomed) phase folds with the best-fit model overlaid,
- odd-vs-even transit comparison (an eclipsing-binary discriminator),
- BLS/TLS periodogram with the peak period and harmonics,
- secondary-eclipse search at phase 0.5,
- a river / waterfall plot across orbital cycles,
- a text summary (period / depth / duration ± uncertainties, SNR/SDE, flags),
- the **calibrated class-probability bar** (transit / eclipsing_binary / blend / other).

Colours follow the colour-blind-safe Okabe–Ito palette and class meaning is
always paired with a text label, never encoded by colour alone.

## How these were produced

The example light curves are drawn from the synthetic generator with clear,
detectable parameters (bright host + sensible depth) so each class is shown by an
unambiguous, **correctly-classified** case, then run through the *real* pipeline
(`detrend → search → vet → features → classify → fit → viz/report`) with the
trained classifier (`models/exopipe_clf.joblib`) loaded:

```bash
PYTHONPATH=src python scripts/make_examples.py
```

## Reproduce from the CLI

```bash
# 1. Train (and calibrate) the 4-class classifier -> models/exopipe_clf.joblib
PYTHONPATH=src python scripts/train_demo_classifier.py --n 260 --n-jobs -1
#    (or the generic trainer:  exopipe train --n 600)

# 2. End-to-end demo: synthetic population -> catalog + vetting sheets + report
PYTHONPATH=src python -m exopipe.cli demo --n 16 --figures --out runs/demo
#    -> runs/demo/catalog.csv, runs/demo/vetting_sheets/*.png, runs/demo/report.*

# 3. Render the report from any catalog
PYTHONPATH=src python -m exopipe.cli report --catalog runs/demo/catalog.csv \
    --figdir runs/demo/vetting_sheets --out examples/example_report
```

The `demo` command auto-loads `models/exopipe_clf.joblib` (when present) so the
catalog reflects calibrated ML + rules + physics-veto predictions; without a
model it cleanly degrades to the rules + physics-veto floor.
