# exopipe

**AI-enabled detection and classification of exoplanet transits in noisy TESS light curves.**
*BAH 2026 — Problem Statement 7.*

## Problem

Transit photometry detects exoplanets by measuring tiny, periodic dips in a
star's brightness. In crowded TESS fields those dips are buried under detector
noise, stellar variability, and contamination from blended neighbours. The same
dips can also be produced by eclipsing binaries or starspots rather than planets.
`exopipe` aims to **detect** periodic dips in noisy light curves, **classify**
them (transit / eclipsing binary / blend / other), report a **significance**
(SNR / SDE), and for genuine transits **fit** the orbital period, transit depth,
and duration with uncertainties — all visualised end to end.

## Status

🚧 **Foundation in place; full pipeline under active development.** This stage
ships the shared package skeleton: typed data structures, configuration,
utilities, and a physically-motivated synthetic TESS-like light-curve generator
(used for demos, tests, training, and injection–recovery). The detection,
detrending, vetting, classification, fitting, and visualisation modules are being
built in parallel against the interfaces in `exopipe.types`.

## Install

Core foundation (numpy/scipy/matplotlib/pandas only):

```bash
pip install -e .
# or:  pip install -r requirements-core.txt
```

Optional capability groups (install only what you need):

```bash
pip install -e ".[science]"   # astropy, lightkurve, transitleastsquares, wotan, batman, emcee, dynesty, astroquery
pip install -e ".[ml]"        # scikit-learn, xgboost, lightgbm
pip install -e ".[dl]"        # torch
pip install -e ".[perf]"      # numba, joblib, pyarrow, zarr, bottleneck, hnswlib
pip install -e ".[app]"       # streamlit, plotly
pip install -e ".[dev]"       # pytest, ruff
```

## Quickstart

Generate and summarise a synthetic light curve (works with the core install):

```bash
python -m exopipe.cli demo --kind transit --seed 1
# also: --kind eclipsing_binary | blend | variable | noise, and --plot out.png
```

In Python:

```python
import exopipe
from exopipe.data import from_arrays, make_synthetic_lightcurve, make_synthetic_population

# a single physically-motivated synthetic transit
lc = make_synthetic_lightcurve("transit", seed=1)
print(lc.meta["true_period"], lc.meta["true_depth"], lc.meta["label"])

# a labelled population for training / evaluation
pop = make_synthetic_population(200, seed=0)

# build a LightCurve from your own arrays (auto dtype + normalisation)
my_lc = from_arrays(time, flux, flux_err)
```

The end-to-end driver `exopipe.process_lightcurve(...)` is re-exported lazily and
will light up as the algorithm modules land.

## Layout

```
exopipe/
├── pyproject.toml            # package metadata, deps, extras, console script
├── requirements-core.txt     # numpy, scipy, matplotlib, pandas
├── requirements-optional.txt # everything else, grouped by extra
├── configs/
│   └── default.yaml          # default configuration (mirrors exopipe.config.Config)
├── src/exopipe/
│   ├── __init__.py           # version + lazy re-exports
│   ├── types.py              # shared dataclasses (the interface contract)
│   ├── config.py             # hierarchical Config + load_config()
│   ├── utils.py              # logging, Timer, seeding, robust stats, phase-fold
│   ├── cli.py                # `exopipe` command-line entry point
│   └── data/
│       ├── lightcurve.py     # LightCurve constructors/operations
│       └── synthetic.py      # synthetic TESS-like generator
└── tests/
    └── test_foundation.py    # foundation test suite (core deps only)
```

## Conventions (for contributors)

- **Time** is in days (`float64`); **flux**/**flux_err** are normalised to a
  median of ~1.0 and stored as `float32`.
- **Depths** are fractional (`0.01` = 1% = 10 000 ppm), positive for a dip;
  **periods** and **durations** are in days.
- Build `LightCurve` objects via `exopipe.data.from_arrays(...)` so dtypes and
  normalisation stay consistent.
- Every pipeline stage exchanges data through the dataclasses in
  `exopipe.types`; `CandidateResult.to_row()` flattens a result into one catalog
  row.

## Data

TESS light curves can be downloaded from the
[MAST/TIC archive](https://archive.stsci.edu/tess/tic_ctl.html) (a single
sector's 2-minute-cadence data contains ~20–30k stars). Until that ingestion is
wired up, the synthetic generator provides labelled, reproducible data for
development and testing.

## License

MIT.
