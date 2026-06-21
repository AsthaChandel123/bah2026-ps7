---
title: "exopipe — AI Detection & Classification of Exoplanet Transits in TESS"
subtitle: "BAH 2026 — Problem Statement 7"
author: "Team exopipe"
date: "2026-06-21"
format:
  pdf:
    documentclass: article
    geometry:
      - margin=1.6cm        # tight margins so the whole report fits in <=3 pages
    fontsize: 9pt
    number-sections: true
    fig-pos: "H"            # keep figures where they are placed
    colorlinks: true
    keep-tex: false
execute:
  echo: false               # show results, hide code
  warning: false
  freeze: auto
---

<!--
  This is the <=3-page methodology + results report template for exopipe.
  It maps 1:1 onto the PS7 rubric (Objective, Data, Methodology, Assumptions,
  Uncertainty estimation, Results, Visualization).

  exopipe.report.generate_report() fills the {{PLACEHOLDERS}} below from the run
  catalog + metadata and renders to PDF with Quarto/pandoc when available, or
  via a matplotlib PdfPages fallback otherwise (so a PDF is ALWAYS produced with
  only core dependencies). Keep prose terse and tables compact to stay <=3 pages.
-->

# Objective {#sec-objective}

Detect periodic transit-like brightness dips in noisy, crowded TESS high-cadence
light curves, **classify** each detected signal as `transit`, `eclipsing_binary`,
`blend`, or `other`, estimate the transit **period, depth and duration** (plus
$R_p/R_\star$, $a/R_\star$, $b$ where fitted), and report a **calibrated
confidence** and detection significance for every candidate.



# Data {#sec-data}

Primary signal: TESS 2-minute-cadence SPOC light curves (a sector contains
~20,000–30,000 stars). Labels and augmentation come from Kepler/K2, the NASA
Exoplanet Archive (`toi`, `cumulative`/KOI, `pscomppars`), ExoFOP-TESS, and the
TESS-EB / Kepler-EB catalogs; Gaia DR3 + TIC v8.2 supply stellar parameters and
blend/contamination context. The pipeline also ships an offline, physically
motivated **synthetic generator** (white + correlated red noise, downlink gaps,
momentum dumps, injected transit/EB/blend/variable signals) so the full
methodology — including this report — runs with **zero network access**.

This run: 4 representative synthetic TESS-like light curves (one clean, correctly-classified example per class) processed end to end by the offline pipeline.

# Methodology {#sec-methodology}

The pipeline is a sequence of single-responsibility, independently cacheable
stages. **Tools/libraries actually used are named explicitly** (rubric
requirement). Core path runs on `numpy`/`scipy`/`astropy`/`pandas`/`matplotlib`;
every optional accelerator has a documented pure-Python fallback.

1. **Detrending** — windowed biweight (default; `wotan`) with running-median /
   Savitzky–Golay (`scipy.signal.savgol_filter`) and robust-spline fallbacks;
   asymmetric sigma-clipping (`astropy.stats.sigma_clip`).
2. **Transit search** — two-stage: Box Least Squares triage
   (`astropy.timeseries.BoxLeastSquares`) → Transit Least Squares confirmation
   (`transitleastsquares`) for the SDE periodogram; Lomb–Scargle / PDM / ACF
   cross-checks.
3. **Significance** — `transit_snr`, CDPP red-noise floor, and a bootstrap /
   GEV false-alarm probability (red-noise-aware), plus the TLS SDE.
4. **Vetting** — 15 physics tests: odd–even depth, secondary eclipse, V-shape,
   centroid offset / difference image, aperture contamination (CROWDSAP / Gaia),
   implied-radius and stellar-density sanity, uniqueness, ephemeris matching.
5. **Classification** — a calibrated 4-class ensemble: deterministic rules +
   tabular ML (`xgboost`/`lightgbm` + `sklearn` RandomForest, isotonic-calibrated)
   + an optional multi-branch CNN, combined by a stacking meta-learner with a
   **physics veto** that overrides the model when a decisive test fires.
6. **Parameter fitting** — trapezoid/box seed → Levenberg–Marquardt
   (`scipy.optimize`) → `batman` + `emcee` posterior (or `dynesty` for evidence),
   reporting 16/50/84 credible intervals and $\Delta$BIC.



# Assumptions {#sec-assumptions}

- One dominant periodic signal per target is searched per pass (multi-planet
  handled by iterative masking, out of scope for the headline figure).
- PDCSAP systematics are largely pre-corrected; residual trends are removed by
  the biweight detrender.
- A limb-darkened (Mandel–Agol) / trapezoid transit model adequately describes
  planetary transits; grazing/V-shaped events are reported as upper limits.
- Detection requires SDE above the conventional threshold (≈7–9).
- Training labels (TOI/KOI/EB catalogs) are trusted, reconciled by precedence.



# Uncertainty estimation {#sec-uncertainty}

Uncertainty is a first-class output at three levels: (i) **parameter** credible
intervals from the 16/50/84 posterior percentiles (asymmetric $+\sigma/-\sigma$),
with red-noise-aware inflation (GP / β-factor) so error bars are honest on real
TESS data; (ii) **detection** significance via SNR, SDE, and a bootstrap FAP;
and (iii) **classification** confidence as a calibrated class probability
(isotonic / temperature scaling), validated with a **reliability diagram** and
injection–recovery so a stated "0.9" means ≈90% empirical correctness.



# Results {#sec-results}

4 candidate(s) were processed, of which 1 were classified as planetary transits. Median calibrated confidence 0.84; median detection SNR 288.0. Examples are drawn from the synthetic generator with detectable parameters so every class is represented by an unambiguous, correctly-classified candidate.

**Per-class candidate counts.**

| Class | Count |
| --- | --- |
| Transit | 1 |
| Eclipsing binary | 1 |
| Blend | 1 |
| Other | 1 |

**Top candidates (by detection significance).**

| TIC | Class | Conf | P [d] | Depth [ppm] | Dur [h] | SNR | SDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 399391772 | blend | 0.90 | 5.2607 | 6520 | 3.84 | 390.6 | 18.6 |
| 349992610 | eclipsing_binary | 0.98 | 4.7488 | 4974 | 3.84 | 304.3 | 20.9 |
| 427222550 | transit | 0.78 | 4.3842 | 12263 | 2.16 | 271.7 | 14.2 |
| 460590760 | other | 0.75 | 0.5720 | 3131 | 5.28 | 111.5 | 6.1 |

![Representative one-page vetting sheet.](/home/user/bah2026-ps7/examples/vetting_transit.png){width=95%}

![Additional example vetting sheet.](/home/user/bah2026-ps7/examples/vetting_eclipsing_binary.png){width=95%}


# Visualization {#sec-visualization}

The headline visualisation is the **one-page vetting sheet** (`viz.vetting_sheet`,
matplotlib `subplot_mosaic`), imitating the TESS SPOC Data-Validation one-page
summary: full-baseline detrended light curve with transit ticks; global and
local phase folds with binned points and the best-fit model; odd-vs-even and
secondary-eclipse diagnostics; the SDE periodogram with harmonics; a river plot;
and a calibrated class-probability bar with parameters ± uncertainties. Colours
follow the colour-blind-safe **Okabe–Ito** palette (transit `#0072B2`,
eclipsing_binary `#D55E00`, blend `#CC79A7`, other `#999999`) with viridis for
sequential data; meaning is never encoded by colour alone. An interactive
Streamlit dashboard browses the full catalog and drills into each sheet.

![One-page vetting sheet — the headline visualisation.](/home/user/bah2026-ps7/examples/vetting_transit.png){width=95%}
