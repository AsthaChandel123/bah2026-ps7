# 06 — Scientific Visualization, Vetting Reports, Dashboards & Final Report

**BAH 2026 — Problem Statement 7: AI-enabled Detection of Exoplanets from Noisy TESS Light Curves**

Domain: *Scientific visualization, vetting-report generation, interactive dashboards, and the ≤3-page methodology report.*

This document is an **implementation-ready reference** for the visualization & reporting layer of the pipeline. It maps directly onto the PS7 deliverables:

- "Visualization of the light curve along with the detected and classified astrophysical signal" → the **one-page vetting sheet** (§A, §1).
- "Provide the confidence level of the detected signal" → **calibrated probability + SNR/SDE panel** (§D).
- "A report (max 3 pages) … methodology, assumptions, tools/libraries, uncertainties" → **§E report template + toolchain**.
- Evaluation rewards **"Visualization and clarity"** → everything here is optimized for that line.

---

## TL;DR — Recommended Stack

| Concern | Recommendation | Why |
|---|---|---|
| Static figures / vetting sheet | **`matplotlib`** (`subplot_mosaic`) + `lightkurve` built-in plotters | Publication quality, full control, embeds in PDF/report |
| Periodogram & phase-fold | `lightkurve` (`to_periodogram`, `.fold`, `plot_river`) + **`transitleastsquares`** (TLS) for SDE & odd/even | Battle-tested transit-specific tooling |
| Aperture / blend vetting | **`tpfplotter`** (Gaia overlay) + in/out-of-transit difference image | Standard ExoFOP-style contamination check |
| Posterior / parameter uncertainty | **`corner`** (corner.py) | De-facto standard for MCMC posteriors |
| Catalog browser (20–30k candidates) | **`streamlit`** (primary) — table → click → vetting sheet | Fastest to build, perfect for a demo; Dash is the fallback if scale/perf bites |
| Color & accessibility | **Okabe–Ito** (categorical classes) + **viridis/cividis** (sequential) | Colorblind-safe, print-safe, Nature-recommended |
| Report rendering | **Quarto** (`.qmd` → PDF) primary; LaTeX/Markdown→pandoc fallback | One-command reproducible PDF, auto-embeds figures+tables, page control |
| Machine-readable output | **one JSON + one CSV row per candidate** (TIC, P, depth, dur ± err, SNR, class, confidence, flags) | Mirrors TOI/ExoFOP schema |

---

## A) Light-curve & Transit Visualizations — What Professional Vetting Reports Show

The gold standard is the **TESS SPOC Data Validation (DV) report**, produced for every Threshold Crossing Event (TCE). SPOC emits three artifacts: a **full report PDF**, a **mini report PDF**, and a **one-page "Report Summary" PDF** per TCE, plus a results XML/FITS ([TESS DV products, HEASARC](https://heasarc.gsfc.nasa.gov/docs/tess/data-validation-products-updated-for-sector-66.html); [Guerrero et al. 2021, TOI Catalog](https://arxiv.org/pdf/2103.12538)). Our per-target "vetting sheet" deliberately imitates the one-page summary. Below is each panel, what it shows, and how to produce it.

### A.1 Raw vs detrended flux time series with transit markers
- **Top, full-width.** Plot SAP/PDCSAP (raw) and the detrended/flattened flux as a time series. Mark each transit mid-time with a downward triangle ("blue triangles" in the DV summary), and mark **sector boundaries** with vertical dashed red lines ([Guerrero et al. 2021](https://arxiv.org/pdf/2103.12538)).
- `lightkurve`: `lc.plot()` for raw; `lc.flatten(window_length=901).remove_outliers()` then `.plot()` for detrended ([Lightkurve transit tutorial](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html)). For detrending use `lightkurve.flatten` (Savitzky–Golay) or **`wotan`** (`flatten(..., method='biweight')`, [Hippke et al. 2019, Wotan](https://arxiv.org/pdf/1906.00966)).

### A.2 Phase-folded light curve — global + zoomed local view, with best-fit model & binned points
- The DV convention: fold on the candidate period; **black points** = all data, **cyan points** = data binned at 1/5 of the fitted transit duration, **red line** = transit model fit ([Guerrero et al. 2021](https://arxiv.org/pdf/2103.12538)).
- **Global** view spans full phase (−0.5…0.5) to expose out-of-transit structure (e.g. secondary eclipse); **local/zoomed** view spans ±2–3 durations around phase 0.
- `lightkurve`: `lc.fold(period=P, epoch_time=t0).scatter()`; overlay the model via `bls.get_transit_model(period=P, transit_time=t0, duration=dur)` then `model.fold(P, t0).plot(ax=ax, c='r', lw=2)`; restrict with `ax.set_xlim(...)` ([Lightkurve tutorial](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html)). For a physical model use the **TLS** `results.model_folded_phase` / `results.model_folded_model`, or **`batman`** for a limb-darkened Mandel–Agol model.

### A.3 Odd vs even transits side-by-side; secondary-eclipse phase view
- **Odd/even test (EB discriminator):** fold odd-numbered and even-numbered transits separately and plot side by side. A statistically significant depth difference flags an **eclipsing binary** (different primary/secondary depths). The DV report prints "the significance of the difference between the odd and even transits" in the panel title ([search synthesis of TESS DV figures](https://arxiv.org/pdf/2103.12538)). TLS directly returns `results.odd_even_mismatch` (in σ), plus per-event depths ([TLS docs](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html); [Hippke & Heller 2019](https://arxiv.org/pdf/1901.02015)).
- **Secondary eclipse:** zoom the global fold around **phase 0.5**. A detected dip at ~0.5 with non-negligible depth indicates a self-luminous/large companion → EB or hot-Jupiter occultation, not a small planet.

### A.4 BLS / TLS periodogram with period & harmonics marked; SDE / power
- Plot power (BLS) or **SDE — Signal Detection Efficiency** (TLS) vs period. Mark the peak period and its **harmonics/aliases** (P/2, 2P, 3P) with vertical lines.
- BLS: `lc.to_periodogram(method='bls', period=np.linspace(1,20,10000), frequency_factor=500)`; `bls.plot()`; read `bls.period_at_max_power`, `bls.transit_time_at_max_power`, `bls.duration_at_max_power` ([Lightkurve tutorial](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html)).
- TLS: `model = transitleastsquares(t, flux); results = model.power()`; plot `results.periods` vs `results.power`; detection threshold typically **SDE > 7–9** (range 6–10 cited) ([TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html); [Hippke & Heller 2019](https://arxiv.org/pdf/1901.02015)). TLS is more sensitive than BLS for shallow, small-planet transits because it fits a realistic limb-darkened transit shape rather than a box.

### A.5 "River" / waterfall plot (transits stacked by epoch)
- 2-D image: each **row = one orbital cycle**, **columns = phase**, **color = flux**. Reveals transit-timing variations (TTVs), missed/odd transits, and whether the signal persists across the full baseline.
- `lightkurve`: `lc.plot_river(period=P, epoch_time=t0, bin_points=10, method='mean', cmap='viridis')` ([Lightkurve `plot_river` API](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.plot_river.html); [river-plot tutorial](https://colab.research.google.com/github/lightkurve/lightkurve/blob/main/docs/source/tutorials/3-science-examples/exoplanets-visualizing-periodic-signals-using-a-river-plot.ipynb)). Signature: `plot_river(period, epoch_time=None, ax=None, bin_points=1, minimum_phase=-0.5, maximum_phase=0.5, method='mean', **kwargs)`.

### A.6 Centroid / in-vs-out-of-transit difference image; sky aperture with Gaia overlay (blend vetting)
- **Difference image (centroid test):** subtract the mean **in-transit** pixel image from the mean **out-of-transit** image. If the transit signal's photocenter is offset from the target star, the dip originates from a **nearby/background blended source** (false positive). DV computes the photocenter by fitting the TESS pixel response function and reports RA/Dec offsets per sector (green crosses), mean offset (magenta cross), target (red star), and a 3σ uncertainty circle ([synthesis of TESS DV centroid figures](https://arxiv.org/pdf/2103.12538)).
- **Sky aperture + Gaia sources (`tpfplotter` style):** plot the Target Pixel File with the SPOC aperture mask overplotted and **Gaia DR3** sources as magnitude-scaled markers to spot contaminants inside the aperture ([jlillo/tpfplotter](https://github.com/jlillo/tpfplotter); [Aller et al. 2020, A&A 635, 128](https://www.aanda.org); [ASCL 2504.018](https://www.ascl.net/2504.018)). CLI: `python tpfplotter.py <TIC> --maglim 6 [--sector N] [--SAVEGAIA] [--PM]`. Dependencies: `numpy>1.20`, `matplotlib>3.2`, `astropy>4.2`, `lightkurve>2.0`.
- Compute in/out-of-transit images directly from a `lightkurve.TargetPixelFile` by masking cadences by phase and differencing the pixel arrays; render with `tpf.plot()` / `imshow` and overlay the aperture (`tpf.pipeline_mask`).

### A.7 Corner / posterior plots for fitted parameters (`corner.py`)
- After MCMC/NUTS fitting (e.g. `emcee`, or `PyMC`+`exoplanet`), draw the joint+marginal posteriors to communicate **parameter covariance and uncertainty**: `corner.corner(samples, labels=[...], truths=[...], quantiles=[0.16,0.5,0.84], show_titles=True)` ([corner.py](https://corner.readthedocs.io/); [exoplanet transit-fit tutorial](https://docs.exoplanet.codes/en/v0.5.0/tutorials/transit/)). Typical fit params: period, t0, Rp/R★, impact parameter b, a/R★, limb-darkening (u1,u2). The 16/50/84th percentiles give the reported value ± asymmetric errors.

### A.8 Per-target one-page "vetting sheet"
Combine A.1–A.7 into a single multi-panel figure mirroring the **DV one-page Report Summary** ([TESS DV one-page summary](https://heasarc.gsfc.nasa.gov/docs/tess/data-validation-products-updated-for-sector-66.html)). A community open-source analogue is [SLSkrzypinski/TESS_diagnosis](https://github.com/SLSkrzypinski/TESS_diagnosis), which generates a one-page PDF per TIC (TPF + SAP/PDCSAP LC + periodogram + phase fold). **Standard layout** (top→bottom, our recommendation):

```
 ┌──────────────────────────── header: TIC, sector, class + confidence, P/depth/dur ± err ─────────────────────────────┐
 │ [ full-baseline detrended light curve, transit ticks, sector dashes ]                                   (full width) │
 ├───────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┤
 │ [ phase fold GLOBAL + model + binned ]         │ [ phase fold LOCAL/zoom + model + binned ]                            │
 ├───────────────────────┬───────────────────────┼───────────────────────────────┬─────────────────────────────────────┤
 │ [ odd transits ]      │ [ even transits ]      │ [ BLS/TLS periodogram, SDE,   │ [ river / waterfall plot ]          │
 │  (depth + Δσ)         │  (depth + Δσ)          │   period + harmonics marked ] │                                     │
 ├───────────────────────┴───────────────────────┼───────────────────────────────┴─────────────────────────────────────┤
 │ [ TPF + aperture + Gaia overlay (tpfplotter) ] │ [ in/out-of-transit difference image + centroid offset ]              │
 ├───────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┤
 │ [ class-probability bar (transit/EB/blend/other) ]  |  [ secondary-eclipse @phase 0.5 ]  |  [ vetting-flags table ]   │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Matplotlib code skeleton — one-page vetting sheet

Uses `subplot_mosaic` (ASCII-art layout, built on `gridspec`, supports `height_ratios`/`width_ratios`/`empty_sentinel`; [subplot_mosaic docs](https://matplotlib.org/stable/users/explain/axes/mosaic.html)).

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # (subplot_mosaic uses gridspec under the hood)

# ---- Okabe-Ito class colors (colorblind-safe) ----
CLASS_COLORS = {
    "transit": "#0072B2",   # blue
    "EB":      "#D55E00",   # vermillion
    "blend":   "#CC79A7",   # reddish-purple
    "other":   "#999999",   # grey
}

def vetting_sheet(cand, lc_raw, lc_flat, fold_global, fold_local,
                  odd, even, periodogram, tpf_img, diff_img, probs,
                  outfile=None):
    """
    cand        : dict of candidate metadata + fitted params (see §F schema)
    lc_*        : lightkurve LightCurve objects (raw, detrended)
    fold_global : (phase, flux, model_phase, model_flux, bin_phase, bin_flux)
    fold_local  : same tuple, zoomed to +/- N durations
    odd, even   : (phase, flux, depth, depth_err) for odd / even transits
    periodogram : (period_array, power_or_sde, peak_period, harmonics_list, sde)
    tpf_img     : (image_2d, aperture_mask, gaia_xy, gaia_mag)
    diff_img    : (diff_image_2d, centroid_offset_arcsec, offset_sigma)
    probs       : dict {class_name: calibrated_probability}
    """
    fig = plt.figure(figsize=(8.27, 11.69), layout="constrained")  # A4 portrait
    mosaic = """
        TTTT
        GGLL
        OEPP
        OERR
        AADD
        CCSF
    """
    ax = fig.subplot_mosaic(
        mosaic,
        height_ratios=[1.1, 1.4, 1.0, 1.0, 1.5, 0.9],
    )

    # ---- Header text ----
    cls = cand["class"]; conf = cand["confidence"]
    fig.suptitle(
        f"TIC {cand['tic_id']}  |  Sector {cand['sector']}  |  "
        f"CLASS: {cls.upper()}  (conf {conf:.2f})\n"
        f"P = {cand['period']:.5f} ± {cand['period_err']:.5f} d   "
        f"depth = {cand['depth_ppm']:.0f} ± {cand['depth_ppm_err']:.0f} ppm   "
        f"dur = {cand['duration_hr']:.2f} ± {cand['duration_hr_err']:.2f} h   "
        f"SNR = {cand['snr']:.1f}",
        fontsize=10, color=CLASS_COLORS.get(cls, "k"), fontweight="bold",
    )

    # ---- T: full-baseline detrended LC with transit ticks + sector dashes ----
    a = ax["T"]
    a.plot(lc_flat.time.value, lc_flat.flux.value, ".", ms=1, color="0.3", rasterized=True)
    for tt in cand["transit_times"]:
        a.plot(tt, lc_flat.flux.value.min(), marker="v", color=CLASS_COLORS["transit"], ms=6)
    for sb in cand.get("sector_bounds", []):
        a.axvline(sb, color="red", ls="--", lw=0.8, alpha=0.6)
    a.set(xlabel="Time [BTJD]", ylabel="Norm. flux", title="Detrended light curve")

    # ---- G / L: phase fold global + local, with binned points + model ----
    for key, data, span in (("G", fold_global, None), ("L", fold_local, (-2, 2))):
        ph, fl, mph, mfl, bph, bfl = data
        a = ax[key]
        a.plot(ph, fl, ".", ms=1, color="0.6", alpha=0.4, rasterized=True)   # all (black)
        a.plot(bph, bfl, "o", ms=4, color="#56B4E9", label="binned")          # binned (cyan)
        a.plot(mph, mfl, "-", color="red", lw=1.6, label="model")            # model (red)
        a.set(xlabel="Phase", ylabel="Norm. flux",
              title="Phase fold (global)" if key == "G" else "Phase fold (zoom)")
        if span: a.set_xlim(*span)
    ax["G"].legend(fontsize=7, loc="lower right")

    # ---- O / E: odd vs even ----
    for key, (ph, fl, d, derr), ttl in (("O", odd, "Odd"), ("E", even, "Even")):
        a = ax[key]
        a.plot(ph, fl, ".", ms=2, color="0.4", rasterized=True)
        a.axhline(1 - d * 1e-6, color="red", lw=1)
        a.set(xlabel="Phase", title=f"{ttl}  (depth {d:.0f}±{derr:.0f} ppm)")
        a.set_xlim(-2, 2)
    # annotate odd/even mismatch significance (EB flag)
    ax["O"].text(0.02, 0.05, f"odd-even Δ = {cand['odd_even_sigma']:.1f}σ",
                 transform=ax["O"].transAxes, fontsize=7,
                 color="red" if cand['odd_even_sigma'] > 3 else "0.3")

    # ---- P: periodogram (BLS power or TLS SDE) ----
    per, pwr, ppk, harmonics, sde = periodogram
    a = ax["P"]
    a.plot(per, pwr, "-", color="0.2", lw=0.8)
    a.axvline(ppk, color=CLASS_COLORS["transit"], lw=1.2, label=f"P={ppk:.4f} d")
    for h in harmonics:
        a.axvline(h, color="orange", ls=":", lw=0.8, alpha=0.7)
    a.set(xlabel="Period [d]", ylabel="SDE / power", xscale="log",
          title=f"Periodogram (SDE={sde:.1f})")
    a.legend(fontsize=7)

    # ---- R: river / waterfall ----
    a = ax["R"]
    # lc_flat.plot_river(period=ppk, epoch_time=cand['t0'], ax=a,
    #                    bin_points=10, method='mean', cmap='viridis')
    a.set_title("River plot")

    # ---- A: TPF + aperture + Gaia (tpfplotter-style) ----
    img, apmask, gxy, gmag = tpf_img
    a = ax["A"]
    a.imshow(img, origin="lower", cmap="viridis")
    a.contour(apmask, levels=[0.5], colors="white", linewidths=1.2)   # aperture outline
    a.scatter(gxy[:, 0], gxy[:, 1], s=200 / (gmag - gmag.min() + 1),
              edgecolor="red", facecolor="none")                       # Gaia sources
    a.set_title("Aperture + Gaia (blend check)")

    # ---- D: in/out-of-transit difference image + centroid ----
    dimg, off_arcsec, off_sig = diff_img
    a = ax["D"]
    im = a.imshow(dimg, origin="lower", cmap="RdBu_r")
    a.plot(dimg.shape[1] / 2, dimg.shape[0] / 2, "*", color="red", ms=12)  # target
    a.set_title(f"Diff image  (centroid off {off_arcsec:.1f}\", {off_sig:.1f}σ)")
    fig.colorbar(im, ax=a, fraction=0.046)

    # ---- C: class-probability bar ----
    a = ax["C"]
    names = list(probs.keys()); vals = [probs[n] for n in names]
    a.barh(names, vals, color=[CLASS_COLORS.get(n, "0.5") for n in names])
    a.set_xlim(0, 1); a.set_xlabel("Calibrated P(class)")
    a.set_title("Classification confidence")
    for i, v in enumerate(vals):
        a.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)

    # ---- S: secondary eclipse @ phase 0.5 ----
    a = ax["S"]
    phg, flg = fold_global[0], fold_global[1]
    m = np.abs(phg - 0.5) < 0.15
    a.plot(phg[m], flg[m], ".", ms=2, color="0.4")
    a.axvline(0.5, color="red", ls="--", lw=0.8)
    a.set(xlabel="Phase", title="Secondary eclipse")

    # ---- F: vetting-flags table ----
    a = ax["F"]; a.axis("off")
    flags = cand.get("vetting_flags", {})
    rows = [[k, "PASS" if v else "FLAG"] for k, v in flags.items()]
    tbl = a.table(cellText=rows, colLabels=["Test", "Result"], loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    a.set_title("Vetting flags")

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    return fig
```

> Notes: use `rasterized=True` on dense scatter points so the saved PDF stays small; `dpi=150` for screen, `dpi=300` if a panel goes into the printed report. `lc.plot_river(...)` plots directly into a supplied `ax`. Swap the placeholder `imshow` aperture call for a real `tpfplotter` figure if you precompute it (it produces its own standalone figure, so for a unified sheet either embed its saved PNG via `ax.imshow(plt.imread(...))` or reproduce its drawing with `astropy.wcs` + Gaia query).

---

## B) Libraries & Tradeoffs

| Library | Role | Strengths | Limitations |
|---|---|---|---|
| **matplotlib** | All static figures, the vetting sheet | Total control, `subplot_mosaic`/`gridspec`, vector PDF, universal | Verbose; not interactive |
| **lightkurve** | LC I/O + built-in plotters (`plot`, `scatter`, `fold`, `to_periodogram`, `plot_river`, `flatten`) | Astronomy-native, returns matplotlib axes (composable), TPF handling | Heavy dependency; opinionated objects |
| **transitleastsquares (TLS)** | SDE periodogram, model, odd/even, per-transit depths | Transit-shaped template → best for shallow/small planets; rich diagnostics | Slower than BLS on huge grids |
| **wotan** | Detrending before plotting | Many robust methods (biweight, GP, splines) | Just detrending |
| **seaborn** | Statistical summary plots across the *catalog* (SNR vs depth, class distributions, `pairplot`, KDE) | Beautiful defaults, one-liners on DataFrames | Not for per-cadence LC; wraps matplotlib |
| **corner** | Posterior corner plots for fitted params | The standard for MCMC uncertainty viz | Single purpose |
| **tpfplotter** | TPF + aperture + Gaia overlay (blend vetting) | Paper-ready, ExoFOP-standard contamination check | CLI-oriented; emits own figure |
| **plotly / bokeh / holoviews** | Interactivity (hover, zoom, linked brushing) | Web-native, embeds in dashboards/HTML report | Larger payloads; PDF export weaker |

**Rule of thumb:** matplotlib + lightkurve for the per-candidate sheet and the PDF report; seaborn for catalog-level statistics; plotly inside the dashboard for interactive drill-down.

References: [Lightkurve docs](https://lightkurve.github.io/lightkurve/), [TLS docs](https://transitleastsquares.readthedocs.io/), [Wotan paper](https://arxiv.org/pdf/1906.00966), [corner.py](https://corner.readthedocs.io/), [tpfplotter](https://github.com/jlillo/tpfplotter).

---

## C) Interactive Dashboard — Browsing 20–30k Results

### Recommendation: **Streamlit** (primary), **Plotly Dash** as the fallback if scale/performance becomes a problem.

**Why Streamlit for this hackathon:** fastest path from a Python script to a working app; pure `.py`, no callback boilerplate; ideal "running today to show stakeholders" tool ([Quansight: Dash/Voila/Panel/Streamlit](https://quansight.com/post/dash-voila-panel-streamlit-our-thoughts-on-the-big-four-dashboarding-tools/); [UI Bakery: Streamlit vs Dash](https://uibakery.io/blog/streamlit-vs-dash)).

**The one caveat — scale.** Streamlit reruns the whole script per interaction and is weaker than Dash on very large datasets / many concurrent users; Dash uses server-side callbacks + `DataTable` pagination and is the better choice for enterprise-scale, high-row-count apps ([UI Bakery](https://uibakery.io/blog/streamlit-vs-dash); [Quansight](https://quansight.com/post/dash-voila-panel-streamlit-our-thoughts-on-the-big-four-dashboarding-tools/)). **Mitigation that makes 30k rows trivial in Streamlit:** the heavy artifacts are precomputed — the dashboard only reads a **summary catalog (CSV/Parquet, one row per candidate)** and loads the **pre-rendered vetting-sheet PNG** on click. Use `@st.cache_data` for the catalog, server-side filtering/sorting on the DataFrame, and pagination. With that pattern, Streamlit handles 20–30k rows comfortably. (Panel is the alternative if you want notebook-native multi-page apps; Dash if you outgrow Streamlit.)

### Layout sketch

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SIDEBAR (filters)            │   MAIN                                          │
│  • Sector select              │   ┌── Candidate table (sortable/paginated) ──┐ │
│  • Class checkboxes           │   │ TIC | P(d) | depth | dur | SNR | SDE |    │ │
│    [x]transit [ ]EB [ ]blend  │   │ class | conf | flags                     │ │
│  • SNR slider  (min–max)      │   │  … 20–30k rows, sort by SNR/score/class  │ │
│  • Confidence slider          │   └──────────────────────────────────────────┘ │
│  • Period range               │           ▼ (user clicks a row)                 │
│  • Vetting-flag toggles       │   ┌── Per-candidate DRILL-DOWN ──────────────┐ │
│  • Search by TIC              │   │  vetting sheet (PNG)  +  param table      │ │
│  • Download filtered CSV/JSON │   │  + interactive plotly phase-fold (zoom)   │ │
│                               │   │  + class-probability bar + flags          │ │
│                               │   └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Minimal Streamlit skeleton

```python
# app.py  — run: streamlit run app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PS7 Exoplanet Candidate Browser", layout="wide")

CLASS_COLORS = {"transit": "#0072B2", "EB": "#D55E00",
                "blend": "#CC79A7", "other": "#999999"}

@st.cache_data
def load_catalog(path="outputs/catalog.parquet"):
    return pd.read_parquet(path)   # one row per candidate (§F schema)

df = load_catalog()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
classes = st.sidebar.multiselect("Class", sorted(df["class"].unique()),
                                 default=sorted(df["class"].unique()))
snr_min, snr_max = st.sidebar.slider("SNR", float(df.snr.min()),
                                     float(df.snr.max()),
                                     (float(df.snr.min()), float(df.snr.max())))
conf_min = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.5, 0.01)
tic_query = st.sidebar.text_input("Search TIC")

mask = (df["class"].isin(classes) & df.snr.between(snr_min, snr_max)
        & (df.confidence >= conf_min))
if tic_query:
    mask &= df.tic_id.astype(str).str.contains(tic_query)
fdf = df[mask].sort_values("snr", ascending=False)

st.sidebar.download_button("Download filtered CSV",
                           fdf.to_csv(index=False), "candidates_filtered.csv")

# ---------------- Main: candidate table ----------------
st.title("Exoplanet Candidate Browser")
st.caption(f"{len(fdf):,} of {len(df):,} candidates match filters")

cols = ["tic_id", "period", "depth_ppm", "duration_hr",
        "snr", "sde", "class", "confidence"]
event = st.dataframe(
    fdf[cols], use_container_width=True, height=420,
    on_select="rerun", selection_mode="single-row",
    column_config={"confidence": st.column_config.ProgressColumn(
        "confidence", min_value=0.0, max_value=1.0, format="%.2f")},
)

# ---------------- Drill-down on selected row ----------------
sel = event.selection.rows
if sel:
    row = fdf.iloc[sel[0]]
    tic = int(row.tic_id)
    st.header(f"TIC {tic} — {row['class'].upper()} (conf {row.confidence:.2f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period [d]", f"{row.period:.5f}", f"± {row.period_err:.5f}")
    c2.metric("Depth [ppm]", f"{row.depth_ppm:.0f}", f"± {row.depth_ppm_err:.0f}")
    c3.metric("Duration [h]", f"{row.duration_hr:.2f}", f"± {row.duration_hr_err:.2f}")
    c4.metric("SNR / SDE", f"{row.snr:.1f}", f"SDE {row.sde:.1f}")

    left, right = st.columns([1.4, 1])
    with left:
        st.image(f"outputs/vetting_sheets/TIC{tic}.png",
                 caption="Vetting sheet", use_container_width=True)
    with right:
        # interactive phase-fold (precomputed arrays per candidate)
        ph = pd.read_parquet(f"outputs/folds/TIC{tic}.parquet")
        fig = px.scatter(ph, x="phase", y="flux", opacity=0.4,
                         render_mode="webgl",
                         title="Phase-folded (interactive)")
        fig.add_scatter(x=ph.model_phase, y=ph.model_flux, mode="lines",
                        line=dict(color="red"), name="model")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Class probabilities")
        probs = {c: row[f"p_{c}"] for c in CLASS_COLORS}
        st.bar_chart(pd.Series(probs))

        st.subheader("Vetting flags")
        st.json({k: bool(row[k]) for k in df.columns if k.startswith("flag_")})
```

> The dashboard reads only precomputed artifacts (catalog Parquet, per-candidate fold Parquet, vetting-sheet PNG). This decouples the (slow) pipeline from the (instant) UI and is what makes 20–30k rows responsive. `render_mode="webgl"` keeps plotly fast for dense scatters.

---

## D) Confidence & Communication

The PS7 deliverable *"confidence level of the detected signal"* should be shown **three complementary ways**, never one:

1. **Calibrated class probability** per candidate over `{transit, EB, blend, other}`. Use a horizontal **stacked or grouped bar** that sums to 1, with the predicted class highlighted. *Calibration matters:* a "0.9" must mean ~90% empirical correctness — apply **Platt scaling / isotonic regression / temperature scaling** to the classifier (`sklearn.calibration.CalibratedClassifierCV`) and show a **reliability diagram** in the report to prove it. Display the numeric value next to the bar.
2. **Detection significance:** SNR and **SDE** (TLS). Show the periodogram SDE peak and annotate the numeric SDE; flag against the SDE>7–9 convention ([TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html)).
3. **Parameter ± uncertainty:** every fitted parameter as `value ± err` (asymmetric `+σ/−σ` from the 16/50/84 posterior percentiles where MCMC is used), and the full covariance via the **corner plot**.

**Color & label conventions (consistent everywhere — sheet, dashboard, report):**
- **Categorical classes → Okabe–Ito** colorblind-safe palette (the Nature-recommended gold standard for categorical data): transit `#0072B2` (blue), EB `#D55E00` (vermillion), blend `#CC79A7` (reddish-purple), other `#999999` (grey) ([Okabe–Ito; PoS colour-blind guidelines](https://pos.sissa.it/guidelines.pdf)). Keep ≤6 categories — beyond six, hues become hard to distinguish ([Towards Data Science: accessible graphs](https://towardsdatascience.com/how-to-create-accessible-graphs-for-colorblind-people-295e517c9b15/)).
- **Sequential/continuous (heatmaps, river plot, diff image magnitude) → viridis or cividis** — perceptually uniform, colorblind-safe, print-safe; cividis looks near-identical to colorblind and non-colorblind viewers ([viridis/cividis discussion](https://adniaconseils.ca/en/the-viridis-color-palette-a-good-choice-in-data-visualization/); [Matplotlib colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)). For the diff image (diverging signed data) use a diverging map (`RdBu_r`).
- **Quick global switch:** `plt.style.use('tableau-colorblind10')` for a colorblind-safe categorical cycle ([Matplotlib 2.2 release notes](https://matplotlib.org/stable/users/prev_whats_new/whats_new_2.2.html)).

**Accessibility:** never encode meaning by color alone — pair color with **markers/linestyles/text labels** (e.g. model = red solid + "model" legend; flags shown as PASS/FLAG text and an icon, not just red/green). Annotate numbers directly on bars. This satisfies the rubric's "clarity" criterion and is robust for colorblind reviewers.

---

## E) The ≤3-Page Methodology Report

### Recommended structure (maps 1:1 to the rubric)

| § | Section | Content | ~Space |
|---|---|---|---|
| 1 | **Objective** | Restate: detect + classify transit/EB/blend/other in noisy TESS LCs; estimate P, depth, duration; give confidence. | 3–4 lines |
| 2 | **Data** | TESS sector high-cadence LCs from MAST (TIC/CTL, ~20–30k stars; [archive.stsci.edu/tess](https://archive.stsci.edu/tess/tic_ctl.html)); curated labeled training set (known planets / FPs / EBs). Cadence, sectors, preprocessing. | short ¶ |
| 3 | **Methodology** | (a) Detrending (`lightkurve`/`wotan`); (b) Transit search (BLS via `lightkurve`, TLS for SDE); (c) Classification (the AI model — features/CNN, training); (d) Parameter fitting (`batman`/`exoplanet` + MCMC). **Name every tool/library** (rubric requires it). 1–2 small flow figures. | ~1 page |
| 4 | **Assumptions** | e.g. single dominant transit per target in scope; PDCSAP systematics already corrected; box/limb-darkened model adequacy; SDE>threshold for detection; training labels trusted. | bullets |
| 5 | **Uncertainty estimation** | How errors are derived: posterior 16/50/84 percentiles (MCMC) → asymmetric ±; periodogram SDE/FAP for detection significance; **classifier-probability calibration** (Platt/isotonic + reliability diagram). | short ¶ |
| 6 | **Results** | Summary table (N detected per class, recovery/precision, example fitted params ± err); 1–2 representative vetting figures. | ~0.5 page |
| 7 | **Visualization** | One compact figure: the vetting sheet (or its key panels) — explicitly addresses the "Visualization and clarity" criterion. | figure |

### Reproducible render-to-PDF toolchain

**Primary: Quarto.** A single `.qmd` mixes narrative + Python code cells; Quarto runs the kernel, captures figures/tables/inline results, and Pandoc renders **PDF/HTML/docx with one command** — minimal config, "batteries included," and you can re-run end-to-end for reproducibility ([Quarto](https://quarto.org/); [Quarto + Python](https://quarto.org/docs/computations/python.html); [Reproducible reports with Jupyter+Quarto](https://www.jumpingrivers.com/blog/reproducible-reports-jupyter-quarto-python/)).

```yaml
---
title: "PS7 — AI Detection & Classification of Exoplanet Transits in TESS"
author: "Team …"
date: today
format:
  pdf:
    documentclass: article
    geometry: [margin=1.8cm]      # tighten margins to fit 3 pages
    fontsize: 9pt
    number-sections: true
    fig-pos: "H"                  # keep figures in place
execute:
  echo: false                     # hide code, show results
  warning: false
---
```
Render: `quarto render report.qmd --to pdf`. Tables auto-generate from DataFrames; figures embed from saved files or live code cells (`#| label: fig-vetting`, `#| fig-cap: ...`).

**Staying within 3 pages:** tighten `geometry` margins + `fontsize: 9pt`; one combined multi-panel figure instead of many; tables via `pandas.DataFrame.to_markdown()` / `tabulate` rather than verbose prose; move extended results to an appendix/repo that you reference but don't paginate. Check length after each render.

**Alternatives:**
- **Jupyter → `nbconvert` → PDF** (`jupyter nbconvert --to pdf report.ipynb`, via LaTeX): works, established, but less layout control and page management than Quarto ([reproducible Jupyter reports](https://python-bloggers.com/2023/09/reproducible-reports-with-jupyter/)).
- **Markdown/LaTeX → `pandoc` → PDF** (`pandoc report.md -o report.pdf --pdf-engine=xelatex`): maximal control over the 3-page constraint if you're comfortable with LaTeX ([Pandoc](https://pandoc.org/)). Pure **LaTeX** (e.g. AASTeX/article) gives the tightest control but is the least "reproducible-from-code".

Recommendation: **Quarto** for the best ratio of reproducibility, auto-embedding, and page control; LaTeX/pandoc as the fallback if a teammate prefers hand-tuned layout.

---

## F) Result Tables / Outputs

Emit both a **machine-readable** per-candidate record and a **human-readable** catalog. Schema mirrors the **TOI / ExoFOP** catalog so it's familiar to evaluators ([TOI column definitions, NExScI](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html); [Guerrero et al. 2021](https://arxiv.org/pdf/2103.12538)).

### Recommended output schema (one row/object per candidate)

| Column | Type | Unit | Meaning | TOI analogue |
|---|---|---|---|---|
| `tic_id` | int | — | TESS Input Catalog ID | `tid` |
| `sector` | int | — | TESS sector | — |
| `cand_id` | str | — | e.g. `TIC<id>.01` (multi-planet index) | `toi` |
| `period` | float | day | Orbital period | `pl_orbper` |
| `period_err` | float | day | 1σ period uncertainty | `pl_orbpererr1/2` |
| `t0` | float | BTJD | Transit epoch (mid-transit) | `pl_tranmid` |
| `t0_err` | float | day | 1σ epoch uncertainty | `pl_tranmiderr1/2` |
| `duration_hr` | float | hour | Transit duration | `pl_trandurh` |
| `duration_hr_err` | float | hour | 1σ duration uncertainty | `pl_trandurherr1/2` |
| `depth_ppm` | float | ppm | Transit depth | `pl_trandep` |
| `depth_ppm_err` | float | ppm | 1σ depth uncertainty | `pl_trandeperr1/2` |
| `rp_rstar` | float | — | Radius ratio Rp/R★ (if fitted) | — |
| `rp_rearth` | float | R⊕ | Planet radius | `pl_rade` |
| `snr` | float | — | Detection signal-to-noise | (~`Planet SNR`) |
| `sde` | float | — | TLS Signal Detection Efficiency | — |
| `odd_even_sigma` | float | σ | Odd/even depth-difference significance | — |
| `secondary_depth_ppm` | float | ppm | Secondary-eclipse depth @phase 0.5 | — |
| `class` | str | — | Predicted class: `transit`/`EB`/`blend`/`other` | (~`tfopwg_disp`) |
| `confidence` | float | — | Calibrated P(predicted class) | — |
| `p_transit` | float | — | Calibrated prob. transit | — |
| `p_eb` | float | — | Calibrated prob. eclipsing binary | — |
| `p_blend` | float | — | Calibrated prob. blend | — |
| `p_other` | float | — | Calibrated prob. other | — |
| `flag_odd_even` | bool | — | Pass odd/even depth test | — |
| `flag_secondary` | bool | — | No significant secondary eclipse | — |
| `flag_centroid` | bool | — | Centroid consistent with target (no offset) | (centroid offset flag) |
| `flag_contamination` | bool | — | No bright Gaia contaminant in aperture | — |
| `flag_vshape` | bool | — | Not V-shaped (planet-like, not grazing EB) | — |
| `st_tmag` | float | mag | TESS magnitude | `st_tmag` |
| `st_teff` | float | K | Stellar effective temperature | `st_teff` |
| `st_rad` | float | R☉ | Stellar radius | `st_rad` |
| `n_transits` | int | — | Number of observed transits | — |
| `vetting_sheet_path` | str | — | Path to the candidate's vetting PNG/PDF | — |

**Formats:**
- **Per-candidate JSON** (full provenance + nested arrays e.g. posterior summaries): `outputs/candidates/TIC<id>.json`.
- **Master CSV / Parquet** (one row per candidate, columns above) for the dashboard and the report's results table: `outputs/catalog.csv` / `catalog.parquet`. Parquet is preferred for the dashboard (fast, typed); CSV for the human-readable deliverable.
- **Human-readable catalog:** the CSV rendered as a sorted table (by SNR/confidence) in the report and exportable from the dashboard. Use `pandas` for round-tripping all three.

Example per-candidate JSON:
```json
{
  "tic_id": 307210830, "sector": 14, "cand_id": "TIC307210830.01",
  "period": 8.13821, "period_err": 0.00042,
  "t0": 1683.4521, "t0_err": 0.0015,
  "duration_hr": 3.21, "duration_hr_err": 0.08,
  "depth_ppm": 1450, "depth_ppm_err": 60,
  "rp_rstar": 0.0381, "rp_rearth": 2.4,
  "snr": 12.7, "sde": 14.2, "odd_even_sigma": 0.6,
  "secondary_depth_ppm": 30,
  "class": "transit", "confidence": 0.93,
  "probs": {"transit": 0.93, "EB": 0.04, "blend": 0.02, "other": 0.01},
  "flags": {"odd_even": true, "secondary": true, "centroid": true,
            "contamination": true, "vshape": true},
  "vetting_sheet_path": "outputs/vetting_sheets/TIC307210830.png"
}
```

---

## Sources

- TESS / vetting & DV reports: [Guerrero et al. 2021 — TOI Catalog (arXiv:2103.12538)](https://arxiv.org/pdf/2103.12538); [TESS DV products, HEASARC](https://heasarc.gsfc.nasa.gov/docs/tess/data-validation-products-updated-for-sector-66.html); [TOI Release Notes (MIT)](https://tess.mit.edu/toi-releases/toi-release-notes/); [TOI column definitions (NExScI)](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html); [PS7 data source: archive.stsci.edu/tess/tic_ctl](https://archive.stsci.edu/tess/tic_ctl.html).
- Lightkurve: [Identifying transiting planet signals tutorial](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html); [`plot_river` API](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.plot_river.html); [river-plot tutorial](https://colab.research.google.com/github/lightkurve/lightkurve/blob/main/docs/source/tutorials/3-science-examples/exoplanets-visualizing-periodic-signals-using-a-river-plot.ipynb).
- TLS / detrending: [Hippke & Heller 2019 — TLS (arXiv:1901.02015)](https://arxiv.org/pdf/1901.02015); [TLS docs/FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html); [Hippke et al. 2019 — Wotan (arXiv:1906.00966)](https://arxiv.org/pdf/1906.00966).
- Aperture/blend & posteriors: [jlillo/tpfplotter](https://github.com/jlillo/tpfplotter); [tpfplotter ASCL 2504.018](https://www.ascl.net/2504.018); [corner.py docs](https://corner.readthedocs.io/); [exoplanet transit-fit tutorial](https://docs.exoplanet.codes/en/v0.5.0/tutorials/transit/); community one-pager [SLSkrzypinski/TESS_diagnosis](https://github.com/SLSkrzypinski/TESS_diagnosis).
- Dashboards: [Quansight — Dash/Voila/Panel/Streamlit](https://quansight.com/post/dash-voila-panel-streamlit-our-thoughts-on-the-big-four-dashboarding-tools/); [UI Bakery — Streamlit vs Dash](https://uibakery.io/blog/streamlit-vs-dash); [Streamlit](https://streamlit.io/).
- Color/accessibility: [Okabe–Ito & colour-blind guidelines (PoS)](https://pos.sissa.it/guidelines.pdf); [viridis/cividis](https://adniaconseils.ca/en/the-viridis-color-palette-a-good-choice-in-data-visualization/); [Matplotlib colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html); [accessible graphs (TDS)](https://towardsdatascience.com/how-to-create-accessible-graphs-for-colorblind-people-295e517c9b15/); [Matplotlib `subplot_mosaic`](https://matplotlib.org/stable/users/explain/axes/mosaic.html).
- Reporting: [Quarto](https://quarto.org/); [Quarto + Python](https://quarto.org/docs/computations/python.html); [Reproducible Jupyter+Quarto reports](https://www.jumpingrivers.com/blog/reproducible-reports-jupyter-quarto-python/); [Pandoc](https://pandoc.org/).
