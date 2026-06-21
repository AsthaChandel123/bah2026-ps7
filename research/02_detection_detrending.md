# PS7 Domain Report — Signal Detrending + Transit Detection / Period-Finding

**Scope:** Robust, fast, scalable detrending and transit/period-search algorithms for noisy, crowded-field TESS light curves (~20–30k targets/sector). For each method: math/intuition, exact Python library + function call, key parameters/hyperparameters, computational complexity, and noise-robustness notes. Ends with a master comparison table and runnable reference pipelines (detrend → BLS → TLS).

> Author note (PS7 team — Detection & Detrending domain). Everything below is implementable with open-source Python: `wotan`, `astropy.timeseries`, `transitleastsquares`, `lightkurve`, `celerite2`, `scipy`, `numpy`. Deep-learning detectors are summarized only and handed off to the Classification team.

---

## 0. TESS data facts that drive algorithm choice

- **Cadences:** 2-min (SPOC PDCSAP), 20-sec (fast), and 30-min/10-min/200-sec FFIs (SPOC/QLP/eleanor). A 27.4-day sector at 2 min ≈ ~20k cadences; FFI 30-min ≈ ~1.3k cadences/sector.
- **Dominant nuisance signals:** stellar variability (rotation/pulsation, hours→days), scattered light (Earth/Moon, strong near perigee), **momentum dumps** (~2.5 days early mission, every ~3.125 days; cause flux jumps/ramps), and **crowding/blending** (TESS pixels are 21″ — aperture contamination from neighbors is the rule, not the exception, hence the need for eclipsing-binary/blend discrimination).
- **Gaps:** mid-sector data downlink gap (~1 day) + quality-flag masking → all folding/period methods must handle gaps; FFT/ACF methods need gap-filling or are penalized.
- **Implication:** Detrending must remove variability/systematics on timescales **longer than a transit** while *preserving the transit shape*; detection must be gap-tolerant, work at low SNR, and be cheap enough to run 20–30k times per sector.

---

# A. Detrending / Noise Removal

Goal: flatten out-of-transit baseline to ~1.0 (normalized) without "eating" or distorting transits. The cardinal rule (Hippke et al. 2019, *wotan*): **set the smoothing window to ≈ 3× the longest transit duration you search**, so the filter passes variability on long timescales but is nearly transparent to the transit (a signal spanning 1/3 of the window is ~fully preserved).

### A1. Median / running-median filter
- **Math/intuition:** replace each point by the median of a sliding window; subtract/divide. Median is robust to outliers/in-transit points (unlike the mean).
- **Library/call:** `wotan.flatten(time, flux, window_length=W, method='median', return_trend=True)` (time-windowed) or `method='medfilt'` (cadence-windowed, `window_length` in points; wraps `scipy.signal.medfilt`). Also `scipy.ndimage.median_filter`.
- **Key params:** `window_length` (time units for `median`; odd integer #points for `medfilt`), `break_tolerance` (split at gaps), `edge_cutoff`.
- **Complexity:** O(N·W) naive; O(N log W) with sliding-window heaps.
- **Robustness:** Very robust to outliers; but a pure boxcar median can clip the bottom of *deep/long* transits and leaves "stair-step" artifacts. Use biweight/spline for shallow transits. Good cheap baseline/first pass.

### A2. Savitzky–Golay (Savgol)
- **Math/intuition:** least-squares fit of a low-order polynomial in a sliding window; preserves higher moments (peak height/width) better than a boxcar mean.
- **Library/call:** `scipy.signal.savgol_filter(flux, window_length, polyorder)`; `wotan.flatten(..., method='savgol', window_length=N_points, cval=polyorder)`; **`lightkurve.LightCurve.flatten(window_length=..., polyorder=2, niters=3, sigma=3)`** (its `flatten` *is* Savgol, splits on gaps, iteratively sigma-clips).
- **Key params:** `window_length` (odd # cadences), `polyorder` (2–3 typical), `niters`/`sigma` (lightkurve iterative clip to keep transits out of the fit).
- **Complexity:** O(N·k) (k = window size).
- **Robustness:** Not robust to outliers by itself → **always** combine with iterative sigma-clipping (lightkurve does). Can ring near sharp transit ingress/egress if window too short. Good for slowly varying stellar trends. **CDPP is conventionally measured on a Savgol-flattened curve** (see C3).

### A3. Biweight / Tukey (recommended default) and other robust M-estimators
- **Math/intuition:** time-windowed **iterative robust location** estimator. Tukey's biweight down-weights points by `(1-(r/(c·MAD))²)²` and zero-weights anything beyond `c` MADs, so in-transit points and outliers barely influence the baseline → the trend "rides over" the transit instead of through it. Solved by Newton–Raphson per window.
- **Library/call:** `wotan.flatten(time, flux, window_length=W, method='biweight', cval=5.0, edge_cutoff=0.5, break_tolerance=0.5, return_trend=True)`.
- **Related robust estimators in wotan** (swap `method=`): `andrewsinewave` (cval≈1.339), `welsch` (cval≈2.11), `huber`/`huber_psi` (cval≈1.5; quadratic-then-linear loss, less aggressive than biweight), `hampel`/`hampelfilt` (3-part redescending, `cval=(1.7,3.4,8.5)`), `tau`, `hodges`, `trim_mean`/`winsorize` (`proportiontocut`).
- **Key params:** `window_length` = **3× max transit duration** (Hippke 2019); `cval` (tuning constant in MAD units; 5 = default, ~4σ Gaussian → tunes robustness vs. efficiency); `break_tolerance` (gap split); `edge_cutoff`.
- **Complexity:** O(N·W·n_iter); n_iter small (converges fast). Vectorized C-accelerated in wotan; ~ms–s per TESS sector light curve.
- **Robustness:** **Best overall.** Hippke et al. (2019) report the time-windowed biweight recovers **99% (Kepler) / 94% (K2)** of the shallowest injected transits — the top performer in their benchmark. **This is the recommended default detrender for PS7.** For extreme young/active stars, switch to robust spline (A4).

### A4. Spline detrending (robust splines, p-spline, cofiam, cosine)
- **Math/intuition:** fit a smooth spline (piecewise polynomial with knots spaced ~`window_length`). Robustness via (a) iterative 2σ-clipping (`rspline`), (b) Huber loss (`hspline`), or (c) penalized splines with automatic knot # selection (`pspline`).
- **Library/call:**
  - `wotan.flatten(..., method='rspline', window_length=W)` — robust spline via sigma-clipping (remove >2σ, refit).
  - `wotan.flatten(..., method='hspline', window_length=W)` — Huber-loss spline (no `edge_cutoff`; use `break_tolerance`; `window_length` = knot spacing).
  - `wotan.flatten(..., method='pspline', max_splines=100, stdev_cut=2, return_nsplines=True)` — penalized B-spline, auto-selects #knots; great when you don't know the variability timescale.
  - `method='cofiam'` (Cosine Filtering with Autocorrelation Minimization — Kepler/HEK heritage), `method='cosine'` (`robust=True` → 2σ clip to convergence), `method='lowess'`, `method='supersmoother'`.
- **Key params:** `window_length`/knot spacing, `max_splines`, `stdev_cut`/`edge_cutoff`, `break_tolerance`.
- **Complexity:** O(N) to O(N·#knots) per iteration; pspline a bit heavier (knot optimization).
- **Robustness:** Splines flex to fast stellar variability better than fixed windows. **Recommended for young/active stars** (Hippke 2019 explicitly recommends robust Huber-estimator splines there). Risk: too-flexible spline (knots too dense) absorbs the transit → keep knot spacing ≥ 3× transit duration.

### A5. Gaussian Process (GP) detrending
- **Math/intuition:** model correlated noise (stellar variability/systematics) as a GP with a covariance kernel; condition on out-of-transit (or sigma-clipped) data; subtract/divide the predictive mean. Best when variability is **quasi-periodic** (rotation) and you want a principled noise model + uncertainty propagation. `celerite2` gives **O(N)** 1-D GP likelihoods (vs O(N³) generic GPs) — essential at TESS scale.
- **Library/call (`celerite2`, fast, recommended):**
  ```python
  import celerite2, numpy as np
  from celerite2 import terms
  kernel = terms.SHOTerm(sigma=σ, rho=ρ, tau=τ)            # damped SHO ≈ stellar granulation/variability
  # kernel = terms.RotationTerm(sigma=σ, period=Prot, Q0=1, dQ=1, f=0.5)  # rotation (two SHO mix)
  # kernel = terms.Matern32Term(sigma=σ, rho=ρ)            # generic red noise
  gp = celerite2.GaussianProcess(kernel, mean=np.median(flux))
  gp.compute(time, yerr=flux_err)                          # O(N) Cholesky
  mu = gp.predict(flux, t=time)                            # predictive mean (the trend)
  flat = flux / mu                                         # detrended
  ```
  - **`george`** (`george.GP(kernel)`, kernels `ExpSquaredKernel`, `Matern32Kernel`, `ExpSine2Kernel`) — O(N³) dense or HODLR solver; flexible but slower; fine for a few targets, **not** for 20–30k at full cadence.
  - **`wotan.flatten(..., method='gp', kernel='matern'|'squared_exp'|'periodic'|'periodic_auto', kernel_size=K, kernel_period=P, robust=True)`** — convenience GP detrend with built-in sigma-clip.
- **Key params:** kernel choice (SHO/Rotation for variability, Matern32 for generic red noise), `sigma` (amplitude), `rho`/`w0` (timescale/period), `Q` (quality factor), `tau` (damping); optimize hyperparameters by maximizing `gp.log_likelihood(flux)` (e.g. `scipy.optimize.minimize`). **Mask in-transit points or iterate** so the GP doesn't soak up the transit.
- **Complexity:** **celerite2 O(N) per likelihood**, ×N_opt optimization steps. george dense O(N³).
- **Robustness:** Excellent for quasi-periodic stellar activity and for **jointly** modeling transit + noise (avoids the "detrend-then-search" bias). Cons: must avoid overfitting transits (mask/iterate); hyperparameter optimization adds cost; over-flexible kernels distort transits.

### A6. Cotrending Basis Vectors (CBV)
- **Math/intuition:** the SPOC/PDC pipeline computes, per CCD per sector, the dominant **shared** systematic trends across many stars (SVD of an ensemble). Removing a fit of these basis vectors removes instrument-correlated systematics without needing pixel data.
- **Library/call:** `lightkurve` — `from lightkurve.correctors import CBVCorrector`; `cbv = CBVCorrector(lc); corrected = cbv.correct(...)`. CBV flavors: **Single-Scale**, **Multi-Scale** (band-split), **Spike** (single-cadence impulses). PDCSAP flux already has CBV-like correction applied.
- **Key params:** which CBV sets, number of vectors, fit regularization (`alpha`).
- **Complexity:** O(N·n_cbv) linear regression.
- **Robustness:** Removes *shared* systematics (scattered light, pointing) well; cannot remove star-specific stellar variability; over-fitting with too many CBVs can inject noise/eat real signals → use a few vectors + regularization.

### A7. Pixel Level Decorrelation (PLD)
- **Math/intuition:** spacecraft-motion/scattered-light systematics correlate with **how flux moves across pixels**. Build regressors from individual pixel time series (and their products for 2nd-order PLD), linear-regress them out of the aperture light curve.
- **Library/call:** `lightkurve` — `from lightkurve.correctors import PLDCorrector`; `pld = PLDCorrector(tpf); corrected = pld.correct(pld_order=2, pca_components=N)`. Built on `RegressionCorrector` + `DesignMatrix`. (FFI alternative: **`unpopular`/Causal Pixel Model**, ASCL; **`eleanor`**.)
- **Key params:** `pld_order` (1–3), `pca_components` (compress pixel regressors), aperture mask, spline component for residual trends.
- **Complexity:** O(N·n_reg²) for the regression (n_reg = #pixel regressors after PCA).
- **Robustness:** Very effective for motion/scattered-light systematics in crowded fields; needs **target pixel files** (not just light curves). Risk of overfitting → use PCA compression + regularization + mask transits.

### A8. Self-Flat-Fielding (SFF) — primarily K2
- **Math/intuition:** K2's two-wheel pointing drift makes flux depend on centroid position along an arc; SFF builds the **flux vs. arclength** correction and divides it out.
- **Library/call:** `lightkurve` — `from lightkurve.correctors import SFFCorrector`; `sff = SFFCorrector(lc); corrected = sff.correct(windows=..., bins=...)`.
- **Key params:** `windows` (time segments), `bins`, `polyorder`.
- **Complexity:** O(N).
- **Robustness:** Designed for K2 roll; for TESS, PLD/CBV are preferred (different systematics). Include for completeness/legacy.

### A9. Sigma-clipping, outlier rejection, gap handling (preprocessing, applies to all)
- **Math/intuition:** iteratively remove points >Nσ from a robust center (median ± MAD) to kill cosmic rays, flares, and bad cadences before/after detrending. **Asymmetric clipping** (clip high outliers harder than low) avoids removing real transit dips.
- **Library/call:** `astropy.stats.sigma_clip(data, sigma=5, maxiters=5, cenfunc='median', stdfunc='mad_std')`; `lightkurve.LightCurve.remove_outliers(sigma_lower=20, sigma_upper=4)` (asymmetric → keep transits), `.remove_nans()`, quality-flag mask via `quality_bitmask='hard'`; `wotan` exposes `slide_clip` (time-windowed clip). **Gaps:** `break_tolerance` (wotan) / `flatten` auto-splits at gaps; momentum-dump cadences should be masked by quality flags before detrending.
- **Complexity:** O(N·maxiters).
- **Robustness:** Essential. **Clip *before* the period search** but **clip the low side gently** (e.g. `sigma_lower` large) so genuine transits survive. Always mask quality-flagged momentum dumps/scattered-light cadences.

### Detrending decision guide (PS7)
| Situation | Recommended detrender |
|---|---|
| Default / quiet–moderate stars, max throughput | **`wotan` biweight, window = 3× max transit duration** |
| Young/active, fast variability | `wotan` `pspline` or `hspline` (robust spline) |
| Quasi-periodic rotation, want noise model | `celerite2` SHO/Rotation GP (mask transits) |
| Shared instrument systematics (scattered light) | PDCSAP flux + `lightkurve` `CBVCorrector` |
| Motion/scattered light, have pixels | `lightkurve` `PLDCorrector` (FFI: `unpopular`) |
| K2 legacy | `lightkurve` `SFFCorrector` |
| Always (pre-step) | quality-flag mask + asymmetric `sigma_clip`/`remove_outliers` |

---

# B. Period Search / Transit Detection (≥8 methods)

### B1. Box Least Squares (BLS)
- **Math/intuition:** model the folded light curve as a periodic rectangular dip (box) with 4 params: period, duration, depth, epoch (t0). For each trial period, fold, slide a box of each trial duration, and find the placement that **maximizes the likelihood / Signal Residue** = best separation of in-transit vs out-of-transit weighted means. Optimizes a χ² statistic (or SNR objective). Foundational (Kovács, Zucker & Mazeh 2002).
- **Library/call (astropy, recommended):**
  ```python
  from astropy.timeseries import BoxLeastSquares
  import numpy as np, astropy.units as u
  bls = BoxLeastSquares(time*u.day, flux, dy=flux_err)
  durations = np.array([0.05,0.075,0.1,0.15,0.2,0.3])*u.day   # grid of trial durations
  pg = bls.autopower(durations, minimum_period=0.5, maximum_period=15,
                     minimum_n_transit=2, frequency_factor=3.0, objective='snr')
  i = np.argmax(pg.power)
  period, t0 = pg.period[i], pg.transit_time[i]
  duration, depth = pg.duration[i], pg.depth[i]
  stats = bls.compute_stats(period, duration, t0)   # depth_odd/even, harmonic_*, per_transit_*
  ```
  Other impls: `lightkurve.LightCurve.to_periodogram(method='bls', ...)` (wraps astropy), classic `eebls`/`bls.py`, GPU `cuvarbase` (B-extra).
- **Period grid design (critical — BLS is grid-sensitive):** astropy `autoperiod` uses frequency spacing **`df = frequency_factor · min(duration) / (max(t)−min(t))²`**, `maximum_period = (max(t)−min(t)) / minimum_n_transit`, `minimum_period = 2·max(duration)`. Smaller `frequency_factor` → finer grid (better recovery, slower). Use a **uniform-in-frequency** grid (transit phase error ∝ frequency), never uniform-in-period.
- **Key params:** `durations` grid (cover expected R_p/orbits), `minimum/maximum_period`, `frequency_factor` (1–5; lower=finer), `objective` ('likelihood' default or 'snr'), `minimum_n_transit` (≥2 for TESS single sector).
- **Complexity:** O(N_period · N_duration · N_bins). FFT-free but embarrassingly parallel over periods. astropy core is C-compiled.
- **Robustness:** Robust, fast, ubiquitous baseline. Box model is a poor match to real (limb-darkened, ingress/egress) transits → ~10–25% lower recovery and lower significance than TLS for small planets. Sensitive to red noise/variability if not detrended. **Use as fast first-pass / sanity check; confirm with TLS.**

### B2. Transit Least Squares (TLS) — recommended primary detector
- **Math/intuition:** like BLS but the template is a **real, limb-darkened transit** with proper ingress/egress (not a box). For >99.9% of known planets a transit fits better than a box → higher significance, ~**+10% detection efficiency at the same false-alarm rate** (Hippke & Heller 2019). Reports **SDE** (signal detection efficiency) as the significance metric.
- **Library/call (`transitleastsquares`):**
  ```python
  from transitleastsquares import transitleastsquares
  model = transitleastsquares(time, flat_flux)             # time in days, normalized flux ~1
  results = model.power(period_min=0.5, period_max=15,
                        n_transits_min=2, oversampling_factor=3,
                        duration_grid_step=1.1, transit_template='default',  # 'grazing'|'box'
                        use_threads=4, R_star=1.0, M_star=1.0, show_progress_bar=True)
  results.period, results.SDE, results.T0, results.duration, results.depth
  results.snr, results.FAP, results.odd_even_mismatch, results.transit_times
  results.periods, results.power            # full SDE periodogram for plotting
  ```
- **Key params:** `period_min/max`, `n_transits_min` (≥2), `oversampling_factor` (2–5; period grid density), `duration_grid_step` (1.05–1.15), `transit_template` ('default' grazing/box variants), `R_star`/`M_star` (+ `_min/_max`) → uses **physical priors** to build a tighter, faster duration grid; `use_threads` for parallelism; `limb_dark`/`u` (limb-darkening). Provides an optimal **period grid** internally (Ofir 2014 / Hippke 2019: `dP ∝ P^(4/3)` from physically allowed durations).
- **Complexity:** Comparable to BLS in practice (~10 s for a K2/TESS light curve on a laptop) thanks to numba/C acceleration + physical grid pruning; heavier than BLS per evaluation but smarter grid.
- **Robustness:** **Best small-planet sensitivity**, better significance, built-in odd/even & FAP diagnostics → ideal for PS7's classify/depth/period/duration deliverables. Needs a **flattened** light curve (run after detrending). Slightly slower than BLS — use BLS to triage, TLS to confirm/measure, or run TLS directly if compute allows.

### B3. Lomb–Scargle periodogram (LS)
- **Math/intuition:** least-squares fit of a sinusoid at each trial frequency for unevenly sampled data; power = variance reduction. Detects **sinusoidal** variability (rotation, pulsation, EB ellipsoidal/sinusoidal) — *not* optimized for narrow transits, but invaluable for **measuring/removing stellar variability** and flagging EBs.
- **Library/call:** `from astropy.timeseries import LombScargle`; `freq, power = LombScargle(time, flux, dy).autopower(minimum_frequency=..., maximum_frequency=..., samples_per_peak=10)`; `LombScargle(...).false_alarm_probability(power.max())`.
- **Key params:** `samples_per_peak` (≥5–10), `nyquist_factor`, `minimum/maximum_frequency`, `normalization` ('standard'/'model'/'log'/'psd'), multi-term via `fit_mean`/`nterms`.
- **Complexity:** O(N log N) with the fast (Press & Rybicki) method.
- **Robustness:** Great for periodic *variability*; narrow transits put power in many harmonics → poor standalone transit detector. **Use it to find the rotation period and feed the GP/biweight window, and to catch EB/variable contaminants** (a strong LS sinusoid at the BLS/TLS period ⇒ likely variable/EB, not a planet).

### B4. Phase Dispersion Minimization (PDM)
- **Math/intuition:** fold at trial period, bin in phase, compute **Θ = (variance within phase bins) / (total variance)**. True period ⇒ tight phase curve ⇒ small bin variance ⇒ Θ ≪ 1 (a *minimum*, not a peak). Non-sinusoidal- and gap-friendly (Stellingwerf 1978).
- **Library/call:** `PyAstronomy.pyTiming.pyPDM` (`Scanner`, `PyPDM`); `Py-PDM` (Cython wrapper of Stellingwerf C — much faster); GPU `cuvarbase` PDM2; `stingray` PDM.
- **Key params:** number of phase bins `Nb`, bin overlap `Nc`, period/frequency scan range.
- **Complexity:** O(N_period · N).
- **Robustness:** Handles arbitrary (non-sinusoidal, EB, transit) shapes and gaps; good cross-check for BLS/TLS periods and for EB folding. Statistic noisier than χ² methods; choose enough bins to resolve the transit but not so many that bins empty.

### B5. Analysis of Variance (AoV) and String-Length minimization
- **AoV (Schwarzenberg-Czerny 1989):** like PDM but uses an **ANOVA F-statistic** across phase bins (statistically more powerful / better-behaved noise than PDM). Multiharmonic AoV (`AoVMHW`) fits Fourier series per period.
  - **Library/call:** `pyaov` (Schwarzenberg-Czerny's `aov`, `aovmh`, `aovw`); `P4J`; some in `astrobase.periodbase`. Complexity O(N_period·N). Robust, powerful for non-sinusoidal signals.
- **String-Length (Dworetsky 1983 / Lafler–Kinman):** fold, sort by phase, **minimize total length of the line connecting consecutive (phase, mag) points**; correct period ⇒ smooth curve ⇒ short string.
  - **Library/call:** `astrobase.periodbase.stringlength` / `P4J`; easy to hand-roll: `L = Σ sqrt(Δphase² + Δmag²)`. Complexity O(N_period·N log N) (the sort). Robust to outliers if magnitudes normalized; best for sparse/sharp signals, weaker at low SNR than χ² methods.

### B6. Autocorrelation Function (ACF) for period
- **Math/intuition:** correlate the (evenly-sampled) light curve with time-lagged copies of itself; **peaks at lags = period and its multiples**. McQuillan et al. (2013/2014) used ACF to measure 34,030 Kepler rotation periods. Primary use here: **rotation period → detrending window & EB period sanity check**; also detects repeating transit dips.
- **Library/call:** `astropy.timeseries.LombScargle` companions; direct `numpy.correlate`/`scipy.signal.correlate` on interpolated, evenly-sampled flux; `statsmodels.tsa.acf`; `lightkurve` `to_periodogram` + custom; irregular-sampling variant **S-ACF** (Kreutzer et al. 2023, MNRAS). Smooth ACF with a Gaussian, take the power spectrum of the ACF, pick the dominant lag.
- **Key params:** lag grid, interpolation/gap-fill, Gaussian smoothing width, peak-selection (first significant peak).
- **Complexity:** O(N log N) via FFT-based autocorrelation.
- **Robustness:** Excellent, robust **rotation-period** finder (more reliable than LS for spotted stars); needs even sampling (interpolate across gaps). Not a primary transit detector but key for crowding/variability disambiguation.

### B7. Matched filtering / template matching
- **Math/intuition:** the **optimal linear detector for a known signal in (Gaussian) noise**. Convolve the (whitened) flux with a transit template `T`; detection statistic `Z(t0) = Σ T·d / sqrt(Σ T·Σ⁻¹·T)`. With a single transit shape this is what Kepler's TPS does (single-event statistic → MES, see C2); TLS is effectively a periodic matched filter over a transit template grid.
- **Library/call:** roll-your-own with `scipy.signal.correlate`/`fftconvolve` + a transit template (e.g. from `batman-package` `TransitModel`); periodic version = **TLS** (B2); Kepler-style **wavelet-whitened** matched filter (Jenkins 2002) → see C2 (MES). `astropy.convolution` for the convolution kernel.
- **Key params:** template duration(s)/depth/limb-darkening, noise whitening (divide by CDPP/red-noise model), threshold.
- **Complexity:** O(N log N) per template (FFT convolution) × N_templates × N_periods.
- **Robustness:** Theoretically optimal at known shape; requires good noise whitening (correlated noise breaks the optimality unless modeled). In practice deployed as TLS or as Kepler/TPS-style wavelet-domain matched filter.

### B8. Wavelet transforms / FFT-based detection
- **Math/intuition:** (a) **FFT** of evenly-sampled flux → quick periodicity/harmonic screen and red-noise spectral characterization (granulation slope), not great for narrow transits (power spread across harmonics). (b) **Wavelet transform** localizes signals in time *and* scale → ideal to separate slow stellar variability (large scale) from transit-scale dips and from white noise; Kepler TPS whitens noise in the wavelet domain (Jenkins 2002) before matched filtering; **Wavelet-Adaptive matched Filter (WAF)** combines both.
- **Library/call:** `numpy.fft`/`scipy.fft`; `scipy.signal.cwt`, `PyWavelets (pywt)` (`pywt.wavedec`, `pywt.cwt`); `astropy` convolution.
- **Key params:** wavelet family (Haar/Daubechies/Morlet), decomposition levels/scales, FFT windowing & gap-filling.
- **Complexity:** FFT O(N log N); discrete wavelet O(N).
- **Robustness:** Wavelets are powerful for **multi-scale denoising / whitening** (great preprocessing for matched filter), and naturally model TESS red noise. Pure FFT is a weak standalone transit detector but cheap for harmonic/alias screening. Needs gap handling.

### B9. (Bonus) Deep-learning detectors on raw/flattened flux
- **Math/intuition:** 1-D CNNs / transformers / contrastive models learn the transit morphology directly, robust to red noise and capable of finding shallow transits; often run **after** BLS/TLS (on folded views) or directly on flux for triage.
- **Examples (hand off detail to Classification team):** **Astronet** (Shallue & Vanderburg 2018, global+local folded views), **ExoMiner**, **GPFC** (GPU Phase Folding + CNN, arXiv:2312.02063 — fast at scale), **DELOS** (contrastive learning, shallow Kepler transits), transformer-based TESS-FFI candidate ID (MNRAS 2025). **Astronet/`exominer`** packages; build with `tensorflow`/`pytorch`.
- **Complexity:** GPU training cost upfront; inference O(N) per light curve — **scales to 20–30k easily** on GPU.
- **Robustness:** Strong on red noise/shallow signals; needs labeled training data and careful false-positive control. **Recommended as a downstream ranker/vetter, not the sole detector.** Detection-domain handoff: provide BLS/TLS folded views + SDE/SNR features as model inputs.

---

# C. Significance / SNR Metrics

### C1. Signal Detection Efficiency (SDE) — TLS/BLS periodogram significance
- **Formula:** `SDE = (P(f_max) − ⟨P⟩) / σ(P)` — how many standard deviations the peak periodogram power (TLS: the χ²-based Signal Residue spectrum, often median-smoothed) sits above the mean of the spectrum.
- **Thresholds (white noise, ~27-day baseline):** **SDE ≳ 7 ⇒ ~1% false-positive rate**; **SDE = 9 ⇒ FAP ≈ 0.01%** (Hippke & Heller 2019). Caveat: calibrated for short (≤~28 d) baselines and white noise; **longer/red-noise data inflate SDE** even for false positives → recalibrate per-survey (bootstrap, below).
- **Where:** `results.SDE` (TLS); for astropy BLS compute it from `pg.power`. Use SDE as the primary candidate-ranking score in PS7.

### C2. Multiple Event Statistic (MES) — Kepler/TPS-style
- **Formula/intuition:** combine **single-event statistics** (SES, the matched-filter detection statistic of each individual transit, computed in the wavelet-whitened domain) across all transits: roughly `MES = Σ SES_i / sqrt(N_transits)` (the folded matched-filter SNR). Kepler's **Threshold Crossing Event** cut is **MES ≥ 7.1σ** (one expected statistical false alarm in ~10⁵ quiet stars).
- **Where:** the Kepler pipeline (`Kepler-TPS`); reproduce with a wavelet-whitened matched filter (B7/B8) or read off TLS `snr`/per-transit SNR (`results.snr_per_transit`). Useful as a second, physically-grounded SNR alongside SDE.

### C3. CDPP — Combined Differential Photometric Precision
- **Formula/intuition:** the effective **white-noise level seen by a transit of duration τ** — i.e. RMS scatter after removing long-term trends, measured on the relevant transit timescale (Christiansen et al. 2012). Converts photometric scatter into a per-transit noise floor.
- **Where:** `lightkurve.LightCurve.estimate_cdpp(transit_duration=13, savgol_window=101, savgol_polyorder=2, sigma=5.0)` (Savgol-detrend then RMS, in **ppm**). Use the CDPP at your transit duration as σ in the SNR formula below.

### C4. Transit SNR (the workhorse number for PS7)
- **Formula:** `SNR = depth / (σ / sqrt(N_in_transit))` for a single binned depth; for a **multi-transit** detection:
  `SNR = (depth / CDPP(τ)) · sqrt(N_transits)` = `((R_p/R_★)² / CDPP(τ)) · sqrt(N_transits)`.
  Equivalent per-point form: `SNR = depth · sqrt(N_in_transit) / σ_point`, where `N_in_transit = (duration/cadence)·N_transits`.
- **Where:** `results.snr` (TLS, full); BLS `pg.depth_snr` and `bls.compute_stats(...)['depth_snr']`; or compute directly from `depth`, `CDPP`, `N_transits`. **PS7 should report this SNR plus SDE per candidate.**

### C5. False-Alarm Probability (FAP)
- **Analytic LS:** `LombScargle.false_alarm_probability(power_peak, method='baluev')` (also 'davies','bootstrap','naive').
- **TLS:** `results.FAP` (lookup-table-calibrated against SDE for white noise).
- **Bootstrap / extreme-value (recommended for real TESS red noise):** scramble or phase-randomize the flux many times, rerun BLS/TLS, build the distribution of peak SDE; fit a **Generalized Extreme Value (GEV)** model to bootstrapped peaks and read off the FAP of the observed SDE. This **recalibrates thresholds for your noise/baseline** (addresses the SDE-inflation caveat in C1). Cost: ×(n_bootstrap) runs → use the cheap BLS for bootstrapping.

---

# D. Robustness & Scaling (running on 20–30k light curves/sector)

### D1. Make the search fast
- **Use compiled/vectorized cores:** astropy BLS (C), TLS (numba/C), `wotan` (C) — avoid pure-Python loops.
- **Sensible period grids (biggest single lever):** uniform **in frequency**; `dν` from astropy's `df = frequency_factor·min(duration)/baseline²` or TLS's physical grid (`dP ∝ P^(4/3)`). Start coarse (`frequency_factor`/`oversampling_factor` low) for triage, refine around peaks. Restrict `period_max` to `baseline/n_transits_min` (need ≥2 transits in a sector).
- **Prune durations with physics:** TLS `R_star`/`M_star` (or stellar-density prior) restricts durations to physically allowed values → fewer evaluations. cuvarbase's `use_fast` mode does the same Keplerian-duration pruning.
- **GPU:** **`cuvarbase`** (PyCUDA) BLS/LS/PDM/CE — **1–2 orders of magnitude** faster than CPU; a 70k-point Kepler curve: 30 s default vs **1.05 s** `use_fast`. QLP uses GPU BLS for the whole TESS FFI search (arXiv:2302.01293). GPU phase-folding + CNN (**GPFC**, arXiv:2312.02063) for end-to-end.
- **Multiprocessing/cluster:** the search is **embarrassingly parallel over targets** — `multiprocessing.Pool`/`joblib.Parallel`/`concurrent.futures` across the 20–30k light curves; TLS also threads internally (`use_threads`). Use `dask`/Slurm array jobs for full-sector runs. Down-bin/limit cadence (e.g. FFI 30-min) for first-pass triage.
- **Two-stage pipeline:** **BLS (fast) to triage** → keep top-N by SDE/SNR → **TLS (accurate) to confirm + measure** depth/period/duration with uncertainties + odd/even/FAP. Cuts total cost dramatically vs TLS-on-everything.

### D2. Multi-sector stitching
- TESS observes many targets in multiple sectors. **Detrend each sector independently** (systematics differ per sector/CCD/orientation), normalize each to 1.0, then **`lightkurve.LightCurveCollection.stitch()`** (handles per-sector normalization) before the period search → longer baseline ⇒ more transits ⇒ higher SNR/SDE and access to longer periods. Watch for per-sector flux offsets and gaps; mask momentum-dump/scattered-light cadences in each sector first.

### D3. Harmonics & aliases
- BLS/TLS/PDM often light up at **P/2, 2P, 3P** and the ~13.7-day TESS orbit / momentum-dump cadence and 1-day aliases. Mitigate: inspect the **odd/even depth test** (`results.odd_even_mismatch`, BLS `depth_odd/even`) to catch EBs masquerading as half-period planets; check the **harmonic_delta_log_likelihood** (BLS `compute_stats`) vs a sinusoid; phase-fold candidate and alternates; cross-check with PDM/ACF; explicitly veto known systematic periods (orbit period, momentum-dump interval, integer days).

### D4. Don't get fooled by stellar variability or momentum dumps
- **Detrend on the right timescale:** window/knot spacing = 3× max transit duration (preserves transits, removes slower variability). For strong rotation, model it (GP/biweight) and **check LS/ACF**: a dominant sinusoid at the candidate period ⇒ likely variability/EB, not a planet.
- **Momentum dumps / scattered light:** mask via **quality flags** (`quality_bitmask='hard'`) before detrending; the ~2.5–3.125-day jump cadence is a classic false-period trap → veto it.
- **Crowding/blending (TESS-specific):** large pixels ⇒ flux contamination. Use **odd/even test, secondary-eclipse search, depth-vs-aperture (centroid) checks, and difference-imaging/centroid offset** to separate true on-target transits from blended EBs — feed these flags to the Classification team. (TLS `transit_depths`/odd-even and BLS `depth_odd/even`, `depth_half` give the first cuts.)
- **Red-noise-aware thresholds:** use bootstrap/GEV FAP (C5) and report **both SDE and physical SNR**; never trust a fixed SDE cut blindly on multi-sector/red-noise data.

---

# Master comparison table

### Detrending methods
| Method | Library / call | Complexity | Noise/transit robustness | Recommended use |
|---|---|---|---|---|
| Running median | `wotan.flatten(method='median'/'medfilt')`, `scipy.ndimage.median_filter` | O(N·W)→O(N log W) | Outlier-robust; can clip deep transits, stair-steps | Cheap first pass |
| Savitzky–Golay | `scipy.signal.savgol_filter`; `lightkurve.flatten`; `wotan method='savgol'` | O(N·k) | Preserves peak shape; **not** outlier-robust (add sigma-clip) | Smooth trends; CDPP baseline |
| **Biweight/Tukey** | **`wotan.flatten(method='biweight', window=3×dur, cval=5)`** | O(N·W·iter) | **Top recovery (99%/94%)**, rides over transits | **Default detrender** |
| Robust spline / pspline | `wotan method='rspline'/'hspline'/'pspline'` | O(N·#knots) | Flexes to fast variability; risk eating transit | Young/active stars |
| GP (celerite2) | `celerite2.GaussianProcess(SHO/Rotation/Matern32)` | **O(N)** /opt step | Best for quasi-periodic noise; mask transits | Rotation, joint noise model |
| GP (george) | `george.GP(Matern32/ExpSine2)` | O(N³) dense | Flexible, slow | Few targets only |
| CBV | `lightkurve.CBVCorrector` | O(N·n_cbv) | Removes shared systematics, not stellar var | PDCSAP scattered-light residuals |
| PLD | `lightkurve.PLDCorrector` (FFI: `unpopular`) | O(N·n_reg²) | Strong on motion/scattered light; needs pixels | Crowded fields, TPFs |
| SFF | `lightkurve.SFFCorrector` | O(N) | K2 roll only | K2 legacy |
| Sigma-clip / gaps | `astropy.stats.sigma_clip`, `lc.remove_outliers`, `break_tolerance` | O(N·iters) | Essential; clip low side gently to keep transits | Always (pre/post) |

### Period-search / detection methods
| Method | Library / function | Complexity | Robustness | Recommended use |
|---|---|---|---|---|
| **BLS** | `astropy.timeseries.BoxLeastSquares.autopower`; `lightkurve to_periodogram('bls')` | O(N_P·N_dur·N_bin) | Fast, grid-sensitive; box ≠ real transit | **Fast triage / first pass** |
| **TLS** | **`transitleastsquares.transitleastsquares().power()`** | ~BLS (numba), smart grid | **+10% recovery, best small planets**, SDE+diagnostics | **Primary detector + measurement** |
| Lomb–Scargle | `astropy.timeseries.LombScargle.autopower` | O(N log N) | Sinusoidal only; weak on narrow transits | Variability/rotation, EB flag |
| PDM | `PyAstronomy.pyPDM`, `Py-PDM`, `cuvarbase` PDM2 | O(N_P·N) | Any shape, gap-friendly; noisier stat | Cross-check, EB folding |
| AoV | `pyaov` (`aov`/`aovmh`), `astrobase` | O(N_P·N) | ANOVA-powerful, non-sinusoidal | Robust period confirm |
| String-length | `astrobase.stringlength`, `P4J` | O(N_P·N log N) | Outlier-tolerant, sparse data | Sparse/sharp signals |
| ACF | `numpy/scipy.correlate`, `statsmodels.acf`, S-ACF | O(N log N) | Robust rotation period; needs even sampling | Rotation, crowding disambig. |
| Matched filter | `scipy.signal.fftconvolve` + `batman` template | O(N log N)·N_tmpl | Optimal at known shape; needs whitening | = TLS / Kepler-TPS style |
| Wavelet / FFT | `pywt`, `scipy.signal.cwt`, `numpy.fft` | O(N log N) | Multi-scale denoise/whiten; FFT weak alone | Preprocessing, harmonic screen |
| Deep learning | Astronet/ExoMiner/GPFC (`tf`/`pytorch`) | O(N) inference (GPU) | Strong on red noise/shallow; needs labels | Downstream vetter/ranker |
| GPU BLS | `cuvarbase` BLS (`use_fast`) | O(N_P·N_dur)/GPU | 1–2 orders faster | Full-sector scale-out |

### Significance / SNR metrics
| Metric | Definition | Tool | Threshold/notes |
|---|---|---|---|
| **SDE** | `(P_max−⟨P⟩)/σ(P)` | `results.SDE` (TLS); from `pg.power` (BLS) | SDE≈7→1% FPR; 9→0.01%; recalibrate for red noise |
| MES | folded matched-filter SNR `Σ SES/√N_tr` | Kepler-TPS; TLS per-transit SNR | TCE cut **7.1σ** |
| CDPP | white-noise floor at duration τ (ppm) | `lc.estimate_cdpp(transit_duration=...)` | use as σ in SNR |
| Transit SNR | `depth/(σ/√N_in) = (depth/CDPP)·√N_tr` | `results.snr` (TLS); `pg.depth_snr` (BLS) | report per candidate |
| FAP | prob. of noise peak ≥ observed | `LombScargle.false_alarm_probability`; `results.FAP`; bootstrap/GEV | bootstrap for real TESS noise |

---

# Recommended default detection pipeline (PS7)

**Per target:** quality-mask + asymmetric sigma-clip → **biweight detrend (window = 3× max duration)** → **BLS triage** → **TLS confirm/measure** → record SDE, SNR, FAP, period/depth/duration (+ uncertainties), odd/even & secondary diagnostics → hand candidates + folded views to the Classification team. Stitch multi-sector before search. Parallelize over the 20–30k targets with `joblib`/`multiprocessing` (or `cuvarbase` BLS on GPU for the triage stage).

### Reference implementation (runnable)

```python
"""PS7 reference: detrend -> BLS triage -> TLS confirm, with significance metrics."""
import numpy as np
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.timeseries import BoxLeastSquares
from wotan import flatten
from transitleastsquares import transitleastsquares, transit_mask

# ---- 0. Inputs: time (days), flux (e-/s or normalized), flux_err -------------
# time, flux, flux_err = load_tess_lightcurve(...)  # e.g. via lightkurve PDCSAP

# ---- 1. Clean: drop NaNs, gently clip outliers (asymmetric keeps transits) ---
m = np.isfinite(time) & np.isfinite(flux)
time, flux = time[m], flux[m]
flux = flux / np.nanmedian(flux)
clip = sigma_clip(flux, sigma_lower=20, sigma_upper=4, maxiters=5, cenfunc='median')
time, flux = time[~clip.mask], flux[~clip.mask]

# ---- 2. Detrend: time-windowed biweight, window = 3x max searched duration ---
MAX_DURATION = 0.3                       # days (longest transit you search)
WINDOW = 3.0 * MAX_DURATION              # Hippke+2019 rule
flat_flux, trend = flatten(
    time, flux, method='biweight',
    window_length=WINDOW, cval=5.0,
    edge_cutoff=0.5, break_tolerance=0.5, return_trend=True)
ok = np.isfinite(flat_flux)
time, flat_flux = time[ok], flat_flux[ok]

# ---- 3. BLS triage: fast first-pass period/duration estimate -----------------
durations = np.array([0.05, 0.075, 0.1, 0.15, 0.2, 0.3]) * u.day
bls = BoxLeastSquares(time * u.day, flat_flux)
pg = bls.autopower(durations, minimum_period=0.5, maximum_period=15.0,
                   minimum_n_transit=2, frequency_factor=3.0, objective='snr')
i = np.argmax(pg.power)
bls_period   = pg.period[i].value
bls_t0       = pg.transit_time[i].value
bls_duration = pg.duration[i].value
bls_depth    = pg.depth[i]
bls_sde      = (pg.power[i] - np.mean(pg.power)) / np.std(pg.power)
bls_stats    = bls.compute_stats(pg.period[i], pg.duration[i], pg.transit_time[i])
print(f"[BLS] P={bls_period:.5f} d  t0={bls_t0:.4f}  dur={bls_duration:.3f} d  "
      f"depth={bls_depth:.5f}  SDE~{bls_sde:.1f}")

# ---- 4. TLS confirm + measure (limb-darkened template, SDE/SNR/FAP) ----------
tls = transitleastsquares(time, flat_flux)
res = tls.power(
    period_min=0.5, period_max=15.0, n_transits_min=2,
    oversampling_factor=3, duration_grid_step=1.1,
    transit_template='default',
    R_star=1.0, M_star=1.0,          # plug in real stellar params for tighter grid
    use_threads=4, show_progress_bar=False)

print(f"[TLS] P={res.period:.5f} +/- {res.period_uncertainty:.5f} d")
print(f"      t0={res.T0:.4f}  duration={res.duration:.3f} d  depth={res.depth:.5f}")
print(f"      SDE={res.SDE:.2f}  SNR={res.snr:.2f}  FAP={res.FAP:.2e}")
print(f"      odd_even_mismatch={res.odd_even_mismatch:.2f} sigma  "
      f"n_transits={res.distinct_transit_count}")

# ---- 5. Candidate gate + iterative multi-planet search -----------------------
DETECTED = (res.SDE >= 7.0) and (res.snr >= 5.0)   # tune on injection tests
print("CANDIDATE" if DETECTED else "no significant signal")

# mask this transit and re-search for additional planets:
# intransit = transit_mask(time, res.period, 2*res.duration, res.T0)
# res2 = transitleastsquares(time[~intransit], flat_flux[~intransit]).power(...)
```

### Multi-sector stitching + parallel scale-out (sketch)

```python
import lightkurve as lk
from joblib import Parallel, delayed

# stitch sectors (each normalized to 1.0) for a longer baseline before searching
sr  = lk.search_lightcurve("TIC 307210830", mission="TESS", author="SPOC")
lcc = sr.download_all()                      # LightCurveCollection
lc  = lcc.stitch().remove_nans().remove_outliers(sigma_lower=20, sigma_upper=4)
time, flux = lc.time.value, lc.flux.value

# scale across 20-30k targets (embarrassingly parallel over light curves)
def search_one(t, f):                        # returns (period, SDE, SNR, FAP, ...)
    ff, _ = flatten(t, f, method='biweight', window_length=0.9, return_trend=True)
    r = transitleastsquares(t, ff).power(period_min=0.5, period_max=15,
                                         n_transits_min=2, use_threads=1)
    return dict(period=r.period, SDE=r.SDE, snr=r.snr, FAP=r.FAP,
                depth=r.depth, duration=r.duration, T0=r.T0)
# results = Parallel(n_jobs=-1)(delayed(search_one)(t, f) for t, f in all_curves)
# For the triage stage at full-sector scale, swap in cuvarbase GPU BLS (use_fast).
```

### GP detrend alternative (active/rotating stars)

```python
import celerite2, numpy as np
from celerite2 import terms
from scipy.optimize import minimize

# (optional) mask in-transit points first so the GP does not absorb the transit
kernel = terms.RotationTerm(sigma=np.std(flux), period=Prot_guess, Q0=1.0, dQ=1.0, f=0.5)
gp = celerite2.GaussianProcess(kernel, mean=np.median(flux))

def neg_ll(p):
    gp.kernel = terms.RotationTerm(sigma=np.exp(p[0]), period=np.exp(p[1]),
                                   Q0=np.exp(p[2]), dQ=np.exp(p[3]), f=p[4])
    gp.compute(time, yerr=flux_err, quiet=True)
    return -gp.log_likelihood(flux)

# soln = minimize(neg_ll, x0, method="L-BFGS-B")   # fit hyperparameters
gp.compute(time, yerr=flux_err)
trend = gp.predict(flux, t=time)                    # O(N) predictive mean
flat_flux = flux / trend                            # detrended -> feed to BLS/TLS
```

---

# Key references / URLs

- wotan (detrending benchmark, biweight, splines, GP): https://wotan.readthedocs.io/en/latest/Usage.html · https://github.com/hippke/wotan · paper arXiv:1906.00966 https://arxiv.org/abs/1906.00966 · IOP https://iopscience.iop.org/article/10.3847/1538-3881/ab3984
- Transit Least Squares (TLS): https://transitleastsquares.readthedocs.io/ · https://github.com/hippke/tls · paper arXiv:1901.02015 https://arxiv.org/abs/1901.02015
- astropy BLS: https://docs.astropy.org/en/stable/timeseries/bls.html · API (autoperiod grid, compute_stats) https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html
- astropy Lomb–Scargle: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
- celerite2 (fast GP, SHO/Rotation/Matern32): https://celerite2.readthedocs.io/en/latest/api/python/
- george GP: https://george.readthedocs.io/
- lightkurve (flatten/CBV/PLD/SFF/CDPP/stitch): https://lightkurve.github.io/lightkurve/ · CBV tutorial https://lightkurve.github.io/lightkurve/tutorials/2-creating-light-curves/2-3-how-to-use-cbvcorrector.html · estimate_cdpp https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.estimate_cdpp.html
- PDM: https://pyastronomy.readthedocs.io/en/latest/pyTimingDoc/pyPDMDoc/pdm.html · Py-PDM https://github.com/ckm3/Py-PDM
- AoV / string-length / period methods: https://astrobase.readthedocs.io/
- ACF rotation (McQuillan): https://iopscience.iop.org/article/10.1088/0067-0049/211/2/24 · S-ACF https://doi.org/10.1093/mnras/stad1223
- CDPP definition (Christiansen 2012): https://iopscience.iop.org/article/10.1086/668847
- Kepler TPS / MES / 7.1σ (Jenkins): https://ntrs.nasa.gov/api/citations/20110010911/downloads/20110010911.pdf
- GPU BLS (cuvarbase): https://cuvarbase.readthedocs.io/en/latest/bls.html · https://github.com/johnh2o2/cuvarbase · QLP GPU search arXiv:2302.01293 https://arxiv.org/pdf/2302.01293
- GPU phase-folding + CNN (GPFC): https://arxiv.org/pdf/2312.02063
- Two-periodogram (BLS vs TLS) study: https://arxiv.org/html/2308.04282
