"""Periodic-transit search / period-finding for ``exopipe`` (PS7 stage SEARCH).

Implements the detection methods of research dossier
`02_detection_detrending.md` §B and the two-stage orchestration of §D, returning
a fully-populated :class:`~exopipe.types.DetectionResult` (period, t0, duration,
depth, SDE, SNR, the full periodogram, and harmonics).

Methods (``search(..., method=...)``)
-------------------------------------
* ``'bls'`` -- Box Least Squares. Uses ``astropy.timeseries.BoxLeastSquares``
  (lazy) when available, else a **vectorised pure-NumPy BLS** over a
  physically-spaced, uniform-in-frequency period grid. **Primary fast triage.**
* ``'tls'`` -- Transit Least Squares. Uses ``transitleastsquares`` (lazy) for the
  limb-darkened template + odd/even/SDE/SNR diagnostics, falling back to BLS when
  the package is missing.
* ``'ls'`` -- Lomb-Scargle (astropy or NumPy DFT) for rotation/variability.
* ``'acf'`` -- autocorrelation period (NumPy FFT) for rotation.

Orchestration
-------------
``search_two_stage`` runs a cheap BLS triage then refines the winning peak with
TLS (if installed) or a fine-grid BLS, returning the refined detection.

Period grid
-----------
Following astropy/Ofir: frequency spacing ``df = oversample * min(duration) /
baseline**2`` (uniform in frequency, never in period); ``period_max`` defaults to
``baseline / 2`` so at least two transits are required. :func:`period_grid` is
``lru_cache``-d so repeated calls with the same parameters are O(1).

Acceleration
-----------
The pure-NumPy BLS inner loop is optionally JIT-compiled with ``numba`` (lazy
import); the NumPy implementation is used unchanged when numba is absent.

All optional dependencies are imported *inside* functions with ``try/except`` so
this module imports with only ``numpy``/``scipy`` present.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from .types import DetectionResult, LightCurve
from .utils import get_logger, robust_std

__all__ = ["search", "search_two_stage", "period_grid"]

_log = get_logger(__name__)

# Default grid of trial transit durations (days), spanning short hot-Jupiter-like
# ingress/egress up to long-period grazing events.
_DEFAULT_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.30], dtype=np.float64)
_MIN_N_TRANSIT = 2  # need >= 2 transits in a single sector


# --------------------------------------------------------------------------- #
# Period grid (cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _period_grid_cached(
    baseline: float,
    period_min: float,
    period_max: float,
    min_duration: float,
    oversample: float,
) -> tuple:
    """Cached core of :func:`period_grid` (operates on hashable scalars).

    Builds a **uniform-in-frequency** grid: ``df = oversample * min_duration /
    baseline**2`` (astropy ``autoperiod`` convention), then converts to periods.
    Returns a tuple so it is hashable/cacheable; the public wrapper returns an
    array.
    """
    baseline = float(baseline)
    period_min = float(period_min)
    period_max = float(period_max)
    min_duration = float(min_duration)
    oversample = float(oversample)

    if baseline <= 0 or period_min <= 0 or period_max <= period_min:
        return tuple()

    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    df = oversample * min_duration / (baseline ** 2)
    if not np.isfinite(df) or df <= 0:
        df = (f_max - f_min) / 10000.0
    n_freq = int(np.ceil((f_max - f_min) / df)) + 1
    # Guard against pathological (memory-blowing) grids.
    n_freq = int(np.clip(n_freq, 16, 2_000_000))
    freqs = f_min + df * np.arange(n_freq)
    periods = 1.0 / freqs[::-1]  # ascending in period
    return tuple(periods.tolist())


def period_grid(
    baseline: float,
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    min_duration: float = 0.04,
    oversample: float = 3.0,
) -> np.ndarray:
    """Physically-spaced trial-period grid (uniform in frequency), cached.

    Parameters
    ----------
    baseline:
        Observing baseline ``max(time) - min(time)`` in days.
    period_min:
        Shortest trial period (days).
    period_max:
        Longest trial period (days). ``None`` -> ``baseline / 2`` (>= 2 transits).
    min_duration:
        Shortest trial transit duration (days); sets the frequency spacing.
    oversample:
        Oversampling/frequency factor (1-5; higher = finer grid, slower).

    Returns
    -------
    np.ndarray
        Trial periods, ascending. Empty array if the inputs are degenerate.
    """
    if period_max is None:
        period_max = baseline / _MIN_N_TRANSIT
    grid = _period_grid_cached(
        float(baseline),
        float(period_min),
        float(period_max),
        float(min_duration),
        float(oversample),
    )
    return np.asarray(grid, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Harmonics helper
# --------------------------------------------------------------------------- #
def _harmonics(period: float) -> list:
    """Return the common alias periods (P/2, 2P, 3P) for harmonic checks."""
    if not np.isfinite(period) or period <= 0:
        return []
    return [0.5 * period, 2.0 * period, 3.0 * period]


# --------------------------------------------------------------------------- #
# Pure-NumPy BLS core (vectorised, optional numba)
# --------------------------------------------------------------------------- #
def _bls_box_binned(
    phase: np.ndarray,
    flux: np.ndarray,
    weight: np.ndarray,
    duration_phases: np.ndarray,
    n_bins: int = 200,
) -> tuple[float, float, float, float]:
    """Best box fit at one folded period via phase-binning (the fast BLS).

    The folded data (phase in ``[0, 1)``) is accumulated into ``n_bins`` equal
    phase bins; the box search then slides over *bins* (not points), giving
    ``O(N + n_bins * n_durations)`` per period instead of an ``O(N log N)`` sort
    plus per-point sweep. For each trial box width (in phase, converted to a whole
    number of bins) the placement maximising the BLS signal residue
    ``s^2 / (r_w (w_total - r_w) / w_total)`` is found by a vectorised
    rolling-window sum over a doubled bin array (so a box wrapping phase 1->0 is a
    contiguous slice).

    Returns ``(best_sr, best_center_phase, best_depth, best_duration_phase)``.
    """
    n = phase.size
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0
    w_total = float(weight.sum())
    if w_total <= 0:
        return 0.0, 0.0, 0.0, 0.0

    # Accumulate weighted flux and weight per phase bin (vectorised histogram).
    idx = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
    bin_w = np.bincount(idx, weights=weight, minlength=n_bins)
    bin_wf = np.bincount(idx, weights=weight * flux, minlength=n_bins)
    wf_total = float(bin_wf.sum())
    global_mean = wf_total / w_total

    # Double the arrays so wrap-around boxes are contiguous; prefix sums for O(1)
    # window sums of any contiguous bin range.
    bw2 = np.concatenate([bin_w, bin_w])
    bwf2 = np.concatenate([bin_wf, bin_wf])
    cum_w = np.concatenate([[0.0], np.cumsum(bw2)])
    cum_wf = np.concatenate([[0.0], np.cumsum(bwf2)])

    starts = np.arange(n_bins)
    best_sr = 0.0
    best_center = 0.0
    best_depth = 0.0
    best_dur = 0.0

    for dphase in duration_phases:
        if dphase <= 0 or dphase >= 1.0:
            continue
        width = max(int(round(dphase * n_bins)), 1)
        if width >= n_bins:
            continue
        # In-box weight / weighted flux for every start bin (vectorised).
        r_w = cum_w[starts + width] - cum_w[starts]
        r_wf = cum_wf[starts + width] - cum_wf[starts]
        valid = (r_w > 1e-12) & (r_w < w_total - 1e-12)
        if not np.any(valid):
            continue
        s = r_wf - (r_w / w_total) * wf_total
        denom = r_w * (w_total - r_w) / w_total
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(valid & (denom > 0), (s ** 2) / denom, -np.inf)
        j = int(np.argmax(sr))
        if not np.isfinite(sr[j]) or sr[j] <= best_sr:
            continue
        best_sr = float(sr[j])
        in_mean = r_wf[j] / r_w[j] if r_w[j] > 0 else global_mean
        best_depth = float(global_mean - in_mean)
        # Box centre in phase = (start_bin + width/2) / n_bins, wrapped to [0,1).
        best_center = float(((j + 0.5 * width) / n_bins) % 1.0)
        best_dur = float(width) / n_bins
    return best_sr, best_center, best_depth, best_dur


# numba-accelerated variant is built lazily on first use and cached here.
_NUMBA_BLS = None
_NUMBA_TRIED = False


def _get_numba_bls():
    """Lazily compile a numba njit per-period box search; None if numba absent."""
    global _NUMBA_BLS, _NUMBA_TRIED
    if _NUMBA_TRIED:
        return _NUMBA_BLS
    _NUMBA_TRIED = True
    try:  # optional dependency
        import numba
    except Exception:
        _NUMBA_BLS = None
        return None

    @numba.njit(cache=True, fastmath=True)
    def _kernel(phase, flux, weight, durations):  # pragma: no cover - jitted
        n = phase.shape[0]
        best_sr = 0.0
        best_center = 0.0
        best_depth = 0.0
        if n < 4:
            return best_sr, best_center, best_depth
        w_total = 0.0
        wf_total = 0.0
        for i in range(n):
            w_total += weight[i]
            wf_total += weight[i] * flux[i]
        if w_total <= 0.0:
            return best_sr, best_center, best_depth
        global_mean = wf_total / w_total
        for d in range(durations.shape[0]):
            dphase = durations[d]
            if dphase <= 0.0 or dphase >= 1.0:
                continue
            for start in range(n):
                lo = phase[start]
                hi = lo + dphase
                r_w = 0.0
                r_wf = 0.0
                # accumulate points in [lo, hi) with wrap
                for k in range(n):
                    pk = phase[k]
                    inside = (pk >= lo and pk < hi)
                    if hi > 1.0:
                        if pk < (hi - 1.0):
                            inside = True
                    if inside:
                        r_w += weight[k]
                        r_wf += weight[k] * flux[k]
                if r_w <= 1e-12 or r_w >= w_total:
                    continue
                s = r_wf - (r_w / w_total) * wf_total
                denom = r_w * (w_total - r_w) / w_total
                if denom <= 0.0:
                    continue
                sr = (s * s) / denom
                if sr > best_sr:
                    best_sr = sr
                    in_mean = r_wf / r_w
                    best_depth = global_mean - in_mean
                    best_center = (lo + 0.5 * dphase) % 1.0
        return best_sr, best_center, best_depth

    _NUMBA_BLS = _kernel
    return _NUMBA_BLS


def _bls_numpy(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
    use_numba: bool = False,
) -> tuple[np.ndarray, int, float, float, float]:
    """Vectorised pure-NumPy BLS over a period grid.

    Returns ``(power, best_index, best_t0, best_duration, best_depth)`` where
    ``power`` is the signal-residue spectrum aligned with ``periods``.
    """
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    # Inverse-variance weights; default to uniform if errors are missing.
    err = np.asarray(flux_err, dtype=np.float64)
    if np.all(~np.isfinite(err)) or np.all(err <= 0):
        w = np.ones_like(f)
    else:
        safe = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[err > 0]))
        w = 1.0 / (safe ** 2)
    w = w / np.sum(w)

    t0_ref = t.min()
    power = np.zeros(periods.size, dtype=np.float64)
    best_centers = np.zeros(periods.size, dtype=np.float64)
    best_depths = np.zeros(periods.size, dtype=np.float64)
    best_durs = np.zeros(periods.size, dtype=np.float64)

    # Resolution of the phase grid; ~3x finer than the shortest box keeps the
    # binning loss negligible while staying cheap.
    n_bins = 200

    kernel = _get_numba_bls() if use_numba else None

    for ip, period in enumerate(periods):
        phase = ((t - t0_ref) / period) % 1.0
        dphases = durations / period
        dphases = dphases[(dphases > 0) & (dphases < 0.5)]
        if dphases.size == 0:
            continue
        if kernel is not None:
            # The numba kernel works on phase-sorted points directly.
            order = np.argsort(phase)
            sr, center, depth = kernel(
                np.ascontiguousarray(phase[order]),
                np.ascontiguousarray(f[order]),
                np.ascontiguousarray(w[order]),
                np.ascontiguousarray(dphases),
            )
            dur = _best_duration_for(phase[order], f[order], w[order], dphases, period, center)
        else:
            sr, center, depth, dphase_win = _bls_box_binned(phase, f, w, dphases, n_bins)
            dur = dphase_win * period
        power[ip] = sr
        best_centers[ip] = center
        best_depths[ip] = depth
        best_durs[ip] = dur

    if power.size == 0 or not np.any(np.isfinite(power)):
        return power, 0, np.nan, np.nan, np.nan
    best = int(np.argmax(power))
    best_period = periods[best]
    # Convert the in-box centre phase to an absolute mid-transit epoch t0.
    best_t0 = t0_ref + best_centers[best] * best_period
    return power, best, float(best_t0), float(best_durs[best]), float(best_depths[best])


def _best_duration_for(
    ph: np.ndarray,
    fl: np.ndarray,
    wt: np.ndarray,
    dphases: np.ndarray,
    period: float,
    center: float,
) -> float:
    """Pick the trial duration giving the deepest box at the winning centre.

    A light helper so the numba/numpy paths both report a duration in days.
    """
    if dphases.size == 0:
        return np.nan
    best_dur = dphases[0] * period
    best_depth = -np.inf
    for dphase in dphases:
        half = 0.5 * dphase
        lo = (center - half) % 1.0
        hi = (center + half) % 1.0
        if lo <= hi:
            inside = (ph >= lo) & (ph < hi)
        else:
            inside = (ph >= lo) | (ph < hi)
        if inside.sum() < 1 or (~inside).sum() < 1:
            continue
        depth = np.average(fl[~inside], weights=wt[~inside]) - np.average(
            fl[inside], weights=wt[inside]
        )
        if depth > best_depth:
            best_depth = depth
            best_dur = dphase * period
    return float(best_dur)


# --------------------------------------------------------------------------- #
# BLS dispatch (astropy preferred, numpy fallback)
# --------------------------------------------------------------------------- #
def _search_bls(
    lc: LightCurve,
    period_min: float,
    period_max: Optional[float],
    oversample: float,
    durations: Optional[np.ndarray],
    use_numba: bool,
    method_label: str = "bls",
) -> DetectionResult:
    """BLS search returning a populated :class:`DetectionResult`."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    flux_err = np.asarray(lc.flux_err, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    flux_err = flux_err[finite] if flux_err.shape == finite.shape else np.full(time.shape, np.nan)

    if time.size < 10:
        return DetectionResult(method=method_label)

    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = baseline / _MIN_N_TRANSIT
    dur = _DEFAULT_DURATIONS if durations is None else np.atleast_1d(np.asarray(durations, dtype=np.float64))
    dur = dur[dur < 0.5 * period_max]
    if dur.size == 0:
        dur = np.array([min(_DEFAULT_DURATIONS[0], 0.1 * period_max)])
    min_dur = float(dur.min())

    # --- preferred path: astropy BoxLeastSquares -------------------------- #
    try:  # optional dependency
        from astropy import units as u
        from astropy.timeseries import BoxLeastSquares

        dy = None
        if np.any(np.isfinite(flux_err)) and np.any(flux_err > 0):
            dy = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, np.nanmedian(flux_err))
            dy = dy * u.dimensionless_unscaled
        bls = BoxLeastSquares(time * u.day, flux, dy=dy)
        pg = bls.autopower(
            dur * u.day,
            minimum_period=period_min * u.day,
            maximum_period=period_max * u.day,
            minimum_n_transit=_MIN_N_TRANSIT,
            frequency_factor=float(oversample),
            objective="snr",
        )
        power = np.asarray(pg.power, dtype=np.float64)
        periods = np.asarray(pg.period.to_value(u.day), dtype=np.float64)
        i = int(np.argmax(power))
        period = float(periods[i])
        t0 = float(pg.transit_time[i].to_value(u.day))
        duration = float(pg.duration[i].to_value(u.day))
        depth = float(np.asarray(pg.depth)[i])
        sde = _sde(power)
        det = DetectionResult(
            period=period,
            t0=t0,
            duration=duration,
            depth=depth,
            sde=sde,
            snr=np.nan,
            method=method_label,
            periods=periods,
            power=power,
            harmonics=_harmonics(period),
            extra={"objective": "snr", "backend": "astropy"},
        )
        # astropy can also report depth_snr via compute_stats.
        try:
            stats = bls.compute_stats(period * u.day, duration * u.day, t0 * u.day)
            ds = stats.get("depth_snr") if isinstance(stats, dict) else None
            if ds is not None:
                det.snr = float(np.asarray(ds).ravel()[0])
            det.extra["depth_odd"] = _safe_first(stats.get("depth_odd"))
            det.extra["depth_even"] = _safe_first(stats.get("depth_even"))
        except Exception:
            pass
        _finalize_snr(det, lc)
        return det
    except Exception as exc:  # noqa: BLE001 - any astropy issue -> numpy fallback
        _log.info("astropy BLS unavailable/failed (%s); using NumPy BLS", type(exc).__name__)

    # --- fallback: pure-NumPy BLS ----------------------------------------- #
    periods = period_grid(baseline, period_min, period_max, min_dur, oversample)
    if periods.size == 0:
        return DetectionResult(method=method_label)
    power, best, t0, duration, depth = _bls_numpy(
        time, flux, flux_err, periods, dur, use_numba=use_numba
    )
    if not np.any(np.isfinite(power)):
        return DetectionResult(method=method_label, periods=periods, power=power)
    period = float(periods[best])
    det = DetectionResult(
        period=period,
        t0=float(t0),
        duration=float(duration),
        depth=float(depth),
        sde=_sde(power),
        snr=np.nan,
        method=method_label,
        periods=periods,
        power=power,
        harmonics=_harmonics(period),
        extra={"objective": "signal_residue", "backend": "numpy" + ("+numba" if use_numba else "")},
    )
    _finalize_snr(det, lc)
    return det


def _safe_first(value) -> float:
    """Return the first scalar of an astropy-stats entry, or nan."""
    if value is None:
        return float("nan")
    try:
        return float(np.asarray(value).ravel()[0])
    except Exception:
        return float("nan")


def _sde(power: np.ndarray) -> float:
    """SDE = (peak - median(power)) / std(power) (median-centred, robust)."""
    p = np.asarray(power, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    std = p.std()
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float((p.max() - np.median(p)) / std)


def _refine_depth(det: DetectionResult, lc: LightCurve) -> None:
    """Recompute ``det.depth`` from the folded light curve at the found ephemeris.

    Box-search backends (notably astropy's ``objective='snr'``) report a model
    amplitude that can under-estimate the true dip. The robust out-of-transit
    minus in-transit median at the detected (period, t0, duration) is a better
    seed for downstream fitting and is stored alongside the original as
    ``extra['depth_box']``. Only overrides when the folded estimate is finite and
    positive; never overwrites a good box depth with noise.
    """
    if not np.isfinite(det.period) or det.period <= 0:
        return
    from .utils import phase_fold

    t = np.asarray(lc.time, dtype=np.float64)
    f = np.asarray(lc.flux, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size < 10:
        return
    duration = det.duration if (np.isfinite(det.duration) and det.duration > 0) else 0.05 * det.period
    phase = phase_fold(t, det.period, det.t0)
    half = 0.5 * duration / det.period
    # Use the central 80% of the box for depth (avoids ingress/egress dilution).
    in_t = np.abs(phase) <= 0.8 * half
    out_t = (np.abs(phase) > half) & (np.abs(phase) < 3.0 * half)
    if in_t.sum() < 1 or out_t.sum() < 3:
        return
    folded_depth = float(np.nanmedian(f[out_t]) - np.nanmedian(f[in_t]))
    det.extra["depth_box"] = float(det.depth)
    det.extra["depth_folded"] = folded_depth
    if np.isfinite(folded_depth) and folded_depth > 0:
        # Prefer the folded depth (less biased) for the headline value.
        det.depth = folded_depth


def _finalize_snr(det: DetectionResult, lc: LightCurve) -> None:
    """Refine depth from the fold, then populate ``det.snr`` if not already set."""
    _refine_depth(det, lc)
    if np.isfinite(det.snr):
        return
    from .significance import transit_snr

    try:
        det.snr = float(transit_snr(lc, det))
    except Exception:
        det.snr = float("nan")


# --------------------------------------------------------------------------- #
# TLS (lazy) -> BLS fallback
# --------------------------------------------------------------------------- #
def _search_tls(
    lc: LightCurve,
    period_min: float,
    period_max: Optional[float],
    oversample: float,
    durations: Optional[np.ndarray],
    use_numba: bool,
) -> DetectionResult:
    """Transit Least Squares search; falls back to BLS if TLS is unavailable."""
    try:  # optional dependency
        from transitleastsquares import transitleastsquares
    except Exception:
        _log.info("transitleastsquares unavailable; TLS falling back to BLS")
        det = _search_bls(lc, period_min, period_max, oversample, durations, use_numba, "bls")
        det.extra["tls_fallback"] = "bls"
        return det

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if time.size < 10:
        return DetectionResult(method="tls")
    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = baseline / _MIN_N_TRANSIT

    kw = dict(
        period_min=float(period_min),
        period_max=float(period_max),
        n_transits_min=_MIN_N_TRANSIT,
        oversampling_factor=int(max(round(oversample), 1)),
        show_progress_bar=False,
    )
    radius = lc.meta.get("radius")
    mass = lc.meta.get("mass")
    if radius is not None and np.isfinite(radius) and radius > 0:
        kw["R_star"] = float(radius)
    if mass is not None and np.isfinite(mass) and mass > 0:
        kw["M_star"] = float(mass)
    try:
        model = transitleastsquares(time, flux)
        res = model.power(**kw)
    except Exception as exc:  # noqa: BLE001
        _log.info("TLS run failed (%s); falling back to BLS", type(exc).__name__)
        det = _search_bls(lc, period_min, period_max, oversample, durations, use_numba, "bls")
        det.extra["tls_fallback"] = "bls"
        return det

    det = DetectionResult(
        period=float(res.period),
        t0=float(res.T0),
        duration=float(res.duration),
        depth=float(1.0 - res.depth) if res.depth <= 1.0 else float(res.depth),
        sde=float(res.SDE),
        snr=float(getattr(res, "snr", np.nan)),
        method="tls",
        periods=np.asarray(res.periods, dtype=np.float64),
        power=np.asarray(res.power, dtype=np.float64),
        harmonics=_harmonics(float(res.period)),
        extra={
            "backend": "transitleastsquares",
            "FAP": float(getattr(res, "FAP", np.nan)),
            "odd_even_mismatch": float(getattr(res, "odd_even_mismatch", np.nan)),
            "depth_mean_odd": _safe_first(getattr(res, "depth_mean_odd", None)),
            "depth_mean_even": _safe_first(getattr(res, "depth_mean_even", None)),
            "transit_count": int(getattr(res, "transit_count", 0) or 0),
            "distinct_transit_count": int(getattr(res, "distinct_transit_count", 0) or 0),
            "transit_times": list(getattr(res, "transit_times", []) or []),
            "rp_rs": float(getattr(res, "rp_rs", np.nan)),
        },
    )
    _finalize_snr(det, lc)
    return det


# --------------------------------------------------------------------------- #
# Lomb-Scargle (rotation / variables)
# --------------------------------------------------------------------------- #
def _search_ls(
    lc: LightCurve,
    period_min: float,
    period_max: Optional[float],
    oversample: float,
) -> DetectionResult:
    """Lomb-Scargle periodogram (astropy preferred, NumPy DFT fallback)."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if time.size < 10:
        return DetectionResult(method="ls")
    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = baseline / _MIN_N_TRANSIT
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min

    flux_c = flux - np.nanmean(flux)
    try:  # optional dependency
        from astropy.timeseries import LombScargle

        ls = LombScargle(time, flux)
        freq, power = ls.autopower(
            minimum_frequency=f_min,
            maximum_frequency=f_max,
            samples_per_peak=int(max(round(oversample * 3), 5)),
        )
        freq = np.asarray(freq, dtype=np.float64)
        power = np.asarray(power, dtype=np.float64)
        backend = "astropy"
        fap = float("nan")
        try:
            fap = float(ls.false_alarm_probability(power.max()))
        except Exception:
            pass
    except Exception:
        # NumPy DFT periodogram fallback.
        n_freq = int(np.clip((f_max - f_min) * baseline * max(oversample * 3, 5), 64, 200000))
        freq = np.linspace(f_min, f_max, n_freq)
        power = _ls_numpy(time, flux_c, freq)
        backend = "numpy"
        fap = float("nan")

    if power.size == 0 or not np.any(np.isfinite(power)):
        return DetectionResult(method="ls", periods=1.0 / freq[::-1] if freq.size else None)
    i = int(np.argmax(power))
    period = float(1.0 / freq[i])
    periods = (1.0 / freq)[::-1]
    power_sorted = power[::-1]
    det = DetectionResult(
        period=period,
        t0=np.nan,
        duration=np.nan,
        depth=np.nan,
        sde=_sde(power),
        snr=np.nan,
        method="ls",
        periods=periods,
        power=power_sorted,
        harmonics=_harmonics(period),
        extra={"backend": backend, "FAP": fap},
    )
    return det


def _ls_numpy(time: np.ndarray, flux: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Classical (Press & Rybicki form) Lomb-Scargle power, pure NumPy."""
    power = np.zeros(freq.size, dtype=np.float64)
    var = np.var(flux)
    if var <= 0:
        return power
    for i, f in enumerate(freq):
        w = 2.0 * np.pi * f
        wt = w * time
        s2 = np.sum(np.sin(2.0 * wt))
        c2 = np.sum(np.cos(2.0 * wt))
        tau = 0.5 * np.arctan2(s2, c2) / w if w != 0 else 0.0
        wtt = w * (time - tau)
        cos = np.cos(wtt)
        sin = np.sin(wtt)
        cc = np.sum(cos ** 2)
        ss = np.sum(sin ** 2)
        yc = np.sum(flux * cos)
        ys = np.sum(flux * sin)
        term_c = (yc ** 2 / cc) if cc > 0 else 0.0
        term_s = (ys ** 2 / ss) if ss > 0 else 0.0
        power[i] = 0.5 * (term_c + term_s) / var
    return power


# --------------------------------------------------------------------------- #
# ACF (rotation)
# --------------------------------------------------------------------------- #
def _search_acf(
    lc: LightCurve,
    period_min: float,
    period_max: Optional[float],
) -> DetectionResult:
    """Autocorrelation period via FFT on an evenly-resampled flux series."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if time.size < 16:
        return DetectionResult(method="acf")
    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = baseline / _MIN_N_TRANSIT

    # Resample onto a uniform grid at the median cadence and interpolate gaps.
    cadence = float(np.median(np.diff(time)))
    if not np.isfinite(cadence) or cadence <= 0:
        return DetectionResult(method="acf")
    n_grid = int(baseline / cadence) + 1
    n_grid = int(np.clip(n_grid, 16, 1_000_000))
    grid = time.min() + cadence * np.arange(n_grid)
    fi = np.interp(grid, time, flux)
    fi = fi - np.mean(fi)

    # FFT autocorrelation.
    acf = np.correlate(fi, fi, mode="full")[fi.size - 1 :]
    if acf[0] == 0 or not np.isfinite(acf[0]):
        return DetectionResult(method="acf")
    acf = acf / acf[0]
    lags = cadence * np.arange(acf.size)

    # Search for the first strong peak within the allowed period window.
    valid = (lags >= period_min) & (lags <= period_max)
    if not np.any(valid):
        return DetectionResult(method="acf", periods=lags, power=acf)
    sub_lags = lags[valid]
    sub_acf = acf[valid]
    # local maxima
    peaks = (
        (sub_acf[1:-1] > sub_acf[:-2])
        & (sub_acf[1:-1] > sub_acf[2:])
    )
    if np.any(peaks):
        peak_idx = np.flatnonzero(peaks) + 1
        best = peak_idx[np.argmax(sub_acf[peak_idx])]
        period = float(sub_lags[best])
    else:
        period = float(sub_lags[np.argmax(sub_acf)])
    det = DetectionResult(
        period=period,
        t0=np.nan,
        duration=np.nan,
        depth=np.nan,
        sde=_sde(sub_acf),
        snr=np.nan,
        method="acf",
        periods=lags,
        power=acf,
        harmonics=_harmonics(period),
        extra={"backend": "numpy"},
    )
    return det


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
def search(
    lc: LightCurve,
    method: str = "bls",
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    oversample: float = 3.0,
    durations: Optional[np.ndarray] = None,
    *,
    use_numba: bool = False,
    **kw,
) -> DetectionResult:
    """Search ``lc`` for a periodic signal and return the best detection.

    Parameters
    ----------
    lc:
        Input light curve (should already be detrended for transit methods).
    method:
        ``'bls'`` (default), ``'tls'``, ``'ls'``, or ``'acf'``. Unknown methods
        raise ``ValueError``.
    period_min:
        Shortest trial period (days).
    period_max:
        Longest trial period (days). ``None`` -> ``baseline / 2`` so at least two
        transits are required.
    oversample:
        Frequency-grid oversampling / ``frequency_factor`` (1-5).
    durations:
        Trial transit durations (days) for BLS/TLS. ``None`` -> a sensible default
        grid.
    use_numba:
        If True, JIT-compile the NumPy BLS inner loop with numba when available
        (ignored on the astropy BLS path and for ls/acf).

    Returns
    -------
    DetectionResult
        Fully populated: ``period``, ``t0``, ``duration``, ``depth``, ``sde``,
        ``snr``, ``method``, the ``periods``/``power`` spectrum, ``harmonics``,
        and method-specific ``extra`` keys.
    """
    method = str(method).lower()
    if method == "bls":
        return _search_bls(lc, period_min, period_max, oversample, durations, use_numba, "bls")
    if method == "tls":
        return _search_tls(lc, period_min, period_max, oversample, durations, use_numba)
    if method == "ls":
        return _search_ls(lc, period_min, period_max, oversample)
    if method == "acf":
        return _search_acf(lc, period_min, period_max)
    raise ValueError(f"unknown search method {method!r}; expected bls/tls/ls/acf")


# --------------------------------------------------------------------------- #
# Two-stage orchestration (BLS triage -> TLS / fine-grid BLS confirm)
# --------------------------------------------------------------------------- #
def search_two_stage(
    lc: LightCurve,
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    oversample: float = 2.0,
    durations: Optional[np.ndarray] = None,
    *,
    refine_window: float = 0.05,
    refine_oversample: float = 5.0,
    use_numba: bool = False,
    **kw,
) -> DetectionResult:
    """Fast BLS triage, then refine the winning peak (TLS if available, else BLS).

    Stage 1 runs a coarse BLS over the full period range to localise the strongest
    box signal. Stage 2 refines that peak: if ``transitleastsquares`` is installed
    it re-searches a *narrow* period window around the triage period with TLS (a
    limb-darkened template -> better significance, odd/even & FAP diagnostics);
    otherwise it re-runs BLS on a fine grid in the same window. The refined
    detection (with the better SDE/SNR) is returned; the triage period is recorded
    in ``extra['triage_period']`` and the periodogram from whichever stage carries
    more information is retained.

    Parameters
    ----------
    refine_window:
        Half-width of the refinement period window as a *fraction* of the triage
        period (e.g. 0.05 -> +/-5%).
    refine_oversample:
        Oversampling for the (fine) refinement search.
    """
    # --- Stage 1: coarse BLS triage --------------------------------------- #
    triage = _search_bls(lc, period_min, period_max, oversample, durations, use_numba, "bls")
    if not np.isfinite(triage.period) or triage.period <= 0:
        return triage

    p0 = float(triage.period)
    lo = max(p0 * (1.0 - refine_window), period_min)
    hi = p0 * (1.0 + refine_window)
    if period_max is not None:
        hi = min(hi, period_max)
    if hi <= lo:
        hi = lo * 1.02

    # --- Stage 2: refine around the triage peak --------------------------- #
    have_tls = False
    try:  # detect TLS without importing heavy machinery yet
        import importlib.util

        have_tls = importlib.util.find_spec("transitleastsquares") is not None
    except Exception:
        have_tls = False

    if have_tls:
        refined = _search_tls(lc, lo, hi, refine_oversample, durations, use_numba)
        refined.method = "bls+tls"
    else:
        refined = _search_bls(lc, lo, hi, refine_oversample, durations, use_numba, "bls+bls")

    # If refinement degenerated (e.g. window too narrow), keep the triage result.
    if not np.isfinite(refined.period) or refined.period <= 0:
        triage.extra["triage_period"] = p0
        triage.method = "bls"
        return triage

    # Prefer the refined ephemeris/depth but carry the full triage periodogram so
    # the vetting sheet can plot the *global* spectrum and mark harmonics. The
    # headline SDE must reflect the global spectrum (a narrow refinement window
    # contains too few trial periods to estimate SDE meaningfully), so keep the
    # refined SDE only when it is the richer spectrum.
    refined.extra["triage_period"] = p0
    refined.extra["triage_sde"] = float(triage.sde)
    refined.extra["triage_backend"] = triage.extra.get("backend")
    refined.extra["refine_sde"] = float(refined.sde)

    triage_n = triage.periods.size if triage.periods is not None else 0
    refine_n = refined.periods.size if refined.periods is not None else 0
    if triage_n > refine_n:
        # Stash the (narrow) refinement spectrum, expose the global one.
        refined.extra["refine_periods"] = refined.periods
        refined.extra["refine_power"] = refined.power
        refined.periods = triage.periods
        refined.power = triage.power
        global_sde = _sde(triage.power) if triage.power is not None else triage.sde
        # The global SDE evaluated at the refined period is the headline number.
        if np.isfinite(global_sde):
            refined.sde = float(global_sde)
        elif np.isfinite(triage.sde):
            refined.sde = float(triage.sde)
    refined.harmonics = _harmonics(refined.period)
    _finalize_snr(refined, lc)
    return refined
