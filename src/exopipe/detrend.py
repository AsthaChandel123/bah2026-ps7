"""Light-curve detrending / flattening for ``exopipe`` (PS7 stage DETREND).

Goal (research dossier `02_detection_detrending.md` §A): flatten the
out-of-transit baseline to ~1.0 by removing stellar variability and instrumental
systematics on timescales *longer than a transit*, **without distorting or eating
the transit itself**. The cardinal rule (Hippke et al. 2019, *wotan*) is to set
the smoothing window to ``~3 x`` the longest transit duration you intend to
search, so the filter passes slow variability but is nearly transparent to a
transit (which spans ~1/3 of the window).

Design notes
------------
* The **default method is a pure-NumPy/SciPy time-windowed Tukey biweight** robust
  trend.  This is the top performer in the wotan benchmark (99% Kepler / 94% K2
  recovery of the shallowest injected transits) and -- crucially for PS7's
  graceful-degradation principle -- it runs with **only numpy/scipy installed**.
* Every optional/heavy method (``wotan``, ``celerite2`` GP) is imported *lazily*
  inside the function with a ``try/except`` and falls back to the biweight if the
  library is missing, so importing this module never requires anything beyond the
  core stack.
* Gaps larger than ``break_tolerance`` days split the series so the trend is never
  smoothed *across* a data-downlink gap or momentum-dump ramp.
* Outliers are rejected **asymmetrically** (positive excursions clipped harder
  than negative ones) so genuine transit/eclipse dips survive the clean-up.

Public API
----------
``detrend(lc, method='biweight', window_length=None, break_tolerance=0.5, **kw)``
returns a **new** :class:`~exopipe.types.LightCurve` (the input is never mutated)
whose flux has been divided by the estimated trend and re-normalised to ~1.0.
``flatten`` is provided as an alias.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import LightCurve
from .utils import get_logger, robust_std, running_median

__all__ = ["detrend", "flatten"]

_log = get_logger(__name__)

# Default longest transit duration we expect to search (days).  TESS planets at
# P <~ 15 d rarely exceed ~0.3 d; the biweight window is 3x this per Hippke 2019.
_DEFAULT_MAX_DURATION = 0.30
# Absolute floor on the smoothing window so a tiny ``max_duration`` cannot produce
# a window that chases (and absorbs) the transit.
_MIN_WINDOW = 0.40


# --------------------------------------------------------------------------- #
# Window-length policy
# --------------------------------------------------------------------------- #
def _default_window_length(
    window_length: Optional[float],
    max_duration: Optional[float],
) -> float:
    """Resolve the smoothing ``window_length`` (days).

    If the caller passes ``window_length`` explicitly it is honoured. Otherwise
    the Hippke (2019) rule ``window = 3 x max_transit_duration`` is applied, with
    a sensible floor of :data:`_MIN_WINDOW` days so shallow/short transits are
    never smoothed away.
    """
    if window_length is not None and np.isfinite(window_length) and window_length > 0:
        return float(window_length)
    md = _DEFAULT_MAX_DURATION if max_duration is None else float(max_duration)
    if not np.isfinite(md) or md <= 0:
        md = _DEFAULT_MAX_DURATION
    return float(max(3.0 * md, _MIN_WINDOW))


# --------------------------------------------------------------------------- #
# Gap handling
# --------------------------------------------------------------------------- #
def _segment_bounds(time: np.ndarray, break_tolerance: float) -> list[tuple[int, int]]:
    """Split ``time`` into contiguous segments at gaps > ``break_tolerance`` days.

    Returns a list of ``(start, stop)`` index pairs (half-open) covering the whole
    array. ``time`` is assumed sorted ascending. A non-positive ``break_tolerance``
    yields a single segment spanning everything.
    """
    n = time.size
    if n == 0:
        return []
    if not np.isfinite(break_tolerance) or break_tolerance <= 0:
        return [(0, n)]
    dt = np.diff(time)
    # A gap is a jump larger than the tolerance.
    breaks = np.where(dt > break_tolerance)[0]
    bounds: list[tuple[int, int]] = []
    start = 0
    for b in breaks:
        bounds.append((start, int(b) + 1))
        start = int(b) + 1
    bounds.append((start, n))
    return bounds


# --------------------------------------------------------------------------- #
# Asymmetric sigma-clip mask (preserve dips)
# --------------------------------------------------------------------------- #
def _outlier_mask(
    resid: np.ndarray,
    sigma_upper: float = 3.0,
    sigma_lower: float = 6.0,
) -> np.ndarray:
    """Boolean ``True == keep`` mask of residuals using asymmetric clipping.

    Residuals are normalised by a robust scale (``1.4826 * MAD``). Positive
    excursions (cosmic rays, flares) are clipped at ``sigma_upper``; negative
    excursions (which include real transit dips) only at the far looser
    ``sigma_lower`` so transits are preserved.
    """
    finite = np.isfinite(resid)
    if finite.sum() < 3:
        return finite
    scale = robust_std(resid[finite])
    if not np.isfinite(scale) or scale == 0:
        return finite
    keep = finite.copy()
    keep[finite] = (resid[finite] <= sigma_upper * scale) & (
        resid[finite] >= -sigma_lower * scale
    )
    return keep


# --------------------------------------------------------------------------- #
# Core: time-windowed Tukey biweight robust trend (pure numpy)
# --------------------------------------------------------------------------- #
def _biweight_location(values: np.ndarray, cval: float) -> float:
    """Tukey's biweight (bisquare) robust location of ``values``.

    Points more than ``cval`` MADs from the median get zero weight; nearer points
    are down-weighted by ``(1 - (r / (cval * MAD))^2)^2``. A couple of
    Newton-style re-centring passes refine the estimate. In-transit points and
    outliers therefore barely influence the trend, so it "rides over" a transit
    instead of through it.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    if v.size < 3:
        return float(np.median(v))
    center = float(np.median(v))
    for _ in range(3):
        mad = np.median(np.abs(v - center))
        if mad <= 0:
            return center
        u = (v - center) / (cval * mad)
        inside = np.abs(u) < 1.0
        if not np.any(inside):
            return center
        w = (1.0 - u[inside] ** 2) ** 2
        wsum = w.sum()
        if wsum <= 0:
            return center
        new_center = center + np.sum(w * (v[inside] - center)) / wsum
        if not np.isfinite(new_center) or abs(new_center - center) < 1e-12:
            center = new_center
            break
        center = new_center
    return float(center)


def _biweight_trend(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: float,
    cval: float,
) -> np.ndarray:
    """Time-windowed biweight trend evaluated at every cadence.

    For each point, gather the cadences within +/- ``window_length / 2`` days and
    take their biweight location. The two-pointer sweep over time-sorted data
    makes this ``O(N * W_points)``. NaNs are ignored inside each window; windows
    with no finite points yield NaN (handled by the caller).
    """
    n = time.size
    trend = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return trend
    half = 0.5 * window_length
    lo = 0
    hi = 0
    for i in range(n):
        # Advance the window so it covers [time[i]-half, time[i]+half].
        while lo < n and time[lo] < time[i] - half:
            lo += 1
        if hi < i:
            hi = i
        while hi < n and time[hi] <= time[i] + half:
            hi += 1
        trend[i] = _biweight_location(flux[lo:hi], cval)
    return trend


# --------------------------------------------------------------------------- #
# Alternative trend estimators (per-segment)
# --------------------------------------------------------------------------- #
def _points_per_window(time: np.ndarray, window_length: float) -> int:
    """Estimate the number of cadences spanned by ``window_length`` days."""
    if time.size < 2:
        return 1
    cadence = float(np.median(np.diff(time)))
    if not np.isfinite(cadence) or cadence <= 0:
        return max(int(time.size), 1)
    return max(int(round(window_length / cadence)), 1)


def _median_trend(time: np.ndarray, flux: np.ndarray, window_length: float) -> np.ndarray:
    """Running-median trend (cadence window derived from ``window_length`` days)."""
    w = _points_per_window(time, window_length)
    if w % 2 == 0:
        w += 1
    return running_median(flux, w)


def _savgol_trend(
    time: np.ndarray, flux: np.ndarray, window_length: float, polyorder: int
) -> np.ndarray:
    """Savitzky-Golay trend with iterative asymmetric clipping (scipy)."""
    from scipy.signal import savgol_filter

    w = _points_per_window(time, window_length)
    if w % 2 == 0:
        w += 1
    if w <= polyorder + 1:
        w = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    if w > flux.size:
        # Window longer than the segment: fall back to a flat robust level.
        return np.full(flux.size, np.nanmedian(flux), dtype=np.float64)

    work = flux.astype(np.float64).copy()
    # Fill NaNs by interpolation so savgol is well-defined, then iterate-clip.
    finite = np.isfinite(work)
    if not finite.all():
        if finite.sum() < 2:
            return np.full(flux.size, np.nanmedian(flux), dtype=np.float64)
        work = np.interp(np.arange(work.size), np.flatnonzero(finite), work[finite])
    trend = savgol_filter(work, w, polyorder)
    for _ in range(3):
        keep = _outlier_mask(work - trend)
        if keep.all():
            break
        work = np.where(keep, work, trend)  # replace outliers with the trend
        trend = savgol_filter(work, w, polyorder)
    return trend


def _spline_trend(
    time: np.ndarray, flux: np.ndarray, window_length: float
) -> np.ndarray:
    """Robust univariate-spline trend (scipy) with iterative 2-sigma clipping.

    Knot spacing is tied to ``window_length`` (>= 3x transit duration) so the
    spline cannot flex tightly enough to absorb the transit.
    """
    from scipy.interpolate import UnivariateSpline

    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 4:
        return np.full(flux.size, np.nanmedian(flux), dtype=np.float64)
    t = time[finite]
    f = flux[finite].astype(np.float64)
    keep = np.ones(t.size, dtype=bool)
    spline = None
    span = float(t[-1] - t[0])
    n_knots = max(int(span / max(window_length, 1e-6)), 1)
    for _ in range(5):
        if keep.sum() < 4:
            break
        tk = t[keep]
        fk = f[keep]
        # Interior knots, uniformly spaced over the kept range.
        if n_knots >= 1 and tk.size > n_knots + 2:
            interior = np.linspace(tk[0], tk[-1], n_knots + 2)[1:-1]
            interior = interior[(interior > tk[0]) & (interior < tk[-1])]
        else:
            interior = None
        try:
            if interior is not None and interior.size > 0:
                spline = UnivariateSpline(tk, fk, k=3, s=0, t=interior, ext=3)
            else:
                spline = UnivariateSpline(tk, fk, k=3, s=fk.size, ext=3)
        except Exception:
            spline = UnivariateSpline(tk, fk, k=3, s=fk.size, ext=3)
        resid = f - spline(t)
        new_keep = _outlier_mask(resid, sigma_upper=2.0, sigma_lower=6.0)
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
    out = np.full(flux.size, np.nan, dtype=np.float64)
    if spline is not None:
        out[finite] = spline(t)
    return out


def _gp_trend(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    window_length: float,
    cval: float,
) -> np.ndarray:
    """Celerite2 SHO-GP trend (lazy); falls back to biweight if unavailable.

    Models correlated stellar variability as a damped simple-harmonic-oscillator
    Gaussian process and returns its predictive mean. ``celerite2`` gives an O(N)
    likelihood, essential at TESS scale. If the import or the fit fails we fall
    straight back to the robust biweight so the pipeline never crashes.
    """
    try:  # optional dependency
        import celerite2
        from celerite2 import terms
    except Exception:
        _log.info("celerite2 unavailable; GP detrend falling back to biweight")
        return _biweight_trend(time, flux, window_length, cval)

    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 8:
        return _biweight_trend(time, flux, window_length, cval)
    t = time[finite]
    f = flux[finite].astype(np.float64)
    err = np.asarray(flux_err, dtype=np.float64)[finite]
    yerr = np.nanmedian(err) if np.isfinite(np.nanmedian(err)) else robust_std(f)
    if not np.isfinite(yerr) or yerr <= 0:
        yerr = float(robust_std(f)) or 1e-3
    try:
        sigma = float(robust_std(f)) or 1e-3
        # Timescale ~ window so the GP captures variability slower than transits.
        rho = max(float(window_length), 5.0 * float(np.median(np.diff(t))))
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0 / np.sqrt(2.0))
        gp = celerite2.GaussianProcess(kernel, mean=float(np.median(f)))
        gp.compute(t, yerr=yerr, quiet=True)
        mu = gp.predict(f, t=t)
    except Exception:
        _log.info("celerite2 GP fit failed; falling back to biweight")
        return _biweight_trend(time, flux, window_length, cval)
    out = np.full(flux.size, np.nan, dtype=np.float64)
    out[finite] = np.asarray(mu, dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Per-segment dispatch
# --------------------------------------------------------------------------- #
def _trend_for_segment(
    method: str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    window_length: float,
    cval: float,
    polyorder: int,
) -> np.ndarray:
    """Compute the trend for one contiguous segment with the chosen estimator."""
    if method in ("biweight", "tukey"):
        return _biweight_trend(time, flux, window_length, cval)
    if method in ("median", "medfilt"):
        return _median_trend(time, flux, window_length)
    if method == "savgol":
        return _savgol_trend(time, flux, window_length, polyorder)
    if method in ("spline", "rspline", "hspline", "pspline"):
        return _spline_trend(time, flux, window_length)
    if method == "gp":
        return _gp_trend(time, flux, flux_err, window_length, cval)
    # 'biweight' is the catch-all fallback.
    return _biweight_trend(time, flux, window_length, cval)


def _wotan_flatten(
    time: np.ndarray,
    flux: np.ndarray,
    method: str,
    window_length: float,
    break_tolerance: float,
    cval: float,
) -> Optional[np.ndarray]:
    """Try ``wotan.flatten`` (lazy). Returns the *trend* array, or None if absent."""
    try:  # optional dependency
        from wotan import flatten as wotan_flatten  # type: ignore
    except Exception:
        return None
    wmethod = method if method != "wotan" else "biweight"
    try:
        _, trend = wotan_flatten(
            time,
            flux,
            method=wmethod,
            window_length=window_length,
            break_tolerance=break_tolerance,
            cval=cval,
            edge_cutoff=0.5,
            return_trend=True,
        )
        return np.asarray(trend, dtype=np.float64)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detrend(
    lc: LightCurve,
    method: str = "biweight",
    window_length: Optional[float] = None,
    break_tolerance: float = 0.5,
    *,
    max_duration: Optional[float] = None,
    cval: float = 5.0,
    polyorder: int = 2,
    sigma_upper: float = 3.0,
    sigma_lower: float = 20.0,
    return_trend: bool = False,
    **kw,
) -> LightCurve:
    """Flatten ``lc`` by dividing out an estimated trend; return a new LightCurve.

    The out-of-transit baseline is brought to ~1.0 while transit dips are
    preserved. The input light curve is **not** mutated.

    Parameters
    ----------
    lc:
        Input :class:`~exopipe.types.LightCurve`.
    method:
        Trend estimator. One of:
        ``'biweight'`` (default, pure-numpy time-windowed Tukey biweight),
        ``'median'`` (running median), ``'savgol'`` (Savitzky-Golay, scipy),
        ``'spline'`` (robust univariate spline, scipy),
        ``'gp'`` (celerite2 SHO Gaussian process, lazy -> biweight fallback),
        ``'wotan'`` (``wotan.flatten`` if installed -> biweight fallback).
        Unknown methods fall back to ``'biweight'``.
    window_length:
        Smoothing window in **days**. ``None`` -> ``max(3 x max_duration, 0.4 d)``
        per Hippke (2019). Pass a value to override.
    break_tolerance:
        Gaps larger than this (days) split the series so the trend is never
        smoothed across a data gap / momentum-dump ramp. ``<= 0`` disables splits.
    max_duration:
        Longest transit duration you intend to search (days); drives the default
        window. Ignored when ``window_length`` is given.
    cval:
        Tukey biweight tuning constant in MAD units (5.0 ~ 4 sigma Gaussian).
    polyorder:
        Polynomial order for the Savitzky-Golay method.
    sigma_upper, sigma_lower:
        Asymmetric clip thresholds (in robust sigma) applied to the flattened
        residuals to flag outliers. Positive excursions (cosmic rays, flares) are
        clipped at ``sigma_upper`` (default 3); negative excursions are clipped
        only at the deliberately lenient ``sigma_lower`` (default 20) so even deep
        transit/eclipse dips are preserved (dossier §A9 convention).
    return_trend:
        If True, store the trend in ``out.meta['detrend_trend']``.

    Returns
    -------
    LightCurve
        New light curve with ``flux = flux / trend`` re-normalised to ~1.0.
        ``meta['detrended'] = True``, ``meta['detrend_method']`` and
        ``meta['detrend_window']`` are recorded. Cadences where the trend is
        non-finite/non-positive, or flagged as outliers, are dropped.
    """
    method = str(method).lower()
    out = lc.copy()
    time = np.asarray(out.time, dtype=np.float64)
    flux = np.asarray(out.flux, dtype=np.float64)
    flux_err = np.asarray(out.flux_err, dtype=np.float64)

    if time.size == 0:
        return out

    win = _default_window_length(window_length, max_duration)

    # --- whole-series methods (wotan handles its own gap splitting) --------- #
    trend = None
    if method == "wotan":
        trend = _wotan_flatten(time, flux, method, win, break_tolerance, cval)
        if trend is None:
            _log.info("wotan unavailable; detrend falling back to biweight")
            method = "biweight"

    # --- per-segment trend (gap-aware) -------------------------------------- #
    if trend is None:
        trend = np.full(time.size, np.nan, dtype=np.float64)
        for start, stop in _segment_bounds(time, break_tolerance):
            if stop - start <= 0:
                continue
            trend[start:stop] = _trend_for_segment(
                method,
                time[start:stop],
                flux[start:stop],
                flux_err[start:stop],
                win,
                cval,
                polyorder,
            )

    # --- divide out the trend ----------------------------------------------- #
    good_trend = np.isfinite(trend) & (trend > 0)
    flat = np.full(flux.size, np.nan, dtype=np.float64)
    np.divide(flux, trend, out=flat, where=good_trend)

    # Re-normalise to a median of exactly ~1.0 (robust to residual offsets).
    med = np.nanmedian(flat[good_trend]) if good_trend.any() else np.nan
    if np.isfinite(med) and med > 0:
        flat = flat / med

    # --- asymmetric outlier rejection on the flattened residual ------------- #
    keep = good_trend & _outlier_mask(flat - 1.0, sigma_upper, sigma_lower)

    # Propagate errors through the same division.
    new_err = np.full(flux_err.size, np.nan, dtype=np.float64)
    np.divide(flux_err, trend, out=new_err, where=good_trend)
    if np.isfinite(med) and med > 0:
        new_err = new_err / med

    out.flux = flat.astype(np.float32)
    out.flux_err = new_err.astype(np.float32)
    out.meta["detrended"] = True
    out.meta["detrend_method"] = method
    out.meta["detrend_window"] = float(win)
    if return_trend:
        out.meta["detrend_trend"] = trend.copy()

    # Drop dropped cadences (non-finite trend or flagged outliers) but keep
    # per-cadence meta aligned via the dataclass helper.
    out = out._apply_mask(keep)
    n_drop = int((~keep).sum())
    if n_drop:
        _log.debug("detrend(%s): dropped %d cadences (window=%.3f d)", method, n_drop, win)
    return out


# Alias required by the build contract / common lightkurve naming.
flatten = detrend
