"""Feature engineering: tabular feature vector + phase-folded CNN views.

Two public entry points (ARCHITECTURE Section 6, research/03 Section F):

* :func:`extract_features` -- a stable, NaN-safe numeric feature dict combining
  transit/orbit, shape/EB-discriminator, blend, implied-physical-sanity, and
  stellar-context features. The ordered key list is exported as
  :data:`FEATURE_NAMES`; the tabular classifier imports it to build its design
  matrix, so the order and membership are part of the module's contract.
* :func:`build_views` -- fixed-length phase-folded, binned views for the
  view-based CNN: ``global`` (full orbit, 2001 bins), ``local`` (transit zoom,
  201 bins, +-~3 durations), ``secondary`` (near phase 0.5), ``odd``, ``even``.
  Each view is normalised so the out-of-transit median is 0 and the transit
  depth is ~ -1 (the Shallue & Vanderburg / AstroNet convention), with empty
  bins filled by interpolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import DetectionResult, LightCurve, TransitFit, VettingReport
from .utils import get_logger, robust_std

__all__ = ["extract_features", "build_views", "FEATURE_NAMES"]

_LOG = get_logger("exopipe.features")

# --------------------------------------------------------------------------- #
# Canonical ordered feature list (imported by the classifier).
# --------------------------------------------------------------------------- #
FEATURE_NAMES: list[str] = [
    # --- transit / orbit ---
    "period",
    "log_period",
    "t0",
    "depth",
    "log_depth",
    "duration",
    "duration_over_period",
    "ingress_egress_ratio",
    "transit_snr",
    "sde",
    "n_transits",
    "delta_bic",
    # --- shape / EB discriminators ---
    "odd_even_depth_sigma",
    "odd_depth",
    "even_depth",
    "secondary_depth",
    "secondary_snr",
    "secondary_phase",
    "secondary_to_primary",
    "v_shape_metric",
    "trapezoid_chi2",
    "uniqueness",
    # --- implied physical sanity ---
    "implied_rp_rjup",
    "stellar_density_ratio",
    "rp_rs",
    "a_rs",
    "b",
    "inclination",
    # --- blend / contamination ---
    "crowdsap",
    "centroid_offset",
    # --- systematics / variability (-> other) ---
    "sweet_metric",
    "flux_mad",
    "oot_rms",
    # --- stellar context ---
    "teff",
    "logg",
    "radius",
    "mass",
    "tmag",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f(value: Any, default: float = np.nan) -> float:
    """Coerce ``value`` to a finite float or ``default`` (NaN-safe)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_log(value: float) -> float:
    """``log10`` that returns NaN for non-positive / non-finite input."""
    v = _f(value)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    return float(np.log10(v))


def _triple0(params: dict | None, name: str) -> float:
    """Median (first element) of a ``(median, lo, hi)`` triple, or NaN."""
    if not params or name not in params:
        return float("nan")
    arr = np.atleast_1d(np.asarray(params[name], dtype=float))
    return float(arr[0]) if arr.size else float("nan")


# --------------------------------------------------------------------------- #
# Tabular features
# --------------------------------------------------------------------------- #
def extract_features(
    lc: LightCurve,
    det: DetectionResult,
    vetting: VettingReport | None,
    fit: TransitFit | None,
) -> dict[str, float]:
    """Build the NaN-safe numeric feature dict for the tabular classifier.

    Pulls detection shape parameters, vetting diagnostics, fit-derived physical
    parameters, and stellar context from ``lc.meta`` into a single flat dict
    keyed by :data:`FEATURE_NAMES`. Every key in ``FEATURE_NAMES`` is guaranteed
    present (missing/invalid values become ``nan``), so callers can vectorise
    without key checks.

    Parameters
    ----------
    lc, det, vetting, fit:
        Pipeline-stage outputs. ``vetting`` and ``fit`` may be ``None`` (the
        corresponding features fall back to detection values or NaN).

    Returns
    -------
    dict[str, float]
        ``len == len(FEATURE_NAMES)``; values are plain Python floats.
    """
    meta = lc.meta if (lc is not None and lc.meta is not None) else {}
    metrics = vetting.metrics if (vetting is not None and vetting.metrics) else {}
    params = fit.params if (fit is not None and fit.params) else {}

    feats: dict[str, float] = {}

    # --- transit / orbit --------------------------------------------------- #
    period = _f(det.period)
    depth = _f(det.depth)
    duration = _f(det.duration)
    feats["period"] = period
    feats["log_period"] = _safe_log(period)
    feats["t0"] = _f(det.t0)
    feats["depth"] = depth
    feats["log_depth"] = _safe_log(depth)
    feats["duration"] = duration
    feats["duration_over_period"] = (
        duration / period if np.isfinite(period) and period > 0 and np.isfinite(duration) else np.nan
    )
    feats["ingress_egress_ratio"] = _f(metrics.get("ingress_egress_ratio"))
    feats["transit_snr"] = _f(metrics.get("transit_snr", det.snr))
    feats["sde"] = _f(det.sde)
    feats["n_transits"] = _f(metrics.get("n_transits"))
    feats["delta_bic"] = _f(fit.delta_bic) if fit is not None else np.nan

    # --- shape / EB discriminators ----------------------------------------- #
    feats["odd_even_depth_sigma"] = _f(metrics.get("odd_even_depth_sigma"))
    feats["odd_depth"] = _f(metrics.get("odd_depth"))
    feats["even_depth"] = _f(metrics.get("even_depth"))
    feats["secondary_depth"] = _f(metrics.get("secondary_depth"))
    feats["secondary_snr"] = _f(metrics.get("secondary_snr"))
    feats["secondary_phase"] = _f(metrics.get("secondary_phase"))
    feats["secondary_to_primary"] = _f(metrics.get("secondary_to_primary"))
    feats["v_shape_metric"] = _f(metrics.get("v_shape_metric"))
    feats["trapezoid_chi2"] = _f(metrics.get("trapezoid_chi2"))
    feats["uniqueness"] = _f(metrics.get("uniqueness"))

    # --- implied physical sanity ------------------------------------------- #
    feats["implied_rp_rjup"] = _f(metrics.get("implied_rp_rjup"))
    feats["stellar_density_ratio"] = _f(metrics.get("stellar_density_ratio"))
    # Prefer fit-derived physical params; fall back to detection-implied rp_rs.
    rp_rs = _triple0(params, "rp_rs")
    if not np.isfinite(rp_rs) and np.isfinite(depth) and depth > 0:
        rp_rs = float(np.sqrt(depth))
    feats["rp_rs"] = rp_rs
    feats["a_rs"] = _triple0(params, "a_rs")
    feats["b"] = _triple0(params, "b")
    feats["inclination"] = _triple0(params, "inclination")

    # --- blend / contamination --------------------------------------------- #
    feats["crowdsap"] = _f(metrics.get("crowdsap", meta.get("crowdsap")))
    feats["centroid_offset"] = _f(meta.get("centroid_offset"))

    # --- systematics / variability ----------------------------------------- #
    feats["sweet_metric"] = _f(metrics.get("sweet_metric"))
    feats["flux_mad"], feats["oot_rms"] = _noise_features(lc, det)

    # --- stellar context --------------------------------------------------- #
    feats["teff"] = _f(meta.get("teff"))
    feats["logg"] = _f(meta.get("logg"))
    feats["radius"] = _f(meta.get("radius"))
    feats["mass"] = _f(meta.get("mass"))
    feats["tmag"] = _f(meta.get("tmag"))

    # Guarantee the exact contract: every FEATURE_NAMES key present, in order.
    return {name: float(feats.get(name, np.nan)) for name in FEATURE_NAMES}


def _noise_features(lc: LightCurve, det: DetectionResult) -> tuple[float, float]:
    """Robust scatter of the flux and the out-of-transit RMS.

    ``flux_mad`` is the global robust sigma; ``oot_rms`` is the standard
    deviation of the cadences *outside* the transit window (a red-noise / clean
    -baseline proxy that helps separate variable stars from clean transits).
    """
    try:
        flux = np.asarray(lc.flux, dtype=np.float64)
        flux = flux[np.isfinite(flux)]
    except Exception:
        return (np.nan, np.nan)
    if flux.size == 0:
        return (np.nan, np.nan)
    mad = _f(robust_std(flux))

    period = _f(det.period)
    t0 = _f(det.t0)
    duration = _f(det.duration)
    oot_rms = np.nan
    if np.isfinite(period) and period > 0 and np.isfinite(duration) and duration > 0:
        time = np.asarray(lc.time, dtype=np.float64)
        good = np.isfinite(time) & np.isfinite(np.asarray(lc.flux, dtype=np.float64))
        time = time[good]
        f = np.asarray(lc.flux, dtype=np.float64)[good]
        phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
        oot = np.abs(phase * period) > 0.5 * duration
        if np.count_nonzero(oot) > 3:
            oot_rms = float(np.nanstd(f[oot]))
    return (mad, _f(oot_rms))


# --------------------------------------------------------------------------- #
# CNN views
# --------------------------------------------------------------------------- #
def _fill_empty(values: np.ndarray) -> np.ndarray:
    """Fill NaN (empty) bins by linear interpolation over bin index.

    Edge NaNs are filled with the nearest valid value; an all-NaN input becomes
    all zeros so downstream tensors stay finite.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    good = np.isfinite(values)
    if not np.any(good):
        return np.zeros(n, dtype=np.float64)
    if np.all(good):
        return values
    idx = np.arange(n)
    return np.interp(idx, idx[good], values[good])


def _bin_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    lo: float,
    hi: float,
    nbins: int,
) -> np.ndarray:
    """Median-bin ``flux`` over ``phase in [lo, hi]`` into ``nbins`` bins.

    Returns the per-bin median (NaN for empty bins, later interpolated). Median
    binning (rather than mean) is robust to the outliers in raw TESS data.
    """
    edges = np.linspace(lo, hi, nbins + 1)
    out = np.full(nbins, np.nan, dtype=np.float64)
    sel = (phase >= lo) & (phase <= hi) & np.isfinite(flux)
    if not np.any(sel):
        return out
    ph, fl = phase[sel], flux[sel]
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nbins - 1)
    # Group by bin with a stable sort + segment medians.
    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    fl_sorted = fl[order]
    # boundaries between bins
    bounds = np.searchsorted(idx_sorted, np.arange(nbins + 1))
    for b in range(nbins):
        s, e = bounds[b], bounds[b + 1]
        if e > s:
            out[b] = np.median(fl_sorted[s:e])
    return out


def _normalize_view(binned: np.ndarray) -> np.ndarray:
    """Normalise to median 0 and transit depth ~ -1 (AstroNet convention).

    Subtract the median, then divide by the absolute minimum so the deepest point
    sits at about -1. Falls back to a plain median subtraction when the signal is
    flat (no dip) to avoid division blow-up.
    """
    filled = _fill_empty(binned)
    med = np.nanmedian(filled)
    centered = filled - med
    min_val = np.nanmin(centered)
    if np.isfinite(min_val) and min_val < 0:
        scale = abs(min_val)
        if scale > 1e-12:
            return centered / scale
    return centered


def build_views(
    lc: LightCurve,
    det: DetectionResult,
    n_global: int = 2001,
    n_local: int = 201,
) -> dict[str, np.ndarray]:
    """Build fixed-length phase-folded views for the view-based CNN.

    Parameters
    ----------
    lc:
        The (detrended) light curve.
    det:
        Detection providing the ephemeris used to fold.
    n_global:
        Bin count for the full-orbit global view (default 2001).
    n_local:
        Bin count for the local (transit-zoom) views (default 201). The local,
        secondary, odd, and even views all share this length.

    Returns
    -------
    dict[str, numpy.ndarray]
        Keys ``global`` (len ``n_global``), ``local``, ``secondary``, ``odd``,
        ``even`` (each len ``n_local``). All arrays are finite, ``float32``,
        normalised to median 0 / depth ~ -1. If the ephemeris is unusable every
        view is returned as zeros of the requested length.
    """
    zeros_g = np.zeros(n_global, dtype=np.float32)
    zeros_l = np.zeros(n_local, dtype=np.float32)
    empty = {
        "global": zeros_g.copy(),
        "local": zeros_l.copy(),
        "secondary": zeros_l.copy(),
        "odd": zeros_l.copy(),
        "even": zeros_l.copy(),
    }

    period = _f(det.period)
    t0 = _f(det.t0)
    duration = _f(det.duration)
    if not (np.isfinite(period) and period > 0):
        return empty

    try:
        time = np.asarray(lc.time, dtype=np.float64)
        flux = np.asarray(lc.flux, dtype=np.float64)
        good = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[good], flux[good]
    except Exception:
        return empty
    if time.size < 8:
        return empty

    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5  # [-0.5, 0.5)

    # Local half-window in phase units: ~3 durations (fallback 5% of period).
    if np.isfinite(duration) and duration > 0:
        local_half = min(0.49, 3.0 * duration / period)
    else:
        local_half = 0.05
    local_half = max(local_half, 1.0 / n_local)  # at least a few bins wide

    views: dict[str, np.ndarray] = {}

    # Global view: full orbit.
    views["global"] = _normalize_view(_bin_phase(phase, flux, -0.5, 0.5, n_global))

    # Local view: zoom on the primary at phase 0.
    views["local"] = _normalize_view(
        _bin_phase(phase, flux, -local_half, local_half, n_local)
    )

    # Secondary view: zoom around phase 0.5 (re-fold with a +0.5 shift).
    phase_sec = (((time - t0) / period) % 1.0) - 0.5  # primary at -0.5/0.5, secondary at 0
    views["secondary"] = _normalize_view(
        _bin_phase(phase_sec, flux, -local_half, local_half, n_local)
    )

    # Odd / even local views.
    cycle = np.floor(((time - t0) / period) + 0.5).astype(np.int64)
    odd = cycle % 2 != 0
    even = ~odd
    views["odd"] = _normalize_view(
        _bin_phase(phase[odd], flux[odd], -local_half, local_half, n_local)
    )
    views["even"] = _normalize_view(
        _bin_phase(phase[even], flux[even], -local_half, local_half, n_local)
    )

    # Enforce dtype + exact lengths.
    out: dict[str, np.ndarray] = {}
    for key, length in (
        ("global", n_global),
        ("local", n_local),
        ("secondary", n_local),
        ("odd", n_local),
        ("even", n_local),
    ):
        arr = np.asarray(views.get(key, np.zeros(length)), dtype=np.float32)
        if arr.size != length:  # defensive: pad/trim to contract length
            fixed = np.zeros(length, dtype=np.float32)
            m = min(length, arr.size)
            fixed[:m] = arr[:m]
            arr = fixed
        out[key] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out
