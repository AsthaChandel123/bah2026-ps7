"""Physics-based vetting diagnostics (the 15 named tests of research/03 Section C).

:func:`vet` consumes a :class:`LightCurve` + :class:`DetectionResult` (and,
optionally, a :class:`TransitFit`) and returns a :class:`VettingReport` of
continuous ``metrics`` and boolean ``flags``. Every test is pure NumPy/SciPy and
degrades gracefully when stellar metadata or a fit is missing (the metric becomes
NaN and any flag it would set stays ``False``).

Vetting test -> discriminating class (ARCHITECTURE Section 6 / research/03 C-table)
----------------------------------------------------------------------------------
* **odd-even depth** (``odd_even_depth_sigma``) -> **eclipsing_binary** (true
  period is 2x; alternating eclipse depths). [C1]
* **secondary eclipse** (``secondary_depth``/``secondary_snr``/``secondary_phase``)
  -> **eclipsing_binary** (a self-luminous companion shows an occultation near
  phase 0.5). [C2]
* **V-shape vs U/box** (``v_shape_metric``/``ingress_egress_ratio``) ->
  **eclipsing_binary** (grazing geometry is V-shaped). [C3]
* **implied radius** (``implied_rp_rjup``) -> **eclipsing_binary**
  (``Rp >~ 2 R_jup`` is stellar). [C7]
* **stellar-density consistency** (``stellar_density_ratio``) ->
  **blend / EB / wrong-period** (transit-implied rho* vs catalogue rho*). [C8]
* **aperture contamination** (``crowdsap``) -> **blend** (diluted depth from a
  crowded aperture). [C5]
* **SWEET sine power** (``sweet_metric``) -> **other** (sinusoidal stellar
  variability at the search period). [C9]
* **uniqueness / secondary-to-primary** (``uniqueness``) -> **other / EB**
  (a strong rival event means the primary is not unique). [C10]
* **transit SNR / n_transits** (``transit_snr``, ``n_transits``) -> **other**
  (low-significance / single-event signals). [C11]

Flags produced: ``eb_odd_even``, ``eb_secondary``, ``eb_vshape``,
``blend_contamination``, ``other_variability``, ``low_snr``,
``implied_radius_too_big``, ``density_inconsistent``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model.transit import (
    a_rs_from_duration,
    density_from_a_rs,
    rp_rjup_from_depth,
    rp_rs_from_depth,
    trapezoid_model,
)
from .types import DetectionResult, LightCurve, TransitFit, VettingReport
from .utils import get_logger, robust_std

__all__ = ["vet"]

_LOG = get_logger("exopipe.vetting")

# Solar mean density (kg/m^3) for the stellar-density consistency test.
_RHO_SUN = 1408.0
_RSUN_M = 6.957e8
_MSUN_KG = 1.98892e30

# Decision thresholds (documented inline; tuned for the synthetic generator and
# the literature values in research/03).
_ODD_EVEN_SIGMA_THRESH = 5.0  # >5 sigma odd/even depth diff -> EB (folded N is huge)
_SECONDARY_SNR_THRESH = 5.0  # significant secondary detection -> EB
# A genuine occultation is a meaningful fraction of the primary; a planet's
# secondary / residual systematics sit far below this. Requiring both a
# significant SNR *and* this depth ratio rejects noise "secondaries" that, when
# binned over a long baseline, would otherwise look statistically significant.
_SECONDARY_RATIO_LO = 0.15  # secondary depth >= 15% of primary -> EB-like
_SECONDARY_RATIO_HI = 0.95  # a real occultation is shallower than the primary;
#                             ratios near/above 1 indicate a noise-corrupted
#                             primary depth, not a genuine secondary.
# Odd-even sigma is inflated by tiny white-noise errors over a long baseline, so
# also require a meaningful *fractional* depth difference (red-noise/systematics
# rarely produce both a large sigma and a several-percent depth asymmetry).
_ODD_EVEN_FRAC_THRESH = 0.10  # >10% odd/even fractional depth difference (a
#                               specific cut: the secondary-eclipse test already
#                               catches EBs, so prefer few odd-even false alarms)
# The EB sub-tests (odd-even, secondary ratio) are only trusted when the primary
# is robustly measured; below this the depth is buried in red noise.
_PRIMARY_SIG_THRESH = 10.0  # measured primary depth must be >10 sigma
_VSHAPE_THRESH = 0.35  # ingress/total >~0.35 (little/no flat floor) -> V-shaped EB
_CROWDSAP_THRESH = 0.9  # CROWDSAP < 0.9 -> blend-prone (diluted aperture)
_SWEET_THRESH = 0.5  # sinusoid explains >50% of variance -> variability
_LOW_SNR_THRESH = 7.1  # Kepler/TESS TCE MES floor
_RP_TOO_BIG_RJUP = 2.0  # Rp > 2 R_jup -> stellar companion
_DENSITY_RATIO_LO = 0.2  # transit rho* far below catalogue -> inconsistent
_DENSITY_RATIO_HI = 5.0
_UNIQUENESS_THRESH = 0.9  # rival event >=90% as deep as primary -> not unique


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _clean(lc: LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite ``(time, flux, flux_err)`` with a fallback error estimate."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    err = np.asarray(lc.flux_err, dtype=np.float64)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    err = err[good] if err.size == good.size else np.full(time.size, np.nan)
    bad = ~np.isfinite(err) | (err <= 0)
    if np.any(bad):
        scatter = robust_std(flux)
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = float(np.nanstd(flux)) or 1e-3
        err = np.where(bad, scatter, err)
    return time, flux, err


def _phase_days(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from nearest mid-transit, in days (phase 0 at the transit)."""
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
    return phase * period


def _depth_in_window(
    flux: np.ndarray,
    err: np.ndarray,
    dt: np.ndarray,
    half_dur: float,
    oot_lo: float,
    oot_hi: float,
) -> tuple[float, float, int]:
    """Measure depth (and its uncertainty) inside a transit window.

    Depth = (median out-of-transit baseline) - (mean in-transit flux). The
    baseline is taken from an annulus ``[oot_lo, oot_hi]`` in |dt|. Returns
    ``(depth, depth_err, n_in)``; depth is positive for a dip.
    """
    in_mask = np.abs(dt) <= half_dur
    oot_mask = (np.abs(dt) >= oot_lo) & (np.abs(dt) <= oot_hi)
    n_in = int(np.count_nonzero(in_mask))
    if n_in == 0 or np.count_nonzero(oot_mask) < 3:
        return (np.nan, np.nan, n_in)
    baseline = float(np.nanmedian(flux[oot_mask]))
    in_flux = flux[in_mask]
    depth = baseline - float(np.nanmean(in_flux))
    # Error on the in-transit mean.
    sigma = float(np.nanmedian(err[in_mask]))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(in_flux))
    depth_err = sigma / np.sqrt(max(n_in, 1))
    return (float(depth), float(depth_err), n_in)


# --------------------------------------------------------------------------- #
# Individual tests
# --------------------------------------------------------------------------- #
def _odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> dict[str, float]:
    """C1 -- odd vs even transit depth. Large difference => EB at half period."""
    half = 0.5 * duration
    oot_lo, oot_hi = 0.75 * duration, 2.5 * duration
    cycle = np.floor(((time - t0) / period) + 0.5).astype(np.int64)
    dt = _phase_days(time, period, t0)

    odd = cycle % 2 != 0
    even = ~odd
    d_odd, e_odd, n_odd = _depth_in_window(
        flux[odd], err[odd], dt[odd], half, oot_lo, oot_hi
    )
    d_even, e_even, n_even = _depth_in_window(
        flux[even], err[even], dt[even], half, oot_lo, oot_hi
    )
    sigma_diff = np.nan
    frac_diff = np.nan
    if _finite(d_odd) and _finite(d_even) and _finite(e_odd) and _finite(e_even):
        denom = np.sqrt(e_odd**2 + e_even**2)
        if denom > 0:
            sigma_diff = abs(d_odd - d_even) / denom
        scale = max(abs(d_odd), abs(d_even))
        if scale > 0:
            frac_diff = abs(d_odd - d_even) / scale
    return {
        "odd_depth": float(d_odd),
        "even_depth": float(d_even),
        "odd_even_depth_sigma": float(sigma_diff),
        "odd_even_frac_diff": float(frac_diff),
    }


def _secondary(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    primary_depth: float,
) -> dict[str, float]:
    """C2 -- search for an occultation. A significant secondary => EB.

    Scans phases away from the primary for the deepest dip, then reports its
    depth, SNR, and phase. The strongest dip near phase 0.5 is the canonical EB
    secondary, but the scan covers all out-of-primary phases so eccentric
    secondaries are caught too.
    """
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5  # [-0.5, 0.5)
    half_phase = 0.5 * duration / period
    baseline = float(np.nanmedian(flux))
    sigma = float(np.nanmedian(err))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(robust_std(flux)) or 1e-3

    # Candidate secondary-centre phases: scan a grid, exclude the primary (and
    # its wings) at phase 0.
    grid = np.linspace(-0.5, 0.5, 101)
    best = {"depth": np.nan, "snr": -np.inf, "phase": np.nan}
    for ph in grid:
        if abs(ph) < 2.0 * half_phase:  # skip primary region
            continue
        sel = np.abs(((phase - ph + 0.5) % 1.0) - 0.5) <= half_phase
        n_in = int(np.count_nonzero(sel))
        if n_in < 3:
            continue
        depth = baseline - float(np.nanmean(flux[sel]))
        snr = depth / (sigma / np.sqrt(n_in))
        if snr > best["snr"]:
            best = {"depth": float(depth), "snr": float(snr), "phase": float(ph)}

    sec_snr = best["snr"] if np.isfinite(best["snr"]) else np.nan
    # Express the secondary relative to the primary depth (albedo-style ratio).
    ratio = np.nan
    if _finite(primary_depth) and primary_depth > 0 and _finite(best["depth"]):
        ratio = best["depth"] / primary_depth
    return {
        "secondary_depth": float(best["depth"]),
        "secondary_snr": float(sec_snr),
        "secondary_phase": float(best["phase"]),
        "secondary_to_primary": float(ratio),
    }


def _v_shape(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
) -> dict[str, float]:
    """C3 -- transit shape. Fit ingress fraction; V-shaped (no floor) => EB.

    Fits a trapezoid's ``ingress_frac`` (and depth) around the folded primary.
    ``ingress_egress_ratio`` ~ 0.5 means the ingress+egress fill the whole event
    (a V), while ~0.1 means a flat-bottomed (planet-like) U. ``v_shape_metric``
    is the chi-square of the best trapezoid normalised by the data scatter (a
    crude goodness measure, larger when the box model fits poorly).
    """
    from scipy.optimize import least_squares

    dt = _phase_days(time, period, t0)
    win = np.abs(dt) <= 3.0 * duration
    if np.count_nonzero(win) < 8 or not _finite(depth) or depth <= 0:
        return {
            "v_shape_metric": np.nan,
            "ingress_egress_ratio": np.nan,
            "trapezoid_chi2": np.nan,
        }
    tw, fw, ew = time[win], flux[win], err[win]

    def resid(theta: np.ndarray) -> np.ndarray:
        d, ing = theta
        model = trapezoid_model(tw, t0, d, duration, ingress_frac=ing, period=period)
        return (fw - model) / ew

    try:
        sol = least_squares(
            resid,
            np.array([depth, 0.2]),
            bounds=([1e-6, 0.0], [0.95, 0.5]),
            method="trf",
            max_nfev=2000,
        )
        ing_frac = float(sol.x[1])
        chi2 = float(2.0 * sol.cost)
        dof = max(1, tw.size - 2)
        chi2_red = chi2 / dof
    except Exception:
        ing_frac, chi2_red = np.nan, np.nan

    return {
        # ingress_egress_ratio: each ramp as a fraction of total duration.
        "ingress_egress_ratio": float(ing_frac),
        # v_shape_metric: 1 = pure V (ing_frac=0.5), 0 = box. Higher => more EB-like.
        "v_shape_metric": float(ing_frac / 0.5) if _finite(ing_frac) else np.nan,
        "trapezoid_chi2": float(chi2_red),
    }


def _sweet(
    time: np.ndarray, flux: np.ndarray, period: float
) -> float:
    """C9 -- SWEET-style sinusoid test at the search period.

    Fits a single sinusoid (sin+cos at ``period``) by linear least squares and
    returns the fraction of the flux variance it explains. A high value means the
    "transit" is really smooth sinusoidal variability (=> ``other``).
    """
    if not _finite(period) or period <= 0 or time.size < 8:
        return float("nan")
    var = float(np.nanvar(flux))
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    omega = 2.0 * np.pi / period
    design = np.column_stack(
        [np.ones_like(time), np.sin(omega * time), np.cos(omega * time)]
    )
    f = flux - np.nanmean(flux)
    try:
        coef, *_ = np.linalg.lstsq(design, flux, rcond=None)
        model = design @ coef
        resid_var = float(np.nanvar(flux - model))
        explained = 1.0 - resid_var / var
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.clip(explained, 0.0, 1.0))


def _uniqueness(
    secondary_snr: float, transit_snr: float
) -> float:
    """C10 -- model-shift uniqueness: strongest rival event vs the primary.

    Ratio of the best secondary/tertiary SNR to the primary transit SNR. Near 1
    means the primary is *not* unique (a rival dip is equally strong) -- a
    systematics or EB signature.
    """
    if not _finite(secondary_snr) or not _finite(transit_snr) or transit_snr <= 0:
        return float("nan")
    return float(max(secondary_snr, 0.0) / transit_snr)


def _n_transits(time: np.ndarray, period: float, t0: float, duration: float) -> int:
    """C11 helper -- count epochs that actually contain in-window cadences."""
    if not _finite(period) or period <= 0:
        return 0
    cycle = np.floor(((time - t0) / period) + 0.5).astype(np.int64)
    dt = _phase_days(time, period, t0)
    in_win = np.abs(dt) <= 0.5 * duration if _finite(duration) and duration > 0 else None
    if in_win is None:
        return int(np.unique(cycle).size)
    return int(np.unique(cycle[in_win]).size)


def _implied_radius(depth: float, stellar_radius: float) -> float:
    """C7 -- implied planet radius (R_jup) from depth & stellar radius."""
    if not _finite(stellar_radius):
        return float("nan")
    return rp_rjup_from_depth(depth, stellar_radius)


def _density_ratio(
    period: float,
    duration: float,
    depth: float,
    fit: TransitFit | None,
    meta: dict[str, Any],
) -> float:
    """C8 -- transit-implied stellar density vs the catalogue value.

    ``rho*_transit`` from ``(P, a/R*)`` (Seager & Mallen-Ornelas) where ``a/R*``
    is taken from the fit if present, else inverted from the duration. The
    catalogue ``rho*`` comes from ``mass``/``radius`` in ``meta``. A ratio far
    from 1 flags a blend, an EB, or the wrong period (e.g. eccentric orbit).
    Returns ``rho*_transit / rho*_catalogue``.
    """
    radius = meta.get("radius")
    mass = meta.get("mass")
    if not _finite(radius) or not _finite(mass) or float(radius) <= 0:
        return float("nan")
    rho_cat = float(mass) * _MSUN_KG / (
        4.0 / 3.0 * np.pi * (float(radius) * _RSUN_M) ** 3
    )
    if rho_cat <= 0:
        return float("nan")

    a_rs = np.nan
    if fit is not None and fit.params:
        a_rs_triple = fit.params.get("a_rs")
        if a_rs_triple is not None:
            a_rs = float(np.atleast_1d(a_rs_triple)[0])
    if not _finite(a_rs) or a_rs <= 1.0:
        k = rp_rs_from_depth(depth) if _finite(depth) and depth > 0 else 0.1
        a_rs = a_rs_from_duration(period, duration, k, 0.3)
    if not _finite(a_rs) or a_rs <= 1.0:
        return float("nan")

    rho_transit = density_from_a_rs(period, a_rs)
    if not _finite(rho_transit) or rho_transit <= 0:
        return float("nan")
    return float(rho_transit / rho_cat)


def _transit_snr_from_fit_or_det(
    fit: TransitFit | None, det: DetectionResult, depth: float, flux: np.ndarray, err: np.ndarray, n_in: int
) -> float:
    """Headline transit SNR: prefer the fit, then detection, then recompute."""
    if fit is not None and _finite(fit.snr):
        return float(fit.snr)
    if _finite(det.snr):
        return float(det.snr)
    if _finite(depth) and depth > 0 and n_in > 0:
        sigma = float(np.nanmedian(err))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(robust_std(flux)) or 1e-3
        return float(depth / sigma * np.sqrt(n_in))
    return float("nan")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def vet(
    lc: LightCurve, det: DetectionResult, fit: TransitFit | None = None
) -> VettingReport:
    """Run the physics-based vetting suite on a detected signal.

    Parameters
    ----------
    lc:
        The light curve (uses ``meta`` for stellar params and ``crowdsap``).
    det:
        The detection providing the ephemeris (period/t0/duration/depth).
    fit:
        Optional :class:`TransitFit`; when present its ``a_rs`` and ``snr`` enrich
        the density and SNR metrics.

    Returns
    -------
    VettingReport
        ``metrics`` (continuous diagnostics, all keys always present, NaN when
        not computable) and ``flags`` (booleans driving the §7 physics veto).
    """
    meta = lc.meta if lc.meta is not None else {}
    period = float(det.period)
    t0 = float(det.t0)
    duration = float(det.duration)
    depth = float(det.depth)

    metrics: dict[str, float] = {}
    flags: dict[str, bool] = {}

    # crowdsap is always available from meta (independent of a valid ephemeris).
    crowdsap = meta.get("crowdsap", np.nan)
    metrics["crowdsap"] = float(crowdsap) if _finite(crowdsap) else np.nan

    # Guard: without a usable period we can only report NaNs + crowdsap/implied R.
    valid_ephemeris = _finite(period) and period > 0 and _finite(duration) and duration > 0
    try:
        time, flux, err = _clean(lc)
    except Exception:
        valid_ephemeris = False
        time = flux = err = np.empty(0)

    if not valid_ephemeris or time.size < 8:
        for key in (
            "odd_even_depth_sigma",
            "odd_even_frac_diff",
            "odd_depth",
            "even_depth",
            "secondary_depth",
            "secondary_snr",
            "secondary_phase",
            "secondary_to_primary",
            "v_shape_metric",
            "trapezoid_chi2",
            "ingress_egress_ratio",
            "stellar_density_ratio",
            "sweet_metric",
            "transit_snr",
            "uniqueness",
            "primary_significance",
        ):
            metrics.setdefault(key, np.nan)
        metrics["n_transits"] = float(_n_transits(time, period, t0, duration)) if time.size else np.nan
        metrics["implied_rp_rjup"] = _implied_radius(depth, meta.get("radius"))
        _populate_flags(metrics, flags, meta)
        return VettingReport(metrics=metrics, flags=flags)

    # --- the tests --------------------------------------------------------- #
    oe = _odd_even(time, flux, err, period, t0, duration)
    metrics.update(oe)

    sec = _secondary(time, flux, err, period, t0, duration, depth)
    metrics.update(sec)

    vs = _v_shape(time, flux, err, period, t0, duration, depth)
    metrics.update(vs)

    metrics["sweet_metric"] = _sweet(time, flux, period)
    metrics["n_transits"] = float(_n_transits(time, period, t0, duration))
    metrics["implied_rp_rjup"] = _implied_radius(depth, meta.get("radius"))
    metrics["stellar_density_ratio"] = _density_ratio(period, duration, depth, fit, meta)

    # Transit SNR (count in-transit cadences for the recompute path).
    dt = _phase_days(time, period, t0)
    n_in = int(np.count_nonzero(np.abs(dt) <= 0.5 * duration))
    metrics["transit_snr"] = _transit_snr_from_fit_or_det(fit, det, depth, flux, err, n_in)
    metrics["uniqueness"] = _uniqueness(
        metrics.get("secondary_snr", np.nan), metrics["transit_snr"]
    )

    # Significance of the *measured* primary depth (from the folded data, not the
    # seed). The odd-even and secondary-ratio sub-tests are only trustworthy when
    # the primary itself is robustly detected -- on faint, low-per-point-SNR stars
    # the measured primary depth is dominated by red noise and those ratios become
    # meaningless. This gate keeps the EB tests honest. [research/03 C1/C2]
    d_prim, e_prim, _ = _depth_in_window(
        flux, err, dt, 0.5 * duration, 0.75 * duration, 2.5 * duration
    )
    prim_sig = abs(d_prim) / e_prim if (_finite(d_prim) and _finite(e_prim) and e_prim > 0) else np.nan
    metrics["primary_significance"] = float(prim_sig)

    _populate_flags(metrics, flags, meta)
    return VettingReport(metrics=metrics, flags=flags)


def _populate_flags(
    metrics: dict[str, float], flags: dict[str, bool], meta: dict[str, Any]
) -> None:
    """Derive boolean false-positive flags from the continuous metrics.

    Each flag maps to the class it discriminates (see module docstring). A flag
    is only ``True`` when its driving metric is finite *and* crosses threshold;
    missing data leaves the flag ``False`` (conservative).
    """
    # Gate the EB sub-tests on a robustly-measured primary (missing -> trust the
    # test, so a fit-free call still flags obvious EBs).
    prim_sig = metrics.get("primary_significance", np.nan)
    primary_ok = (not _finite(prim_sig)) or prim_sig >= _PRIMARY_SIG_THRESH

    oe_sigma = metrics.get("odd_even_depth_sigma", np.nan)
    oe_frac = metrics.get("odd_even_frac_diff", np.nan)
    flags["eb_odd_even"] = bool(
        primary_ok
        and _finite(oe_sigma)
        and oe_sigma > _ODD_EVEN_SIGMA_THRESH
        and _finite(oe_frac)
        and oe_frac > _ODD_EVEN_FRAC_THRESH
    )

    sec_snr = metrics.get("secondary_snr", np.nan)
    sec_ratio = metrics.get("secondary_to_primary", np.nan)
    flags["eb_secondary"] = bool(
        primary_ok
        and _finite(sec_snr)
        and sec_snr > _SECONDARY_SNR_THRESH
        and _finite(sec_ratio)
        and _SECONDARY_RATIO_LO < sec_ratio < _SECONDARY_RATIO_HI
    )

    vshape = metrics.get("ingress_egress_ratio", np.nan)
    flags["eb_vshape"] = bool(_finite(vshape) and vshape > _VSHAPE_THRESH)

    crowdsap = metrics.get("crowdsap", np.nan)
    flags["blend_contamination"] = bool(_finite(crowdsap) and crowdsap < _CROWDSAP_THRESH)

    sweet = metrics.get("sweet_metric", np.nan)
    flags["other_variability"] = bool(_finite(sweet) and sweet > _SWEET_THRESH)

    snr = metrics.get("transit_snr", np.nan)
    flags["low_snr"] = bool(_finite(snr) and snr < _LOW_SNR_THRESH)

    rp = metrics.get("implied_rp_rjup", np.nan)
    flags["implied_radius_too_big"] = bool(_finite(rp) and rp > _RP_TOO_BIG_RJUP)

    ratio = metrics.get("stellar_density_ratio", np.nan)
    flags["density_inconsistent"] = bool(
        _finite(ratio) and (ratio < _DENSITY_RATIO_LO or ratio > _DENSITY_RATIO_HI)
    )
