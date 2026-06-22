"""Detection significance / SNR metrics for ``exopipe`` (PS7 requirement R4).

Implements the four headline significance numbers from research dossier
`02_detection_detrending.md` §C, each with a pure-NumPy fallback so the core path
runs with no optional dependencies:

* :func:`transit_snr` -- the workhorse per-candidate SNR,
  ``depth / (sigma / sqrt(N_in_transit))`` with ``N_in_transit`` derived from the
  transit duration, the observing cadence, and the number of transits in the
  baseline.
* :func:`cdpp` -- Combined Differential Photometric Precision (ppm): the effective
  white-noise floor seen by a transit of a given duration. Uses
  ``lightkurve.estimate_cdpp`` when available, else a Savitzky-Golay-flatten +
  windowed-RMS fallback (the conventional CDPP recipe).
* :func:`bootstrap_fap` -- a permutation / scramble false-alarm probability of the
  detected SDE peak, recalibrating the (white-noise) SDE threshold for the real
  (red) noise in the data.

Plus small helpers :func:`sde_from_power` and :func:`mes`.

All functions accept the shared :class:`~exopipe.types.LightCurve` /
:class:`~exopipe.types.DetectionResult` dataclasses and never mutate them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import DetectionResult, LightCurve
from .utils import get_logger, phase_fold, robust_std

__all__ = [
    "transit_snr",
    "cdpp",
    "bootstrap_fap",
    "sde_from_power",
    "mes",
]

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def sde_from_power(power: np.ndarray) -> float:
    """Signal Detection Efficiency of a periodogram: ``(P_max - <P>) / sigma(P)``.

    This is the standard TLS/BLS significance statistic (Hippke & Heller 2019):
    how many standard deviations the peak power sits above the spectrum mean.
    Returns ``nan`` for an empty/degenerate spectrum.
    """
    p = np.asarray(power, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    std = p.std()
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float((p.max() - p.mean()) / std)


def _cadence_days(lc: LightCurve) -> float:
    """Best estimate of the sampling cadence in days.

    Prefers ``meta['cadence_s']`` (seconds), else the median time difference.
    """
    cadence_s = lc.meta.get("cadence_s")
    if cadence_s is not None and np.isfinite(cadence_s) and cadence_s > 0:
        return float(cadence_s) / 86400.0
    t = np.asarray(lc.time, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    dt = np.median(np.diff(np.sort(t)))
    return float(dt) if np.isfinite(dt) and dt > 0 else float("nan")


def _count_transits(lc: LightCurve, det: DetectionResult) -> int:
    """Number of *distinct* transit epochs with at least one in-transit cadence.

    Counts how many integer transit cycles within the data baseline actually have
    a cadence falling inside ``+/- duration/2`` of mid-transit -- the physically
    meaningful ``N_transits`` (gaps can remove some).
    """
    period = float(det.period)
    duration = float(det.duration)
    t0 = float(det.t0)
    t = np.asarray(lc.time, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size == 0 or not np.isfinite(period) or period <= 0:
        return 0
    if not np.isfinite(duration) or duration <= 0:
        duration = 0.05 * period
    phase = phase_fold(t, period, t0)  # in [-0.5, 0.5), transit at 0
    in_transit = np.abs(phase) <= (0.5 * duration / period)
    if not np.any(in_transit):
        return 0
    cycles = np.round((t[in_transit] - t0) / period).astype(np.int64)
    return int(np.unique(cycles).size)


# --------------------------------------------------------------------------- #
# CDPP
# --------------------------------------------------------------------------- #
def cdpp(lc: LightCurve, duration_hours: float = 2.0) -> float:
    """Combined Differential Photometric Precision in **ppm** at ``duration_hours``.

    CDPP is the RMS scatter a transit of the given duration "sees" after long-term
    trends are removed (Christiansen et al. 2012). It is the natural noise floor
    ``sigma`` for the transit-SNR formula.

    Uses ``lightkurve.LightCurve.estimate_cdpp`` when ``lightkurve`` is importable
    (the reference implementation); otherwise falls back to the conventional
    recipe: Savitzky-Golay flatten, then the RMS of the flux binned to the transit
    duration, expressed in ppm.

    Parameters
    ----------
    lc:
        Light curve (already normalised to ~1.0). It need not be detrended; the
        estimator removes a smooth trend internally.
    duration_hours:
        Transit timescale over which to measure the precision (default 2 h).
    """
    flux = np.asarray(lc.flux, dtype=np.float64)
    time = np.asarray(lc.time, dtype=np.float64)
    finite = np.isfinite(flux) & np.isfinite(time)
    if finite.sum() < 5:
        return float("nan")

    cadence_d = _cadence_days(lc)

    # --- preferred path: lightkurve --------------------------------------- #
    try:  # optional dependency
        import lightkurve as lk  # type: ignore

        # lightkurve expects ``transit_duration`` in *cadences*, so convert from
        # the requested duration (hours) using the actual sampling cadence.
        if np.isfinite(cadence_d) and cadence_d > 0:
            n_cad = max(int(round((duration_hours / 24.0) / cadence_d)), 1)
        else:
            n_cad = max(int(round(duration_hours / 0.5)), 1)  # assume 30-min
        klc = lk.LightCurve(time=time[finite], flux=flux[finite])
        value = klc.estimate_cdpp(transit_duration=n_cad)
        val = float(getattr(value, "value", value))
        if np.isfinite(val):
            return val
    except Exception:
        pass

    # --- fallback: Savgol flatten + windowed RMS -------------------------- #
    f = flux[finite]
    t = time[finite]
    if not np.isfinite(cadence_d) or cadence_d <= 0:
        cadence_d = float(np.median(np.diff(np.sort(t)))) if t.size > 1 else 1.0 / 720.0

    # Number of cadences spanned by the transit duration.
    n_bin = max(int(round((duration_hours / 24.0) / cadence_d)), 1)

    # Flatten with Savgol (robust to absent scipy? scipy is a core dep).
    try:
        from scipy.signal import savgol_filter

        w = max(n_bin * 6 + 1, 11)
        if w % 2 == 0:
            w += 1
        if w >= f.size:
            w = f.size - 1 if (f.size - 1) % 2 == 1 else f.size - 2
        if w >= 5:
            trend = savgol_filter(f, w, 2)
            flat = f / np.where(trend != 0, trend, 1.0)
        else:
            flat = f / np.nanmedian(f)
    except Exception:
        flat = f / np.nanmedian(f)

    # RMS of the duration-binned flux: bin then take robust std, scale to ppm.
    n_full = (flat.size // n_bin) * n_bin
    if n_full < n_bin:
        binned = np.atleast_1d(np.nanmean(flat))
    else:
        binned = flat[:n_full].reshape(-1, n_bin).mean(axis=1)
    scatter = robust_std(binned)
    if not np.isfinite(scatter):
        scatter = float(np.nanstd(binned))
    return float(scatter * 1e6)


# --------------------------------------------------------------------------- #
# Transit SNR
# --------------------------------------------------------------------------- #
def transit_snr(lc: LightCurve, det: DetectionResult) -> float:
    """Per-candidate transit signal-to-noise ratio.

    Computes ``SNR = depth / (sigma_eff / sqrt(N_in_transit))`` where

    * ``depth`` is the fractional transit depth from ``det`` (falls back to the
      folded in/out flux difference if ``det.depth`` is missing),
    * ``sigma_eff`` is the robust per-point out-of-transit scatter,
    * ``N_in_transit = (duration / cadence) * n_transits`` is the total number of
      in-transit cadences, with ``n_transits`` counted from the data.

    Equivalent to the dossier's ``(depth / CDPP) * sqrt(N_tr)`` form but computed
    directly from the per-point scatter so it needs no extra assumptions. Returns
    ``nan`` if the detection is too degenerate to evaluate.
    """
    period = float(det.period)
    duration = float(det.duration)
    t0 = float(det.t0)
    depth = float(det.depth)

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if time.size < 5 or not np.isfinite(period) or period <= 0:
        return float("nan")
    if not np.isfinite(duration) or duration <= 0:
        duration = 0.05 * period

    phase = phase_fold(time, period, t0)
    half_dur_phase = 0.5 * duration / period
    in_transit = np.abs(phase) <= half_dur_phase
    out_transit = ~in_transit

    # Depth fallback from the fold if the detector did not provide one.
    if not np.isfinite(depth) or depth <= 0:
        if in_transit.sum() >= 1 and out_transit.sum() >= 3:
            depth = float(np.nanmedian(flux[out_transit]) - np.nanmedian(flux[in_transit]))
        else:
            return float("nan")
    if depth <= 0:
        return float("nan")

    # Out-of-transit robust scatter is the per-point noise.
    sigma = robust_std(flux[out_transit]) if out_transit.sum() >= 3 else robust_std(flux)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")

    cadence_d = _cadence_days(lc)
    n_transits = _count_transits(lc, det)
    if n_transits < 1:
        n_transits = 1
    if np.isfinite(cadence_d) and cadence_d > 0:
        n_in = (duration / cadence_d) * n_transits
    else:
        n_in = float(max(int(in_transit.sum()), 1))
    n_in = max(n_in, 1.0)

    return float(depth / (sigma / np.sqrt(n_in)))


def mes(
    depth: float,
    cdpp_ppm: float,
    n_transits: int,
) -> float:
    """Multiple Event Statistic (Kepler/TPS-style folded matched-filter SNR).

    ``MES ~ (depth / CDPP) * sqrt(N_transits)`` -- the per-transit single-event
    statistics combined in quadrature. ``depth`` is fractional, ``cdpp_ppm`` is in
    ppm. Kepler's threshold-crossing-event cut is ``MES >= 7.1``. Returns ``nan``
    on degenerate input.
    """
    if not np.isfinite(depth) or depth <= 0:
        return float("nan")
    if not np.isfinite(cdpp_ppm) or cdpp_ppm <= 0:
        return float("nan")
    n = max(int(n_transits), 1)
    return float((depth / (cdpp_ppm * 1e-6)) * np.sqrt(n))


# --------------------------------------------------------------------------- #
# Bootstrap FAP
# --------------------------------------------------------------------------- #
def bootstrap_fap(
    lc: LightCurve,
    det: DetectionResult,
    n: int = 200,
    *,
    method: Optional[str] = None,
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    seed: int = 0,
) -> float:
    """Permutation false-alarm probability of the detected SDE peak.

    Scrambles the flux (destroying any real periodicity while preserving its noise
    distribution), re-runs the period search ``n`` times, and returns the fraction
    of scrambles whose peak SDE meets or exceeds the observed ``det.sde``. This
    recalibrates the (white-noise) SDE threshold for the actual (often red) noise
    in the data (dossier §C5).

    ``+1`` smoothing is applied to numerator and denominator so a never-exceeded
    peak reports ``1 / (n + 1)`` rather than an over-confident exact zero.

    Parameters
    ----------
    lc, det:
        The light curve and the detection whose significance is being assessed.
    n:
        Number of scramble trials (default 200; raise for a finer FAP floor).
    method:
        Search method to use for the null trials; defaults to ``det.method`` or
        ``'bls'``. The cheap ``'bls'`` is recommended for bootstrapping.
    period_min, period_max:
        Period bounds for the null searches (should match the original search).
    seed:
        RNG seed for reproducible scrambles.

    Returns
    -------
    float
        FAP in ``(0, 1]``. ``nan`` if the observed SDE is unavailable.
    """
    observed_sde = float(det.sde)
    if not np.isfinite(observed_sde):
        # Recompute from the stored power spectrum if possible.
        if det.power is not None:
            observed_sde = sde_from_power(det.power)
        if not np.isfinite(observed_sde):
            return float("nan")

    # Lazy import to avoid a circular import at module load time.
    from .search import search

    search_method = (method or det.method or "bls").lower()
    # Use the cheap BLS for the null distribution regardless of a fancy original.
    if search_method not in ("bls", "ls", "acf"):
        search_method = "bls"

    flux = np.asarray(lc.flux, dtype=np.float64)
    finite = np.isfinite(flux) & np.isfinite(np.asarray(lc.time, dtype=np.float64))
    if finite.sum() < 10:
        return float("nan")

    rng = np.random.default_rng(seed)
    base = lc.copy()
    n_exceed = 0
    n_done = 0
    for _ in range(max(int(n), 1)):
        trial = base.copy()
        permuted = trial.flux.copy()
        idx = np.flatnonzero(np.isfinite(permuted))
        if idx.size < 3:
            continue
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        permuted[idx] = permuted[shuffled]
        trial.flux = permuted
        try:
            res = search(
                trial,
                method=search_method,
                period_min=period_min,
                period_max=period_max,
            )
        except Exception:
            continue
        null_sde = res.sde if np.isfinite(res.sde) else sde_from_power(res.power)
        n_done += 1
        if np.isfinite(null_sde) and null_sde >= observed_sde:
            n_exceed += 1

    if n_done == 0:
        return float("nan")
    return float((n_exceed + 1) / (n_done + 1))
