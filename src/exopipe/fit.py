"""Two-stage transit parameter estimation with calibrated uncertainties.

Implements :func:`fit_transit`, the PS7 parameter-estimation engine
(ARCHITECTURE Section 8, research/04). The fit proceeds in up to two stages:

1. **Trapezoid least-squares seed (always).** A pure-NumPy/SciPy trapezoid model
   (:func:`exopipe.model.transit.trapezoid_model`) is fit by
   ``scipy.optimize.least_squares`` seeded from the :class:`DetectionResult`,
   giving refined point estimates of depth/duration/t0/period and
   covariance-matrix 1-sigma errors. This also de-multimodalises and seeds the
   sampler.

2. **Full transit-model MCMC (optional).** When ``batman`` *and* ``emcee`` are
   importable and ``method != 'fast'``, a Mandel & Agol quadratic model is
   sampled with ``emcee`` in the physical parameters
   ``(period, t0, rp_rs, a_rs, b, q1, q2)`` -- limb darkening in the
   **Kipping (2013)** ``(q1, q2)`` reparameterisation -- producing a posterior
   summarised at the 16/50/84 percentiles. The equal-weight chain is stored in
   :attr:`TransitFit.samples`. (If ``dynesty`` is importable an evidence estimate
   may be added to ``extra`` -- not required, kept lightweight.)

The function **never raises**: any failure degrades to the trapezoid-only result
(or, in the worst case, to a :class:`TransitFit` seeded directly from the
detection with NaN errors), annotating ``extra['warnings']``.

Reported parameters (each ``(median, err_lo, err_hi)``):
``period, t0, depth, duration, rp_rs, a_rs, b, inclination, u1, u2``.
``delta_bic = bic_flat - bic_transit`` quantifies transit-vs-flat significance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model.transit import (
    a_rs_from_duration,
    incl_from_impact,
    rp_rs_from_depth,
    transit_model,
    trapezoid_model,
)
from .types import DetectionResult, LightCurve, TransitFit
from .utils import get_logger, robust_std

__all__ = ["fit_transit"]

_LOG = get_logger("exopipe.fit")

# Parameter order for the trapezoid least-squares solve.
_TRAP_NAMES = ("t0", "depth", "duration", "ingress_frac")
# Parameter order for the emcee sampler.
_MCMC_NAMES = ("period", "t0", "rp_rs", "a_rs", "b", "q1", "q2")


# --------------------------------------------------------------------------- #
# Limb-darkening reparameterisation (Kipping 2013)
# --------------------------------------------------------------------------- #
def _q_to_u(q1: float, q2: float) -> tuple[float, float]:
    """Kipping (2013) ``(q1, q2) -> (u1, u2)`` quadratic-LD inverse map.

    ``u1 = 2 sqrt(q1) q2``; ``u2 = sqrt(q1) (1 - 2 q2)``. Sampling ``q1, q2`` in
    the unit square spans exactly the physically valid LD triangle.
    """
    sq1 = np.sqrt(max(q1, 0.0))
    u1 = 2.0 * sq1 * q2
    u2 = sq1 * (1.0 - 2.0 * q2)
    return float(u1), float(u2)


def _u_to_q(u1: float, u2: float) -> tuple[float, float]:
    """Forward Kipping map ``(u1, u2) -> (q1, q2)`` (for seeding the sampler)."""
    s = u1 + u2
    q1 = s * s
    q2 = 0.5 * u1 / s if abs(s) > 1e-12 else 0.5
    return float(np.clip(q1, 0.0, 1.0)), float(np.clip(q2, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _clean_arrays(lc: LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite ``(time, flux, flux_err)`` with a usable error estimate."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    err = np.asarray(lc.flux_err, dtype=np.float64)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux, err = time[good], flux[good], err[good]
    err = err[: time.size] if err.size >= time.size else np.full(time.size, np.nan)
    # Replace missing / non-positive errors with a robust scatter estimate.
    bad_err = ~np.isfinite(err) | (err <= 0)
    if np.any(bad_err):
        scatter = robust_std(flux)
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = float(np.nanstd(flux)) or 1e-3
        err = np.where(bad_err, scatter, err)
    return time, flux, err


# --------------------------------------------------------------------------- #
# Stage 1: trapezoid least-squares
# --------------------------------------------------------------------------- #
def _trapezoid_lsq(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    period: float,
    seed: dict[str, float],
) -> dict[str, Any]:
    """Least-squares trapezoid fit at fixed period; returns params + covariance.

    Fits ``t0, depth, duration, ingress_frac`` with
    ``scipy.optimize.least_squares`` (TRF, bounded). The covariance is recovered
    from the Jacobian, ``Cov = (J^T J)^-1 * chi2_red``, giving 1-sigma point
    errors per Sec. 3.1 of research/04.
    """
    from scipy.optimize import least_squares

    t0_0 = float(seed["t0"])
    depth0 = max(float(seed["depth"]), 1e-5)
    dur0 = float(seed["duration"])
    if not np.isfinite(dur0) or dur0 <= 0:
        dur0 = 0.1 * period if np.isfinite(period) else 0.1
    theta0 = np.array([t0_0, depth0, dur0, 0.15], dtype=np.float64)

    # The fit is *seeded from a detection that already localised the transit*, so
    # we refine locally rather than re-search: keep t0 within ~1 duration, the
    # duration within a few x the seed, and the depth within ~10x the seed. These
    # tight bounds stop the LM from latching onto a deeper, wider red-noise
    # excursion elsewhere in a low-per-point-SNR light curve.
    dt_window = max(dur0, 0.02 * period if np.isfinite(period) and period > 0 else dur0)
    dur_lo = max(1e-3, 0.2 * dur0)
    dur_hi = min(0.6 * period if np.isfinite(period) and period > 0 else 5.0 * dur0, 5.0 * dur0)
    dur_hi = max(dur_hi, dur_lo * 1.5)
    depth_hi = min(0.9, max(10.0 * depth0, 5e-3))
    lb = np.array([t0_0 - dt_window, 1e-6, dur_lo, 0.0])
    ub = np.array([t0_0 + dt_window, depth_hi, dur_hi, 0.5])
    theta0 = np.clip(theta0, lb + 1e-9, ub - 1e-9)

    def residuals(theta: np.ndarray) -> np.ndarray:
        t0, depth, dur, ing = theta
        model = trapezoid_model(time, t0, depth, dur, ingress_frac=ing, period=period)
        return (flux - model) / err

    sol = least_squares(residuals, theta0, bounds=(lb, ub), method="trf", max_nfev=4000)

    # Covariance from the Jacobian, scaled by the reduced chi-square.
    dof = max(1, time.size - theta0.size)
    chi2 = float(2.0 * sol.cost)
    perr = np.full(theta0.size, np.nan)
    try:
        jtj = sol.jac.T @ sol.jac
        cov = np.linalg.inv(jtj) * (chi2 / dof)
        diag = np.diag(cov)
        perr = np.sqrt(np.clip(diag, 0.0, None))
    except np.linalg.LinAlgError:
        pass

    params = dict(zip(_TRAP_NAMES, sol.x))
    errors = dict(zip(_TRAP_NAMES, perr))
    return {
        "params": params,
        "errors": errors,
        "chi2": chi2,
        "dof": dof,
        "success": bool(sol.success),
    }


# --------------------------------------------------------------------------- #
# BIC / significance
# --------------------------------------------------------------------------- #
def _chi2(flux: np.ndarray, model: np.ndarray, err: np.ndarray) -> float:
    return float(np.sum(((flux - model) / err) ** 2))


def _bic(chi2: float, k: int, n: int) -> float:
    """Bayesian Information Criterion ``BIC = k ln n - 2 ln L`` (Gaussian)."""
    return float(k * np.log(max(n, 1)) + chi2)


def _transit_snr(
    flux: np.ndarray, err: np.ndarray, model: np.ndarray, depth: float
) -> float:
    """Phase-folded transit SNR ``= depth / sigma * sqrt(N_in)`` (white box).

    ``N_in`` is the number of in-transit cadences (where the best-fit model dips
    by more than 10% of its depth). Robust to missing depth.
    """
    if not np.isfinite(depth) or depth <= 0:
        return float("nan")
    in_transit = model < (1.0 - 0.1 * depth)
    n_in = int(np.count_nonzero(in_transit))
    if n_in == 0:
        return float("nan")
    sigma = float(np.nanmedian(err))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(robust_std(flux)) or 1e-3
    return float(depth / sigma * np.sqrt(n_in))


# --------------------------------------------------------------------------- #
# Stage 2: emcee posterior
# --------------------------------------------------------------------------- #
def _run_emcee(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    seed_geom: dict[str, float],
    nsteps: int,
    nwalkers: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Sample the batman transit posterior with emcee; return chain + summaries.

    Parameters are ``(period, t0, rp_rs, a_rs, b, q1, q2)`` with uniform priors;
    LD is sampled in Kipping ``(q1, q2)``. Burn-in/thinning use the integrated
    autocorrelation time. Returns ``None`` if batman/emcee are unavailable or the
    run fails.
    """
    try:
        import batman  # type: ignore
        import emcee  # type: ignore
    except Exception:
        return None

    period0 = float(seed_geom["period"])
    t0_0 = float(seed_geom["t0"])
    k0 = float(np.clip(seed_geom["rp_rs"], 1e-3, 0.49))
    a0 = float(np.clip(seed_geom["a_rs"], 1.5, 49.0))
    b0 = float(np.clip(seed_geom["b"], 0.0, 0.95))
    dur0 = float(seed_geom.get("duration", np.nan))
    q1_0, q2_0 = _u_to_q(0.4, 0.3)

    # Prior windows. The trapezoid seed already localises the transit, so we
    # refine locally: t0 within ~2 durations of the seed (NOT a large fraction of
    # the period, which would let a long-period walker drift onto a red-noise
    # excursion), period within +-2%, and a/Rs / k confined to a generous band
    # around the seed so the sampler refines geometry rather than re-searching it.
    if np.isfinite(dur0) and dur0 > 0:
        dt0 = max(2.0 * dur0, 0.005 * period0 if period0 > 0 else 2.0 * dur0)
    else:
        dt0 = 0.02 * period0 if period0 > 0 else 0.1
    dp = 0.02 * period0 if period0 > 0 else 0.05
    a_lo = max(1.5, 0.3 * a0)
    a_hi = min(50.0, 3.0 * a0)
    k_lo = max(1e-3, 0.2 * k0)
    k_hi = min(0.5, 5.0 * k0)
    lb = np.array([period0 - dp, t0_0 - dt0, k_lo, a_lo, 0.0, 0.0, 0.0])
    ub = np.array([period0 + dp, t0_0 + dt0, k_hi, a_hi, 1.0, 1.0, 1.0])

    # Build the batman model once; only update params inside the likelihood.
    bp = batman.TransitParams()
    bp.t0 = t0_0
    bp.per = period0
    bp.rp = k0
    bp.a = a0
    bp.inc = incl_from_impact(a0, b0)
    bp.ecc = 0.0
    bp.w = 90.0
    bp.limb_dark = "quadratic"
    bp.u = [0.4, 0.3]
    bmodel = batman.TransitModel(bp, time)

    def log_prob(theta: np.ndarray) -> float:
        period, t0, k, a_rs, b, q1, q2 = theta
        if not (
            lb[0] < period < ub[0]
            and lb[1] < t0 < ub[1]
            and lb[2] < k < ub[2]
            and lb[3] < a_rs < ub[3]
            and 0.0 <= b < 1.0 + k
            and 0.0 < q1 < 1.0
            and 0.0 < q2 < 1.0
        ):
            return -np.inf
        u1, u2 = _q_to_u(q1, q2)
        bp.per = period
        bp.t0 = t0
        bp.rp = k
        bp.a = a_rs
        bp.inc = incl_from_impact(a_rs, b)
        bp.u = [u1, u2]
        model = bmodel.light_curve(bp)
        if not np.all(np.isfinite(model)):
            return -np.inf
        return -0.5 * np.sum(((flux - model) / err) ** 2 + np.log(2.0 * np.pi * err * err))

    ndim = len(_MCMC_NAMES)
    nwalkers = max(int(nwalkers), 2 * ndim)
    center = np.array([period0, t0_0, k0, a0, b0, q1_0, q2_0])
    scale = np.array([dp * 0.1, dt0 * 0.1, 0.01, 0.5, 0.05, 0.05, 0.05])
    p0 = center + scale * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, int(nsteps), progress=False)

    # Burn-in / thin via autocorrelation time (robust to short chains).
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burn = int(np.nanmax(tau) * 3) if np.all(np.isfinite(tau)) else nsteps // 3
        thin = max(1, int(np.nanmin(tau) * 0.5)) if np.all(np.isfinite(tau)) else 1
    except Exception:
        burn, thin = nsteps // 3, 1
    burn = int(np.clip(burn, 1, max(1, nsteps - 2)))
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if chain.size == 0 or chain.shape[0] < 8:
        chain = sampler.get_chain(discard=nsteps // 2, flat=True)
    if chain.size == 0:
        return None

    # Acceptance fraction for diagnostics.
    try:
        acc = float(np.mean(sampler.acceptance_fraction))
    except Exception:
        acc = float("nan")

    return {
        "chain": chain,
        "labels": list(_MCMC_NAMES),
        "acceptance": acc,
        "burn": burn,
        "thin": thin,
        "n_eff": int(chain.shape[0]),
        "bp": bp,
        "bmodel": bmodel,
    }


def _summarize_mcmc(
    chain: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    """Turn an equal-weight chain into ``name -> (median, err_lo, err_hi)`` triples.

    Derives ``depth = k^2``, ``duration`` (Winn 2010 per-sample), ``inclination``,
    and ``u1, u2`` (Kipping) in addition to the sampled parameters, propagating
    the full ``b``-``a/Rs``-``k`` correlation into every derived error bar.
    """
    period_s = chain[:, 0]
    t0_s = chain[:, 1]
    k_s = chain[:, 2]
    a_s = chain[:, 3]
    b_s = chain[:, 4]
    q1_s = chain[:, 5]
    q2_s = chain[:, 6]

    depth_s = k_s**2
    inc_s = np.degrees(np.arccos(np.clip(b_s / a_s, 0.0, 1.0)))
    # Winn (2010) total duration per sample.
    num = np.clip((1.0 + k_s) ** 2 - b_s**2, 0.0, None)
    sini = np.sin(np.radians(inc_s))
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = np.where(sini > 0, np.sqrt(num) / (a_s * sini), np.nan)
    dur_s = (period_s / np.pi) * np.arcsin(np.clip(arg, -1.0, 1.0))
    sq1 = np.sqrt(np.clip(q1_s, 0.0, None))
    u1_s = 2.0 * sq1 * q2_s
    u2_s = sq1 * (1.0 - 2.0 * q2_s)

    def triple(arr: np.ndarray) -> tuple[float, float, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (np.nan, np.nan, np.nan)
        lo, med, hi = np.percentile(arr, [16, 50, 84])
        return (float(med), float(med - lo), float(hi - med))

    return {
        "period": triple(period_s),
        "t0": triple(t0_s),
        "depth": triple(depth_s),
        "duration": triple(dur_s),
        "rp_rs": triple(k_s),
        "a_rs": triple(a_s),
        "b": triple(b_s),
        "inclination": triple(inc_s),
        "u1": triple(u1_s),
        "u2": triple(u2_s),
    }


# --------------------------------------------------------------------------- #
# Fallback constructors
# --------------------------------------------------------------------------- #
def _seed_only_fit(det: DetectionResult, warning: str) -> TransitFit:
    """Worst-case fallback: a TransitFit straight from the detection, NaN errors."""
    period = float(det.period)
    depth = float(det.depth)
    duration = float(det.duration)
    t0 = float(det.t0)
    k = rp_rs_from_depth(depth) if np.isfinite(depth) and depth > 0 else np.nan
    params: dict[str, tuple[float, float, float]] = {
        "period": (period, np.nan, np.nan),
        "t0": (t0, np.nan, np.nan),
        "depth": (depth, np.nan, np.nan),
        "duration": (duration, np.nan, np.nan),
        "rp_rs": (k, np.nan, np.nan),
        "a_rs": (np.nan, np.nan, np.nan),
        "b": (np.nan, np.nan, np.nan),
        "inclination": (np.nan, np.nan, np.nan),
        "u1": (np.nan, np.nan, np.nan),
        "u2": (np.nan, np.nan, np.nan),
    }
    return TransitFit(
        params=params,
        model_time=None,
        model_flux=None,
        bic_transit=np.nan,
        bic_flat=np.nan,
        delta_bic=np.nan,
        snr=float(det.snr),
        method="seed_only",
        samples=np.empty((0, 0)),
        extra={"warnings": [warning]},
    )


def _phase_grid_model(
    period: float, t0: float, geom: dict[str, float], n: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit model sampled on a dense phase grid (one period centred on t0)."""
    if not np.isfinite(period) or period <= 0:
        return np.empty(0), np.empty(0)
    grid = t0 + np.linspace(-0.5, 0.5, n) * period
    model_params = {
        "period": period,
        "t0": t0,
        "rp_rs": geom.get("rp_rs", 0.1),
        "a_rs": geom.get("a_rs", 10.0),
        "b": geom.get("b", 0.3),
        "u1": geom.get("u1", 0.4),
        "u2": geom.get("u2", 0.3),
    }
    model_flux = transit_model(grid, model_params)
    return grid, model_flux


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def fit_transit(
    lc: LightCurve,
    det: DetectionResult,
    method: str = "auto",
    sampler: str = "emcee",
    nsteps: int = 1500,
    nwalkers: int = 32,
    seed: int | None = 42,
    **kw: Any,
) -> TransitFit:
    """Estimate transit parameters with uncertainties (two-stage fit).

    Parameters
    ----------
    lc:
        The (detrended) light curve.
    det:
        Detection seeding the fit (period/t0/duration/depth).
    method:
        ``'auto'`` (default) runs trapezoid LM then the MCMC stage if its
        dependencies are present; ``'fast'`` forces trapezoid-only;
        ``'mcmc'``/``'full'`` request the MCMC stage explicitly.
    sampler:
        Posterior sampler for stage 2. ``'emcee'`` is implemented; other values
        fall back to ``'emcee'`` (and annotate ``extra``).
    nsteps, nwalkers:
        emcee chain length and walker count.
    seed:
        RNG seed for reproducible walker initialisation.
    **kw:
        Accepted and ignored for forward compatibility (e.g. extra sampler kw).

    Returns
    -------
    TransitFit
        Populated ``params`` (median, err_lo, err_hi) for
        ``period, t0, depth, duration, rp_rs, a_rs, b, inclination, u1, u2``;
        ``model_time``/``model_flux`` (dense phase-grid best fit);
        ``bic_transit``/``bic_flat``/``delta_bic``; ``snr``; ``method``;
        ``samples`` (equal-weight chain or empty); diagnostics in ``extra``.
    """
    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    period = float(det.period)
    if not np.isfinite(period) or period <= 0:
        return _seed_only_fit(det, "detection period is invalid; returning seed-only fit")

    try:
        time, flux, err = _clean_arrays(lc)
    except Exception as exc:  # pragma: no cover - defensive
        return _seed_only_fit(det, f"could not prepare light curve: {exc!r}")

    if time.size < 8:
        return _seed_only_fit(det, "too few finite cadences to fit")

    # ----- Stage 1: trapezoid least-squares -------------------------------- #
    seed_dict = {
        "t0": float(det.t0) if np.isfinite(det.t0) else float(time[0]),
        "depth": float(det.depth) if np.isfinite(det.depth) and det.depth > 0 else 1e-3,
        "duration": float(det.duration),
    }
    try:
        trap = _trapezoid_lsq(time, flux, err, period, seed_dict)
    except Exception as exc:
        warnings.append(f"trapezoid least-squares failed ({exc!r}); using detection seed")
        trap = {
            "params": dict(
                t0=seed_dict["t0"],
                depth=seed_dict["depth"],
                duration=seed_dict["duration"],
                ingress_frac=0.15,
            ),
            "errors": dict(t0=np.nan, depth=np.nan, duration=np.nan, ingress_frac=np.nan),
            "chi2": np.nan,
            "dof": max(1, time.size - 4),
            "success": False,
        }

    tp = trap["params"]
    te = trap["errors"]
    trap_t0 = float(tp["t0"])
    trap_depth = float(np.clip(tp["depth"], 1e-6, 0.95))
    trap_dur = float(tp["duration"])
    trap_k = rp_rs_from_depth(trap_depth)

    # Geometry seed for batman / model rendering: derive a/Rs from the duration.
    b_seed = float(kw.get("b_seed", 0.3))
    a_rs_seed = a_rs_from_duration(period, trap_dur, trap_k, b_seed)
    if not np.isfinite(a_rs_seed) or a_rs_seed <= 1.5:
        a_rs_seed = 10.0
    geom_seed = {
        "period": period,
        "t0": trap_t0,
        "rp_rs": float(np.clip(trap_k, 1e-3, 0.49)),
        "a_rs": float(np.clip(a_rs_seed, 1.5, 49.0)),
        "b": b_seed,
        "duration": trap_dur,
        "u1": 0.4,
        "u2": 0.3,
    }

    # Trapezoid model on the data grid -> BIC vs flat + SNR.
    trap_model_data = trapezoid_model(
        time, trap_t0, trap_depth, trap_dur, ingress_frac=float(tp["ingress_frac"]), period=period
    )
    chi2_transit = _chi2(flux, trap_model_data, err)
    chi2_flat = _chi2(flux, np.ones_like(flux), err)
    n = time.size
    bic_transit = _bic(chi2_transit, 4, n)
    bic_flat = _bic(chi2_flat, 0, n)
    delta_bic = bic_flat - bic_transit
    snr = _transit_snr(flux, err, trap_model_data, trap_depth)

    # Stage-1 parameter triples (symmetric covariance errors).
    def trip(value: float, error: float) -> tuple[float, float, float]:
        e = float(error) if np.isfinite(error) else np.nan
        return (float(value), e, e)

    stage1_params: dict[str, tuple[float, float, float]] = {
        "period": (period, np.nan, np.nan),  # fixed in stage 1
        "t0": trip(trap_t0, te.get("t0", np.nan)),
        "depth": trip(trap_depth, te.get("depth", np.nan)),
        "duration": trip(trap_dur, te.get("duration", np.nan)),
        "rp_rs": trip(trap_k, 0.5 * te.get("depth", np.nan) / max(trap_k, 1e-6)),
        "a_rs": (geom_seed["a_rs"], np.nan, np.nan),
        "b": (b_seed, np.nan, np.nan),
        "inclination": (incl_from_impact(geom_seed["a_rs"], b_seed), np.nan, np.nan),
        "u1": (0.4, np.nan, np.nan),
        "u2": (0.3, np.nan, np.nan),
    }

    # Decide whether to run stage 2.
    want_mcmc = method not in ("fast", "trapezoid", "lsq")
    if sampler not in ("emcee", "auto", None):
        warnings.append(f"sampler {sampler!r} not implemented; using emcee")

    mcmc_result = None
    if want_mcmc:
        try:
            mcmc_result = _run_emcee(
                time, flux, err, geom_seed, nsteps=nsteps, nwalkers=nwalkers, rng=rng
            )
        except Exception as exc:
            warnings.append(f"emcee stage failed ({exc!r}); falling back to trapezoid")
            mcmc_result = None
        if mcmc_result is None and method in ("mcmc", "full"):
            warnings.append("batman/emcee unavailable; trapezoid-only fit returned")

    # ----- Assemble result -------------------------------------------------- #
    if mcmc_result is not None:
        chain = mcmc_result["chain"]
        params = _summarize_mcmc(chain)
        # Best-fit geometry from posterior medians for model rendering + BIC.
        best_geom = {
            "period": params["period"][0],
            "t0": params["t0"][0],
            "rp_rs": params["rp_rs"][0],
            "a_rs": params["a_rs"][0],
            "b": params["b"][0],
            "u1": params["u1"][0],
            "u2": params["u2"][0],
        }
        # Recompute BIC with the full transit model (7 free params).
        model_on_data = transit_model(time, best_geom)
        chi2_full = _chi2(flux, model_on_data, err)
        bic_transit = _bic(chi2_full, 7, n)
        delta_bic = bic_flat - bic_transit
        depth_med = params["depth"][0]
        snr = _transit_snr(flux, err, model_on_data, depth_med)
        model_time, model_flux = _phase_grid_model(
            best_geom["period"], best_geom["t0"], best_geom
        )
        extra = {
            "warnings": warnings,
            "stage": "trapezoid+batman_emcee",
            "acceptance_fraction": mcmc_result["acceptance"],
            "n_eff": mcmc_result["n_eff"],
            "burn": mcmc_result["burn"],
            "thin": mcmc_result["thin"],
            "trapezoid": {
                "depth": trap_depth,
                "duration": trap_dur,
                "t0": trap_t0,
                "ingress_frac": float(tp["ingress_frac"]),
                "chi2": chi2_transit,
            },
            "chi2_transit": chi2_full,
            "chi2_flat": chi2_flat,
            "reduced_chi2": chi2_full / max(1, n - 7),
        }
        # Optional dynesty evidence (best-effort; never required).
        _maybe_add_evidence(extra, time, flux, err, geom_seed, kw)
        fit_method = "batman_emcee"
        samples = chain
    else:
        params = stage1_params
        best_geom = geom_seed
        model_time, model_flux = _phase_grid_model(period, trap_t0, geom_seed)
        extra = {
            "warnings": warnings,
            "stage": "trapezoid",
            "trapezoid": {
                "depth": trap_depth,
                "duration": trap_dur,
                "t0": trap_t0,
                "ingress_frac": float(tp["ingress_frac"]),
            },
            "chi2_transit": chi2_transit,
            "chi2_flat": chi2_flat,
            "reduced_chi2": chi2_transit / max(1, n - 4),
        }
        fit_method = "trapezoid"
        samples = np.empty((0, 0))

    return TransitFit(
        params=params,
        model_time=model_time,
        model_flux=model_flux,
        bic_transit=float(bic_transit),
        bic_flat=float(bic_flat),
        delta_bic=float(delta_bic),
        snr=float(snr) if np.isfinite(snr) else float(det.snr),
        method=fit_method,
        samples=samples,
        extra=extra,
    )


def _maybe_add_evidence(
    extra: dict[str, Any],
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    geom: dict[str, float],
    kw: dict[str, Any],
) -> None:
    """Best-effort Bayesian evidence (ln Z) via dynesty, stored in ``extra``.

    Only runs when ``kw.get('evidence')`` is truthy *and* dynesty + batman are
    importable; silently skipped otherwise so the common path stays fast.
    """
    if not kw.get("evidence", False):
        return
    try:  # pragma: no cover - optional + slow
        import batman  # type: ignore
        import dynesty  # type: ignore
    except Exception:
        extra.setdefault("warnings", []).append("dynesty unavailable; no ln Z computed")
        return
    try:  # pragma: no cover - optional + slow
        period0 = geom["period"]
        t0_0 = geom["t0"]
        dp = 0.02 * period0
        dt0 = 0.05 * period0

        bp = batman.TransitParams()
        bp.t0 = t0_0
        bp.per = period0
        bp.rp = geom["rp_rs"]
        bp.a = geom["a_rs"]
        bp.inc = incl_from_impact(geom["a_rs"], geom["b"])
        bp.ecc = 0.0
        bp.w = 90.0
        bp.limb_dark = "quadratic"
        bp.u = [0.4, 0.3]
        bmodel = batman.TransitModel(bp, time)

        def ptform(u: np.ndarray) -> np.ndarray:
            x = np.empty_like(u)
            x[0] = period0 - dp + 2 * dp * u[0]
            x[1] = t0_0 - dt0 + 2 * dt0 * u[1]
            x[2] = 1e-3 + (0.5 - 1e-3) * u[2]
            x[3] = 1.5 + (50.0 - 1.5) * u[3]
            x[4] = (1.0 + x[2]) * u[4]
            x[5] = u[5]
            x[6] = u[6]
            return x

        def loglike(theta: np.ndarray) -> float:
            period, t0, k, a_rs, b, q1, q2 = theta
            bp.per, bp.t0, bp.rp, bp.a = period, t0, k, a_rs
            bp.inc = incl_from_impact(a_rs, b)
            bp.u = list(_q_to_u(q1, q2))
            m = bmodel.light_curve(bp)
            return float(-0.5 * np.sum(((flux - m) / err) ** 2 + np.log(2 * np.pi * err * err)))

        ds = dynesty.NestedSampler(loglike, ptform, ndim=7, nlive=int(kw.get("nlive", 250)))
        ds.run_nested(dlogz=0.5, print_progress=False)
        res = ds.results
        extra["ln_z"] = float(res.logz[-1])
        extra["ln_z_err"] = float(res.logzerr[-1])
    except Exception as exc:
        extra.setdefault("warnings", []).append(f"dynesty evidence failed: {exc!r}")
