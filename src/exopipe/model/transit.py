"""Forward transit-light-curve models and the geometric helper relations.

Two forward models are provided:

* :func:`trapezoid_model` -- a pure-NumPy trapezoid (flat bottom + linear
  ingress/egress). Fast, dependency-free; the workhorse for the least-squares
  seed and for V-shape vetting.
* :func:`transit_model` -- a physically-motivated Mandel & Agol (2002)
  quadratically-limb-darkened model. Uses ``batman`` when it is importable and
  falls back to a self-contained NumPy small-planet approximation otherwise.

A bank of analytic helpers translates between the observable shape parameters
(period, depth, duration) and the physical transit geometry
(``Rp/R*``, ``a/R*``, ``b``, ``i``, stellar density). All formulas are from
**Winn (2010), "Transits and Occultations"** (arXiv:1001.2010) and
**Seager & Mallen-Ornelas (2003)**; the relevant equations are reproduced in the
individual docstrings.

Conventions
-----------
* ``depth`` is the *fractional, positive* flux decrement at mid-transit
  (``delta == k**2`` in the flat-bottom limit).
* ``period``, ``duration``, ``t0`` are in **days**.
* ``rp_rs == k == Rp/R*`` is the radius ratio; ``a_rs == a/R*`` is the scaled
  semi-major axis; ``b`` is the (dimensionless) impact parameter; inclination is
  in **degrees** unless a function name says otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "trapezoid_model",
    "transit_model",
    "winn_duration",
    "impact_from_incl",
    "incl_from_impact",
    "a_rs_from_duration",
    "density_from_a_rs",
    "a_rs_from_density",
    "rp_rjup_from_depth",
    "rp_rearth_from_depth",
    "depth_from_rp_rs",
    "rp_rs_from_depth",
]

# --------------------------------------------------------------------------- #
# Physical constants (SI) for the stellar-density relations
# --------------------------------------------------------------------------- #
_G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
_DAY_S = 86_400.0  # seconds per day
_RSUN_M = 6.957e8  # solar radius, metres
_RJUP_M = 7.1492e7  # Jupiter equatorial radius, metres
_REARTH_M = 6.3781e6  # Earth equatorial radius, metres
_RSUN_RJUP = _RSUN_M / _RJUP_M  # ~9.7311 R_jup per R_sun
_RSUN_REARTH = _RSUN_M / _REARTH_M  # ~109.08 R_earth per R_sun


# --------------------------------------------------------------------------- #
# Trapezoid model (pure NumPy)
# --------------------------------------------------------------------------- #
def trapezoid_model(
    time: np.ndarray,
    t0: float,
    depth: float,
    duration: float,
    ingress_frac: float = 0.1,
    period: float | None = None,
) -> np.ndarray:
    """Symmetric trapezoidal transit (flat bottom + linear ingress/egress).

    The model is normalised so the out-of-transit level is 1.0 and the bottom of
    the flat floor is ``1 - depth``. ``duration`` is the *total* (first-to-fourth
    contact) duration ``T14``; the flat (second-to-third contact) portion is
    ``T23 = (1 - 2 * ingress_frac) * T14`` and each ingress/egress ramp spans
    ``ingress_frac * T14``.

    Parameters
    ----------
    time:
        Observation times in days.
    t0:
        Mid-transit time in days.
    depth:
        Fractional transit depth (positive), e.g. ``0.01`` for 1%.
    duration:
        Total transit duration ``T14`` in days.
    ingress_frac:
        Fraction of ``T14`` occupied by *each* of ingress and egress, in
        ``[0, 0.5]``. ``0`` gives a box; ``0.5`` gives a pure V shape (no floor).
    period:
        If given, the model is periodic: the dip repeats every ``period`` days
        (phase computed relative to ``t0``). If ``None`` a single transit at
        ``t0`` is modelled.

    Returns
    -------
    numpy.ndarray
        Normalised flux, same shape as ``time``.

    Notes
    -----
    A best-fit ``T23 -> 0`` (i.e. ``ingress_frac -> 0.5``) means there is no flat
    bottom -- the classic V-shaped, grazing signature of an eclipsing binary or a
    grazing planet (Winn 2010, sect. 3; see :mod:`exopipe.vetting`).
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.ones_like(time)

    duration = float(duration)
    depth = float(depth)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(depth) or depth == 0:
        return flux

    ingress_frac = float(np.clip(ingress_frac, 0.0, 0.5))

    if period is not None and np.isfinite(period) and period > 0:
        # Time from the nearest mid-transit, in days.
        phase = (((time - t0) / float(period) + 0.5) % 1.0) - 0.5
        dt = np.abs(phase * float(period))
    else:
        dt = np.abs(time - t0)

    half14 = 0.5 * duration
    half23 = half14 * (1.0 - 2.0 * ingress_frac)
    ramp = half14 - half23  # width of one ingress/egress

    # Flat floor.
    flux[dt <= half23] = 1.0 - depth
    # Linear ingress/egress between half23 and half14.
    if ramp > 0:
        edge = (dt > half23) & (dt < half14)
        flux[edge] = 1.0 - depth * (half14 - dt[edge]) / ramp
    return flux


# --------------------------------------------------------------------------- #
# Mandel & Agol model (batman, with NumPy fallback)
# --------------------------------------------------------------------------- #
def _params_to_geometry(params: dict[str, Any]) -> dict[str, float]:
    """Normalise a loose ``params`` dict into a complete geometry description.

    Fills missing physical parameters from whatever observables are supplied,
    using the helper relations below. Returns a dict with keys
    ``period, t0, rp_rs, a_rs, inc, b, u1, u2`` (inc in degrees).
    """
    g: dict[str, float] = {}
    g["period"] = float(params.get("period", np.nan))
    g["t0"] = float(params.get("t0", 0.0))

    # Radius ratio: explicit rp_rs, else from depth.
    rp_rs = params.get("rp_rs")
    if rp_rs is None:
        depth = params.get("depth")
        rp_rs = rp_rs_from_depth(depth) if depth is not None else 0.1
    g["rp_rs"] = float(rp_rs)

    # Impact parameter and a/Rs are intertwined; resolve what we can.
    a_rs = params.get("a_rs")
    b = params.get("b")
    incl = params.get("inclination", params.get("inc"))

    if a_rs is None:
        # Try to derive a/Rs from the duration (needs period, duration, b).
        period = g["period"]
        duration = params.get("duration")
        b_guess = float(b) if b is not None else 0.3
        if duration is not None and np.isfinite(period) and period > 0:
            a_rs = a_rs_from_duration(period, float(duration), g["rp_rs"], b_guess)
        if a_rs is None or not np.isfinite(a_rs) or a_rs <= 1.0:
            a_rs = 10.0
    g["a_rs"] = float(a_rs)

    if b is None:
        if incl is not None:
            b = impact_from_incl(g["a_rs"], float(incl))
        else:
            b = 0.3
    g["b"] = float(np.clip(b, 0.0, 1.0 + g["rp_rs"]))

    if incl is None:
        incl = incl_from_impact(g["a_rs"], g["b"])
    g["inc"] = float(incl)

    g["u1"] = float(params.get("u1", 0.4))
    g["u2"] = float(params.get("u2", 0.3))
    return g


def _transit_model_numpy(time: np.ndarray, g: dict[str, float]) -> np.ndarray:
    """Self-contained quadratic-LD transit model (no ``batman`` required).

    Implements the Mandel & Agol (2002) small-planet ("uniform-source")
    occultation with a quadratic limb-darkening correction evaluated at the
    transit chord. This is an *approximation* -- it is exact for the uniform case
    and applies the quadratic-law surface-brightness weighting at the local
    impact parameter -- accurate to well under the photometric noise for the
    shallow TESS transits this pipeline targets, and it degrades gracefully where
    ``batman`` is unavailable.

    Geometry follows Winn (2010): the sky-projected separation of the centres is
    ``z(t) = (a/R*) * sqrt(sin^2(2*pi*phi) + (cos i * cos(2*pi*phi))^2)`` with
    ``phi = (t - t0) / P``.
    """
    time = np.asarray(time, dtype=np.float64)
    period = g["period"]
    if not np.isfinite(period) or period <= 0:
        return np.ones_like(time)

    k = g["rp_rs"]
    a_rs = g["a_rs"]
    inc = np.radians(g["inc"])
    u1, u2 = g["u1"], g["u2"]

    phi = 2.0 * np.pi * (time - g["t0"]) / period
    # Projected centre-to-centre separation in stellar radii.
    z = a_rs * np.sqrt(np.sin(phi) ** 2 + (np.cos(inc) * np.cos(phi)) ** 2)
    # Only the near side (planet in front) eclipses; far side is the secondary.
    on_front = np.cos(phi) > 0

    # Uniform-source occulted area fraction (Mandel & Agol 2002, eq. 1).
    lam_uniform = _uniform_occultation(z, k)

    # Quadratic limb-darkening weighting: evaluate the normalised stellar surface
    # brightness at the planet's location and weight the blocked flux by it.
    # mu = sqrt(1 - r^2) where r is the radial position of the planet centre.
    r = np.clip(z, 0.0, 1.0)
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    # Normalised intensity I(mu)/<I>; the disc-averaged factor keeps the total
    # stellar flux at unity for the quadratic law.
    norm = 1.0 - u1 / 3.0 - u2 / 6.0
    local_i = (1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2) / norm

    blocked = lam_uniform * local_i
    blocked = np.where(on_front, blocked, 0.0)
    return 1.0 - blocked


def _uniform_occultation(z: np.ndarray, p: float) -> np.ndarray:
    """Fraction of a uniform stellar disc occulted by an opaque planet.

    Mandel & Agol (2002), eq. (1): the area of overlap of two circles (stellar
    disc radius 1, planet radius ``p``) at centre separation ``z``, expressed as
    a fraction of the stellar area. Returns the blocked flux fraction (``= p**2``
    deep inside full transit).
    """
    z = np.asarray(z, dtype=np.float64)
    p = float(abs(p))
    out = np.zeros_like(z)
    if p <= 0:
        return out

    # Fully outside: no overlap.
    outside = z >= 1.0 + p
    # Planet fully inside the disc: blocks p^2.
    full = (z <= 1.0 - p) & ~outside
    out[full] = p * p

    # Partial overlap (ingress/egress): lens area between the two circles.
    partial = (~outside) & (~full) & (z > 0)
    if np.any(partial):
        zp = z[partial]
        # Guard the arccos arguments into [-1, 1].
        k0 = np.arccos(np.clip((p * p + zp * zp - 1.0) / (2.0 * p * zp), -1.0, 1.0))
        k1 = np.arccos(np.clip((1.0 - p * p + zp * zp) / (2.0 * zp), -1.0, 1.0))
        tri = 0.25 * (4.0 * zp * zp - (1.0 + zp * zp - p * p) ** 2)
        tri = np.sqrt(np.clip(tri, 0.0, None))
        area = (p * p * k0 + k1 - tri) / np.pi
        out[partial] = np.clip(area, 0.0, p * p)

    # Degenerate z == 0 (concentric): planet entirely within disc.
    out[z <= 0] = p * p
    return out


def transit_model(
    time: np.ndarray,
    params: dict[str, Any],
    supersample: int = 1,
    exp_time: float | None = None,
) -> np.ndarray:
    """Quadratically-limb-darkened transit light curve (Mandel & Agol 2002).

    Uses ``batman`` when it is importable (the reference Mandel & Agol quadratic
    solution with the fast EXOFAST elliptic-integral evaluation); otherwise falls
    back to the self-contained NumPy approximation :func:`_transit_model_numpy`,
    so the model always returns a result even with only NumPy installed.

    Parameters
    ----------
    time:
        Observation times in days.
    params:
        Loose parameter dict. Recognised keys (any subset; missing physical
        parameters are inferred via the helper relations):
        ``period``, ``t0``, ``rp_rs`` (or ``depth``), ``a_rs`` (or ``duration``),
        ``inclination``/``inc`` (or ``b``), ``u1``, ``u2``.
    supersample:
        Integer supersampling factor for finite-exposure smearing. Each cadence
        is evaluated at this many sub-samples spanning ``exp_time`` and averaged
        (important for 30-min TESS FFI cadence).
    exp_time:
        Exposure time in days used for supersampling. If ``None`` while
        ``supersample > 1`` a small default (2-min cadence) is assumed.

    Returns
    -------
    numpy.ndarray
        Normalised flux, same shape as ``time``.
    """
    time = np.asarray(time, dtype=np.float64)
    g = _params_to_geometry(params)
    supersample = max(1, int(supersample))
    if supersample > 1 and (exp_time is None or not np.isfinite(exp_time)):
        exp_time = 2.0 / 60.0 / 24.0  # 2-min cadence in days

    # Try batman first.
    try:  # pragma: no cover - exercised when batman present
        import batman  # type: ignore

        bp = batman.TransitParams()
        bp.t0 = g["t0"]
        bp.per = g["period"]
        bp.rp = g["rp_rs"]
        bp.a = max(g["a_rs"], 1.0 + g["rp_rs"] + 1e-6)
        bp.inc = float(np.clip(g["inc"], 0.0, 90.0))
        bp.ecc = 0.0
        bp.w = 90.0
        bp.limb_dark = "quadratic"
        bp.u = [g["u1"], g["u2"]]
        if supersample > 1:
            model = batman.TransitModel(
                bp, time, supersample_factor=supersample, exp_time=float(exp_time)
            )
        else:
            model = batman.TransitModel(bp, time)
        flux = np.asarray(model.light_curve(bp), dtype=np.float64)
        if np.all(np.isfinite(flux)):
            return flux
    except Exception:
        pass

    # NumPy fallback (optionally supersampled by hand).
    if supersample > 1:
        offsets = (np.arange(supersample) - 0.5 * (supersample - 1)) / supersample
        acc = np.zeros_like(time)
        for off in offsets:
            acc += _transit_model_numpy(time + off * float(exp_time), g)
        return acc / supersample
    return _transit_model_numpy(time, g)


# --------------------------------------------------------------------------- #
# Geometry helpers (Winn 2010)
# --------------------------------------------------------------------------- #
def winn_duration(
    period: float,
    rp_rs: float,
    a_rs: float,
    b: float,
    ecc: float = 0.0,
    omega: float = 90.0,
) -> float:
    """Total transit duration ``T14`` (days) -- Winn (2010), Eq. 14.

    .. math::

        T_{14} = \\frac{P}{\\pi}\\,\\arcsin\\!\\left[
            \\frac{1}{a/R_\\star}\\,
            \\frac{\\sqrt{(1+k)^2 - b^2}}{\\sin i}\\right]
            \\cdot \\frac{\\sqrt{1-e^2}}{1+e\\sin\\omega}

    where ``k = Rp/R*`` and ``i = arccos(b / (a/R*))`` for a circular orbit. The
    eccentricity factor ``sqrt(1-e^2)/(1+e sin omega)`` rescales the duration via
    the instantaneous orbital velocity (Winn 2010, Eq. 16).

    Returns ``nan`` if the geometry is non-transiting (``b > 1 + k``) or the
    inputs are unphysical.
    """
    period = float(period)
    a_rs = float(a_rs)
    k = float(rp_rs)
    b = float(b)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(a_rs) or a_rs <= 0:
        return float("nan")

    inc = np.radians(incl_from_impact(a_rs, b))
    num = (1.0 + k) ** 2 - b * b
    if num <= 0:
        return float("nan")
    sini = np.sin(inc)
    if sini <= 0:
        return float("nan")
    arg = np.sqrt(num) / (a_rs * sini)
    arg = np.clip(arg, -1.0, 1.0)
    t14 = (period / np.pi) * np.arcsin(arg)

    ecc = float(ecc)
    if ecc > 0:
        ecc_factor = np.sqrt(max(1.0 - ecc * ecc, 0.0)) / (
            1.0 + ecc * np.sin(np.radians(omega))
        )
        t14 *= ecc_factor
    return float(t14)


def impact_from_incl(a_rs: float, incl: float) -> float:
    """Impact parameter from ``a/R*`` and inclination (deg) -- Winn (2010), Eq. 7.

    ``b = (a/R*) * cos i`` for a circular orbit.
    """
    return float(float(a_rs) * np.cos(np.radians(float(incl))))


def incl_from_impact(a_rs: float, b: float) -> float:
    """Inclination (deg) from ``a/R*`` and impact parameter ``b``.

    Inverse of :func:`impact_from_incl`: ``i = arccos(b / (a/R*))`` (circular).
    """
    a_rs = float(a_rs)
    if a_rs <= 0:
        return 90.0
    ratio = np.clip(float(b) / a_rs, -1.0, 1.0)
    return float(np.degrees(np.arccos(ratio)))


def a_rs_from_duration(
    period: float, duration: float, rp_rs: float, b: float
) -> float:
    """Invert the duration equation for ``a/R*`` -- from Winn (2010), Eq. 14.

    Solving :func:`winn_duration` (circular, ``sin i = 1`` small-angle form) for
    ``a/R*``:

    .. math::

        \\frac{a}{R_\\star} \\approx
            \\frac{\\sqrt{(1+k)^2 - b^2}}{\\sin\\!\\big(\\pi\\,T_{14}/P\\big)}

    Returns ``nan`` for non-transiting / degenerate geometry.
    """
    period = float(period)
    duration = float(duration)
    k = float(rp_rs)
    b = float(b)
    if (
        not np.isfinite(period)
        or period <= 0
        or not np.isfinite(duration)
        or duration <= 0
    ):
        return float("nan")
    num = (1.0 + k) ** 2 - b * b
    if num <= 0:
        return float("nan")
    sin_arg = np.sin(np.pi * duration / period)
    if sin_arg <= 0:
        return float("nan")
    return float(np.sqrt(num) / sin_arg)


def density_from_a_rs(period: float, a_rs: float) -> float:
    """Mean stellar density (kg/m^3) from ``P`` and ``a/R*``.

    Seager & Mallen-Ornelas (2003) + Kepler's third law (circular orbit):

    .. math::

        \\rho_\\star \\approx \\frac{3\\pi}{G P^2}\\left(\\frac{a}{R_\\star}\\right)^3

    ``period`` is in days (converted to seconds internally).
    """
    period = float(period)
    a_rs = float(a_rs)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(a_rs) or a_rs <= 0:
        return float("nan")
    p_s = period * _DAY_S
    return float(3.0 * np.pi / (_G * p_s * p_s) * a_rs**3)


def a_rs_from_density(period: float, density: float) -> float:
    """Scaled semi-major axis from ``P`` and mean stellar density.

    Inverse of :func:`density_from_a_rs` (Seager & Mallen-Ornelas 2003):

    .. math::

        \\frac{a}{R_\\star} = \\left[\\frac{G P^2 \\rho_\\star}{3\\pi}\\right]^{1/3}

    ``period`` in days, ``density`` in kg/m^3.
    """
    period = float(period)
    density = float(density)
    if (
        not np.isfinite(period)
        or period <= 0
        or not np.isfinite(density)
        or density <= 0
    ):
        return float("nan")
    p_s = period * _DAY_S
    return float((_G * p_s * p_s * density / (3.0 * np.pi)) ** (1.0 / 3.0))


def rp_rjup_from_depth(depth: float, stellar_radius_rsun: float) -> float:
    """Implied planet radius in Jupiter radii from depth and stellar radius.

    ``Rp = sqrt(depth) * R* `` with ``R*`` in solar radii converted to Jupiter
    radii (1 R_sun = 9.731 R_jup). Used by the implied-radius sanity vetting test
    (``Rp >~ 2 R_jup`` strongly favours an eclipsing binary).
    """
    depth = float(depth)
    rstar = float(stellar_radius_rsun)
    if not np.isfinite(depth) or depth <= 0 or not np.isfinite(rstar) or rstar <= 0:
        return float("nan")
    k = np.sqrt(depth)
    return float(k * rstar * _RSUN_RJUP)


def rp_rearth_from_depth(depth: float, stellar_radius_rsun: float) -> float:
    """Implied planet radius in Earth radii (companion to :func:`rp_rjup_from_depth`)."""
    depth = float(depth)
    rstar = float(stellar_radius_rsun)
    if not np.isfinite(depth) or depth <= 0 or not np.isfinite(rstar) or rstar <= 0:
        return float("nan")
    k = np.sqrt(depth)
    return float(k * rstar * _RSUN_REARTH)


def depth_from_rp_rs(rp_rs: float) -> float:
    """Flat-bottom transit depth from the radius ratio -- Winn (2010), Eq. 22.

    ``delta = k**2 = (Rp/R*)**2``.
    """
    k = float(rp_rs)
    if not np.isfinite(k):
        return float("nan")
    return float(k * k)


def rp_rs_from_depth(depth: float) -> float:
    """Radius ratio from the flat-bottom depth -- inverse of :func:`depth_from_rp_rs`.

    ``k = sqrt(delta)``.
    """
    depth = float(depth)
    if not np.isfinite(depth) or depth < 0:
        return float("nan")
    return float(np.sqrt(depth))
