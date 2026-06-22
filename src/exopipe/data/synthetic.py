"""Physically-motivated synthetic TESS-like light-curve generator.

This is the workhorse used for demos, unit tests, classifier training, and
injection--recovery experiments. It produces :class:`~exopipe.types.LightCurve`
objects that *look* like 2-minute-cadence TESS Sector data: a ~27.4 day baseline,
realistic white + correlated red noise, a mid-sector downlink gap, a couple of
momentum-dump discontinuities, occasional outliers, and -- depending on ``kind``
-- an injected transit, eclipsing binary, blend, stellar variability, or pure
noise.

Five kinds, mapped to four science classes via ``meta['label']``:

==================  ==================  ============================================
``kind``            ``meta['label']``   what it contains
==================  ==================  ============================================
``transit``         ``transit``         planetary transit; small symmetric dips
``eclipsing_binary````eclipsing_binary`` deep primary + secondary eclipse, V-ish
``blend``           ``blend``           diluted transit/eclipse, crowdsap < 1
``variable``        ``other``           stellar variability only, no transit
``noise``           ``other``           pure white + red noise, no signal
==================  ==================  ============================================

Everything is vectorised NumPy and fully reproducible through ``seed``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import LightCurve

__all__ = [
    "make_synthetic_lightcurve",
    "make_synthetic_population",
    "KINDS",
]

KINDS = ("transit", "eclipsing_binary", "blend", "variable", "noise")

# Map each generator kind to the canonical science-class label.
_KIND_TO_LABEL = {
    "transit": "transit",
    "eclipsing_binary": "eclipsing_binary",
    "blend": "blend",
    "variable": "other",
    "noise": "other",
}

_DAY_TO_SEC = 86_400.0


# --------------------------------------------------------------------------- #
# Noise / brightness model
# --------------------------------------------------------------------------- #
def _tmag_to_sigma(tmag: float) -> float:
    """Per-point white-noise sigma (fractional) from a TESS-like magnitude.

    A smooth heuristic anchored to TESS reality: a bright Tmag~8 star reaches
    roughly a few hundred ppm per 2-min cadence, climbing toward a percent for
    faint Tmag~15 stars. Not an official noise model -- just monotonic and
    realistic enough to make detection non-trivial.
    """
    # ~150 ppm at Tmag 8, growing ~exponentially toward the faint end.
    sigma = 1.5e-4 * 10.0 ** (0.28 * (tmag - 8.0))
    return float(np.clip(sigma, 5e-5, 5e-2))


def _red_noise(
    n: int, rng: np.random.Generator, sigma_white: float, strength: float = 0.6
) -> np.ndarray:
    """Correlated (red) noise as a sum of low-frequency sines + an AR(1) term.

    Returns a zero-mean fractional series whose amplitude scales with the white
    sigma, reproducing the time-correlated systematics that make naive transit
    searches throw false positives.
    """
    t = np.arange(n, dtype=np.float64)

    # (a) handful of slow sinusoids (instrument/stellar correlated trends)
    red = np.zeros(n, dtype=np.float64)
    n_components = int(rng.integers(2, 5))
    for _ in range(n_components):
        # periods of a few hours to several days, expressed in cadences
        period_cad = rng.uniform(0.1 * n / 6.0, 0.6 * n)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        red += amp * np.sin(2.0 * np.pi * t / period_cad + phase)
    if np.std(red) > 0:
        red /= np.std(red)

    # (b) AR(1) process for short-timescale correlated wander
    phi = rng.uniform(0.85, 0.98)
    innov = rng.standard_normal(n)
    ar = np.empty(n, dtype=np.float64)
    ar[0] = innov[0]
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + np.sqrt(1.0 - phi**2) * innov[i]
    if np.std(ar) > 0:
        ar /= np.std(ar)

    red_total = 0.6 * red + 0.4 * ar
    return strength * sigma_white * red_total


def _add_outliers(
    flux: np.ndarray, rng: np.random.Generator, sigma_white: float, rate: float = 0.002
) -> None:
    """Sprinkle in cosmic-ray-like positive (and a few negative) spikes in place."""
    n = flux.size
    n_out = rng.poisson(rate * n)
    if n_out <= 0:
        return
    idx = rng.choice(n, size=min(n_out, n), replace=False)
    amp = rng.uniform(5.0, 12.0, size=idx.size) * sigma_white
    sign = rng.choice(np.array([1.0, 1.0, 1.0, -1.0]), size=idx.size)  # mostly positive
    flux[idx] += amp * sign


# --------------------------------------------------------------------------- #
# Transit / eclipse shape
# --------------------------------------------------------------------------- #
def _limb_darkened_dip(
    phase_time: np.ndarray,
    depth: float,
    duration: float,
    ingress_frac: float = 0.15,
    limb: float = 0.4,
) -> np.ndarray:
    """A quasi-trapezoidal, limb-darkened dip in fractional flux (>= 0 == dip).

    Models a flat-bottomed transit with finite ingress/egress (the trapezoid)
    and a gentle limb-darkening curvature across the floor, without requiring
    ``batman``. ``phase_time`` is time relative to mid-transit in days; returns
    the *positive* flux decrement to subtract from the baseline.
    """
    duration = float(duration)
    if duration <= 0 or depth <= 0:
        return np.zeros_like(phase_time)

    half = 0.5 * duration
    ingress = max(ingress_frac * duration, 1e-6)
    x = np.abs(phase_time)
    dip = np.zeros_like(phase_time)

    # Flat (curved) interior: |x| <= half - ingress
    flat = half - ingress
    inside = x <= max(flat, 0.0)
    # Limb-darkening: deeper at centre, shallower toward the limb (edge of floor).
    if flat > 0:
        mu = np.sqrt(np.clip(1.0 - (x[inside] / flat) ** 2, 0.0, 1.0))
    else:
        mu = np.ones(int(np.count_nonzero(inside)))
    ld = (1.0 - limb * (1.0 - mu)) / (1.0 - limb / 3.0)
    dip[inside] = depth * ld

    # Linear ingress/egress ramp between (half-ingress) and half.
    ramp = (x > max(flat, 0.0)) & (x < half)
    frac = (half - x[ramp]) / ingress
    dip[ramp] = depth * np.clip(frac, 0.0, 1.0)

    return dip


def _v_shaped_dip(
    phase_time: np.ndarray, depth: float, duration: float
) -> np.ndarray:
    """A V-shaped (grazing/EB-like) dip: linear walls meeting at a point."""
    duration = float(duration)
    if duration <= 0 or depth <= 0:
        return np.zeros_like(phase_time)
    half = 0.5 * duration
    x = np.abs(phase_time)
    inside = x < half
    dip = np.zeros_like(phase_time)
    dip[inside] = depth * (1.0 - x[inside] / half)
    return dip


def _inject_periodic(
    time: np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
    shape: str = "trapezoid",
    ingress_frac: float = 0.15,
) -> np.ndarray:
    """Return the summed positive flux decrement of all transits/eclipses.

    Computes phase relative to ``t0`` over the whole series at once (vectorised),
    so a single call injects every transit in the light curve.
    """
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
    phase_time = phase * period  # days from nearest mid-transit
    if shape == "v":
        return _v_shaped_dip(phase_time, depth, duration)
    return _limb_darkened_dip(phase_time, depth, duration, ingress_frac=ingress_frac)


def _duration_from_period(
    period: float, a_rs: float, b: float, rng: np.random.Generator
) -> float:
    """Transit duration (days) from period, scaled semi-major axis, and impact b.

    Uses the standard total-duration approximation
    ``T = (P/pi) * arcsin( (1/a_rs) * sqrt((1+k)^2 - b^2) / sin i )`` reduced to
    the central-transit form, with a small Rp/Rs term folded in. Falls back to a
    plausible fraction of the period if the geometry is grazing/degenerate.
    """
    arg = max((1.0 - b**2), 1e-3) / a_rs**2
    arg = np.sqrt(arg)
    arg = np.clip(arg, 1e-4, 0.999)
    duration = (period / np.pi) * np.arcsin(arg)
    if not np.isfinite(duration) or duration <= 0:
        duration = period * rng.uniform(0.02, 0.06)
    # keep within a sane window (a few hours to ~1 day)
    return float(np.clip(duration, 0.02, min(0.6, 0.2 * period)))


# --------------------------------------------------------------------------- #
# Stellar-variability model (for 'variable')
# --------------------------------------------------------------------------- #
def _stellar_variability(
    time: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Smooth out-of-transit modulation: spots and/or pulsations.

    Returns ``(signal, dominant_period_days)`` where ``signal`` is a zero-mean
    fractional flux variation (amplitude 0.1%--3%).
    """
    span = time[-1] - time[0]
    mode = rng.choice(["spots", "pulsation", "hybrid"])
    signal = np.zeros_like(time)

    if mode in ("spots", "hybrid"):
        p_rot = rng.uniform(0.5, 0.5 * span)
        amp = rng.uniform(2e-3, 3e-2)
        # rotation + first harmonic gives a non-sinusoidal spotted profile
        signal += amp * np.sin(2 * np.pi * (time - time[0]) / p_rot + rng.uniform(0, 2 * np.pi))
        signal += 0.4 * amp * np.sin(
            4 * np.pi * (time - time[0]) / p_rot + rng.uniform(0, 2 * np.pi)
        )
        dominant = p_rot
    else:
        dominant = rng.uniform(0.05, 2.0)

    if mode in ("pulsation", "hybrid"):
        n_modes = int(rng.integers(2, 5))
        for _ in range(n_modes):
            p_puls = rng.uniform(0.05, 2.0)
            amp = rng.uniform(2e-4, 3e-3)
            signal += amp * np.sin(
                2 * np.pi * (time - time[0]) / p_puls + rng.uniform(0, 2 * np.pi)
            )

    return signal, float(dominant)


# --------------------------------------------------------------------------- #
# Stellar parameters
# --------------------------------------------------------------------------- #
def _draw_star(rng: np.random.Generator, params: dict[str, Any]) -> dict[str, Any]:
    """Draw a plausible main-sequence-ish host star, honouring overrides."""
    teff = float(params.get("teff", rng.uniform(3500.0, 7000.0)))
    # crude main-sequence radius from Teff (solar radii)
    radius = float(params.get("radius", np.clip((teff / 5772.0) ** 0.8, 0.3, 2.5)))
    mass = float(params.get("mass", np.clip((teff / 5772.0) ** 1.0, 0.3, 1.8)))
    logg = float(params.get("logg", np.clip(4.5 - 0.5 * (radius - 1.0), 3.8, 4.7)))
    tmag = float(params.get("tmag", rng.uniform(8.0, 15.0)))
    ra = float(params.get("ra", rng.uniform(0.0, 360.0)))
    dec = float(params.get("dec", rng.uniform(-90.0, 90.0)))
    return {
        "teff": teff,
        "radius": radius,
        "mass": mass,
        "logg": logg,
        "tmag": tmag,
        "ra": ra,
        "dec": dec,
    }


# --------------------------------------------------------------------------- #
# Time base + instrumental artefacts
# --------------------------------------------------------------------------- #
def _build_time(
    n_days: float, cadence_min: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the time grid and a TESS-like quality flag array.

    Inserts a mid-sector downlink gap (data removed) and flags a few cadences
    around two momentum-dump-like times. Returns ``(time, quality)`` *before*
    the downlink gap is removed; the caller applies the gap mask.
    """
    cadence_days = cadence_min / (24.0 * 60.0)
    t0 = 1325.0 + rng.uniform(0.0, 5.0)  # TESS BTJD-ish start
    n = int(round(n_days / cadence_days))
    time = t0 + np.arange(n, dtype=np.float64) * cadence_days
    quality = np.zeros(n, dtype=np.int32)
    return time, quality


def _apply_artifacts(
    time: np.ndarray,
    flux: np.ndarray,
    quality: np.ndarray,
    rng: np.random.Generator,
    sigma_white: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply downlink gap (removal), momentum-dump steps, and quality flags.

    Returns possibly-shortened ``(time, flux, quality)``.
    """
    n = time.size
    span = time[-1] - time[0]

    # --- momentum-dump-like discontinuities: small flux steps + flags ------- #
    n_dumps = int(rng.integers(1, 4))
    dump_idx = np.sort(rng.choice(np.arange(int(0.1 * n), int(0.9 * n)), size=n_dumps, replace=False))
    for idx in dump_idx:
        step = rng.uniform(-3.0, 3.0) * sigma_white
        flux[idx:] += step  # persistent offset after the dump
        lo = max(idx - 2, 0)
        hi = min(idx + 3, n)
        quality[lo:hi] |= 0x20  # bit set near a dump

    # --- mid-sector downlink gap: physically remove a chunk ----------------- #
    gap_center = time[0] + span * rng.uniform(0.45, 0.55)
    gap_width = span * rng.uniform(0.02, 0.05)
    in_gap = np.abs(time - gap_center) < (0.5 * gap_width)
    keep = ~in_gap

    return time[keep], flux[keep], quality[keep]


# --------------------------------------------------------------------------- #
# Public: single light curve
# --------------------------------------------------------------------------- #
def make_synthetic_lightcurve(
    kind: str = "transit",
    seed: int | None = None,
    n_days: float = 27.4,
    cadence_min: float = 2.0,
    **params: Any,
) -> LightCurve:
    """Generate one synthetic TESS-like light curve of the requested ``kind``.

    Parameters
    ----------
    kind:
        One of :data:`KINDS`. See module docstring for what each contains.
    seed:
        Seed for reproducibility. ``None`` draws a fresh random series.
    n_days:
        Baseline length in days (default ~ one TESS sector).
    cadence_min:
        Sampling cadence in minutes (default 2-min, TESS high cadence).
    **params:
        Optional overrides. Recognised keys include stellar (``teff``, ``radius``,
        ``mass``, ``logg``, ``tmag``, ``ra``, ``dec``) and signal parameters
        (``period``, ``depth``, ``duration``, ``t0``, ``rp_rs``, ``a_rs``, ``b``,
        ``crowdsap``). Anything supplied overrides the random draw.

    Returns
    -------
    LightCurve
        With canonical dtypes and a richly-populated ``meta`` including the
        ground-truth injected parameters and ``label``.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")

    rng = np.random.default_rng(seed)

    # --- star + brightness/noise level ------------------------------------- #
    star = _draw_star(rng, params)
    sigma_white = _tmag_to_sigma(star["tmag"])

    # --- time base + flux baseline ----------------------------------------- #
    time, quality = _build_time(n_days, cadence_min, rng)
    n = time.size
    flux = np.ones(n, dtype=np.float64)

    # ground-truth holders (populated per-kind)
    truth: dict[str, Any] = {
        "true_period": np.nan,
        "true_depth": np.nan,
        "true_duration": np.nan,
        "true_t0": np.nan,
    }
    crowdsap = float(params.get("crowdsap", 1.0))
    centroid_offset = 0.0
    secondary_depth = 0.0

    # --- inject the per-kind astrophysical signal -------------------------- #
    if kind == "transit":
        period = float(params.get("period", rng.uniform(1.0, 12.0)))
        rp_rs = float(params.get("rp_rs", np.sqrt(rng.uniform(5e-5, 2e-2))))
        depth = float(params.get("depth", rp_rs**2))
        a_rs = float(params.get("a_rs", rng.uniform(6.0, 30.0)))
        b = float(params.get("b", rng.uniform(0.0, 0.7)))
        duration = float(params.get("duration", _duration_from_period(period, a_rs, b, rng)))
        t0 = float(params.get("t0", time[0] + rng.uniform(0.0, period)))
        dip = _inject_periodic(time, period, t0, depth, duration, shape="trapezoid")
        flux -= dip
        truth.update(true_period=period, true_depth=depth, true_duration=duration, true_t0=t0)

    elif kind == "eclipsing_binary":
        period = float(params.get("period", rng.uniform(0.5, 12.0)))
        depth = float(params.get("depth", rng.uniform(1e-2, 0.4)))
        a_rs = float(params.get("a_rs", rng.uniform(3.0, 15.0)))
        b = float(params.get("b", rng.uniform(0.0, 0.9)))
        duration = float(params.get("duration", _duration_from_period(period, a_rs, b, rng)))
        t0 = float(params.get("t0", time[0] + rng.uniform(0.0, period)))
        # often V-shaped / grazing for EBs
        shape = "v" if rng.random() < 0.6 else "trapezoid"
        primary = _inject_periodic(time, period, t0, depth, duration, shape=shape)
        # SECONDARY eclipse near phase 0.5, shallower, possibly offset for eccentricity
        sec_frac = rng.uniform(0.3, 0.9)
        secondary_depth = depth * sec_frac
        sec_phase_offset = rng.uniform(-0.03, 0.03)  # eccentric orbits shift it
        t0_secondary = t0 + period * (0.5 + sec_phase_offset)
        secondary = _inject_periodic(
            time, period, t0_secondary, secondary_depth, duration * rng.uniform(0.7, 1.0),
            shape=shape,
        )
        # odd/even depth mismatch: alternate transits differ slightly
        cycle = np.floor(((time - t0) / period) + 0.5).astype(np.int64)
        odd_even = np.where(cycle % 2 == 0, 1.0, rng.uniform(0.75, 0.95))
        flux -= primary * odd_even
        flux -= secondary
        truth.update(true_period=period, true_depth=depth, true_duration=duration, true_t0=t0)

    elif kind == "blend":
        # underlying eclipse/transit signal then DILUTED by crowding
        period = float(params.get("period", rng.uniform(0.5, 12.0)))
        intrinsic_depth = float(params.get("depth", rng.uniform(2e-3, 0.2)))
        a_rs = float(params.get("a_rs", rng.uniform(4.0, 20.0)))
        b = float(params.get("b", rng.uniform(0.0, 0.8)))
        duration = float(params.get("duration", _duration_from_period(period, a_rs, b, rng)))
        t0 = float(params.get("t0", time[0] + rng.uniform(0.0, period)))
        crowdsap = float(params.get("crowdsap", rng.uniform(0.2, 0.85)))
        # observed depth is diluted by the fraction of light from the target
        observed_depth = intrinsic_depth * crowdsap
        shape = "v" if rng.random() < 0.5 else "trapezoid"
        dip = _inject_periodic(time, period, t0, observed_depth, duration, shape=shape)
        flux -= dip
        centroid_offset = float(rng.uniform(0.5, 4.0))  # pixels (simulated)
        truth.update(
            true_period=period,
            true_depth=observed_depth,
            true_duration=duration,
            true_t0=t0,
        )

    elif kind == "variable":
        variability, dominant = _stellar_variability(time, rng)
        flux += variability
        # record the dominant variability period in meta but no transit truth
        truth["variability_period"] = dominant

    elif kind == "noise":
        # pure noise: nothing injected here; noise added below
        pass

    # --- correlated red noise + white noise (all kinds) -------------------- #
    flux += _red_noise(n, rng, sigma_white, strength=rng.uniform(0.4, 1.0))
    flux += rng.normal(0.0, sigma_white, size=n)
    _add_outliers(flux, rng, sigma_white)

    # --- instrumental artefacts: dumps + downlink gap ---------------------- #
    time, flux, quality = _apply_artifacts(time, flux, quality, rng, sigma_white)

    # --- per-point error estimate ------------------------------------------ #
    flux_err = np.full(flux.shape, sigma_white, dtype=np.float64)

    # --- assemble meta ----------------------------------------------------- #
    label = _KIND_TO_LABEL[kind]
    tic_id = int(params.get("tic_id", rng.integers(1_000_000, 500_000_000)))
    sector = int(params.get("sector", rng.integers(1, 80)))
    meta: dict[str, Any] = {
        "tic_id": tic_id,
        "sector": sector,
        "cadence_s": float(cadence_min * 60.0),
        "mission": "TESS-sim",
        "teff": star["teff"],
        "logg": star["logg"],
        "radius": star["radius"],
        "mass": star["mass"],
        "crowdsap": crowdsap,
        "ra": star["ra"],
        "dec": star["dec"],
        "tmag": star["tmag"],
        "sigma_white": sigma_white,
        "quality": quality,
        "kind": kind,
        "label": label,
        "centroid_offset": centroid_offset,
        "secondary_depth": secondary_depth,
    }
    meta.update(truth)

    return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)


# --------------------------------------------------------------------------- #
# Public: population
# --------------------------------------------------------------------------- #
def make_synthetic_population(
    n: int,
    seed: int = 0,
    fractions: dict[str, float] | None = None,
) -> list[LightCurve]:
    """Generate a labelled mixed population for training/evaluation.

    Parameters
    ----------
    n:
        Number of light curves to generate.
    seed:
        Master seed; each light curve gets a distinct derived seed so the whole
        population is reproducible yet not identical.
    fractions:
        Mapping ``kind -> fraction`` (need not sum to 1; it is normalised). The
        default leans toward transits/EBs/noise so a classifier sees a balanced
        but realistically imbalanced mix.

    Returns
    -------
    list[LightCurve]
        Shuffled list; each element's class is in ``meta['label']`` (and the
        generator kind in ``meta['kind']``).
    """
    if n <= 0:
        return []

    if fractions is None:
        fractions = {
            "transit": 0.30,
            "eclipsing_binary": 0.25,
            "blend": 0.15,
            "variable": 0.15,
            "noise": 0.15,
        }
    # restrict to known kinds and normalise
    fractions = {k: float(v) for k, v in fractions.items() if k in KINDS and v > 0}
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("fractions must contain at least one positive known kind")
    fractions = {k: v / total for k, v in fractions.items()}

    rng = np.random.default_rng(seed)

    # Allocate integer counts via largest-remainder so they sum exactly to n.
    kinds = list(fractions.keys())
    exact = np.array([fractions[k] * n for k in kinds])
    counts = np.floor(exact).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        order = np.argsort(-(exact - counts))  # largest fractional parts first
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    # Build the kind list, then assign per-curve seeds deterministically.
    kind_sequence: list[str] = []
    for kind, count in zip(kinds, counts):
        kind_sequence.extend([kind] * int(count))
    rng.shuffle(kind_sequence)

    child_seeds = rng.integers(0, 2**31 - 1, size=len(kind_sequence))
    population = [
        make_synthetic_lightcurve(kind=kind_sequence[i], seed=int(child_seeds[i]))
        for i in range(len(kind_sequence))
    ]
    return population
