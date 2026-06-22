"""Shared dataclasses for the ``exopipe`` pipeline.

This module is the **interface contract** for the whole package: every other
module (detrending, transit search, vetting, classification, fitting,
visualisation, CLI) imports the dataclasses defined here and reads/writes their
fields. Keep field names and method signatures stable.

Conventions
-----------
* ``time`` is in **days** (TESS BTJD-like), ``float64``.
* ``flux`` / ``flux_err`` are **normalised** (median ~ 1.0) and ``float32``.
* Depths are *fractional* (e.g. ``0.01`` == 1% == 10 000 ppm), positive for a dip.
* Durations and periods are in **days**.
* ``meta`` is a free-form dict; well-known keys are documented on ``LightCurve``.

All dataclasses are plain ``@dataclass`` objects (no third-party deps) so they
import cleanly even when the optional science/ML stacks are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "LightCurve",
    "DetectionResult",
    "VettingReport",
    "TransitFit",
    "Classification",
    "CandidateResult",
]


# --------------------------------------------------------------------------- #
# LightCurve
# --------------------------------------------------------------------------- #
@dataclass
class LightCurve:
    """A single (possibly multi-sector) photometric time series.

    Parameters
    ----------
    time:
        Observation times in days, ``float64``, ideally sorted ascending.
    flux:
        Relative flux, ``float32``, normalised so the out-of-transit level is
        ~1.0.
    flux_err:
        Per-point flux uncertainty, ``float32``. May be all-NaN if unknown.
    meta:
        Free-form metadata. Well-known keys produced by the synthetic generator
        and consumed downstream include:
        ``tic_id``, ``sector``, ``cadence_s``, ``mission``, ``teff``, ``logg``,
        ``radius``, ``mass``, ``crowdsap``, ``ra``, ``dec``, ``tmag``,
        ``quality`` (int flag array), ``label`` (ground-truth class), and the
        injected truth ``true_period``, ``true_depth``, ``true_duration``,
        ``true_t0``.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    meta: dict = field(default_factory=dict)

    # -- construction helpers ------------------------------------------------ #
    def __post_init__(self) -> None:
        # Be permissive about inputs (lists, tuples) but settle on canonical
        # dtypes so downstream code never has to guess.
        self.time = np.asarray(self.time, dtype=np.float64)
        self.flux = np.asarray(self.flux, dtype=np.float32)
        if self.flux_err is None:
            self.flux_err = np.full(self.flux.shape, np.nan, dtype=np.float32)
        else:
            self.flux_err = np.asarray(self.flux_err, dtype=np.float32)

    # -- size / dunder helpers ---------------------------------------------- #
    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def n(self) -> int:
        """Number of cadences."""
        return int(self.time.size)

    # -- operations ---------------------------------------------------------- #
    def normalize(self, inplace: bool = False) -> LightCurve:
        """Divide flux (and errors) by the median so the baseline is ~1.0.

        Uses ``np.nanmedian`` so masked/NaN cadences do not bias the level. If
        the median is non-finite or zero the light curve is returned unchanged.
        """
        med = np.nanmedian(self.flux)
        target = self if inplace else self.copy()
        if not np.isfinite(med) or med == 0:
            return target
        target.flux = (np.asarray(target.flux, dtype=np.float64) / med).astype(np.float32)
        with np.errstate(invalid="ignore"):
            target.flux_err = (
                np.asarray(target.flux_err, dtype=np.float64) / med
            ).astype(np.float32)
        return target

    def remove_nans(self) -> LightCurve:
        """Return a copy with non-finite ``time`` or ``flux`` cadences dropped.

        Any array-valued ``meta`` entry of matching length (e.g. ``quality``) is
        masked consistently so per-cadence metadata stays aligned.
        """
        good = np.isfinite(self.time) & np.isfinite(self.flux)
        return self._apply_mask(good)

    def quality_mask(self, bad_bits: int | None = None) -> np.ndarray:
        """Boolean array (``True`` == keep) from a TESS-like ``quality`` flag.

        If ``meta['quality']`` is absent, every finite cadence is kept. When
        ``bad_bits`` is given, cadences whose quality value shares any bit with
        ``bad_bits`` are rejected; otherwise any non-zero quality is rejected.
        """
        finite = np.isfinite(self.time) & np.isfinite(self.flux)
        quality = self.meta.get("quality")
        if quality is None:
            return finite
        quality = np.asarray(quality)
        if quality.shape != self.flux.shape:
            return finite
        if bad_bits is None:
            bad = quality != 0
        else:
            bad = (quality.astype(np.int64) & int(bad_bits)) != 0
        return finite & ~bad

    def fold(self, period: float, t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Phase-fold onto ``period`` and return ``(phase, flux)`` sorted by phase.

        Phase is in ``[-0.5, 0.5)`` with transit centred at 0. The returned flux
        is reordered to match the sorted phase so it can be plotted directly.
        """
        period = float(period)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be a positive, finite number")
        phase = (((self.time - t0) / period + 0.5) % 1.0) - 0.5
        order = np.argsort(phase)
        return phase[order], np.asarray(self.flux, dtype=np.float64)[order]

    def bin(self, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin in time into ``bins`` equal-width bins.

        Returns ``(bin_centers, bin_flux, bin_err)`` where ``bin_flux`` is the
        NaN-safe mean per bin and ``bin_err`` is the standard error of the mean
        (``std / sqrt(count)``). Empty bins are NaN.
        """
        bins = int(bins)
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
        t = self.time
        f = np.asarray(self.flux, dtype=np.float64)
        finite = np.isfinite(t) & np.isfinite(f)
        t = t[finite]
        f = f[finite]
        if t.size == 0:
            nan = np.full(bins, np.nan)
            return nan.copy(), nan.copy(), nan.copy()
        edges = np.linspace(t.min(), t.max(), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
        counts = np.bincount(idx, minlength=bins).astype(np.float64)
        sums = np.bincount(idx, weights=f, minlength=bins)
        sumsq = np.bincount(idx, weights=f * f, minlength=bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / counts
            var = sumsq / counts - mean**2
            var = np.where(var < 0, 0.0, var)  # guard fp roundoff
            err = np.sqrt(var) / np.sqrt(counts)
        mean[counts == 0] = np.nan
        err[counts == 0] = np.nan
        return centers, mean, err

    def copy(self) -> LightCurve:
        """Deep-ish copy: arrays are copied, ``meta`` is shallow-copied.

        Array-valued ``meta`` entries are copied so masking a copy never mutates
        the original's per-cadence metadata.
        """
        new_meta: dict[str, Any] = {}
        for key, value in self.meta.items():
            if isinstance(value, np.ndarray):
                new_meta[key] = value.copy()
            else:
                new_meta[key] = value
        return LightCurve(
            time=self.time.copy(),
            flux=self.flux.copy(),
            flux_err=self.flux_err.copy(),
            meta=new_meta,
        )

    # -- internal ------------------------------------------------------------ #
    def _apply_mask(self, mask: np.ndarray) -> LightCurve:
        """Return a new ``LightCurve`` keeping cadences where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        new_meta: dict[str, Any] = {}
        for key, value in self.meta.items():
            if isinstance(value, np.ndarray) and value.shape == self.flux.shape:
                new_meta[key] = value[mask].copy()
            elif isinstance(value, np.ndarray):
                new_meta[key] = value.copy()
            else:
                new_meta[key] = value
        return LightCurve(
            time=self.time[mask],
            flux=self.flux[mask],
            flux_err=self.flux_err[mask],
            meta=new_meta,
        )


# --------------------------------------------------------------------------- #
# DetectionResult
# --------------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """Output of a periodic-transit search (BLS / TLS / custom box search)."""

    period: float = np.nan
    t0: float = np.nan
    duration: float = np.nan
    depth: float = np.nan
    sde: float = np.nan
    snr: float = np.nan
    method: str = ""
    periods: np.ndarray | None = None
    power: np.ndarray | None = None
    harmonics: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# VettingReport
# --------------------------------------------------------------------------- #
@dataclass
class VettingReport:
    """Diagnostic metrics and boolean false-positive flags.

    ``metrics`` holds continuous diagnostics (e.g. ``odd_even_depth_ratio``,
    ``secondary_depth``, ``v_shape``, ``centroid_offset``). ``flags`` holds
    boolean verdicts (e.g. ``is_eb``, ``secondary_detected``,
    ``odd_even_mismatch``, ``centroid_shift``).
    """

    metrics: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# TransitFit
# --------------------------------------------------------------------------- #
@dataclass
class TransitFit:
    """Result of a transit-model fit with uncertainty estimates.

    ``params`` maps a parameter name to a ``(median, err_lo, err_hi)`` tuple,
    where ``err_lo``/``err_hi`` are the lower/upper 1-sigma (or 16th/84th
    percentile) uncertainties. Expected keys include ``period``, ``t0``,
    ``depth``, ``duration``, ``rp_rs``, ``a_rs``, ``inc``, ``b``.
    """

    params: dict = field(default_factory=dict)
    model_time: np.ndarray | None = None
    model_flux: np.ndarray | None = None
    bic_transit: float = np.nan
    bic_flat: float = np.nan
    delta_bic: float = np.nan
    snr: float = np.nan
    method: str = ""
    samples: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@dataclass
class Classification:
    """Predicted astrophysical category for a detected signal.

    ``label`` is one of ``{'transit', 'eclipsing_binary', 'blend', 'other'}``.
    ``probabilities`` maps each class label to its predicted probability.
    """

    label: str = "other"
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    method: str = ""
    rationale: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# CandidateResult
# --------------------------------------------------------------------------- #
@dataclass
class CandidateResult:
    """Everything known about one candidate, end to end.

    Bundles the source light curve and the outputs of every pipeline stage so a
    single object can be serialised, visualised, or flattened into a catalog row.
    """

    lightcurve: LightCurve
    detection: DetectionResult
    vetting: VettingReport
    fit: TransitFit
    classification: Classification
    features: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flatten into a single dict suitable for a CSV/Parquet catalog.

        Pulls best-estimate parameters (period/depth/duration with ± errors)
        preferentially from the model :class:`TransitFit`, falls back to the
        :class:`DetectionResult` when the fit is missing a value, copies
        significance (``snr``/``sde``) from detection, the predicted
        class/confidence/probabilities from :class:`Classification`, the most
        informative vetting flags, and the star identifiers from
        ``lightcurve.meta``.

        All values are plain Python scalars/strings so the resulting dict feeds
        straight into ``pandas.DataFrame([...])``.
        """
        row: dict[str, Any] = {}

        # -- identifiers from the light-curve metadata ----------------------- #
        meta = self.lightcurve.meta if self.lightcurve is not None else {}
        row["tic_id"] = meta.get("tic_id")
        row["sector"] = meta.get("sector")

        # -- helper: read a (median, lo, hi) param triple ------------------- #
        def _fit_triple(name: str) -> tuple[float, float, float]:
            value = self.fit.params.get(name) if self.fit is not None else None
            if value is None:
                return (np.nan, np.nan, np.nan)
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            median = float(arr[0]) if arr.size >= 1 else np.nan
            lo = float(arr[1]) if arr.size >= 2 else np.nan
            hi = float(arr[2]) if arr.size >= 3 else lo
            return (median, lo, hi)

        def _coalesce(primary: float, fallback: float) -> float:
            return primary if np.isfinite(primary) else float(fallback)

        det = self.detection if self.detection is not None else DetectionResult()

        # -- period / depth / duration (fit first, detection fallback) ------- #
        for name, det_fallback in (
            ("period", det.period),
            ("depth", det.depth),
            ("duration", det.duration),
        ):
            median, lo, hi = _fit_triple(name)
            row[name] = _coalesce(median, det_fallback)
            row[f"{name}_err_lo"] = lo
            row[f"{name}_err_hi"] = hi

        # epoch / mid-transit time
        t0_med, t0_lo, t0_hi = _fit_triple("t0")
        row["t0"] = _coalesce(t0_med, det.t0)
        row["t0_err_lo"] = t0_lo
        row["t0_err_hi"] = t0_hi

        # -- significance from detection ------------------------------------- #
        row["snr"] = float(self.fit.snr) if (
            self.fit is not None and np.isfinite(self.fit.snr)
        ) else float(det.snr)
        row["sde"] = float(det.sde)
        row["detection_method"] = det.method

        # -- classification -------------------------------------------------- #
        cls = self.classification if self.classification is not None else Classification()
        row["class"] = cls.label
        row["confidence"] = float(cls.confidence)
        row["classify_method"] = cls.method
        for class_name in ("transit", "eclipsing_binary", "blend", "other"):
            row[f"prob_{class_name}"] = float(cls.probabilities.get(class_name, np.nan))

        # -- key vetting flags ----------------------------------------------- #
        vet = self.vetting if self.vetting is not None else VettingReport()
        for flag_name in (
            "is_eb",
            "secondary_detected",
            "odd_even_mismatch",
            "centroid_shift",
            "is_blend",
        ):
            row[f"flag_{flag_name}"] = bool(vet.flags.get(flag_name, False))
        # a couple of headline continuous diagnostics, when present
        for metric_name in ("odd_even_depth_ratio", "secondary_depth", "centroid_offset"):
            if metric_name in vet.metrics:
                row[metric_name] = float(vet.metrics[metric_name])

        # -- model goodness-of-fit ------------------------------------------- #
        if self.fit is not None:
            row["delta_bic"] = float(self.fit.delta_bic)
            row["fit_method"] = self.fit.method

        return row
