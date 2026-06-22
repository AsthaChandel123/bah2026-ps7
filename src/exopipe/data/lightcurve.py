"""Helper constructors and operations for :class:`~exopipe.types.LightCurve`.

These functions wrap and extend the dataclass methods with the dtype coercion,
normalisation, multi-sector stitching, and outlier handling that the rest of the
pipeline relies on. Importing modules should prefer these over building
``LightCurve`` objects by hand so the dtype/normalisation conventions stay
consistent.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from ..types import LightCurve

__all__ = [
    "from_arrays",
    "stitch",
    "quality_mask",
    "sigma_clip",
]


def from_arrays(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
) -> LightCurve:
    """Build a :class:`LightCurve` from raw arrays with canonical conventions.

    * ``time`` is coerced to ``float64``, ``flux``/``flux_err`` to ``float32``.
    * Arrays are sorted by time (keeping ``flux``/``flux_err`` aligned).
    * Flux is normalised to a median of ~1.0 *unless it already is* (within 1%),
      which avoids re-dividing data that arrives pre-normalised.

    Parameters
    ----------
    time, flux:
        Equal-length 1-D sequences.
    flux_err:
        Optional per-point uncertainty; defaults to all-NaN.
    meta:
        Optional metadata dict (copied by reference into the light curve).
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError(
            f"time and flux must have the same shape, got {time.shape} vs {flux.shape}"
        )
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)
        if flux_err.shape != flux.shape:
            raise ValueError("flux_err must match the shape of flux")

    # Sort chronologically so downstream folding/binning is well-defined.
    if time.size > 1 and not np.all(np.diff(time) >= 0):
        order = np.argsort(time, kind="stable")
        time = time[order]
        flux = flux[order]
        if flux_err is not None:
            flux_err = flux_err[order]

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta=dict(meta or {}))

    # Normalise only if it is not already ~1.0 (robust to NaNs).
    med = np.nanmedian(lc.flux)
    if np.isfinite(med) and med > 0 and not np.isclose(med, 1.0, atol=1e-2):
        lc = lc.normalize(inplace=True)
    return lc


def stitch(lcs: Iterable[LightCurve]) -> LightCurve:
    """Concatenate multiple (multi-sector) light curves into one.

    Each input is normalised *independently* (per-segment) before concatenation
    so sector-to-sector flux-level offsets do not introduce artificial steps.
    The result is sorted by time. ``meta`` is taken from the first segment and
    augmented with a ``sectors`` list and a per-cadence ``segment`` index array;
    a concatenated ``quality`` array is preserved when every segment has one.

    Empty inputs raise ``ValueError``; a single input is returned (normalised).
    """
    lcs = [lc for lc in lcs if lc is not None]
    if not lcs:
        raise ValueError("stitch() requires at least one LightCurve")

    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    errs: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    sectors: list[Any] = []
    have_all_quality = True

    for seg_index, lc in enumerate(lcs):
        norm = lc.normalize(inplace=False)
        times.append(np.asarray(norm.time, dtype=np.float64))
        fluxes.append(np.asarray(norm.flux, dtype=np.float64))
        errs.append(np.asarray(norm.flux_err, dtype=np.float64))
        seg_ids.append(np.full(norm.time.shape, seg_index, dtype=np.int32))
        sectors.append(lc.meta.get("sector"))
        quality = lc.meta.get("quality")
        if isinstance(quality, np.ndarray) and quality.shape == norm.flux.shape:
            qualities.append(np.asarray(quality))
        else:
            have_all_quality = False

    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    flux_err = np.concatenate(errs)
    segment = np.concatenate(seg_ids)

    order = np.argsort(time, kind="stable")
    time = time[order]
    flux = flux[order]
    flux_err = flux_err[order]
    segment = segment[order]

    meta: dict[str, Any] = dict(lcs[0].meta)
    meta["sectors"] = sectors
    meta["segment"] = segment
    meta["n_segments"] = len(lcs)
    if have_all_quality:
        meta["quality"] = np.concatenate(qualities)[order]
    else:
        meta.pop("quality", None)

    return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)


def quality_mask(
    lc: LightCurve,
    bad_bits: int | None = None,
    finite_only: bool = True,
) -> LightCurve:
    """Return a copy of ``lc`` with bad-quality / non-finite cadences removed.

    Delegates to :meth:`LightCurve.quality_mask` for the boolean selection. When
    ``finite_only`` is True (default) non-finite ``time``/``flux`` cadences are
    always dropped even if no quality array is present.
    """
    mask = lc.quality_mask(bad_bits=bad_bits)
    if finite_only:
        mask = mask & np.isfinite(lc.time) & np.isfinite(lc.flux)
    return lc._apply_mask(mask)


def sigma_clip(
    lc: LightCurve,
    sigma: float = 5.0,
    asymmetric: bool = True,
    iters: int = 5,
) -> LightCurve:
    """Iterative robust sigma-clipping that preserves transits.

    Outliers are measured against the median using a robust scale
    (``1.4826 * MAD``). When ``asymmetric`` is True, *positive* excursions
    (brightenings, cosmic rays) are clipped at ``sigma`` while *negative*
    excursions (which include genuine transit/eclipse dips) are clipped far more
    leniently (``3 * sigma``), so real transits survive.

    Parameters
    ----------
    sigma:
        Threshold for positive outliers (and the base for the lenient negative
        threshold).
    asymmetric:
        If False, clip symmetrically at ``sigma`` on both sides.
    iters:
        Maximum number of clipping iterations; stops early once stable.
    """
    from ..utils import robust_std

    keep = np.isfinite(lc.time) & np.isfinite(lc.flux)
    flux = np.asarray(lc.flux, dtype=np.float64)
    neg_factor = 3.0 if asymmetric else 1.0

    for _ in range(max(int(iters), 1)):
        current = flux[keep]
        if current.size < 3:
            break
        med = np.nanmedian(current)
        scale = robust_std(current)
        if not np.isfinite(scale) or scale == 0:
            break
        resid = flux - med
        upper = resid <= (sigma * scale)
        lower = resid >= (-neg_factor * sigma * scale)
        new_keep = keep & upper & lower
        if new_keep.sum() == keep.sum():  # converged
            keep = new_keep
            break
        keep = new_keep

    return lc._apply_mask(keep)
