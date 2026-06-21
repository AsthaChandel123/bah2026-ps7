"""Scientific visualisation for ``exopipe`` -- the one-page vetting sheet.

This module is the **headline visualisation deliverable** for PS7 requirement R6
("visualise the light curve along with the detected and classified signal") and
R7 ("confidence level of the detected signal"). It imitates the TESS SPOC Data
Validation *one-page Report Summary* using a single multi-panel matplotlib
figure built with :func:`matplotlib.figure.Figure.subplot_mosaic`.

Design goals
------------
* **matplotlib only.** No optional dependency is required -- matplotlib is the
  always-available backend per the foundation conventions, so the vetting sheet
  renders in the pure-core environment.
* **Robust to partial results.** Every panel degrades gracefully: a missing or
  all-NaN field draws an "n/a" placeholder rather than raising. A degraded
  :class:`~exopipe.types.CandidateResult` (e.g. NaN ``fit``) never crashes the
  sheet.
* **Colour-blind safe.** Categorical classes use the Okabe--Ito palette
  (:data:`CLASS_COLORS`); sequential data (river plot) uses ``viridis``. Meaning
  is never encoded by colour alone -- markers, line styles and text labels are
  always paired with colour.

Public API
----------
``vetting_sheet(result, save_path=None, dpi=130) -> Figure``
    The full one-page sheet. Returns the :class:`~matplotlib.figure.Figure`;
    writes a PNG when ``save_path`` is given.

Reusable per-panel helpers (each accepts an optional ``ax``)::

    plot_phasefold(lc, det, fit=None, ax=None, zoom=False)
    plot_periodogram(det, ax=None)
    plot_oddeven(lc, det, ax=None)
    plot_secondary(lc, det, ax=None)
    plot_river(lc, det, ax=None)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

# Use a non-interactive backend so the sheet renders in headless / batch runs
# (CI, the driver fan-out, the report generator) without a display server.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
from matplotlib.figure import Figure  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from .types import CandidateResult, DetectionResult, LightCurve, TransitFit

__all__ = [
    "CLASS_COLORS",
    "CLASS_LABELS",
    "vetting_sheet",
    "plot_phasefold",
    "plot_periodogram",
    "plot_oddeven",
    "plot_secondary",
    "plot_river",
]

# --------------------------------------------------------------------------- #
# Colour & label conventions (consistent with ARCHITECTURE.md §11 / dossier 06)
# --------------------------------------------------------------------------- #
#: Okabe--Ito colour-blind-safe palette keyed by the canonical class labels.
CLASS_COLORS: dict[str, str] = {
    "transit": "#0072B2",           # blue
    "eclipsing_binary": "#D55E00",  # vermillion
    "blend": "#CC79A7",             # reddish-purple
    "other": "#999999",             # grey
}

#: Canonical class ordering used for probability bars / tables.
CLASS_LABELS: tuple[str, ...] = ("transit", "eclipsing_binary", "blend", "other")

#: Short human-friendly names for compact panels.
_CLASS_SHORT: dict[str, str] = {
    "transit": "transit",
    "eclipsing_binary": "EB",
    "blend": "blend",
    "other": "other",
}

# Reused styling constants.
_MODEL_COLOR = "#D55E00"   # vermillion solid line == "model" (also labelled)
_BIN_COLOR = "#56B4E9"     # sky-blue binned points (Okabe--Ito)
_DATA_COLOR = "0.55"       # neutral grey for raw cadences
_TRANSIT_TICK = "#0072B2"


# --------------------------------------------------------------------------- #
# Small numeric / safety helpers
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> bool:
    """True iff ``value`` is a finite real scalar."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _as_array(value: Any) -> np.ndarray:
    """Coerce to a 1-D float array, returning an empty array on failure."""
    if value is None:
        return np.empty(0, dtype=float)
    try:
        return np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)


def _clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the subset of ``(x, y)`` where both are finite."""
    n = min(x.size, y.size)
    if n == 0:
        return x[:0], y[:0]
    x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _na_panel(ax: Axes, title: str, message: str = "n/a") -> None:
    """Render a titled but empty panel carrying an ``n/a`` placeholder."""
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="0.5",
        style="italic",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _fit_param(fit: TransitFit | None, name: str) -> tuple[float, float, float]:
    """Read a ``(median, err_lo, err_hi)`` triple from a fit, NaNs if absent."""
    if fit is None or getattr(fit, "params", None) is None:
        return (np.nan, np.nan, np.nan)
    value = fit.params.get(name)
    if value is None:
        return (np.nan, np.nan, np.nan)
    arr = _as_array(value)
    median = float(arr[0]) if arr.size >= 1 else np.nan
    lo = float(arr[1]) if arr.size >= 2 else np.nan
    hi = float(arr[2]) if arr.size >= 3 else lo
    return (median, lo, hi)


def _coalesce(*values: Any) -> float:
    """First finite scalar among ``values`` (NaN if none)."""
    for value in values:
        if _finite(value):
            return float(value)
    return float("nan")


def _binned_phase(
    phase: np.ndarray, flux: np.ndarray, n_bins: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe mean of ``flux`` in ``n_bins`` equal-width phase bins."""
    phase, flux = _clean_xy(phase, flux)
    if phase.size == 0:
        return np.empty(0), np.empty(0)
    lo, hi = float(phase.min()), float(phase.max())
    if hi <= lo:
        return np.empty(0), np.empty(0)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    sums = np.bincount(idx, weights=flux, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    mean[counts == 0] = np.nan
    keep = np.isfinite(mean)
    return centers[keep], mean[keep]


def _transit_times(
    det: DetectionResult | None, t_min: float, t_max: float, max_marks: int = 400
) -> np.ndarray:
    """Predicted transit mid-times within ``[t_min, t_max]`` from ``period``/``t0``."""
    if det is None:
        return np.empty(0)
    period = getattr(det, "period", np.nan)
    t0 = getattr(det, "t0", np.nan)
    if not (_finite(period) and period > 0 and _finite(t0)):
        return np.empty(0)
    if not (_finite(t_min) and _finite(t_max)) or t_max <= t_min:
        return np.empty(0)
    n_lo = int(np.floor((t_min - t0) / period))
    n_hi = int(np.ceil((t_max - t0) / period))
    if (n_hi - n_lo) > max_marks:  # avoid pathological short periods flooding the axis
        return np.empty(0)
    epochs = np.arange(n_lo, n_hi + 1)
    times = t0 + epochs * period
    return times[(times >= t_min) & (times <= t_max)]


# --------------------------------------------------------------------------- #
# Reusable per-panel plotters
# --------------------------------------------------------------------------- #
def plot_phasefold(
    lc: LightCurve,
    det: DetectionResult,
    fit: TransitFit | None = None,
    ax: Axes | None = None,
    zoom: bool = False,
    n_durations: float = 2.5,
) -> Axes:
    """Phase-folded light curve with binned points and the best-fit model.

    Parameters
    ----------
    lc, det:
        Light curve and detection providing ``period``/``t0`` for the fold.
    fit:
        Optional :class:`~exopipe.types.TransitFit`; if it carries
        ``model_time``/``model_flux`` the model is folded on the same ephemeris
        and overlaid in red.
    ax:
        Target axis (created if ``None``).
    zoom:
        When ``True`` restrict the x-axis to ``±n_durations`` transit durations
        (the "local" view); otherwise show the full ``[-0.5, 0.5)`` phase range
        (the "global" view, which exposes any secondary eclipse).
    """
    if ax is None:
        _, ax = plt.subplots()
    title = "Phase fold (local)" if zoom else "Phase fold (global)"

    period = getattr(det, "period", np.nan)
    t0 = getattr(det, "t0", 0.0)
    if not (_finite(period) and period > 0):
        _na_panel(ax, title, "no period")
        return ax

    try:
        phase, flux = lc.fold(period, t0 if _finite(t0) else 0.0)
    except Exception:
        _na_panel(ax, title, "fold failed")
        return ax
    phase, flux = _clean_xy(np.asarray(phase, float), np.asarray(flux, float))
    if phase.size == 0:
        _na_panel(ax, title, "no data")
        return ax

    ax.plot(
        phase, flux, ".", ms=1.6, color=_DATA_COLOR, alpha=0.45,
        rasterized=True, label="data",
    )
    bph, bfl = _binned_phase(phase, flux, n_bins=80 if not zoom else 50)
    if bph.size:
        ax.plot(bph, bfl, "o", ms=3.5, color=_BIN_COLOR, label="binned", zorder=4)

    # --- best-fit model overlay (folded on the same ephemeris) ------------- #
    if fit is not None:
        mtime = _as_array(getattr(fit, "model_time", None))
        mflux = _as_array(getattr(fit, "model_flux", None))
        mtime, mflux = _clean_xy(mtime, mflux)
        if mtime.size:
            mphase = (((mtime - (t0 if _finite(t0) else 0.0)) / period + 0.5) % 1.0) - 0.5
            order = np.argsort(mphase)
            ax.plot(
                mphase[order], mflux[order], "-", color=_MODEL_COLOR, lw=1.6,
                label="model", zorder=5,
            )

    ax.set_xlabel("Phase")
    ax.set_ylabel("Norm. flux")
    ax.set_title(title, fontsize=9)
    ax.axvline(0.0, color="0.8", lw=0.6, zorder=0)

    if zoom:
        duration = _coalesce(
            getattr(det, "duration", np.nan),
            _fit_param(fit, "duration")[0],
        )
        if _finite(duration) and duration > 0:
            half = n_durations * duration / period
            ax.set_xlim(-half, half)
        else:
            ax.set_xlim(-0.1, 0.1)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    return ax


def plot_periodogram(det: DetectionResult, ax: Axes | None = None) -> Axes:
    """Periodogram (BLS power / TLS SDE) with the peak period and harmonics."""
    if ax is None:
        _, ax = plt.subplots()
    title = "Periodogram"

    periods = _as_array(getattr(det, "periods", None))
    power = _as_array(getattr(det, "power", None))
    periods, power = _clean_xy(periods, power)
    if periods.size == 0:
        _na_panel(ax, title, "no periodogram")
        return ax

    order = np.argsort(periods)
    periods, power = periods[order], power[order]
    ax.plot(periods, power, "-", color="0.25", lw=0.8)

    peak = getattr(det, "period", np.nan)
    if _finite(peak) and peak > 0:
        ax.axvline(
            peak, color=_TRANSIT_TICK, lw=1.3, label=f"P = {peak:.4f} d",
        )
        # Mark harmonics/aliases P/2, 2P, 3P (those that fall on the axis).
        harmonics = getattr(det, "harmonics", None) or [peak / 2.0, 2.0 * peak, 3.0 * peak]
        lo, hi = float(periods.min()), float(periods.max())
        labelled = False
        for h in harmonics:
            if _finite(h) and lo <= h <= hi:
                ax.axvline(
                    h, color="#E69F00", ls=":", lw=0.8, alpha=0.8,
                    label="harmonics" if not labelled else None,
                )
                labelled = True

    sde = getattr(det, "sde", np.nan)
    if _finite(sde):
        title = f"Periodogram (SDE = {sde:.1f})"
    try:
        if periods.min() > 0 and (periods.max() / max(periods.min(), 1e-9)) > 20:
            ax.set_xscale("log")
    except (ValueError, ZeroDivisionError):
        pass
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("Power / SDE")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    return ax


def _odd_even_split(
    lc: LightCurve, det: DetectionResult
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Phase-fold odd- and even-numbered transits separately.

    Returns ``((odd_phase, odd_flux), (even_phase, even_flux))`` or ``None`` when
    the ephemeris is unusable.
    """
    period = getattr(det, "period", np.nan)
    t0 = getattr(det, "t0", np.nan)
    if not (_finite(period) and period > 0 and _finite(t0)):
        return None
    time = np.asarray(lc.time, dtype=float)
    flux = np.asarray(lc.flux, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if time.size == 0:
        return None
    cycle = np.floor((time - t0) / period + 0.5).astype(np.int64)
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
    is_odd = (cycle % 2) != 0
    odd = (phase[is_odd], flux[is_odd])
    even = (phase[~is_odd], flux[~is_odd])
    return odd, even


def plot_oddeven(
    lc: LightCurve, det: DetectionResult, ax: Axes | None = None,
    n_durations: float = 2.5,
) -> Axes:
    """Overlay odd vs even folded transits (an eclipsing-binary discriminator)."""
    if ax is None:
        _, ax = plt.subplots()
    title = "Odd vs even"

    split = _odd_even_split(lc, det)
    if split is None:
        _na_panel(ax, title, "no ephemeris")
        return ax
    (odd_ph, odd_fl), (even_ph, even_fl) = split

    drew = False
    for (ph, fl), color, marker, lbl in (
        (_clean_xy(odd_ph, odd_fl), "#0072B2", "o", "odd"),
        (_clean_xy(even_ph, even_fl), "#D55E00", "s", "even"),
    ):
        bph, bfl = _binned_phase(ph, fl, n_bins=40)
        if ph.size:
            ax.plot(ph, fl, ".", ms=1.2, color=color, alpha=0.18, rasterized=True)
            drew = True
        if bph.size:
            ax.plot(bph, bfl, marker, ms=3.0, color=color, label=lbl, zorder=4)
            drew = True

    if not drew:
        _na_panel(ax, title, "no data")
        return ax

    # Annotate odd/even mismatch significance when present.
    sigma = None
    if getattr(det, "extra", None):
        sigma = det.extra.get("odd_even_sigma")
    if sigma is not None and _finite(sigma):
        ax.text(
            0.02, 0.04, f"odd-even Δ = {float(sigma):.1f}σ",
            transform=ax.transAxes, fontsize=6.5,
            color="#D55E00" if float(sigma) > 3 else "0.35",
        )

    period = getattr(det, "period", np.nan)
    duration = getattr(det, "duration", np.nan)
    if _finite(duration) and duration > 0 and _finite(period) and period > 0:
        half = n_durations * duration / period
        ax.set_xlim(-half, half)
    else:
        ax.set_xlim(-0.1, 0.1)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Norm. flux")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    return ax


def plot_secondary(
    lc: LightCurve, det: DetectionResult, ax: Axes | None = None,
    window: float = 0.15,
) -> Axes:
    """Zoom the global phase fold around phase 0.5 (secondary-eclipse search)."""
    if ax is None:
        _, ax = plt.subplots()
    title = "Secondary @ φ=0.5"

    period = getattr(det, "period", np.nan)
    t0 = getattr(det, "t0", 0.0)
    if not (_finite(period) and period > 0):
        _na_panel(ax, title, "no period")
        return ax
    try:
        phase, flux = lc.fold(period, t0 if _finite(t0) else 0.0)
    except Exception:
        _na_panel(ax, title, "fold failed")
        return ax
    # Re-wrap phase to [0, 1) so phase 0.5 is centred and contiguous.
    phase = np.mod(phase, 1.0)
    phase, flux = _clean_xy(np.asarray(phase, float), np.asarray(flux, float))
    sel = np.abs(phase - 0.5) < window
    if not np.any(sel):
        _na_panel(ax, title, "no data")
        return ax
    ax.plot(phase[sel], flux[sel], ".", ms=2.0, color=_DATA_COLOR, rasterized=True)
    bph, bfl = _binned_phase(phase[sel], flux[sel], n_bins=25)
    if bph.size:
        ax.plot(bph, bfl, "o", ms=3.0, color=_BIN_COLOR, zorder=4)
    ax.axvline(0.5, color="#D55E00", ls="--", lw=0.8, label="φ=0.5")

    # Annotate secondary depth if the vetting/detection recorded one.
    sec_depth = None
    if getattr(det, "extra", None):
        sec_depth = det.extra.get("secondary_depth")
    if sec_depth is not None and _finite(sec_depth):
        ax.text(
            0.02, 0.05, f"sec depth ≈ {float(sec_depth) * 1e6:.0f} ppm",
            transform=ax.transAxes, fontsize=6.5, color="0.35",
        )
    ax.set_xlabel("Phase")
    ax.set_ylabel("Norm. flux")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    return ax


def plot_river(
    lc: LightCurve, det: DetectionResult, ax: Axes | None = None,
    n_phase_bins: int = 60, max_cycles: int = 200,
) -> Axes:
    """River / waterfall plot: rows = orbital cycles, columns = phase, colour = flux.

    Reveals transit-timing variations, missed transits, and whether the signal
    persists across the baseline. Implemented in pure NumPy (no ``lightkurve``)
    so it works with core deps only; uses the perceptually-uniform ``viridis``
    sequential colormap.
    """
    if ax is None:
        _, ax = plt.subplots()
    title = "River plot"

    period = getattr(det, "period", np.nan)
    t0 = getattr(det, "t0", np.nan)
    if not (_finite(period) and period > 0 and _finite(t0)):
        _na_panel(ax, title, "no ephemeris")
        return ax

    time = np.asarray(lc.time, dtype=float)
    flux = np.asarray(lc.flux, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if time.size == 0:
        _na_panel(ax, title, "no data")
        return ax

    cycle = np.floor((time - t0) / period + 0.5).astype(np.int64)
    cycle -= cycle.min()
    n_cycles = int(cycle.max()) + 1
    if n_cycles < 2 or n_cycles > max_cycles:
        _na_panel(ax, title, "n/a (baseline)")
        return ax
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5

    # Accumulate a (cycle x phase) grid of mean flux.
    edges = np.linspace(-0.5, 0.5, n_phase_bins + 1)
    pbin = np.clip(np.digitize(phase, edges) - 1, 0, n_phase_bins - 1)
    grid_sum = np.zeros((n_cycles, n_phase_bins))
    grid_cnt = np.zeros((n_cycles, n_phase_bins))
    np.add.at(grid_sum, (cycle, pbin), flux)
    np.add.at(grid_cnt, (cycle, pbin), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = grid_sum / grid_cnt
    grid = np.where(grid_cnt > 0, grid, np.nan)

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        _na_panel(ax, title, "no data")
        return ax
    vmin, vmax = np.nanpercentile(finite, [3, 97])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=cmap,
        extent=(-0.5, 0.5, 0, n_cycles), vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.axvline(0.0, color="white", lw=0.6, alpha=0.7)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Cycle")
    ax.set_title(title, fontsize=9)
    # Attach the colorbar to a child inset_axes rather than via ``figure.colorbar
    # (ax=ax)``: a constrained-layout-managed colorbar can steal all the width
    # from a small mosaic cell and collapse it to zero size (raising a layout
    # warning and producing an empty panel). An inset is ignored by the
    # constrained solver, so the river panel keeps its full size.
    try:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
    except Exception:  # pragma: no cover - colorbar is cosmetic
        pass
    return ax


# --------------------------------------------------------------------------- #
# Full-baseline light-curve panel + text/summary panels (sheet-internal)
# --------------------------------------------------------------------------- #
def _plot_fullcurve(lc: LightCurve, det: DetectionResult, ax: Axes) -> None:
    """Full detrended flux vs time with predicted transit epochs marked."""
    time = np.asarray(getattr(lc, "time", None), dtype=float)
    flux = np.asarray(getattr(lc, "flux", None), dtype=float)
    time, flux = _clean_xy(time, flux)
    if time.size == 0:
        _na_panel(ax, "Light curve", "no data")
        return
    ax.plot(time, flux, ".", ms=1.2, color="0.35", alpha=0.6, rasterized=True)

    marks = _transit_times(det, float(time.min()), float(time.max()))
    if marks.size:
        y = float(np.nanpercentile(flux, 1.0))
        ax.plot(
            marks, np.full_like(marks, y), marker="v", ls="none",
            color=_TRANSIT_TICK, ms=6, label="transits", zorder=5,
        )
        ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Norm. flux")
    ax.set_title("Detrended light curve", fontsize=9)


def _plot_probbar(classification: Any, ax: Axes) -> None:
    """Horizontal calibrated class-probability bar (sums to ~1)."""
    title = "Class probabilities"
    probs = getattr(classification, "probabilities", None) if classification else None
    if not probs:
        _na_panel(ax, title, "n/a")
        return
    names = list(CLASS_LABELS)
    vals = [float(probs.get(n, np.nan)) for n in names]
    vals = [v if _finite(v) else 0.0 for v in vals]
    colors = [CLASS_COLORS.get(n, "0.5") for n in names]
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([_CLASS_SHORT[n] for n in names], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(class)", fontsize=8)
    ax.set_title(title, fontsize=9)
    predicted = getattr(classification, "label", None)
    for i, (name, v) in enumerate(zip(names, vals, strict=True)):
        weight = "bold" if name == predicted else "normal"
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center", fontsize=6.5,
                fontweight=weight)
    ax.invert_yaxis()


def _plot_textsummary(result: CandidateResult, ax: Axes) -> None:
    """Text panel: TIC, calibrated class + confidence, parameters ± err, flags."""
    ax.axis("off")
    lc = getattr(result, "lightcurve", None)
    det = getattr(result, "detection", None)
    fit = getattr(result, "fit", None)
    classification = getattr(result, "classification", None)
    vetting = getattr(result, "vetting", None)
    meta = getattr(lc, "meta", {}) if lc is not None else {}

    label = getattr(classification, "label", "n/a") if classification else "n/a"
    confidence = getattr(classification, "confidence", np.nan) if classification else np.nan
    color = CLASS_COLORS.get(label, "k")

    tic = meta.get("tic_id", "n/a")
    sector = meta.get("sector", "n/a")

    # Parameters: prefer the fit, fall back to detection.
    period = _coalesce(_fit_param(fit, "period")[0], getattr(det, "period", np.nan))
    p_lo, p_hi = _fit_param(fit, "period")[1:]
    depth_frac = _coalesce(_fit_param(fit, "depth")[0], getattr(det, "depth", np.nan))
    d_lo, d_hi = _fit_param(fit, "depth")[1:]
    dur = _coalesce(_fit_param(fit, "duration")[0], getattr(det, "duration", np.nan))
    du_lo, du_hi = _fit_param(fit, "duration")[1:]
    snr = _coalesce(getattr(fit, "snr", np.nan), getattr(det, "snr", np.nan))
    sde = getattr(det, "sde", np.nan)

    def _pm(med: float, lo: float, hi: float, scale: float = 1.0, unit: str = "") -> str:
        if not _finite(med):
            return "n/a"
        base = f"{med * scale:.4g}"
        if _finite(lo) and _finite(hi):
            return f"{base} (+{hi * scale:.2g}/-{lo * scale:.2g}){unit}"
        return f"{base}{unit}"

    lines: list[tuple[str, str]] = []
    lines.append(("HEADER", f"TIC {tic}    Sector {sector}"))
    conf_str = f"{float(confidence):.2f}" if _finite(confidence) else "n/a"
    lines.append(("CLASS", f"{str(label).upper()}   (confidence {conf_str})"))
    lines.append(("BLANK", ""))
    depth_ppm = depth_frac * 1e6 if _finite(depth_frac) else np.nan
    dl_ppm = d_lo * 1e6 if _finite(d_lo) else np.nan
    dh_ppm = d_hi * 1e6 if _finite(d_hi) else np.nan
    lines.append(("ROW", f"Period   : {_pm(period, p_lo, p_hi, 1.0, ' d')}"))
    lines.append(("ROW", f"Depth    : {_pm(depth_ppm, dl_ppm, dh_ppm, 1.0, ' ppm')}"))
    lines.append(("ROW", f"Duration : {_pm(dur, du_lo, du_hi, 24.0, ' h')}"))
    snr_str = f"{snr:.1f}" if _finite(snr) else "n/a"
    sde_str = f"{sde:.1f}" if _finite(sde) else "n/a"
    lines.append(("ROW", f"SNR / SDE: {snr_str} / {sde_str}"))

    # Key vetting flags (raised ones first).
    flags = getattr(vetting, "flags", {}) if vetting else {}
    raised = [k for k, v in flags.items() if v]
    lines.append(("BLANK", ""))
    if raised:
        lines.append(("ROW", "Flags raised: " + ", ".join(raised[:5])))
    else:
        lines.append(("ROW", "Flags raised: none"))

    # Rationale (truncated).
    rationale = getattr(classification, "rationale", None) if classification else None
    if rationale:
        text = "; ".join(str(r) for r in rationale)
        if len(text) > 90:
            text = text[:87] + "..."
        lines.append(("ROW", f"Rationale: {text}"))

    # Render top-to-bottom.
    y = 0.97
    for kind, text in lines:
        if kind == "BLANK":
            y -= 0.05
            continue
        if kind == "HEADER":
            ax.text(0.02, y, text, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="top")
            y -= 0.11
        elif kind == "CLASS":
            ax.text(0.02, y, text, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", color=color, va="top")
            y -= 0.11
        else:
            ax.text(0.02, y, text, transform=ax.transAxes, fontsize=8.5,
                    family="monospace", va="top")
            y -= 0.085
    # Coloured border keyed to the class for instant visual class read-out.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)
    ax.set_title("Summary & confidence", fontsize=9, color=color)


# --------------------------------------------------------------------------- #
# The one-page vetting sheet
# --------------------------------------------------------------------------- #
# Mosaic layout (each letter is a panel). Mirrors the DV one-page summary:
#   T  full light curve (full width)
#   G  global phase fold        L  local zoom
#   O  odd/even                 P  periodogram
#   S  secondary @0.5           R  river
#   X  text summary (2 rows)    B  class-probability bar
_MOSAIC = [
    ["T", "T", "T", "T"],
    ["G", "G", "L", "L"],
    ["O", "O", "P", "P"],
    ["S", "S", "R", "R"],
    ["X", "X", "B", "B"],
]
_HEIGHT_RATIOS = [1.0, 1.25, 1.1, 1.1, 1.0]


def vetting_sheet(
    result: CandidateResult,
    save_path: str | None = None,
    dpi: int = 130,
) -> Figure:
    """Render the one-page vetting sheet for a candidate.

    Builds an A4-portrait multi-panel matplotlib figure (via ``subplot_mosaic``)
    summarising detection, phase-folded shape, odd/even and secondary-eclipse
    diagnostics, the periodogram, a river plot, the calibrated class-probability
    bar, and a text panel with parameters ± uncertainties and vetting flags.

    The function is **defensive**: any missing/NaN field degrades to an "n/a"
    placeholder for its panel and never raises, so it can be called on partial or
    degraded :class:`~exopipe.types.CandidateResult` objects (e.g. when the fit
    failed).

    Parameters
    ----------
    result:
        The candidate to visualise.
    save_path:
        If given, the figure is written to this path (PNG inferred from the
        extension) at ``dpi`` resolution.
    dpi:
        Output resolution for the saved PNG (the on-screen figure is vector).

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (always returned, even when saved).
    """
    lc = getattr(result, "lightcurve", None)
    det = getattr(result, "detection", None)
    fit = getattr(result, "fit", None)
    classification = getattr(result, "classification", None)

    # A4 portrait. We use explicit gridspec spacing (not constrained/tight
    # layout): the dense 5x4 mosaic plus a colorbar inset and the coloured-spine
    # text panel can make the auto layout engines collapse a cell to zero size
    # (which emits a warning and blanks a panel). Fixed margins/spacing are
    # deterministic and warning-free for this known geometry.
    fig = plt.figure(figsize=(8.27, 11.69))
    axd = fig.subplot_mosaic(
        _MOSAIC,
        height_ratios=_HEIGHT_RATIOS,
        gridspec_kw={
            "left": 0.085,
            "right": 0.93,
            "top": 0.935,
            "bottom": 0.05,
            "hspace": 0.55,
            "wspace": 0.32,
        },
    )

    # Suptitle gives an at-a-glance header even if individual panels degrade.
    meta = getattr(lc, "meta", {}) if lc is not None else {}
    label = getattr(classification, "label", "n/a") if classification else "n/a"
    confidence = getattr(classification, "confidence", np.nan) if classification else np.nan
    conf_str = f"{float(confidence):.2f}" if _finite(confidence) else "n/a"
    fig.suptitle(
        f"exopipe vetting sheet  |  TIC {meta.get('tic_id', 'n/a')}  |  "
        f"Sector {meta.get('sector', 'n/a')}  |  "
        f"{str(label).upper()} (conf {conf_str})",
        fontsize=12, fontweight="bold",
        color=CLASS_COLORS.get(label, "k"),
    )

    # Each panel is wrapped so a single panel failure cannot kill the sheet.
    def _safe(fn, key, *args, **kw):
        ax = axd[key]
        try:
            fn(*args, ax=ax, **kw)
        except Exception as exc:  # pragma: no cover - last-resort guard
            ax.clear()
            _na_panel(ax, key, f"error: {type(exc).__name__}")

    if lc is None or det is None:
        # Without a light curve or detection most panels are meaningless; still
        # emit the text summary so the sheet carries whatever is known.
        for key in ("T", "G", "L", "O", "P", "S", "R"):
            _na_panel(axd[key], key, "n/a")
    else:
        _safe(_plot_fullcurve, "T", lc, det)
        _safe(plot_phasefold, "G", lc, det, fit, zoom=False)
        _safe(plot_phasefold, "L", lc, det, fit, zoom=True)
        _safe(plot_oddeven, "O", lc, det)
        _safe(plot_periodogram, "P", det)
        _safe(plot_secondary, "S", lc, det)
        _safe(plot_river, "R", lc, det)

    try:
        _plot_textsummary(result, axd["X"])
    except Exception as exc:  # pragma: no cover
        _na_panel(axd["X"], "Summary", f"error: {type(exc).__name__}")
    try:
        _plot_probbar(classification, axd["B"])
    except Exception as exc:  # pragma: no cover
        _na_panel(axd["B"], "Class probabilities", f"error: {type(exc).__name__}")

    if save_path is not None:
        # NOTE: do NOT pass ``bbox_inches='tight'`` here -- it re-runs the
        # constrained-layout solver at save time which can collapse a panel that
        # carries a colorbar (the river plot). Constrained layout already trims
        # the padding, so a plain savefig yields a clean, full-size A4 PNG.
        fig.savefig(save_path, dpi=dpi)
    return fig
