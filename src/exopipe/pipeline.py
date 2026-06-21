"""Per-light-curve orchestration: ``process_lightcurve``.

This is the single-light-curve entry point named in the build contract
(``ARCHITECTURE.md`` §3, §6.2). It runs the linear stage sequence

    detrend → search → vet → featurize → fit → classify → assemble

over one :class:`~exopipe.types.LightCurve` and returns a fully-populated
:class:`~exopipe.types.CandidateResult`.

Robustness contract
-------------------
The pipeline must survive a single bad light curve so that a 20–30k-LC sector run
never crashes mid-way (``ARCHITECTURE`` §2 "graceful degradation", §14). To that
end **every stage is wrapped in its own ``try/except``**: on failure the stage is
logged and a *safe empty* dataclass is substituted, so the chain always reaches
the end and returns a ``CandidateResult``. The per-stage modules owned by B1–B4
are imported **lazily inside the helpers** so that importing this module never
executes an optional dependency and never fails just because, say, ``batman`` or
``transitleastsquares`` is missing.

Config toggles honoured (via :class:`~exopipe.config.Config`):

* ``detrend.method == "none"`` skips detrending (search runs on the raw LC).
* ``classify`` label gates the fit: the expensive transit fit runs only when the
  candidate is classified ``transit`` (or is a high-SNR borderline), matching the
  "fit gated on classification" control-flow note in §3. Because the ML
  classifier needs the feature vector (which can use the fit), we classify with a
  cheap *pre-pass*, fit if warranted, then finalise the classification with the
  fit-aware features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Config, default_config
from .types import (
    CandidateResult,
    Classification,
    DetectionResult,
    LightCurve,
    TransitFit,
    VettingReport,
)
from .utils import get_logger

__all__ = ["process_lightcurve"]

_LOG = get_logger("exopipe.pipeline")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def process_lightcurve(
    lc: LightCurve,
    config: Config | None = None,
    models: Any | None = None,
) -> CandidateResult:
    """Run the full detection→classification pipeline for one light curve.

    Parameters
    ----------
    lc:
        Input :class:`LightCurve` (raw or pre-cleaned). Never mutated.
    config:
        Pipeline :class:`Config`; ``None`` uses :func:`default_config`.
    models:
        Optional pre-loaded classifier/CNN bundle passed straight through to the
        classification ensemble (e.g. from
        ``exopipe.classify.ensemble.load_models``). ``None`` ⇒ the ensemble falls
        back to its rules/heuristic floor.

    Returns
    -------
    CandidateResult
        Always returned, even if individual stages fail (failed stages contribute
        empty-but-valid dataclasses and an annotation in
        ``features['stage_errors']``).

    Notes
    -----
    The function is deterministic given ``(lc, config)`` and side-effect-free, so
    it is safe to fan out across processes in :func:`exopipe.driver.run_batch`.
    """
    cfg = config if config is not None else default_config()
    if lc is None:
        raise ValueError("process_lightcurve requires a LightCurve, got None")

    stage_errors: dict[str, str] = {}
    tic = _tic_label(lc)

    # -- 1. detrend --------------------------------------------------------- #
    detrended = _run_detrend(lc, cfg, stage_errors, tic)

    # -- 2. search ---------------------------------------------------------- #
    detection = _run_search(detrended, cfg, stage_errors, tic)

    # -- 3. vet ------------------------------------------------------------- #
    vetting = _run_vet(detrended, detection, None, cfg, stage_errors, tic)

    # -- 4. features (fit-agnostic pre-pass) -------------------------------- #
    features = _run_features(detrended, detection, vetting, None, cfg, stage_errors, tic)

    # -- 5. pre-classification (cheap, decides whether to fit) -------------- #
    pre_class = _run_classify(detection, vetting, features, detrended, models, cfg, stage_errors, tic)

    # -- 6. fit (gated on the pre-classification / SNR) --------------------- #
    if _should_fit(pre_class, detection, vetting, cfg):
        fit = _run_fit(detrended, detection, cfg, stage_errors, tic)
        # Recompute features now that a fit is available (richer ML inputs), then
        # finalise the classification with the fit-aware feature vector.
        features = _run_features(detrended, detection, vetting, fit, cfg, stage_errors, tic)
        classification = _run_classify(
            detection, vetting, features, detrended, models, cfg, stage_errors, tic
        )
    else:
        fit = _empty_fit()
        classification = pre_class

    if stage_errors:
        features = dict(features)
        features["stage_errors"] = stage_errors

    return CandidateResult(
        lightcurve=lc,
        detection=detection,
        vetting=vetting,
        fit=fit,
        classification=classification,
        features=features,
    )


# --------------------------------------------------------------------------- #
# Stage runners (each lazily imports its module and never raises)
# --------------------------------------------------------------------------- #
def _run_detrend(
    lc: LightCurve, cfg: Config, errors: dict, tic: str
) -> LightCurve:
    """Stage 1 — flatten the baseline. Returns the input LC on failure/skip."""
    if getattr(cfg.detrend, "method", "biweight") == "none":
        return lc
    try:
        from . import detrend as detrend_mod

        out = detrend_mod.detrend(
            lc,
            method=cfg.detrend.method,
            window_length=cfg.detrend.window_length,
        )
        if isinstance(out, LightCurve) and len(out) > 0:
            return out
        _LOG.debug("[%s] detrend returned empty/invalid; using raw LC.", tic)
        return lc
    except Exception as exc:
        _record(errors, "detrend", exc)
        _LOG.warning("[%s] detrend failed (%s); continuing with raw LC.", tic, exc)
        return lc


def _run_search(
    lc: LightCurve, cfg: Config, errors: dict, tic: str
) -> DetectionResult:
    """Stage 2 — periodic-transit search (two-stage BLS→TLS)."""
    try:
        from . import search as search_mod

        det = search_mod.search_two_stage(
            lc,
            period_min=cfg.search.period_min,
            period_max=cfg.search.period_max,
        )
        if isinstance(det, DetectionResult):
            return det
        _LOG.debug("[%s] search returned non-DetectionResult; using empty.", tic)
    except Exception as exc:
        _record(errors, "search", exc)
        _LOG.warning("[%s] search failed (%s); using empty detection.", tic, exc)
    return _empty_detection()


def _run_vet(
    lc: LightCurve,
    det: DetectionResult,
    fit: TransitFit | None,
    cfg: Config,
    errors: dict,
    tic: str,
) -> VettingReport:
    """Stage 3 — physics vetting tests."""
    try:
        from . import vetting as vetting_mod

        report = vetting_mod.vet(lc, det, fit=fit)
        if isinstance(report, VettingReport):
            return report
    except Exception as exc:
        _record(errors, "vetting", exc)
        _LOG.warning("[%s] vetting failed (%s); using empty report.", tic, exc)
    return VettingReport(metrics={}, flags={})


def _run_features(
    lc: LightCurve,
    det: DetectionResult,
    vetting: VettingReport,
    fit: TransitFit | None,
    cfg: Config,
    errors: dict,
    tic: str,
) -> dict:
    """Stage 4 — engineered feature vector for the classifier."""
    try:
        from . import features as features_mod

        feats = features_mod.extract_features(lc, det, vetting, fit)
        if isinstance(feats, dict):
            return feats
    except Exception as exc:
        _record(errors, "features", exc)
        _LOG.warning("[%s] feature extraction failed (%s); using empty dict.", tic, exc)
    return {}


def _run_fit(
    lc: LightCurve, det: DetectionResult, cfg: Config, errors: dict, tic: str
) -> TransitFit:
    """Stage 5 — transit model fit with uncertainties (gated upstream)."""
    try:
        from . import fit as fit_mod

        result = fit_mod.fit_transit(
            lc,
            det,
            sampler=cfg.fit.sampler,
        )
        if isinstance(result, TransitFit):
            return result
    except Exception as exc:
        _record(errors, "fit", exc)
        _LOG.warning("[%s] transit fit failed (%s); using empty fit.", tic, exc)
    return _empty_fit()


def _run_classify(
    det: DetectionResult,
    vetting: VettingReport,
    features: dict,
    lc: LightCurve,
    models: Any | None,
    cfg: Config,
    errors: dict,
    tic: str,
) -> Classification:
    """Stage 6 — final 4-class ensemble decision (rules + ML + veto)."""
    try:
        from .classify import ensemble as ensemble_mod

        result = ensemble_mod.classify(det, vetting, features, lc=lc, models=models)
        if isinstance(result, Classification):
            return result
    except Exception as exc:
        _record(errors, "classify", exc)
        _LOG.warning("[%s] classification failed (%s); using fallback.", tic, exc)
    return _fallback_classification(det, vetting)


# --------------------------------------------------------------------------- #
# Fit gating
# --------------------------------------------------------------------------- #
def _should_fit(
    classification: Classification,
    det: DetectionResult,
    vetting: VettingReport,
    cfg: Config,
) -> bool:
    """Decide whether to run the (expensive) transit fit for this candidate.

    Per the §3 control-flow note, full Bayesian sampling is gated on
    classification to keep the sector budget low: fit when the candidate looks
    like a ``transit`` **or** is a high-SNR borderline worth measuring. We also
    require a usable period so the fit has something to fold on.
    """
    period = getattr(det, "period", np.nan)
    if not np.isfinite(period) or period <= 0:
        return False

    label = getattr(classification, "label", "other")
    if label == "transit":
        return True

    # High-SNR borderline: measure it even if the cheap pass said EB/blend/other.
    snr = _finite(getattr(det, "snr", np.nan))
    min_snr = float(getattr(cfg.search, "min_snr", 7.0))
    p_transit = 0.0
    try:
        p_transit = float(classification.probabilities.get("transit", 0.0))
    except Exception:  # pragma: no cover - probabilities may be missing
        p_transit = 0.0
    return snr >= max(min_snr, 1.0) and p_transit >= 0.25


# --------------------------------------------------------------------------- #
# Safe empty dataclasses + fallbacks
# --------------------------------------------------------------------------- #
def _empty_detection() -> DetectionResult:
    """A valid, signal-free :class:`DetectionResult` placeholder."""
    return DetectionResult(
        period=np.nan,
        t0=np.nan,
        duration=np.nan,
        depth=np.nan,
        sde=np.nan,
        snr=np.nan,
        method="none",
        periods=np.empty(0, dtype=np.float64),
        power=np.empty(0, dtype=np.float64),
        harmonics=[],
        extra={"empty": True},
    )


def _empty_fit() -> TransitFit:
    """A valid, empty :class:`TransitFit` placeholder (no model run)."""
    return TransitFit(
        params={},
        model_time=None,
        model_flux=None,
        bic_transit=np.nan,
        bic_flat=np.nan,
        delta_bic=np.nan,
        snr=np.nan,
        method="none",
        samples=None,
        extra={"empty": True},
    )


def _fallback_classification(
    det: DetectionResult, vetting: VettingReport
) -> Classification:
    """A minimal always-available classification when the ensemble is absent.

    Uses only the vetting flags / detection SNR to make a defensible call so the
    pipeline still classifies with zero ML deps (the graceful-degradation floor,
    ``ARCHITECTURE`` §7.1). Probabilities are coarse but valid (sum to 1).
    """
    classes = ("transit", "eclipsing_binary", "blend", "other")
    flags = getattr(vetting, "flags", {}) or {}

    def flagged(*names: str) -> bool:
        return any(bool(flags.get(name, False)) for name in names)

    label = "other"
    rationale: list[str] = ["fallback classifier (ensemble unavailable)"]

    if flagged("is_eb", "secondary_detected", "odd_even_mismatch", "eb_secondary", "eb_odd_even"):
        label = "eclipsing_binary"
        rationale.append("EB flag set by vetting")
    elif flagged("is_blend", "centroid_shift", "blend_contamination"):
        label = "blend"
        rationale.append("blend/centroid flag set by vetting")
    else:
        snr = _finite(getattr(det, "snr", np.nan))
        sde = _finite(getattr(det, "sde", np.nan))
        if (snr >= 7.0 or sde >= 7.0) and np.isfinite(getattr(det, "period", np.nan)):
            label = "transit"
            rationale.append(f"significant detection (SNR={snr:.1f}, SDE={sde:.1f})")
        else:
            rationale.append("no significant transit-like signal")

    probabilities = {name: (0.7 if name == label else 0.1) for name in classes}
    return Classification(
        label=label,
        confidence=float(probabilities[label]),
        probabilities=probabilities,
        method="fallback",
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _record(errors: dict, stage: str, exc: Exception) -> None:
    errors[stage] = f"{type(exc).__name__}: {exc}"


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def _tic_label(lc: LightCurve) -> str:
    meta = getattr(lc, "meta", {}) or {}
    tic = meta.get("tic_id")
    return f"TIC {tic}" if tic is not None else "lc"
