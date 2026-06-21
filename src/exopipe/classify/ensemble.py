"""Final 4-class decision: combine streams, then apply the physics veto.

:func:`classify` implements ``ARCHITECTURE.md`` Section 7. It always computes the
rule-based stream, optionally folds in a trained :class:`MLClassifier` (primary)
and a :class:`CNNClassifier` (when ``torch`` and a light curve are available),
combines their probability vectors by a **weighted average** (rules as a weak
prior, ML as the primary, CNN as a supporting view-based vote), and finally
applies a high-confidence **vetting veto**: a confirmed deep secondary forces
``eclipsing_binary`` and confirmed strong contamination/centroid offset forces
``blend`` -- but only when the underlying flag is decisive. The result is
renormalised, the arg-max becomes the label, ``confidence`` is that class's
probability, ``method='ensemble'``, and the rationales of every contributing
stream are merged.

``models`` contract
-------------------
:func:`classify` accepts an optional ``models`` mapping with these keys::

    {
        "ml":  MLClassifier      # trained tabular model (optional)
        "cnn": CNNClassifier     # trained view-based model (optional)
        "weights": {             # optional per-stream blend weights (optional)
            "rules": float, "ml": float, "cnn": float
        }
    }

Any key may be absent. :func:`load_models` builds such a dict from a model
directory (loading ``exopipe_clf.joblib`` into ``models['ml']`` when present).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np

from ..types import Classification, DetectionResult, VettingReport
from .rules import CLASSES, _first, _flag, classify_rules

__all__ = ["classify", "load_models", "CLASSES"]

logger = logging.getLogger(__name__)

# Default blend weights for the available streams (renormalised over whatever is
# present). ML is primary; rules a weak prior; CNN a supporting vote.
_DEFAULT_WEIGHTS = {"rules": 0.25, "ml": 1.0, "cnn": 0.75}

# Probability floor used when forcing a class via the physics veto.
_VETO_FLOOR = 0.90


def _index(label: str) -> int:
    return CLASSES.index(label)


def _normalise(vec: np.ndarray) -> np.ndarray:
    """Return a non-negative vector that sums to 1 (uniform if degenerate)."""
    vec = np.clip(np.asarray(vec, dtype=np.float64), 0.0, None)
    total = vec.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(CLASSES), 1.0 / len(CLASSES))
    return vec / total


def _probs_to_vec(probabilities: Mapping[str, float]) -> np.ndarray:
    """Pull a :class:`Classification`-style dict into a canonical-order vector."""
    return np.asarray([float(probabilities.get(cls, 0.0)) for cls in CLASSES], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def classify(
    det: DetectionResult | None,
    vetting: VettingReport | None,
    features: Mapping[str, Any] | None,
    lc: Any = None,
    models: Mapping[str, Any] | None = None,
) -> Classification:
    """Combine all available classifier streams + physics veto -> Classification.

    Parameters
    ----------
    det:
        Detection result (passed to the rule stream for SNR context).
    vetting:
        Vetting report; drives both the rule stream and the final veto.
    features:
        Engineered feature dict for the ML stream and rules.
    lc:
        Optional :class:`~exopipe.types.LightCurve`. Required to build CNN views
        (only used when a CNN model is supplied and ``torch`` is present).
    models:
        Optional mapping with keys ``'ml'``, ``'cnn'``, ``'weights'`` (see the
        module docstring). ``None`` => rules only.

    Returns
    -------
    Classification
        ``method='ensemble'`` with calibrated, vetoed 4-class ``probabilities``
        (sum 1), ``confidence == probabilities[label]``, and merged rationales.
    """
    models = models or {}
    weights_cfg = {**_DEFAULT_WEIGHTS, **(models.get("weights") or {})}

    stream_vecs: list[np.ndarray] = []
    stream_weights: list[float] = []
    rationale: list[str] = []

    # -- Stream 1: rules (always) --------------------------------------- #
    rules_cls = classify_rules(det, vetting, features)
    stream_vecs.append(_probs_to_vec(rules_cls.probabilities))
    stream_weights.append(float(weights_cfg.get("rules", 0.25)))
    rationale.extend(f"[rules] {r}" for r in rules_cls.rationale)

    methods_used = ["rules"]

    # -- Stream 2: tabular ML (if a trained model is provided) ---------- #
    ml_model = models.get("ml")
    if ml_model is not None and features is not None:
        try:
            ml_cls = ml_model.predict(features)
            stream_vecs.append(_probs_to_vec(ml_cls.probabilities))
            stream_weights.append(float(weights_cfg.get("ml", 1.0)))
            rationale.extend(f"[ml] {r}" for r in ml_cls.rationale)
            methods_used.append("ml")
        except Exception as exc:  # pragma: no cover - robustness net
            logger.warning("ensemble: ML stream failed (%s); skipping.", exc)
            rationale.append(f"[ml] skipped (error: {exc})")

    # -- Stream 3: CNN (if a model + torch + light curve are available) - #
    cnn_model = models.get("cnn")
    if cnn_model is not None and getattr(cnn_model, "available", False) and lc is not None:
        try:
            views = _build_views(lc, det)
            if views is not None:
                cnn_cls = cnn_model.predict(views)
                stream_vecs.append(_probs_to_vec(cnn_cls.probabilities))
                stream_weights.append(float(weights_cfg.get("cnn", 0.75)))
                rationale.extend(f"[cnn] {r}" for r in cnn_cls.rationale)
                methods_used.append("cnn")
        except Exception as exc:  # pragma: no cover - robustness net
            logger.warning("ensemble: CNN stream failed (%s); skipping.", exc)
            rationale.append(f"[cnn] skipped (error: {exc})")

    # -- Weighted average of available streams -------------------------- #
    weights = np.asarray(stream_weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    combined = np.zeros(len(CLASSES), dtype=np.float64)
    for vec, w in zip(stream_vecs, weights):
        combined += w * _normalise(vec)
    combined = _normalise(combined)

    rationale.insert(0, f"combined streams: {', '.join(methods_used)}")

    # -- Physics veto (applied last, only when decisive) ---------------- #
    combined, veto_notes = _apply_veto(combined, det, vetting, features)
    rationale.extend(veto_notes)

    combined = _normalise(combined)
    label_idx = int(np.argmax(combined))
    label = CLASSES[label_idx]
    prob_map = {cls: float(p) for cls, p in zip(CLASSES, combined)}

    return Classification(
        label=label,
        confidence=float(prob_map[label]),
        probabilities=prob_map,
        method="ensemble",
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Physics veto
# --------------------------------------------------------------------------- #
def _apply_veto(
    probs: np.ndarray,
    det: DetectionResult | None,
    vetting: VettingReport | None,
    features: Mapping[str, Any] | None,
) -> tuple[np.ndarray, list[str]]:
    """Override the combined probabilities when a decisive physics test fires.

    Only *high-confidence* flags trigger an override (we do not want a marginal
    metric to overrule a calibrated model). Blend is checked before EB so that a
    confirmed off-target source wins even if a secondary is also present
    (a blended NEB *is* a blend, reported as such).

    Returns the (possibly forced) probability vector and a list of rationale
    strings describing any veto that fired.
    """
    metrics: Mapping[str, Any] = vetting.metrics if vetting is not None else {}
    flags: Mapping[str, Any] = vetting.flags if vetting is not None else {}
    feats: Mapping[str, Any] = features or {}
    notes: list[str] = []

    # -- Decisive BLEND: explicit confirmed flag, or very low crowdsap +
    #    a clear centroid offset. ------------------------------------------ #
    blend_confirmed_flag = _flag(
        flags, keys=("is_blend", "blend_confirmed", "centroid_shift", "blend_contamination")
    )
    crowdsap = _first(metrics, feats, keys=("crowdsap", "CROWDSAP"), default=1.0)
    centroid_offset = _first(
        metrics, feats,
        keys=("centroid_offset", "centroid_offset_arcsec", "centroid_offset_sigma",
              "offset_arcsec"),
        default=0.0,
    )
    strong_blend = blend_confirmed_flag or (
        math.isfinite(crowdsap) and crowdsap < 0.5
        and math.isfinite(centroid_offset) and centroid_offset >= 2.0
    )
    if strong_blend:
        notes.append(
            "VETO: confirmed contamination/centroid offset => force blend"
        )
        return _force_class(probs, "blend"), notes

    # -- Decisive EB: confirmed deep secondary, huge radius, or huge
    #    odd-even difference. ---------------------------------------------- #
    eb_secondary_flag = _flag(
        flags, keys=("eb_secondary", "secondary_detected", "deep_secondary")
    )
    secondary_snr = _first(
        metrics, feats,
        keys=("secondary_snr", "secondary_significance", "secondary_depth_sigma", "MS4"),
        default=0.0,
    )
    secondary_depth = _first(
        metrics, feats, keys=("secondary_depth", "secondary_depth_ppm"), default=0.0
    )
    implied_rp_rjup = _first(
        metrics, feats, keys=("implied_rp_rjup", "rp_rjup"), default=float("nan")
    )
    odd_even_sigma = _first(
        metrics, feats, keys=("odd_even_depth_sigma", "odd_even_sigma"), default=0.0
    )

    radius_flag = _flag(flags, keys=("implied_radius_too_big",))
    deep_secondary = eb_secondary_flag or (
        math.isfinite(secondary_snr) and secondary_snr >= 5.0
    ) or (math.isfinite(secondary_depth) and secondary_depth >= 5e-3)
    huge_radius = radius_flag or (
        math.isfinite(implied_rp_rjup) and implied_rp_rjup > 2.5
    )
    huge_oe = math.isfinite(odd_even_sigma) and odd_even_sigma >= 5.0

    if deep_secondary or huge_radius or huge_oe:
        reason = (
            "deep secondary eclipse" if deep_secondary
            else "implied radius > 2.5 R_Jup" if huge_radius
            else "odd-even depth diff >= 5 sigma"
        )
        notes.append(f"VETO: {reason} => force eclipsing_binary")
        return _force_class(probs, "eclipsing_binary"), notes

    return probs, notes


def _force_class(probs: np.ndarray, label: str) -> np.ndarray:
    """Force ``label`` to dominate while preserving the relative tail mass.

    Sets the forced class to at least :data:`_VETO_FLOOR` and rescales the
    remaining probability across the other classes (keeping their relative
    proportions), so the override is decisive but still reports a sensible runner
    -up distribution.
    """
    out = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None).copy()
    idx = _index(label)
    out = _normalise(out)
    if out[idx] >= _VETO_FLOOR:
        return out
    remaining = 1.0 - _VETO_FLOOR
    new = out.copy()
    mask = np.ones(len(out), dtype=bool)
    mask[idx] = False
    others_sum = out[mask].sum()
    new[idx] = _VETO_FLOOR
    if others_sum > 0:
        # Preserve the relative proportions of the non-forced classes.
        new[mask] = out[mask] * (remaining / others_sum)
    else:
        # No tail mass to distribute: spread the remainder uniformly.
        new[mask] = remaining / (len(out) - 1)
    return _normalise(new)


# --------------------------------------------------------------------------- #
# CNN view construction (lazy; degrades to None)
# --------------------------------------------------------------------------- #
def _build_views(lc: Any, det: DetectionResult | None) -> Mapping[str, Any] | None:
    """Build phase-folded views for the CNN, or ``None`` if not possible.

    Prefers :func:`exopipe.features.build_views` (owned by another engineer);
    returns ``None`` on any failure so the ensemble simply drops the CNN stream.
    """
    try:
        from ..features import build_views  # type: ignore
    except Exception:
        return None
    try:
        return build_views(lc, det)
    except Exception as exc:  # pragma: no cover - depends on integration state
        logger.debug("ensemble: build_views failed (%s); skipping CNN.", exc)
        return None


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_models(model_dir: str = "models") -> dict[str, Any]:
    """Load trained models from ``model_dir`` into the ``models`` dict.

    Currently loads the tabular classifier from ``exopipe_clf.joblib`` (if it
    exists) into ``models['ml']``. Returns an empty dict when nothing is found
    or scikit-learn/joblib is unavailable, so the ensemble cleanly degrades to
    the rules-only path.

    Parameters
    ----------
    model_dir:
        Directory to search for ``exopipe_clf.joblib``.

    Returns
    -------
    dict
        Suitable to pass straight to :func:`classify` as ``models=...``.
    """
    import os

    models: dict[str, Any] = {}
    ml_path = os.path.join(model_dir, "exopipe_clf.joblib")
    if os.path.exists(ml_path):
        try:
            from .ml import MLClassifier

            models["ml"] = MLClassifier.load(ml_path)
            logger.info("load_models: loaded ML classifier from %s", ml_path)
        except Exception as exc:
            logger.warning("load_models: failed to load %s (%s)", ml_path, exc)

    # Optional CNN weights (only if torch is present).
    cnn_path = os.path.join(model_dir, "exopipe_cnn.pt")
    if os.path.exists(cnn_path):
        try:
            from .cnn import CNNClassifier

            cnn = CNNClassifier()
            if cnn.available:
                models["cnn"] = cnn.load(cnn_path)
                logger.info("load_models: loaded CNN from %s", cnn_path)
        except Exception as exc:
            logger.warning("load_models: failed to load %s (%s)", cnn_path, exc)

    return models
