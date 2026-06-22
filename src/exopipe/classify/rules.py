"""Transparent rule-based classifier (the graceful-degradation floor).

:func:`classify_rules` encodes the decisive astrophysical thresholds that
astronomers use to separate planet / eclipsing-binary / blend / other directly
from the :class:`~exopipe.types.VettingReport` flags and metrics (plus the
engineered feature dict). It needs **no training data** and never imports a
heavy dependency, so it is always available -- the pipeline can classify even
with no ML model and no network.

Decision policy (first decisive trigger wins; see :func:`classify_rules`)
-------------------------------------------------------------------------
1. **Strong blend contamination** -- low ``crowdsap`` and/or a significant
   centroid offset -> ``blend``.
2. **Eclipsing-binary signatures** -- a significant deep secondary eclipse, an
   implausibly large implied radius (``Rp > ~2 R_Jup``), a strong odd--even
   depth difference, or a clearly V-shaped (grazing) event -> ``eclipsing_binary``.
3. **Non-transit / systematics** -- a dominant out-of-transit sinusoid (SWEET),
   a very low transit SNR, or a wildly inconsistent stellar density ->
   ``other``.
4. Otherwise -> ``transit``.

Rather than emit a hard one-hot vector, the policy turns each diagnostic's
*margin past its threshold* into a soft per-class score via a logistic squash,
so the reported ``probabilities`` (and hence ``confidence``) degrade smoothly
near a boundary instead of flipping discontinuously. The ``rationale`` lists the
metrics that actually fired.

Robustness to upstream naming
-----------------------------
The vetting module (owned by another engineer) may use slightly different
metric/flag key names than this module expects. Every lookup therefore consults
a list of aliases and tolerates missing keys, so the rules never raise on a
partially-populated report.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from ..types import Classification, DetectionResult, VettingReport

__all__ = ["classify_rules", "CLASSES", "softmax", "logistic"]

# Canonical class order. Defined here (the dependency-free leaf module) and
# re-exported by :mod:`exopipe.classify.ensemble` and the package ``__init__``.
CLASSES: list[str] = ["transit", "eclipsing_binary", "blend", "other"]


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def logistic(x: float, midpoint: float = 0.0, scale: float = 1.0) -> float:
    """Numerically-stable logistic squash ``1 / (1 + exp(-(x-mid)/scale))``.

    Returns a value in ``(0, 1)``; ``scale`` controls how sharply the response
    saturates around ``midpoint``. Non-finite inputs map to ``0.5`` (maximum
    ignorance) so a missing diagnostic never dominates.
    """
    if not math.isfinite(x):
        return 0.5
    if scale <= 0:
        scale = 1e-6
    z = (float(x) - float(midpoint)) / float(scale)
    # Guard against overflow in exp for large |z|.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert raw class scores to a normalised probability vector.

    Uses the standard max-subtraction trick for stability. A non-positive
    ``temperature`` is clamped to a tiny positive number. If every score is
    non-finite the result is uniform.
    """
    if temperature <= 0:
        temperature = 1e-6
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        n = len(scores)
        return [1.0 / n] * n if n else []
    m = max(finite)
    exps = []
    for s in scores:
        if not math.isfinite(s):
            exps.append(0.0)
        else:
            exps.append(math.exp((float(s) - m) / temperature))
    total = sum(exps)
    if total <= 0:
        n = len(scores)
        return [1.0 / n] * n if n else []
    return [e / total for e in exps]


def _first(
    *mappings: Mapping[str, Any] | None,
    keys: Iterable[str],
    default: float = float("nan"),
) -> float:
    """Return the first finite numeric value found under any ``keys`` alias.

    Searches each mapping in order for each key; booleans are accepted and
    coerced to ``0.0``/``1.0`` so a flag stored as a number still works. Missing
    or non-finite values are skipped; ``default`` is returned if nothing hits.
    """
    for mapping in mappings:
        if not mapping:
            continue
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if isinstance(value, bool):
                    return 1.0 if value else 0.0
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fval):
                    return fval
    return default


def _flag(
    *mappings: Mapping[str, Any] | None,
    keys: Iterable[str],
    default: bool = False,
) -> bool:
    """Return the first truthy/falsey boolean flag found under any alias."""
    for mapping in mappings:
        if not mapping:
            continue
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                return bool(value)
    return default


def _make_classification(
    scores: Mapping[str, float],
    rationale: list[str],
    method: str = "rules",
) -> Classification:
    """Build a :class:`Classification` from per-class scores via softmax.

    Guarantees the ``probabilities`` dict covers exactly :data:`CLASSES`, sums
    to ~1, picks ``label`` as the arg-max, and sets ``confidence`` to that
    class's probability.
    """
    raw = [float(scores.get(cls, 0.0)) for cls in CLASSES]
    probs = softmax(raw)
    prob_map = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    # Renormalise defensively against floating-point drift.
    total = sum(prob_map.values())
    if total > 0:
        prob_map = {k: v / total for k, v in prob_map.items()}
    label = max(prob_map, key=prob_map.get)
    return Classification(
        label=label,
        confidence=float(prob_map[label]),
        probabilities=prob_map,
        method=method,
        rationale=list(rationale),
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def classify_rules(
    det: DetectionResult | None,
    vetting: VettingReport | None,
    features: Mapping[str, Any] | None = None,
) -> Classification:
    """Classify a candidate from vetting diagnostics with a transparent policy.

    Parameters
    ----------
    det:
        The :class:`~exopipe.types.DetectionResult` (used for SNR / SDE context).
        May be ``None``.
    vetting:
        The :class:`~exopipe.types.VettingReport`; its ``metrics`` and ``flags``
        drive the decision. May be ``None`` (everything falls back to neutral).
    features:
        Optional engineered feature dict (:func:`exopipe.features.extract_features`).
        Consulted as an additional source for any metric the report omits.

    Returns
    -------
    Classification
        ``method='rules'`` with soft 4-class ``probabilities`` (sum 1),
        ``confidence == probabilities[label]``, and a ``rationale`` citing the
        triggering metrics.

    Notes
    -----
    The four class scores start from a mild prior that favours ``transit`` (the
    desired default when nothing fires) and are then *boosted* by the logistic
    margin of any diagnostic that crosses its threshold. The arg-max therefore
    only leaves ``transit`` when a diagnostic provides positive evidence for an
    alternative, which keeps the policy conservative and explainable.
    """
    metrics: Mapping[str, Any] = vetting.metrics if vetting is not None else {}
    flags: Mapping[str, Any] = vetting.flags if vetting is not None else {}
    feats: Mapping[str, Any] = features or {}

    rationale: list[str] = []
    # Base scores: gentle prior toward transit so "no evidence" => transit.
    scores: dict[str, float] = {
        "transit": 1.0,
        "eclipsing_binary": 0.0,
        "blend": 0.0,
        "other": 0.0,
    }

    # ------------------------------------------------------------------ #
    # Gather diagnostics (defensive against alias / missing keys)
    # ------------------------------------------------------------------ #
    crowdsap = _first(
        metrics, feats,
        keys=("crowdsap", "CROWDSAP", "crowd_sap", "blend_crowdsap"),
        default=1.0,
    )
    centroid_offset = _first(
        metrics, feats,
        keys=("centroid_offset", "centroid_offset_arcsec", "centroid_offset_sigma",
              "offset_arcsec", "centroid_shift_sigma"),
        default=0.0,
    )
    secondary_depth = _first(
        metrics, feats,
        keys=("secondary_depth", "secondary_depth_ppm", "depth_secondary",
              "sec_depth"),
        default=0.0,
    )
    secondary_snr = _first(
        metrics, feats,
        keys=("secondary_snr", "secondary_significance", "sec_snr", "MS4",
              "secondary_depth_sigma"),
        default=0.0,
    )
    odd_even_sigma = _first(
        metrics, feats,
        keys=("odd_even_depth_sigma", "odd_even_sigma", "odd_even_depth_diff_sigma",
              "oe_sigma", "odd_even_mismatch_sigma"),
        default=0.0,
    )
    v_shape = _first(
        metrics, feats,
        keys=("v_shape_metric", "v_shape", "vshape_metric", "vshape"),
        default=float("nan"),
    )
    ingress_egress_ratio = _first(
        metrics, feats,
        keys=("ingress_egress_ratio", "ingress_total_ratio", "t12_t14"),
        default=float("nan"),
    )
    implied_rp_rjup = _first(
        metrics, feats,
        keys=("implied_rp_rjup", "implied_rp_rJup", "rp_rjup", "implied_radius_rjup"),
        default=float("nan"),
    )
    stellar_density_ratio = _first(
        metrics, feats,
        keys=("stellar_density_ratio", "rho_star_ratio", "density_ratio",
              "rho_ratio"),
        default=float("nan"),
    )
    sweet_metric = _first(
        metrics, feats,
        keys=("sweet_metric", "sweet", "SWEET", "sweet_snr", "sinusoid_snr"),
        default=0.0,
    )
    transit_snr = _first(
        metrics, feats,
        keys=("transit_snr", "snr", "mes", "depth_snr"),
        default=(det.snr if det is not None else float("nan")),
    )

    # ------------------------------------------------------------------ #
    # 1) BLEND -- strong aperture contamination / off-target source
    # ------------------------------------------------------------------ #
    blend_flag = _flag(
        flags,
        keys=("is_blend", "blend_contamination", "blend", "centroid_shift",
              "blend_detected", "contamination"),
    )
    blend_score = 0.0
    # Low CROWDSAP: target light is heavily diluted (<~0.7 is suspicious,
    # <~0.5 strong). crowdsap in [0,1]; smaller => more contamination.
    if math.isfinite(crowdsap) and crowdsap < 0.9:
        margin = logistic(0.9 - crowdsap, midpoint=0.2, scale=0.12)
        blend_score += 2.6 * margin
        if crowdsap < 0.7:
            rationale.append(
                f"low CROWDSAP {crowdsap:.2f} (<0.70) => aperture contamination"
            )
    # Centroid offset: photocentre moves in-transit toward a neighbour. Units
    # may be arcsec or sigma; either way larger => more off-target.
    if math.isfinite(centroid_offset) and centroid_offset > 0:
        margin = logistic(centroid_offset, midpoint=2.0, scale=1.0)
        blend_score += 2.4 * margin
        if centroid_offset >= 2.0:
            rationale.append(
                f"centroid offset {centroid_offset:.2f} in/out of transit => off-target (blend)"
            )
    if blend_flag:
        blend_score += 2.0
        rationale.append("vetting blend/centroid flag set => blend")
    scores["blend"] += blend_score

    # ------------------------------------------------------------------ #
    # 2) ECLIPSING BINARY -- secondary, big radius, odd/even, V-shape
    # ------------------------------------------------------------------ #
    eb_flag = _flag(
        flags,
        keys=("is_eb", "eb_secondary", "eb_odd_even", "eb_vshape", "secondary_detected",
              "odd_even_mismatch", "is_eclipsing_binary", "implied_radius_too_big"),
    )
    eb_score = 0.0
    # 2a) Deep / significant secondary eclipse.
    sec_is_deep = (math.isfinite(secondary_snr) and secondary_snr >= 3.0) or (
        math.isfinite(secondary_depth) and secondary_depth >= 1e-3
    )
    if sec_is_deep:
        # Weight by whichever evidence is available.
        sig = secondary_snr if math.isfinite(secondary_snr) and secondary_snr > 0 else 5.0
        margin = logistic(sig, midpoint=3.0, scale=2.0)
        eb_score += 3.0 * margin
        if math.isfinite(secondary_depth) and secondary_depth > 0:
            rationale.append(
                f"secondary eclipse depth {secondary_depth * 1e6:.0f} ppm "
                f"(snr {sig:.1f}) at phase ~0.5 => EB"
            )
        else:
            rationale.append(f"significant secondary eclipse (snr {sig:.1f}) => EB")
    # 2b) Implausibly large implied radius (> ~2 R_Jup) => stellar companion.
    if math.isfinite(implied_rp_rjup) and implied_rp_rjup > 2.0:
        margin = logistic(implied_rp_rjup, midpoint=2.0, scale=0.6)
        eb_score += 2.8 * margin
        rationale.append(
            f"implied radius {implied_rp_rjup:.1f} R_Jup (>2) => too big for a planet (EB)"
        )
    # 2c) Strong odd-even depth difference (true period is 2x).
    if math.isfinite(odd_even_sigma) and odd_even_sigma >= 3.0:
        margin = logistic(odd_even_sigma, midpoint=3.0, scale=1.5)
        eb_score += 2.6 * margin
        rationale.append(
            f"odd-even depth difference {odd_even_sigma:.1f}sigma (>=3) => EB at half period"
        )
    # 2d) V-shape / grazing geometry. v_shape_metric = Rp/R* + b; <1.5 is the
    # classic EB-grazing fail. ingress/egress ~ total duration (ratio -> 0.5)
    # also flags V-shape.
    if math.isfinite(v_shape) and v_shape < 1.5:
        margin = logistic(1.5 - v_shape, midpoint=0.3, scale=0.4)
        eb_score += 1.6 * margin
        rationale.append(f"V-shape metric {v_shape:.2f} (<1.5) => grazing/EB")
    elif math.isfinite(ingress_egress_ratio) and ingress_egress_ratio > 0.4:
        margin = logistic(ingress_egress_ratio, midpoint=0.4, scale=0.1)
        eb_score += 1.2 * margin
        rationale.append(
            f"ingress/egress fraction {ingress_egress_ratio:.2f} => V-shaped (EB)"
        )
    if eb_flag:
        eb_score += 1.8
        rationale.append("vetting EB flag set => eclipsing_binary")
    scores["eclipsing_binary"] += eb_score

    # ------------------------------------------------------------------ #
    # 3) OTHER -- variability / systematics / low significance
    # ------------------------------------------------------------------ #
    other_flag = _flag(
        flags,
        keys=("other_variability", "low_snr", "is_variable", "systematics",
              "not_unique", "sweet_fail", "density_inconsistent"),
    )
    other_score = 0.0
    # 3a) Dominant out-of-transit sinusoid (SWEET): the signal is just stellar
    # variability / a sine, not a transit. SWEET >~3 typically fails.
    if math.isfinite(sweet_metric) and sweet_metric >= 3.0:
        margin = logistic(sweet_metric, midpoint=3.0, scale=2.0)
        other_score += 2.6 * margin
        rationale.append(
            f"SWEET sinusoid significance {sweet_metric:.1f} (>=3) => variability (other)"
        )
    # 3b) Very low transit SNR: not a believable event.
    if math.isfinite(transit_snr) and transit_snr < 7.1:
        margin = logistic(7.1 - transit_snr, midpoint=2.0, scale=2.0)
        other_score += 2.2 * margin
        if transit_snr < 5.0:
            rationale.append(
                f"low transit SNR {transit_snr:.1f} (<7.1 MES cut) => marginal (other)"
            )
    # 3c) Stellar-density grossly inconsistent (factor >~5 either way) without a
    # blend/centroid explanation => wrong period / systematics => other.
    if math.isfinite(stellar_density_ratio) and stellar_density_ratio > 0:
        log_ratio = abs(math.log10(stellar_density_ratio))
        if log_ratio > math.log10(5.0):
            margin = logistic(log_ratio, midpoint=math.log10(5.0), scale=0.4)
            other_score += 1.6 * margin
            rationale.append(
                f"stellar density ratio {stellar_density_ratio:.2g} "
                f"(>5x off) => inconsistent (other/wrong-P)"
            )
    if other_flag:
        other_score += 1.6
        rationale.append("vetting variability/low-SNR flag set => other")
    scores["other"] += other_score

    # ------------------------------------------------------------------ #
    # If nothing fired, state that the candidate passes as a transit.
    # ------------------------------------------------------------------ #
    if not rationale:
        rationale.append(
            "no decisive EB/blend/other diagnostic fired => consistent with transit"
        )

    return _make_classification(scores, rationale, method="rules")
