"""Tests for ``exopipe.classify`` -- rules, ML, and the ensemble.

* ``classify_rules`` returns the expected label on crafted clear-cut inputs
  (deep secondary -> EB, heavy contamination -> blend, clean strong dip ->
  transit), and a normalised 4-class probability vector.
* A freshly-trained :class:`MLClassifier` on a small *separable* feature table
  predicts a valid 4-class :class:`Classification` (sklearn-only path).
* ``ensemble.classify(models=None)`` degrades cleanly to the rules+veto path and
  still returns a normalised :class:`Classification`.
"""

from __future__ import annotations

import numpy as np
import pytest

from exopipe.classify.ensemble import classify
from exopipe.classify.rules import classify_rules
from exopipe.types import Classification, DetectionResult, VettingReport

_CLASSES = ("transit", "eclipsing_binary", "blend", "other")


def _strong_detection() -> DetectionResult:
    return DetectionResult(
        period=3.0, t0=0.1, duration=0.1, depth=0.05, sde=15.0, snr=50.0, method="bls"
    )


def _assert_valid_classification(c: Classification):
    assert isinstance(c, Classification)
    assert c.label in _CLASSES
    assert set(c.probabilities) == set(_CLASSES)
    assert abs(sum(c.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= c.confidence <= 1.0


def test_rules_flags_eclipsing_binary_on_deep_secondary():
    vet = VettingReport(
        metrics={
            "secondary_snr": 12.0,
            "secondary_to_primary": 0.5,
            "primary_significance": 20.0,
        },
        flags={"eb_secondary": True},
    )
    c = classify_rules(_strong_detection(), vet, {})
    _assert_valid_classification(c)
    assert c.label == "eclipsing_binary"


def test_rules_flags_blend_on_contamination():
    vet = VettingReport(
        metrics={"crowdsap": 0.4},
        flags={"blend_contamination": True},
    )
    c = classify_rules(_strong_detection(), vet, {})
    _assert_valid_classification(c)
    assert c.label == "blend"


def test_rules_calls_clean_strong_dip_a_transit():
    vet = VettingReport(metrics={"transit_snr": 30.0}, flags={})
    c = classify_rules(_strong_detection(), vet, {"snr": 30.0})
    _assert_valid_classification(c)
    assert c.label == "transit"


def test_ml_classifier_predicts_valid_class_on_separable_data():
    pytest.importorskip("sklearn")
    from exopipe.classify.ml import MLClassifier

    rng = np.random.default_rng(0)
    centers = {"transit": 0.0, "eclipsing_binary": 3.0, "blend": 6.0, "other": 9.0}
    X: list[dict] = []
    y: list[str] = []
    for label, center in centers.items():
        for _ in range(25):
            X.append(
                {
                    "f0": float(center + rng.normal(0.0, 0.3)),
                    "f1": float(rng.normal(0.0, 0.3)),
                    "snr": float(20.0 + rng.normal(0.0, 1.0)),
                }
            )
            y.append(label)

    clf = MLClassifier()
    clf.fit(X, y)

    # a clearly transit-like point (centroid 0.0) should be classified transit.
    pred = clf.predict({"f0": 0.0, "f1": 0.0, "snr": 20.0})
    _assert_valid_classification(pred)
    assert pred.method == "ml"
    assert pred.label == "transit"


def test_ensemble_classify_without_models_uses_rules_path():
    vet = VettingReport(
        metrics={
            "secondary_snr": 12.0,
            "secondary_to_primary": 0.5,
            "primary_significance": 20.0,
        },
        flags={"eb_secondary": True},
    )
    c = classify(_strong_detection(), vet, {}, models=None)
    _assert_valid_classification(c)
    assert c.method == "ensemble"
    # the decisive secondary still forces the EB call through the physics veto.
    assert c.label == "eclipsing_binary"
