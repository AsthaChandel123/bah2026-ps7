"""Classification subsystem for ``exopipe``.

This package turns a detected periodic signal into one of four astrophysical
classes -- ``{'transit', 'eclipsing_binary', 'blend', 'other'}`` -- together
with a **calibrated** confidence and a human-readable rationale. It implements
the three-stream ensemble described in ``ARCHITECTURE.md`` Section 7:

1. :func:`~exopipe.classify.rules.classify_rules` -- a transparent, always
   available physics decision policy over the :class:`~exopipe.types.VettingReport`
   flags/metrics. This is the graceful-degradation floor: with no trained model
   and no network the pipeline still classifies.
2. :class:`~exopipe.classify.ml.MLClassifier` -- a calibrated gradient-boosted
   tabular classifier (XGBoost / LightGBM / scikit-learn fallback) over the
   engineered feature vector from :func:`exopipe.features.extract_features`.
3. :class:`~exopipe.classify.cnn.CNNClassifier` -- an optional AstroNet-style
   multi-branch 1-D CNN over the phase-folded views from
   :func:`exopipe.features.build_views`. Present only when ``torch`` is
   installed.

:func:`~exopipe.classify.ensemble.classify` combines whatever streams are
available by a weighted average of their probability vectors and then applies a
high-confidence physics **veto** (a confirmed deep secondary forces
``eclipsing_binary``; confirmed strong contamination/centroid offset forces
``blend``).

The canonical class order is :data:`CLASSES`; every probability dictionary in
this package is a mapping over exactly those four labels that sums to ``1.0``,
and ``confidence == probabilities[label]``.
"""

from __future__ import annotations

from .cnn import CNNClassifier
from .ensemble import CLASSES, classify, load_models
from .ml import MLClassifier
from .rules import classify_rules

__all__ = [
    "CLASSES",
    "classify",
    "classify_rules",
    "MLClassifier",
    "CNNClassifier",
    "load_models",
]
