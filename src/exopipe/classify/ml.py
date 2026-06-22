"""Calibrated tabular ML classifier for the four astrophysical classes.

:class:`MLClassifier` wraps a gradient-boosted tree model over the engineered
feature vector (:func:`exopipe.features.extract_features`) and produces a
**probability-calibrated** 4-class :class:`~exopipe.types.Classification`.

Backend selection (all imported lazily, with graceful fallback)
---------------------------------------------------------------
* ``xgboost.XGBClassifier`` (``multi:softprob``) -- preferred primary.
* ``lightgbm.LGBMClassifier`` -- preferred alternative.
* ``sklearn.ensemble.HistGradientBoostingClassifier`` / ``RandomForestClassifier``
  -- always-available scikit-learn fallback.
* **No scikit-learn at all** -> :meth:`MLClassifier.predict` delegates to
  :func:`exopipe.classify.rules.classify_rules` and :meth:`fit` is a logged
  no-op, so the object is still importable and usable.

Calibration uses :class:`sklearn.calibration.CalibratedClassifierCV` with
isotonic regression (sigmoid/Platt fallback for tiny folds) inside a grouped or
stratified cross-validation, which is what makes the reported ``confidence``
trustworthy. Class imbalance is handled with balanced ``sample_weight`` /
``class_weight``. Persistence is via :mod:`joblib`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import Classification
from .rules import CLASSES, classify_rules

__all__ = ["MLClassifier"]

logger = logging.getLogger(__name__)

# Class label <-> integer index (canonical order from :data:`CLASSES`).
_LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}
_INDEX_TO_LABEL = {i: label for i, label in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# Optional-dependency probes (lazy; never raise at import time)
# --------------------------------------------------------------------------- #
def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def _make_base_estimator(model: str, n_classes: int) -> tuple[Any, str]:
    """Instantiate the requested (or best available) base estimator.

    Returns ``(estimator, backend_name)``. ``model='auto'`` tries
    XGBoost -> LightGBM -> sklearn HistGradientBoosting. Anything else selects a
    specific backend by name (falling back to sklearn if it is missing).
    """
    want = (model or "auto").lower()

    def _try_xgboost() -> tuple[Any, str] | None:
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None
        est = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            verbosity=0,
        )
        return est, "xgboost"

    def _try_lightgbm() -> tuple[Any, str] | None:
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            return None
        est = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
        return est, "lightgbm"

    def _try_hgb() -> tuple[Any, str] | None:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except Exception:
            return None
        est = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            class_weight="balanced",
        )
        return est, "sklearn-histgb"

    def _try_rf() -> tuple[Any, str] | None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except Exception:
            return None
        est = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        return est, "sklearn-rf"

    if want in ("xgboost", "xgb"):
        order = [_try_xgboost, _try_hgb, _try_rf]
    elif want in ("lightgbm", "lgbm", "lgb"):
        order = [_try_lightgbm, _try_hgb, _try_rf]
    elif want in ("rf", "random_forest", "randomforest"):
        order = [_try_rf, _try_hgb]
    elif want in ("histgb", "hgb", "sklearn"):
        order = [_try_hgb, _try_rf]
    else:  # auto
        order = [_try_xgboost, _try_lightgbm, _try_hgb, _try_rf]

    for factory in order:
        result = factory()
        if result is not None:
            return result
    raise RuntimeError("no usable ML backend (scikit-learn is required)")


# --------------------------------------------------------------------------- #
# MLClassifier
# --------------------------------------------------------------------------- #
class MLClassifier:
    """Calibrated 4-class gradient-boosted classifier over engineered features.

    Parameters
    ----------
    feature_names:
        Ordered list of feature keys to pull out of each feature dict. If
        ``None`` it is inferred from the first ``fit`` call (sorted union of the
        keys present, or :data:`exopipe.features.FEATURE_NAMES` when training on
        2-D arrays). Stored on the model so ``predict`` vectorises consistently.
    model:
        Backend selector -- ``'auto'`` (default), ``'xgboost'``, ``'lightgbm'``,
        ``'rf'``, or ``'histgb'``.

    Attributes
    ----------
    available:
        ``True`` when scikit-learn is importable (so a real model can be fit).
        When ``False`` the classifier transparently delegates to the rule-based
        policy.
    backend:
        Human-readable name of the selected base estimator (set after ``fit``).
    """

    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        model: str = "auto",
    ) -> None:
        self.feature_names: list[str] | None = (
            list(feature_names) if feature_names is not None else None
        )
        self.model_spec = model
        self.available = _has_sklearn()
        self.backend: str | None = None
        self._clf: Any = None  # fitted (possibly calibrated) estimator
        self._calibrated = False
        self._classes_seen: list[int] = []  # integer class ids present in train

    # ------------------------------------------------------------------ #
    # Vectorisation helpers
    # ------------------------------------------------------------------ #
    def _infer_feature_names(self, X: Any) -> None:
        """Populate ``feature_names`` from the training data if not given."""
        if self.feature_names is not None:
            return
        if isinstance(X, np.ndarray) and X.ndim == 2:
            # Try the canonical feature list; else generic positional names.
            names = _canonical_feature_names(X.shape[1])
            self.feature_names = names
            return
        if isinstance(X, Sequence) and len(X) and isinstance(X[0], Mapping):
            keys: set[str] = set()
            for row in X:
                keys.update(k for k, v in row.items() if _is_number(v))
            self.feature_names = sorted(keys)
            return
        raise ValueError("cannot infer feature_names from the provided X")

    def _to_matrix(self, X: Any) -> np.ndarray:
        """Convert ``X`` (list[dict] | 2-D array) to a float matrix.

        Columns follow ``self.feature_names``; missing keys become ``NaN`` (the
        tree backends handle NaN, and the sklearn fallbacks get NaNs imputed to
        column medians at fit time -- see :meth:`_impute`).
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return X.astype(np.float64, copy=False)
        if isinstance(X, Mapping):
            X = [X]
        assert self.feature_names is not None
        rows = []
        for item in X:
            if isinstance(item, Mapping):
                rows.append(
                    [_safe_float(item.get(name, np.nan)) for name in self.feature_names]
                )
            else:  # assume already a vector aligned to feature_names
                rows.append([_safe_float(v) for v in np.atleast_1d(item)])
        return np.asarray(rows, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Sequence[Mapping[str, Any]] | np.ndarray,
        y: Sequence[Any],
        groups: Sequence[Any] | None = None,
    ) -> "MLClassifier":
        """Fit (and calibrate) the classifier.

        Parameters
        ----------
        X:
            Either a list of feature dicts or a 2-D array aligned to
            ``feature_names``.
        y:
            Labels -- either canonical class strings (``'transit'`` ...) or
            integer ids in ``range(4)``.
        groups:
            Optional group ids (e.g. TIC) so calibration folds never split a star
            across train/test (uses :class:`GroupKFold` when given).

        Returns ``self``. If scikit-learn is unavailable this logs a warning and
        is a no-op (``predict`` then falls back to rules).
        """
        if not self.available:
            warnings.warn(
                "scikit-learn unavailable: MLClassifier.fit() is a no-op; "
                "predict() will fall back to classify_rules.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self

        self._infer_feature_names(X)
        Xmat = self._to_matrix(X)
        yint = np.asarray([_label_to_int(label) for label in y], dtype=int)

        self._impute_fit(Xmat)
        Xmat = self._impute(Xmat)

        self._classes_seen = sorted(set(int(v) for v in yint))

        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight = compute_sample_weight(class_weight="balanced", y=yint)

        base, backend = _make_base_estimator(self.model_spec, n_classes=len(CLASSES))
        self.backend = backend

        # Decide whether calibration is feasible: need >= 2 classes and enough
        # samples per class for a small CV.
        self._clf = self._fit_calibrated(base, Xmat, yint, sample_weight, groups)
        return self

    def _fit_calibrated(
        self,
        base: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        groups: Sequence[Any] | None,
    ) -> Any:
        """Fit ``base`` wrapped in calibration when there is enough data.

        Falls back to an uncalibrated fit (still a valid probability model) when
        the class counts are too small for cross-validated calibration.
        """
        from sklearn.calibration import CalibratedClassifierCV

        classes, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if counts.size else 0
        n_splits = int(min(3, min_count)) if min_count >= 2 else 0

        # Not enough per-class data to calibrate: fit the bare estimator.
        if classes.size < 2 or n_splits < 2:
            logger.info(
                "MLClassifier: insufficient per-class data (min_count=%d); "
                "fitting uncalibrated %s.",
                min_count,
                self.backend,
            )
            self._calibrated = False
            _fit_with_optional_weight(base, X, y, sample_weight)
            return base

        cv = self._make_cv(y, groups, n_splits)
        method = "isotonic" if min_count >= 5 else "sigmoid"
        try:
            calibrator = CalibratedClassifierCV(base, method=method, cv=cv)
            # sample_weight is routed to the base estimator where supported.
            try:
                calibrator.fit(X, y, sample_weight=sample_weight)
            except (TypeError, ValueError):
                calibrator.fit(X, y)
            self._calibrated = True
            logger.info(
                "MLClassifier: fitted %s with %s calibration (%d folds).",
                self.backend,
                method,
                n_splits,
            )
            return calibrator
        except Exception as exc:  # pragma: no cover - robustness net
            logger.warning(
                "MLClassifier: calibration failed (%s); fitting uncalibrated %s.",
                exc,
                self.backend,
            )
            self._calibrated = False
            _fit_with_optional_weight(base, X, y, sample_weight)
            return base

    @staticmethod
    def _make_cv(y: np.ndarray, groups: Sequence[Any] | None, n_splits: int) -> Any:
        """Pick a CV splitter: grouped if ``groups`` given and feasible."""
        if groups is not None:
            groups_arr = np.asarray(groups)
            n_groups = len(np.unique(groups_arr))
            if n_groups >= n_splits:
                from sklearn.model_selection import GroupKFold

                return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups_arr))
        from sklearn.model_selection import StratifiedKFold

        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # -- NaN imputation for the sklearn fallbacks ------------------------ #
    def _impute_fit(self, X: np.ndarray) -> None:
        """Record per-column medians used to fill NaNs (RF/calibration safe)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self._impute_values = med

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Replace non-finite entries with the stored column medians."""
        med = getattr(self, "_impute_values", None)
        if med is None:
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        out = X.copy()
        bad = ~np.isfinite(out)
        if bad.any():
            cols = np.where(bad)[1]
            out[bad] = med[cols]
        return out

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_proba_batch(
        self, X: Sequence[Mapping[str, Any]] | np.ndarray
    ) -> np.ndarray:
        """Return an ``(n, 4)`` calibrated probability matrix over :data:`CLASSES`.

        If the model was not (or could not be) fit, falls back to the rule-based
        policy per row. Classes absent from the training set get probability 0
        and the row is renormalised over :data:`CLASSES`.
        """
        if self._clf is None:
            return self._rules_proba_batch(X)

        Xmat = self._to_matrix(X)
        Xmat = self._impute(Xmat)
        raw = self._clf.predict_proba(Xmat)
        model_classes = list(getattr(self._clf, "classes_", self._classes_seen))

        out = np.zeros((Xmat.shape[0], len(CLASSES)), dtype=np.float64)
        for col, cls_id in enumerate(model_classes):
            idx = int(cls_id)
            if 0 <= idx < len(CLASSES):
                out[:, idx] = raw[:, col]
        # Renormalise (rows where every present class is 0 -> uniform).
        row_sums = out.sum(axis=1, keepdims=True)
        safe = row_sums.squeeze(-1) > 0
        out[safe] /= row_sums[safe]
        out[~safe] = 1.0 / len(CLASSES)
        return out

    def predict(self, features: Mapping[str, Any]) -> Classification:
        """Classify a single feature dict into a :class:`Classification`.

        ``method='ml'``; ``rationale`` lists the top contributing features when
        the backend exposes ``feature_importances_``. Falls back to
        :func:`classify_rules` (with ``method='ml'`` annotation) when no model is
        available.
        """
        if self._clf is None:
            result = classify_rules(None, None, features)
            result.method = "ml"
            result.rationale = [
                "ML model unavailable; fell back to rule-based policy"
            ] + list(result.rationale)
            return result

        proba = self.predict_proba_batch([features])[0]
        prob_map = {cls: float(p) for cls, p in zip(CLASSES, proba)}
        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: v / total for k, v in prob_map.items()}
        label = max(prob_map, key=prob_map.get)

        rationale = [f"ML ({self.backend}) prediction: {label} p={prob_map[label]:.2f}"]
        if not self._calibrated:
            rationale.append("probabilities uncalibrated (insufficient data)")
        rationale.extend(self._top_feature_rationale(features))

        return Classification(
            label=label,
            confidence=float(prob_map[label]),
            probabilities=prob_map,
            method="ml",
            rationale=rationale,
        )

    def _top_feature_rationale(self, features: Mapping[str, Any], k: int = 3) -> list[str]:
        """Best-effort "top features" rationale from importances, if exposed."""
        importances = self._feature_importances()
        if importances is None or self.feature_names is None:
            return []
        order = np.argsort(importances)[::-1][:k]
        parts = []
        for idx in order:
            if idx < len(self.feature_names):
                name = self.feature_names[idx]
                val = features.get(name) if isinstance(features, Mapping) else None
                if _is_number(val):
                    parts.append(f"{name}={float(val):.3g} (imp {importances[idx]:.2f})")
                else:
                    parts.append(f"{name} (imp {importances[idx]:.2f})")
        if parts:
            return ["top features: " + ", ".join(parts)]
        return []

    def _feature_importances(self) -> np.ndarray | None:
        """Pull ``feature_importances_`` from the (possibly calibrated) model."""
        est = self._clf
        if est is None:
            return None
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float)
        # CalibratedClassifierCV: average importances across fold estimators.
        cc = getattr(est, "calibrated_classifiers_", None)
        if cc:
            imps = []
            for fold in cc:
                base = getattr(fold, "estimator", None)
                if base is not None and hasattr(base, "feature_importances_"):
                    imps.append(np.asarray(base.feature_importances_, dtype=float))
            if imps:
                return np.mean(imps, axis=0)
        return None

    def _rules_proba_batch(
        self, X: Sequence[Mapping[str, Any]] | np.ndarray
    ) -> np.ndarray:
        """Rule-based probability matrix used when no model is fitted."""
        if isinstance(X, Mapping):
            X = [X]
        rows = []
        for item in X:
            feats = item if isinstance(item, Mapping) else {}
            c = classify_rules(None, None, feats)
            rows.append([c.probabilities.get(cls, 0.0) for cls in CLASSES])
        return np.asarray(rows, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist the fitted model (and metadata) with :mod:`joblib`."""
        import joblib

        payload = {
            "clf": self._clf,
            "feature_names": self.feature_names,
            "backend": self.backend,
            "calibrated": self._calibrated,
            "classes_seen": self._classes_seen,
            "impute_values": getattr(self, "_impute_values", None),
            "model_spec": self.model_spec,
            "classes": CLASSES,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "MLClassifier":
        """Load a model previously written by :meth:`save`."""
        import joblib

        payload = joblib.load(path)
        obj = cls(feature_names=payload.get("feature_names"), model=payload.get("model_spec", "auto"))
        obj._clf = payload.get("clf")
        obj.backend = payload.get("backend")
        obj._calibrated = bool(payload.get("calibrated", False))
        obj._classes_seen = payload.get("classes_seen", [])
        impute = payload.get("impute_values")
        if impute is not None:
            obj._impute_values = impute
        obj.available = _has_sklearn()
        return obj


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    """True for a real, finite (or NaN-but-numeric) scalar -- not bool/None/str."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_float(value: Any) -> float:
    """Coerce to float; non-numeric / None -> NaN; bool -> 0/1."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _label_to_int(label: Any) -> int:
    """Map a class label (string or int) to its canonical integer index."""
    if isinstance(label, str):
        if label in _LABEL_TO_INDEX:
            return _LABEL_TO_INDEX[label]
        raise ValueError(f"unknown class label {label!r}; expected one of {CLASSES}")
    idx = int(label)
    if 0 <= idx < len(CLASSES):
        return idx
    raise ValueError(f"class index {idx} out of range for {CLASSES}")


def _canonical_feature_names(n: int) -> list[str]:
    """Return ``exopipe.features.FEATURE_NAMES`` if it matches ``n``, else f0..f{n-1}."""
    try:
        from ..features import FEATURE_NAMES  # type: ignore

        names = list(FEATURE_NAMES)
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"f{i}" for i in range(n)]


def _fit_with_optional_weight(
    estimator: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
) -> None:
    """``estimator.fit`` with ``sample_weight`` where the backend accepts it."""
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        estimator.fit(X, y)
