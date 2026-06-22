#!/usr/bin/env python
"""Train and evaluate the ``exopipe`` tabular classifier on synthetic data.

This is the runnable trainer behind ``exopipe train``. It builds a labelled
synthetic population, runs the *real* detection -> vetting -> feature path for
each light curve, collects ``(features, label, group=tic_id)``, trains an
:class:`~exopipe.classify.ml.MLClassifier` with **grouped** cross-validation
(``GroupKFold`` by TIC so a star never spans train/test), prints accuracy +
confusion matrix + per-class precision/recall, and saves the calibrated model to
``models/exopipe_clf.joblib``.

It depends on modules owned by other engineers (``exopipe.detrend``,
``exopipe.search``, ``exopipe.vetting``, ``exopipe.features``); those exist at
integration time. Each per-light-curve stage is wrapped defensively so a single
failure (or a not-yet-implemented module) does not abort the whole run -- the
light curve is skipped and counted.

Examples
--------
::

    PYTHONPATH=src python scripts/train_classifier.py --n 600 --seed 0
    PYTHONPATH=src python scripts/train_classifier.py --n 2000 --out models/clf.joblib
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import numpy as np

# Make ``src`` importable when run directly (PYTHONPATH=src is the convention,
# but this also works from a plain ``python scripts/train_classifier.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logger = logging.getLogger("train_classifier")


# --------------------------------------------------------------------------- #
# Per-light-curve feature extraction (real pipeline path)
# --------------------------------------------------------------------------- #
def _extract_one(lc: Any) -> tuple[dict, str, Any] | None:
    """Run detrend -> search -> vet -> features for one light curve.

    Returns ``(features, label, tic_id)`` or ``None`` if any stage fails or the
    light curve carries no ground-truth label.
    """
    from exopipe.detrend import detrend
    from exopipe.features import extract_features
    from exopipe.search import search_two_stage
    from exopipe.vetting import vet

    label = lc.meta.get("label")
    if label is None:
        return None
    tic_id = lc.meta.get("tic_id")

    try:
        flat = detrend(lc)
        det = search_two_stage(flat)
        vetting = vet(flat, det)
        features = extract_features(flat, det, vetting, None)
    except Exception as exc:
        logger.debug("skip TIC %s: pipeline error %s", tic_id, exc)
        return None

    if not isinstance(features, dict) or not features:
        return None
    return features, str(label), tic_id


def build_dataset(
    n: int, seed: int, n_jobs: int = 1
) -> tuple[list[dict], list[str], list[Any]]:
    """Generate a population and extract features for every light curve.

    Returns parallel lists ``(X, y, groups)``. Uses joblib for fan-out when
    ``n_jobs != 1`` and joblib is importable; otherwise a serial loop.
    """
    from exopipe.data import make_synthetic_population

    logger.info("generating %d synthetic light curves (seed=%d)...", n, seed)
    population = make_synthetic_population(n, seed=seed)

    results: list[Any] = []
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            logger.info("extracting features with joblib (n_jobs=%d)...", n_jobs)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_extract_one)(lc) for lc in population
            )
        except Exception as exc:
            logger.warning("joblib fan-out failed (%s); falling back to serial.", exc)
            results = []
    if not results:
        logger.info("extracting features serially...")
        results = []
        for i, lc in enumerate(population):
            results.append(_extract_one(lc))
            if (i + 1) % max(1, n // 10) == 0:
                logger.info("  %d/%d processed", i + 1, n)

    X: list[dict] = []
    y: list[str] = []
    groups: list[Any] = []
    for item in results:
        if item is None:
            continue
        features, label, tic_id = item
        X.append(features)
        y.append(label)
        groups.append(tic_id)

    logger.info("usable training rows: %d / %d", len(X), n)
    return X, y, groups


# --------------------------------------------------------------------------- #
# Grouped cross-validated evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    X: list[dict],
    y: list[str],
    groups: list[Any],
    model: str = "auto",
    n_splits: int = 5,
) -> None:
    """Grouped-CV evaluation: accuracy + confusion matrix + per-class P/R.

    Predictions are collected out-of-fold (each star is predicted only by a model
    that never saw it), then scored once. Prints a human-readable report.
    """
    from exopipe.classify.ml import MLClassifier
    from exopipe.classify.rules import CLASSES

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import GroupKFold
    except Exception:
        logger.warning("scikit-learn unavailable: skipping grouped-CV evaluation.")
        return

    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    n_groups = len(np.unique(groups_arr))
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        logger.warning("not enough groups (%d) for CV; skipping evaluation.", n_groups)
        return

    gkf = GroupKFold(n_splits=splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(y_arr)), y_arr, groups_arr)):
        X_tr = [X[i] for i in tr]
        y_tr = [y[i] for i in tr]
        g_tr = [groups[i] for i in tr]
        clf = MLClassifier(model=model)
        clf.fit(X_tr, y_tr, groups=g_tr)
        for i in te:
            pred = clf.predict(X[i])
            y_true_all.append(y[i])
            y_pred_all.append(pred.label)
        logger.info("fold %d/%d done (%d test rows)", fold + 1, splits, len(te))

    acc = float(np.mean(np.asarray(y_true_all) == np.asarray(y_pred_all)))
    print("\n" + "=" * 64)
    print(f"GROUPED CROSS-VALIDATION ({splits} folds, group=TIC)")
    print("=" * 64)
    print(f"Overall accuracy: {acc:.3f}  (n={len(y_true_all)})\n")

    labels_present = [c for c in CLASSES if c in set(y_true_all) | set(y_pred_all)]
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_present)
    header = "            " + "".join(f"{c[:10]:>12}" for c in labels_present)
    print(header)
    for row_label, row in zip(labels_present, cm):
        print(f"{row_label[:10]:>10}  " + "".join(f"{v:>12d}" for v in row))

    print("\nPer-class precision / recall / F1:")
    print(
        classification_report(
            y_true_all, y_pred_all, labels=labels_present, digits=3, zero_division=0
        )
    )


# --------------------------------------------------------------------------- #
# Final model fit + save
# --------------------------------------------------------------------------- #
def train_and_save(
    X: list[dict],
    y: list[str],
    groups: list[Any],
    out_path: str,
    model: str = "auto",
) -> None:
    """Fit the final calibrated model on ALL data and persist it."""
    from exopipe.classify.ml import MLClassifier

    clf = MLClassifier(model=model)
    clf.fit(X, y, groups=groups)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    clf.save(out_path)
    print(
        f"\nSaved calibrated classifier (backend={clf.backend}, "
        f"calibrated={clf._calibrated}) -> {out_path}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the exopipe tabular classifier on synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=600, help="number of synthetic light curves")
    parser.add_argument("--seed", type=int, default=0, help="master random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="models/exopipe_clf.joblib",
        help="output path for the saved model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "xgboost", "lightgbm", "rf", "histgb"],
        help="ML backend",
    )
    parser.add_argument("--splits", type=int, default=5, help="grouped-CV folds")
    parser.add_argument("--n-jobs", type=int, default=1, help="joblib workers for feature extraction")
    parser.add_argument(
        "--no-eval", action="store_true", help="skip the grouped-CV evaluation"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    X, y, groups = build_dataset(args.n, args.seed, n_jobs=args.n_jobs)
    if len(X) < 8:
        logger.error("too few usable rows (%d) to train; aborting.", len(X))
        return 1

    if not args.no_eval:
        evaluate(X, y, groups, model=args.model, n_splits=args.splits)

    train_and_save(X, y, groups, args.out, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
