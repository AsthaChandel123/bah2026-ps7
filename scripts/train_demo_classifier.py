#!/usr/bin/env python
"""Fast, demo-quality trainer for the ``exopipe`` 4-class tabular classifier.

This is a thin, *fast* driver around the same machinery as
``scripts/train_classifier.py`` (``MLClassifier`` + ``GroupKFold`` by TIC + the
real detrend -> search -> vet -> features path). Two practical differences keep
it inside a few-minutes budget while still producing an honest, skilful model:

1. **Low-resolution training light curves.** Full-resolution (27.4 d / 2 min)
   BLS->TLS is ~10 s/LC. We generate ``n_days=20, cadence_min=8`` curves
   (~1.5 s/LC) and fan the feature extraction out with joblib. The detrend ->
   search -> vet -> features path is *identical* to inference, so the train and
   inference feature distributions match (no truth-seeding).

2. **Detectable-signal bias.** The synthetic generator's faintest transits are
   genuinely undetectable (low SNR -> correctly 'other'). A classifier learns
   nothing useful from those, so we draw a *healthy fraction* of transits / EBs /
   blends with brighter hosts and sensible depths (so the real search recovers
   them), while still keeping some hard / faint cases for realism. The label is
   still the ground-truth astrophysical class; only the *parameter priors* are
   nudged toward detectability.

The model is saved to ``models/exopipe_clf.joblib`` -- the path
``exopipe.classify.ensemble.load_models`` reads -- and a grouped-CV report
(accuracy + 4x4 confusion + per-class precision/recall) is printed.

Examples
--------
::

    PYTHONPATH=src python scripts/train_demo_classifier.py --n 260 --n-jobs -1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logger = logging.getLogger("train_demo_classifier")


# --------------------------------------------------------------------------- #
# Detectable-signal-biased population
# --------------------------------------------------------------------------- #
def _draw_lightcurve(kind: str, seed: int, rng: np.random.Generator,
                     n_days: float, cadence_min: float) -> Any:
    """Draw one synthetic LC of ``kind`` with detectability-biased parameters.

    A fraction of transit/EB/blend signals are drawn "easy" (bright host +
    sensible depth so the real BLS->TLS search recovers them); the rest are left
    to the generator's natural (sometimes hard / faint) priors. ``variable`` and
    ``noise`` (both label == 'other') always use the natural priors.
    """
    from exopipe.data.synthetic import make_synthetic_lightcurve

    params: dict[str, Any] = {}

    if kind == "transit":
        # 75% clearly detectable, 25% hard (faint / shallow) for realism.
        if rng.random() < 0.75:
            params["tmag"] = float(rng.uniform(8.0, 12.0))
            # 700 ppm .. 2.2% depth -> recoverable at this brightness.
            params["depth"] = float(rng.uniform(7e-4, 2.2e-2))
            params["period"] = float(rng.uniform(1.5, 9.0))
            params["b"] = float(rng.uniform(0.0, 0.6))
        else:
            params["tmag"] = float(rng.uniform(12.5, 14.5))
    elif kind == "eclipsing_binary":
        if rng.random() < 0.8:
            params["tmag"] = float(rng.uniform(8.0, 12.5))
            params["depth"] = float(rng.uniform(3e-2, 0.35))
            params["period"] = float(rng.uniform(1.0, 9.0))
        else:
            params["tmag"] = float(rng.uniform(12.5, 14.5))
    elif kind == "blend":
        if rng.random() < 0.8:
            params["tmag"] = float(rng.uniform(8.0, 12.5))
            # intrinsic depth before dilution; crowdsap dilutes it in-generator.
            params["depth"] = float(rng.uniform(2e-2, 0.2))
            params["crowdsap"] = float(rng.uniform(0.25, 0.7))
            params["period"] = float(rng.uniform(1.0, 9.0))
        else:
            params["tmag"] = float(rng.uniform(12.5, 14.5))
    # 'variable' / 'noise' -> natural priors (these are the 'other' class).

    return make_synthetic_lightcurve(
        kind=kind, seed=seed, n_days=n_days, cadence_min=cadence_min, **params
    )


def build_population(n: int, seed: int, n_days: float, cadence_min: float,
                     fractions: dict[str, float]) -> list[Any]:
    """Build a labelled, detectability-biased, reproducible population."""
    from exopipe.data.synthetic import KINDS

    fractions = {k: float(v) for k, v in fractions.items() if k in KINDS and v > 0}
    total = sum(fractions.values())
    fractions = {k: v / total for k, v in fractions.items()}

    rng = np.random.default_rng(seed)
    kinds = list(fractions.keys())
    exact = np.array([fractions[k] * n for k in kinds])
    counts = np.floor(exact).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        order = np.argsort(-(exact - counts))
        for i in range(rem):
            counts[order[i % len(order)]] += 1

    sequence: list[str] = []
    for kind, count in zip(kinds, counts):
        sequence.extend([kind] * int(count))
    rng.shuffle(sequence)

    child_seeds = rng.integers(0, 2**31 - 1, size=len(sequence))
    pop = [
        _draw_lightcurve(sequence[i], int(child_seeds[i]), rng, n_days, cadence_min)
        for i in range(len(sequence))
    ]
    return pop


# --------------------------------------------------------------------------- #
# Real-path feature extraction (identical to inference)
# --------------------------------------------------------------------------- #
def _extract_one(lc: Any) -> tuple[dict, str, Any] | None:
    """detrend -> search -> vet -> features for one LC (no truth-seeding)."""
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
    except Exception as exc:  # pragma: no cover - per-LC robustness
        logger.debug("skip TIC %s: %s", tic_id, exc)
        return None
    if not isinstance(features, dict) or not features:
        return None
    return features, str(label), tic_id


def build_dataset(pop: list[Any], n_jobs: int) -> tuple[list[dict], list[str], list[Any]]:
    """Extract features for every LC, in parallel when possible."""
    results: list[Any] = []
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            logger.info("extracting features with joblib (n_jobs=%d)...", n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_extract_one)(lc) for lc in pop)
        except Exception as exc:
            logger.warning("joblib fan-out failed (%s); serial fallback.", exc)
            results = []
    if not results:
        results = [_extract_one(lc) for lc in pop]

    X: list[dict] = []
    y: list[str] = []
    groups: list[Any] = []
    for item in results:
        if item is None:
            continue
        feats, label, tic = item
        X.append(feats)
        y.append(label)
        groups.append(tic)
    logger.info("usable training rows: %d / %d", len(X), len(pop))
    return X, y, groups


# --------------------------------------------------------------------------- #
# Grouped-CV evaluation
# --------------------------------------------------------------------------- #
def evaluate(X: list[dict], y: list[str], groups: list[Any], model: str,
             n_splits: int) -> dict[str, Any]:
    """Out-of-fold grouped-CV: accuracy + confusion + per-class P/R/F1."""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold

    from exopipe.classify.ml import MLClassifier
    from exopipe.classify.rules import CLASSES

    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    n_groups = len(np.unique(groups_arr))
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        logger.warning("not enough groups for CV (%d); skipping eval.", n_groups)
        return {}

    gkf = GroupKFold(n_splits=splits)
    y_true: list[str] = []
    y_pred: list[str] = []
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(y_arr)), y_arr, groups_arr)):
        clf = MLClassifier(model=model)
        clf.fit([X[i] for i in tr], [y[i] for i in tr], groups=[groups[i] for i in tr])
        for i in te:
            y_true.append(y[i])
            y_pred.append(clf.predict(X[i]).label)
        logger.info("fold %d/%d done (%d test rows)", fold + 1, splits, len(te))

    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    labels = list(CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 68)
    print(f"GROUPED CROSS-VALIDATION  ({splits} folds, group=TIC)   n={len(y_true)}")
    print("=" * 68)
    print(f"Overall accuracy : {acc:.3f}   (4-class chance baseline = 0.250)\n")
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "".join(f"{c[:11]:>13}" for c in labels))
    for lbl, row in zip(labels, cm):
        print(f"{lbl[:11]:>11} " + "".join(f"{v:>13d}" for v in row))
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))

    return {"accuracy": acc, "confusion": cm.tolist(), "labels": labels,
            "y_true": y_true, "y_pred": y_pred}


# --------------------------------------------------------------------------- #
# Final fit + save
# --------------------------------------------------------------------------- #
def train_and_save(X: list[dict], y: list[str], groups: list[Any],
                   out_path: str, model: str) -> None:
    from exopipe.classify.ml import MLClassifier

    clf = MLClassifier(model=model)
    clf.fit(X, y, groups=groups)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    clf.save(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved calibrated classifier (backend={clf.backend}, "
          f"calibrated={clf._calibrated}) -> {out_path}  ({size_mb:.2f} MB)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=260, help="number of synthetic LCs")
    p.add_argument("--seed", type=int, default=0, help="master seed")
    p.add_argument("--out", default="models/exopipe_clf.joblib", help="model output path")
    p.add_argument("--model", default="auto",
                   choices=["auto", "xgboost", "lightgbm", "rf", "histgb"])
    p.add_argument("--n-days", type=float, default=20.0, dest="n_days")
    p.add_argument("--cadence-min", type=float, default=8.0, dest="cadence_min")
    p.add_argument("--splits", type=int, default=5, help="grouped-CV folds")
    p.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Healthy fraction of detectable transits/EBs/blends; some hard 'other'.
    fractions = {
        "transit": 0.34,
        "eclipsing_binary": 0.24,
        "blend": 0.16,
        "variable": 0.13,
        "noise": 0.13,
    }

    t0 = time.time()
    logger.info("building detectability-biased population (n=%d, %.0fd/%.0fmin)...",
                args.n, args.n_days, args.cadence_min)
    pop = build_population(args.n, args.seed, args.n_days, args.cadence_min, fractions)
    X, y, groups = build_dataset(pop, n_jobs=args.n_jobs)
    logger.info("dataset built in %.1fs", time.time() - t0)

    if len(X) < 8:
        logger.error("too few usable rows (%d); aborting.", len(X))
        return 1

    # Report the realised label mix.
    uniq, cnts = np.unique(np.asarray(y), return_counts=True)
    print("\nTraining-set label mix: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnts)))

    if not args.no_eval:
        evaluate(X, y, groups, model=args.model, n_splits=args.splits)

    train_and_save(X, y, groups, args.out, model=args.model)
    logger.info("total wall time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
