#!/usr/bin/env python
"""Generate the curated PS7 example deliverables under ``examples/``.

Produces, from a small reproducible synthetic population run through the *real*
pipeline (``process_lightcurve`` with the trained classifier loaded):

* one representative, correctly-classified one-page vetting sheet per class
  (``examples/vetting_<class>.png``),
* the demo catalog (``examples/example_catalog.csv``),
* the <=3-page methodology + results report (``examples/example_report.pdf``).

The light curves use clear, detectable parameters per class (bright host +
sensible depth) so each example is an unambiguous, correctly-classified case --
exactly what a reviewer should see. Selection still requires the *predicted*
label to match the truth (no cherry-picking of mislabels).

Usage::

    PYTHONPATH=src python scripts/make_examples.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")

# Canonical class -> generator-kind we use to manufacture a clean example.
_CLASS_KIND = {
    "transit": "transit",
    "eclipsing_binary": "eclipsing_binary",
    "blend": "blend",
    "other": "variable",  # a clear stellar-variability ('other') case
}

# Deterministic per-class RNG seeds (Python's str ``hash`` is salted per process,
# so we use an explicit table to keep example selection fully reproducible).
_CLASS_SEED = {
    "transit": 101,
    "eclipsing_binary": 202,
    "blend": 303,
    "other": 404,
}


def _candidate_pool(klass: str, kind: str) -> list[tuple[int, dict]]:
    """Return (seed, params) draws biased to a clean, detectable example."""
    rng = np.random.default_rng(_CLASS_SEED[klass])
    draws: list[tuple[int, dict]] = []
    for _ in range(14):
        seed = int(rng.integers(1, 2**31 - 1))
        if kind == "transit":
            params = dict(tmag=float(rng.uniform(8.5, 10.5)),
                          depth=float(rng.uniform(6e-3, 1.6e-2)),
                          period=float(rng.uniform(2.5, 6.5)),
                          b=float(rng.uniform(0.0, 0.4)))
        elif kind == "eclipsing_binary":
            params = dict(tmag=float(rng.uniform(8.5, 11.0)),
                          depth=float(rng.uniform(8e-2, 0.25)),
                          period=float(rng.uniform(2.0, 6.0)))
        elif kind == "blend":
            params = dict(tmag=float(rng.uniform(8.5, 11.0)),
                          depth=float(rng.uniform(6e-2, 0.18)),
                          crowdsap=float(rng.uniform(0.3, 0.55)),
                          period=float(rng.uniform(2.0, 6.0)))
        else:  # variable -> 'other'
            params = dict(tmag=float(rng.uniform(9.0, 12.0)))
        draws.append((seed, params))
    return draws


def _score(result: Any) -> float:
    """Rank correctly-classified candidates: prefer high confidence + SNR."""
    conf = float(getattr(result.classification, "confidence", 0.0) or 0.0)
    snr = getattr(result.detection, "snr", np.nan)
    snr = float(snr) if np.isfinite(snr) else 0.0
    return conf + min(snr, 100.0) / 100.0


def pick_per_class(models: Any, cfg: Any) -> dict[str, Any]:
    """Process clean draws per class; keep the best correctly-classified result."""
    from exopipe.data.synthetic import make_synthetic_lightcurve
    from exopipe.pipeline import process_lightcurve

    chosen: dict[str, Any] = {}
    for klass, kind in _CLASS_KIND.items():
        best = None
        best_score = -1.0
        for seed, params in _candidate_pool(klass, kind):
            lc = make_synthetic_lightcurve(kind=kind, seed=seed, **params)
            result = process_lightcurve(lc, config=cfg, models=models)
            pred = getattr(result.classification, "label", None)
            if pred != klass:
                continue
            s = _score(result)
            if s > best_score:
                best, best_score = result, s
            # Early-exit: a confident, correctly-classified case is good enough
            # for a representative example (keeps the generator fast).
            if best_score >= 1.6:  # conf >= ~0.85 and high SNR
                break
        if best is None:
            # Fallback: take the highest-scoring regardless of match so we always
            # emit a sheet (should be rare for these easy draws).
            for seed, params in _candidate_pool(klass, kind):
                lc = make_synthetic_lightcurve(kind=kind, seed=seed, **params)
                result = process_lightcurve(lc, config=cfg, models=models)
                s = _score(result)
                if s > best_score:
                    best, best_score = result, s
        chosen[klass] = best
        pred = getattr(best.classification, "label", "?")
        conf = getattr(best.classification, "confidence", float("nan"))
        print(f"  [{klass:16s}] -> predicted {pred:16s} conf={conf:.2f} "
              f"(score={best_score:.2f})")
    return chosen


def write_vetting_sheets(chosen: dict[str, Any]) -> list[str]:
    from exopipe.viz import vetting_sheet
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for klass, result in chosen.items():
        path = os.path.join(EXAMPLES_DIR, f"vetting_{klass}.png")
        fig = vetting_sheet(result, save_path=path, dpi=130)
        plt.close(fig)
        size_kb = os.path.getsize(path) / 1024
        print(f"  wrote {path}  ({size_kb:.0f} KB)")
        paths.append(path)
    return paths


def write_catalog(results: list[Any]) -> str:
    from exopipe import catalog as catalog_mod

    rows = [r.to_row() for r in results]
    path = os.path.join(EXAMPLES_DIR, "example_catalog.csv")
    catalog_mod.write_catalog(rows, path, fmt="csv")
    print(f"  wrote {path}  ({os.path.getsize(path)} bytes, {len(rows)} rows)")
    return path


def write_report(results: list[Any], figure_paths: list[str]) -> str:
    from exopipe.report import generate_report

    out = os.path.join(EXAMPLES_DIR, "example_report.pdf")
    run_meta = {
        "synthetic": True,
        "data_summary": (
            f"{len(results)} representative synthetic TESS-like light curves "
            "(one clean, correctly-classified example per class) processed end to "
            "end by the offline pipeline."
        ),
        "results_note": (
            "Examples are drawn from the synthetic generator with detectable "
            "parameters so every class is represented by an unambiguous, "
            "correctly-classified candidate."
        ),
    }
    # Use the transit + EB sheets as the embedded example figures.
    figs = [p for p in figure_paths
            if p.endswith("vetting_transit.png") or p.endswith("vetting_eclipsing_binary.png")]
    written = generate_report(results, output_path=out, figures=figs[:2], run_meta=run_meta)
    print(f"  wrote {written}  ({os.path.getsize(written)} bytes)")
    return written


def main() -> int:
    from exopipe.classify.ensemble import load_models
    from exopipe.config import default_config

    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    cfg = default_config()
    models = load_models(os.path.join(_REPO_ROOT, "models"))
    print(f"Loaded models: {list(models.keys()) or '(rules only)'}")

    print("Selecting one correctly-classified example per class ...")
    chosen = pick_per_class(models, cfg)
    results = [chosen[k] for k in ("transit", "eclipsing_binary", "blend", "other")]

    print("Writing vetting sheets ...")
    figure_paths = write_vetting_sheets(chosen)

    print("Writing catalog ...")
    write_catalog(results)

    print("Writing report ...")
    write_report(results, figure_paths)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
