"""Command-line entry point for ``exopipe`` (``exopipe ...`` / ``python -m``).

A self-contained ``argparse`` CLI (kept core-safe so it imports and runs with
only ``numpy``/``scipy``/``pandas``/``matplotlib``) wiring up the deployment
surface from ``ARCHITECTURE.md`` §12:

    demo       offline synthetic run → catalog + vetting sheets + report
    run        process a directory of FITS / a CSV → catalog + figures
    fetch      download a TESS sector / TIC from MAST (network)
    train      train + calibrate the classifier (grouped CV)
    report     render the ≤3-page report from a catalog
    dashboard  launch the Streamlit catalog browser
    version    print the package version

Every heavy/optional stage module is imported **lazily inside its handler**, so
``exopipe --help`` and ``exopipe version`` work without any optional dependency,
and ``exopipe demo`` runs the full pipeline on synthetic data with zero network.
If ``typer`` is installed it is *not* required — argparse is the safe default —
keeping the command surface stable across environments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__

# Canonical four science classes (kept local so the CLI never has to import the
# classify subpackage just to print a summary table).
_CLASSES = ("transit", "eclipsing_binary", "blend", "other")


# =========================================================================== #
# demo -- offline, self-contained acceptance run
# =========================================================================== #
def _cmd_demo(args: argparse.Namespace) -> int:
    """Synthesize a labelled population, run the pipeline, write artifacts.

    This is the first thing a reviewer runs: it exercises every module end to end
    with **only core dependencies** and no network (``ARCHITECTURE`` §12.3).
    """
    import numpy as np

    from .config import default_config
    from .data.synthetic import make_synthetic_population
    from .driver import run_batch
    from .utils import Timer, get_logger

    log = get_logger("exopipe.cli")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"exopipe {__version__} — demo")
    print(f"  synthesizing {args.n} light curve(s) (seed={args.seed}) …")
    population = make_synthetic_population(args.n, seed=args.seed)

    cfg = default_config()
    cfg.perf.n_jobs = args.n_jobs

    print(f"  running pipeline → {out_dir} (figures={'on' if args.figures else 'off'}) …")
    with Timer("demo.run_batch", logger=log):
        results = run_batch(
            population,
            config=cfg,
            n_jobs=args.n_jobs,
            out_dir=out_dir,
            make_figures=args.figures,
            top_k_figures=args.top_k,
            catalog_fmt=args.fmt,
        )

    # -- report (best-effort; degrades to a printed summary) ---------------- #
    _try_write_report(out_dir, fmt=args.fmt)

    _print_summary_table(results, population)
    print(f"\nArtifacts written under: {out_dir.resolve()}")
    print("  - catalog.%s" % ("parquet" if args.fmt == "parquet" else "csv"))
    if args.figures:
        print("  - vetting_sheets/*.png")
    return 0


def _print_summary_table(results: Sequence[Any], population: Sequence[Any]) -> None:
    """Print per-class counts (predicted vs. truth) and a few example candidates."""
    import numpy as np

    if not results:
        print("\n(no results to summarise)")
        return

    pred_counts = {c: 0 for c in _CLASSES}
    true_counts = {c: 0 for c in _CLASSES}
    correct = 0
    total_labeled = 0

    truth_by_tic = {}
    for lc in population:
        meta = getattr(lc, "meta", {}) or {}
        truth_by_tic[meta.get("tic_id")] = meta.get("label")

    for result in results:
        label = getattr(result.classification, "label", "other")
        pred_counts[label] = pred_counts.get(label, 0) + 1
        tic = (getattr(result.lightcurve, "meta", {}) or {}).get("tic_id")
        truth = truth_by_tic.get(tic)
        if truth in true_counts:
            true_counts[truth] += 1
            total_labeled += 1
            if truth == label:
                correct += 1

    print("\n  Per-class counts (predicted vs. truth):")
    print(f"    {'class':<18} {'predicted':>10} {'truth':>8}")
    for c in _CLASSES:
        print(f"    {c:<18} {pred_counts.get(c, 0):>10} {true_counts.get(c, 0):>8}")
    if total_labeled:
        print(f"    {'-' * 36}")
        print(f"    accuracy vs. truth: {correct}/{total_labeled} = {correct / total_labeled:.1%}")

    # -- a few example candidates, highest SNR first ------------------------ #
    def snr_of(r: Any) -> float:
        s = getattr(r.detection, "snr", float("nan"))
        d = getattr(r.detection, "sde", float("nan"))
        s = float(s) if np.isfinite(s) else 0.0
        d = float(d) if np.isfinite(d) else 0.0
        return max(s, d)

    top = sorted(results, key=snr_of, reverse=True)[:5]
    print("\n  Example candidates (highest SNR/SDE):")
    print(f"    {'TIC':>12} {'class':<18} {'conf':>5} {'period':>8} {'SNR':>7}")
    for r in top:
        meta = getattr(r.lightcurve, "meta", {}) or {}
        period = getattr(r.detection, "period", float("nan"))
        period_s = f"{period:.3f}" if np.isfinite(period) else "  n/a "
        print(
            f"    {str(meta.get('tic_id')):>12} "
            f"{getattr(r.classification, 'label', 'other'):<18} "
            f"{getattr(r.classification, 'confidence', 0.0):>5.2f} "
            f"{period_s:>8} {snr_of(r):>7.1f}"
        )


# =========================================================================== #
# run -- process a local dataset (FITS dir or CSV)
# =========================================================================== #
def _cmd_run(args: argparse.Namespace) -> int:
    """Process a directory of FITS files or a single CSV through the pipeline."""
    from .config import load_config
    from .driver import run_batch
    from .utils import get_logger

    log = get_logger("exopipe.cli")
    cfg = load_config(args.config)
    cfg.perf.n_jobs = args.n_jobs

    lightcurves = _load_input(args.input, log)
    if not lightcurves:
        print(f"error: no light curves loaded from {args.input!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    print(f"exopipe {__version__} — run: {len(lightcurves)} light curve(s) → {out_dir}")
    run_batch(
        lightcurves,
        config=cfg,
        n_jobs=args.n_jobs,
        out_dir=out_dir,
        make_figures=args.figures,
        top_k_figures=args.top_k,
        catalog_fmt=args.fmt,
        resume=args.resume,
    )
    _try_write_report(out_dir, fmt=args.fmt)
    print(f"\nDone. Artifacts under: {out_dir.resolve()}")
    return 0


def _load_input(input_path: str, log: Any) -> list[Any]:
    """Load light curves from a CSV file or a directory of FITS files."""
    from .data.loaders import DataUnavailable, load_from_csv, load_from_fits

    path = Path(input_path)
    lightcurves: list[Any] = []

    if not path.exists():
        log.error("Input path does not exist: %s", path)
        return lightcurves

    if path.is_file() and path.suffix.lower() == ".csv":
        try:
            lightcurves.append(load_from_csv(path))
        except Exception as exc:
            log.error("Failed to read CSV %s (%s).", path, exc)
        return lightcurves

    if path.is_file() and path.suffix.lower() in (".fits", ".fit", ".fz"):
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in (".fits", ".fit", ".fz")
        )
        csvs = sorted(path.glob("*.csv")) if path.is_dir() else []
        for csv in csvs:
            try:
                lightcurves.append(load_from_csv(csv))
            except Exception as exc:
                log.warning("Skipping CSV %s (%s).", csv, exc)

    for fits_path in files:
        try:
            lightcurves.append(load_from_fits(fits_path))
        except (DataUnavailable, Exception) as exc:
            log.warning("Skipping FITS %s (%s).", fits_path, exc)

    return lightcurves


# =========================================================================== #
# fetch -- download a TESS TIC / sector from MAST
# =========================================================================== #
def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download a TESS light curve for one TIC (network; offline-aware)."""
    from .data.loaders import DataUnavailable, load_tess
    from .utils import get_logger

    log = get_logger("exopipe.cli")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"exopipe {__version__} — fetch TIC {args.tic} (sector={args.sector}) …")
    try:
        loaded = load_tess(args.tic, sector=args.sector, author=args.author)
    except DataUnavailable as exc:
        print(f"error: could not fetch TIC {args.tic}: {exc}", file=sys.stderr)
        print(
            "hint: install the 'science' extra (lightkurve/astroquery) and ensure "
            "network access, or use 'exopipe demo' for an offline run.",
            file=sys.stderr,
        )
        return 1

    lcs = loaded if isinstance(loaded, list) else [loaded]
    written = 0
    for lc in lcs:
        sector = (lc.meta or {}).get("sector", "NA")
        path = out_dir / f"tic{args.tic}_s{sector}.csv"
        if _save_lightcurve_csv(lc, path):
            written += 1
            print(f"  wrote {path}  ({len(lc)} cadences)")
    print(f"\nFetched {len(lcs)} sector(s); wrote {written} file(s) to {out_dir.resolve()}.")
    return 0


def _save_lightcurve_csv(lc: Any, path: Path) -> bool:
    """Persist a LightCurve to CSV (time, flux, flux_err). Best-effort."""
    try:
        import pandas as pd

        pd.DataFrame(
            {"time": lc.time, "flux": lc.flux, "flux_err": lc.flux_err}
        ).to_csv(path, index=False)
        return True
    except Exception as exc:  # pragma: no cover - non-fatal
        from .utils import get_logger

        get_logger("exopipe.cli").warning("Failed to write %s (%s).", path, exc)
        return False


# =========================================================================== #
# train -- train + calibrate the classifier
# =========================================================================== #
def _cmd_train(args: argparse.Namespace) -> int:
    """Train the tabular classifier on a synthetic (or provided) labelled set."""
    from .utils import get_logger

    log = get_logger("exopipe.cli")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"exopipe {__version__} — train (n={args.n}, seed={args.seed}) → {out_dir}")

    try:
        from .classify.ml import MLClassifier  # lazy: owned by B4
    except Exception as exc:
        print(
            f"error: classifier trainer unavailable ({exc}).\n"
            "The classification subpackage (exopipe.classify.ml) is required for "
            "training; install the 'ml' extra and ensure the module is present.",
            file=sys.stderr,
        )
        return 1

    # Build a labelled training set: prefer a labels CSV if given, else synthesise.
    try:
        X, y = _build_training_set(args)
    except Exception as exc:
        log.error("Failed to build training set (%s).", exc)
        return 1

    try:
        clf = MLClassifier()
        clf.fit(X, y)
        # Standardized model path: ``ensemble.load_models`` and
        # ``scripts/train_classifier.py`` both load/save ``exopipe_clf.joblib``.
        model_path = out_dir / "exopipe_clf.joblib"
        clf.save(str(model_path))
        print(f"  trained on {len(y)} example(s); saved model → {model_path}")
    except Exception as exc:
        log.error("Training failed (%s).", exc)
        return 1
    return 0


def _build_training_set(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    """Construct ``(features, labels)`` for the classifier from synthetic data.

    Runs the (cheap, fit-free) front of the pipeline over a synthetic labelled
    population to produce feature dicts aligned with ground-truth labels. Used by
    ``exopipe train`` so training works fully offline.
    """
    from .config import default_config
    from .data.synthetic import make_synthetic_population
    from .pipeline import process_lightcurve

    cfg = default_config()
    population = make_synthetic_population(args.n, seed=args.seed)

    features: list[dict] = []
    labels: list[str] = []
    for lc in population:
        truth = (lc.meta or {}).get("label")
        if truth is None:
            continue
        result = process_lightcurve(lc, config=cfg)
        features.append(result.features)
        labels.append(truth)
    return features, labels


# =========================================================================== #
# report -- render the <=3-page report from a catalog
# =========================================================================== #
def _cmd_report(args: argparse.Namespace) -> int:
    """Render the project report from an existing catalog (best-effort)."""
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    ok = _render_report(catalog_path, out_path, figdir=args.figdir)
    if ok:
        print(f"exopipe {__version__} — report written → {out_path}")
        return 0
    print(
        "warning: no report renderer available; printed a text summary instead.",
        file=sys.stderr,
    )
    _print_catalog_summary(catalog_path)
    return 0


def _render_report(catalog_path: Path, out_path: Path, figdir: str | None) -> bool:
    """Try ``exopipe.report`` then a markdown fallback. Returns True if written."""
    # Preferred: a dedicated report module (may be added by the reporting owner).
    try:
        from . import report as report_mod  # type: ignore

        if hasattr(report_mod, "render_report"):
            report_mod.render_report(str(catalog_path), str(out_path), figdir=figdir)
            return True
    except Exception:
        pass

    # Fallback: emit a small markdown report from the catalog table.
    try:
        import pandas as pd

        frame = _read_catalog(catalog_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = _catalog_to_markdown(frame)
        out_md = out_path.with_suffix(".md")
        out_md.write_text(md, encoding="utf-8")
        return True
    except Exception:
        return False


def _catalog_to_markdown(frame: Any) -> str:
    """Render a compact markdown report (counts + top candidates) from a catalog."""
    lines = ["# exopipe candidate report", ""]
    lines.append(f"Total candidates: **{len(frame)}**")
    lines.append("")
    if "class" in frame.columns:
        lines.append("## Per-class counts")
        lines.append("")
        counts = frame["class"].value_counts()
        for label, count in counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    sort_col = "snr" if "snr" in frame.columns else frame.columns[0]
    lines.append("## Top candidates")
    lines.append("")
    cols = [c for c in ("tic_id", "class", "confidence", "period", "snr") if c in frame.columns]
    try:
        top = frame.sort_values(sort_col, ascending=False).head(15)[cols]
        lines.append(top.to_markdown(index=False))
    except Exception:
        lines.append(frame.head(15).to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def _print_catalog_summary(catalog_path: Path) -> None:
    try:
        frame = _read_catalog(catalog_path)
    except Exception as exc:
        print(f"  (could not read catalog: {exc})")
        return
    print(f"  catalog rows: {len(frame)}")
    if "class" in frame.columns:
        print("  per-class counts:")
        for label, count in frame["class"].value_counts().items():
            print(f"    {label}: {count}")


# =========================================================================== #
# dashboard -- launch the Streamlit catalog browser
# =========================================================================== #
def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard via ``streamlit run`` (if installed)."""
    import shutil
    import subprocess

    app_path = _find_dashboard_app()
    if app_path is None:
        print(
            "error: dashboard app not found (looked for app/dashboard.py and app.py).",
            file=sys.stderr,
        )
        return 2

    if shutil.which("streamlit") is None and _module_missing("streamlit"):
        print(
            "error: streamlit is not installed. Install the 'app' extra:\n"
            "    pip install 'exopipe[app]'",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--"]
    if args.catalog:
        cmd += ["--catalog", str(args.catalog)]
    if args.figdir:
        cmd += ["--figdir", str(args.figdir)]

    print(f"exopipe {__version__} — launching dashboard: {' '.join(cmd)}")
    try:
        return int(subprocess.call(cmd))
    except Exception as exc:  # pragma: no cover - subprocess/env dependent
        print(f"error: failed to launch streamlit ({exc}).", file=sys.stderr)
        return 1


def _find_dashboard_app() -> Path | None:
    """Locate the Streamlit entry script (repo ``app/dashboard.py`` or ``app.py``)."""
    here = Path(__file__).resolve()
    # Walk up to the repo root looking for a dashboard script.
    for base in [Path.cwd(), *here.parents]:
        for candidate in (base / "app" / "dashboard.py", base / "app.py", base / "dashboard.py"):
            if candidate.exists():
                return candidate
    return None


# =========================================================================== #
# version
# =========================================================================== #
def _cmd_version(_args: argparse.Namespace) -> int:
    """Print the package version and the optional-dependency availability."""
    print(f"exopipe {__version__}")
    print(f"  python {sys.version.split()[0]}")
    available = _optional_status()
    on = [name for name, ok in available.items() if ok]
    off = [name for name, ok in available.items() if not ok]
    print(f"  optional deps present: {', '.join(on) if on else '(none)'}")
    print(f"  optional deps missing: {', '.join(off) if off else '(none)'}")
    return 0


def _optional_status() -> dict[str, bool]:
    """Report which optional accelerators/loaders are importable (no imports run)."""
    import importlib.util

    names = [
        "lightkurve",
        "astroquery",
        "joblib",
        "numba",
        "pyarrow",
        "zarr",
        "hnswlib",
        "faiss",
        "xgboost",
        "torch",
        "streamlit",
        "typer",
    ]
    status: dict[str, bool] = {}
    for name in names:
        try:
            status[name] = importlib.util.find_spec(name) is not None
        except Exception:  # pragma: no cover
            status[name] = False
    return status


# =========================================================================== #
# shared helpers
# =========================================================================== #
def _read_catalog(path: Path) -> Any:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _try_write_report(out_dir: Path, fmt: str) -> None:
    """Best-effort report rendering after a run/demo; never fails the command."""
    catalog_path = out_dir / ("catalog.parquet" if fmt == "parquet" else "catalog.csv")
    if not catalog_path.exists():
        return
    try:
        _render_report(catalog_path, out_dir / "report", figdir=str(out_dir / "vetting_sheets"))
    except Exception:  # pragma: no cover - report is optional
        pass


def _module_missing(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is None
    except Exception:  # pragma: no cover
        return True


# =========================================================================== #
# parser
# =========================================================================== #
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="exopipe",
        description=(
            "AI detection & classification of exoplanet transits in noisy TESS "
            "light curves (BAH 2026 PS7)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"exopipe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # -- demo --------------------------------------------------------------- #
    demo = sub.add_parser(
        "demo",
        help="Offline synthetic run: population → pipeline → catalog + sheets + report.",
    )
    demo.add_argument("--n", type=int, default=12, help="Number of synthetic light curves.")
    demo.add_argument("--out", default="runs/demo", help="Output directory.")
    demo.add_argument("--seed", type=int, default=0, help="Random seed.")
    demo.add_argument("--figures", action="store_true", help="Render vetting sheets.")
    demo.add_argument("--top-k", type=int, default=12, dest="top_k", help="Figures for top-K candidates.")
    demo.add_argument("--n-jobs", type=int, default=1, dest="n_jobs", help="Parallel workers (1=serial).")
    demo.add_argument("--fmt", default="csv", choices=["csv", "parquet"], help="Catalog format.")
    demo.set_defaults(func=_cmd_demo)

    # -- run ---------------------------------------------------------------- #
    run = sub.add_parser("run", help="Process a directory of FITS or a CSV through the pipeline.")
    run.add_argument("--input", required=True, help="Path to a FITS directory or a CSV file.")
    run.add_argument("--out", default="runs/run", help="Output directory.")
    run.add_argument("--config", default=None, help="Path to a YAML config file.")
    run.add_argument("--figures", action="store_true", help="Render vetting sheets.")
    run.add_argument("--top-k", type=int, default=50, dest="top_k", help="Figures for top-K candidates.")
    run.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs", help="Parallel workers (-1=all cores).")
    run.add_argument("--fmt", default="csv", choices=["csv", "parquet"], help="Catalog format.")
    run.add_argument("--resume", action="store_true", help="Skip already-completed TICs (O(1) manifest).")
    run.set_defaults(func=_cmd_run)

    # -- fetch -------------------------------------------------------------- #
    fetch = sub.add_parser("fetch", help="Download a TESS light curve for a TIC from MAST.")
    fetch.add_argument("--tic", required=True, help="TESS Input Catalog id (e.g. 307210830).")
    fetch.add_argument("--sector", type=int, default=None, help="Restrict to one sector.")
    fetch.add_argument("--author", default="SPOC", help="Pipeline/HLSP author (SPOC/QLP/...).")
    fetch.add_argument("--out", default="data/raw", help="Output directory.")
    fetch.set_defaults(func=_cmd_fetch)

    # -- train -------------------------------------------------------------- #
    train = sub.add_parser("train", help="Train + calibrate the classifier (offline synthetic by default).")
    train.add_argument("--n", type=int, default=600, help="Training-population size.")
    train.add_argument("--seed", type=int, default=0, help="Random seed.")
    train.add_argument("--labels", default=None, help="Optional labels CSV (reserved; synthetic by default).")
    train.add_argument("--out", default="models", help="Where to save the trained model.")
    train.set_defaults(func=_cmd_train)

    # -- report ------------------------------------------------------------- #
    report = sub.add_parser("report", help="Render the ≤3-page report from a catalog.")
    report.add_argument("--catalog", required=True, help="Path to catalog.csv / catalog.parquet.")
    report.add_argument("--figdir", default=None, help="Directory of pre-rendered vetting sheets.")
    report.add_argument("--out", default="report/report", help="Output path (extension added).")
    report.set_defaults(func=_cmd_report)

    # -- dashboard ---------------------------------------------------------- #
    dashboard = sub.add_parser("dashboard", help="Launch the Streamlit catalog browser.")
    dashboard.add_argument("--catalog", default=None, help="Catalog to browse.")
    dashboard.add_argument("--figdir", default=None, help="Directory of pre-rendered vetting sheets.")
    dashboard.set_defaults(func=_cmd_dashboard)

    # -- version ------------------------------------------------------------ #
    version = sub.add_parser("version", help="Print the package version and dependency status.")
    version.set_defaults(func=_cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``exopipe`` console script and ``python -m``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
