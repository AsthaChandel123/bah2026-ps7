"""Command-line entry point for ``exopipe``.

This is the **foundation-stage** CLI: it wires up argument parsing and a working
``demo`` subcommand built only on numpy/scipy so ``python -m exopipe.cli demo``
runs out of the box. The full ``run`` / ``train`` pipelines are owned by the
algorithm modules and are stubbed here until those land.

Implemented with the stdlib ``argparse`` to keep the core dependency-free; the
production CLI may migrate to Typer (see the architecture notes) without
changing this command surface.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__


def _cmd_demo(args: argparse.Namespace) -> int:
    """Generate a synthetic light curve and print a short summary."""
    import numpy as np

    from .data.synthetic import make_synthetic_lightcurve
    from .utils import phase_fold

    lc = make_synthetic_lightcurve(kind=args.kind, seed=args.seed)
    print(f"exopipe {__version__} -- synthetic '{args.kind}' light curve")
    print(f"  TIC {lc.meta.get('tic_id')}  sector {lc.meta.get('sector')}  "
          f"Tmag {lc.meta.get('tmag'):.2f}")
    print(f"  cadences      : {len(lc)}")
    print(f"  time span     : {lc.time.min():.3f} -> {lc.time.max():.3f} d "
          f"({lc.time.max() - lc.time.min():.2f} d)")
    print(f"  flux dtype    : {lc.flux.dtype}   time dtype: {lc.time.dtype}")
    print(f"  median flux   : {np.nanmedian(lc.flux):.6f}")
    print(f"  robust scatter: {np.nanstd(lc.flux):.3e}")
    print(f"  label (truth) : {lc.meta.get('label')}")

    true_period = lc.meta.get("true_period")
    if true_period is not None and np.isfinite(true_period):
        phase = phase_fold(lc.time, true_period, lc.meta.get("true_t0", 0.0))
        in_tr = np.abs(phase) < (0.5 * lc.meta["true_duration"] / true_period)
        depth = float(np.nanmedian(lc.flux[~in_tr]) - np.nanmedian(lc.flux[in_tr]))
        print(f"  injected      : P={true_period:.4f} d  "
              f"depth={lc.meta.get('true_depth'):.2e}  "
              f"dur={lc.meta.get('true_duration'):.4f} d")
        print(f"  recovered dip : {depth:.2e} (in- vs out-of-transit median)")

    if args.plot:
        _try_plot(lc, args.plot)

    print("\nNote: full detection/classification/fitting pipeline is under "
          "active development.")
    return 0


def _try_plot(lc, path: str) -> None:
    """Best-effort save of a quick-look plot (no hard matplotlib dependency)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional
        print(f"  (plot skipped: matplotlib unavailable: {exc})")
        return
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(lc.time, lc.flux, ".", ms=2, alpha=0.6)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalised flux")
    ax.set_title(f"exopipe demo: {lc.meta.get('label')} (TIC {lc.meta.get('tic_id')})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved plot    : {path}")


def _cmd_run(args: argparse.Namespace) -> int:  # pragma: no cover - stub
    """Placeholder for the end-to-end pipeline (owned by algorithm modules)."""
    print(
        "exopipe run: the end-to-end pipeline (detrend -> search -> vet -> "
        "classify -> fit) is under active development and not wired up yet."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="exopipe",
        description="AI detection of exoplanet transits in noisy TESS light curves.",
    )
    parser.add_argument("--version", action="version", version=f"exopipe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate and summarise a synthetic light curve.")
    demo.add_argument(
        "--kind",
        default="transit",
        choices=["transit", "eclipsing_binary", "blend", "variable", "noise"],
        help="Type of synthetic signal to generate.",
    )
    demo.add_argument("--seed", type=int, default=1, help="Random seed.")
    demo.add_argument("--plot", default=None, help="Optional path to save a quick-look PNG.")
    demo.set_defaults(func=_cmd_demo)

    run = sub.add_parser("run", help="Run the full pipeline (under construction).")
    run.add_argument("--config", default=None, help="Path to a YAML config file.")
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``exopipe`` console script and ``python -m``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
