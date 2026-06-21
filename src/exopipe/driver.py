"""Batch driver: fan ``process_lightcurve`` across a whole sector.

Implements the embarrassingly-parallel batch layer from ``ARCHITECTURE.md`` §3,
§6.2, §9.1 and ``research/05_performance_architecture.md`` §C. The pipeline is
independent per light curve, so the driver maps
:func:`exopipe.pipeline.process_lightcurve` over the inputs with
``joblib.Parallel`` (loky process backend) when available, and falls back to a
plain serial loop otherwise — same results either way, only speed differs.

O(1) levers wired in (research/05 D.3, D.6, §E4):

* **Skip-completed via a :class:`~exopipe.cache.Manifest`.** On resume the driver
  checks ``manifest.contains(tic_id)`` — an O(1) dict lookup — and skips work
  already recorded, so re-running a partially-finished sector only processes the
  delta.
* **Constant-time TIC→record bookkeeping.** Each completed candidate is recorded
  in the manifest (TIC → class/SNR/output paths) for downstream O(1) retrieval.

If an output directory is given the driver also writes the machine-readable
catalog (``exopipe.catalog.write_catalog``) and, optionally, renders vetting
sheets for the top-K candidates ranked by SNR/score.

All heavy/optional modules (``joblib``, ``exopipe.catalog``, ``exopipe.viz``) are
imported lazily inside functions, so importing this module needs only the
foundation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .cache import Manifest
from .config import Config, default_config
from .types import CandidateResult, LightCurve
from .utils import Timer, get_logger

__all__ = ["run_batch", "run_on_tics"]

_LOG = get_logger("exopipe.driver")


# --------------------------------------------------------------------------- #
# Core batch driver
# --------------------------------------------------------------------------- #
def run_batch(
    lightcurves: Sequence[LightCurve],
    config: Config | None = None,
    n_jobs: int = -1,
    models: Any | None = None,
    out_dir: str | Path | None = None,
    make_figures: bool = False,
    top_k_figures: int = 50,
    manifest: Manifest | None = None,
    resume: bool = False,
    catalog_fmt: str = "csv",
    process_fn: Callable[..., CandidateResult] | None = None,
) -> list[CandidateResult]:
    """Process many light curves in parallel and (optionally) write outputs.

    Parameters
    ----------
    lightcurves:
        Sequence of :class:`LightCurve` to process.
    config:
        Pipeline :class:`Config`; ``None`` uses :func:`default_config`.
    n_jobs:
        Worker count for ``joblib.Parallel`` (``-1`` == all cores). ``1`` forces
        the serial path. Ignored gracefully when joblib is absent.
    models:
        Optional classifier bundle forwarded to every ``process_lightcurve`` call.
    out_dir:
        If given, write ``catalog.<fmt>`` (and vetting sheets when
        ``make_figures``) under this directory; created if missing.
    make_figures:
        Render vetting sheets for the top-``top_k_figures`` candidates.
    top_k_figures:
        How many candidates (ranked by SNR then confidence) get a figure.
    manifest:
        A :class:`~exopipe.cache.Manifest` to consult/update. Created fresh when
        ``None``. Records each completed candidate (TIC → summary) for O(1) reuse.
    resume:
        When ``True``, skip any light curve whose ``tic_id`` is already in
        ``manifest`` (O(1) membership test) — idempotent restart.
    catalog_fmt:
        ``"csv"`` or ``"parquet"`` for the written catalog.
    process_fn:
        Override for the per-LC function (mainly for testing/injection). Defaults
        to :func:`exopipe.pipeline.process_lightcurve`. Must accept
        ``(lc, config=..., models=...)`` and return a ``CandidateResult``.

    Returns
    -------
    list[CandidateResult]
        One result per *processed* light curve (skipped ones are omitted), in
        input order.
    """
    cfg = config if config is not None else default_config()
    manifest = manifest if manifest is not None else Manifest()
    lightcurves = list(lightcurves or [])

    # -- O(1) skip-completed gate ------------------------------------------- #
    if resume:
        pending = [lc for lc in lightcurves if not _is_done(lc, manifest)]
        skipped = len(lightcurves) - len(pending)
        if skipped:
            _LOG.info("Resume: skipping %d already-completed light curve(s).", skipped)
    else:
        pending = lightcurves

    if not pending:
        _LOG.info("Nothing to process (0 pending light curves).")
        return []

    fn = process_fn if process_fn is not None else _default_process_fn()

    _LOG.info(
        "Processing %d light curve(s) with n_jobs=%s (backend=%s).",
        len(pending),
        n_jobs,
        cfg.perf.backend,
    )
    with Timer("run_batch", logger=_LOG):
        results = _map(fn, pending, cfg, models, n_jobs)

    # -- record completions for O(1) resume next time ---------------------- #
    for lc, result in zip(pending, results):
        if result is not None:
            _record_completion(manifest, lc, result)

    results = [r for r in results if r is not None]

    # -- outputs ------------------------------------------------------------ #
    if out_dir is not None:
        _write_outputs(
            results,
            Path(out_dir),
            make_figures=make_figures,
            top_k_figures=top_k_figures,
            catalog_fmt=catalog_fmt,
            manifest=manifest,
        )

    _log_summary(results)
    return results


def run_on_tics(
    tic_ids: Iterable[int | str],
    config: Config | None = None,
    n_jobs: int = -1,
    models: Any | None = None,
    out_dir: str | Path | None = None,
    make_figures: bool = False,
    top_k_figures: int = 50,
    sector: int | None = None,
    author: str = "SPOC",
    fallback_to_synthetic: bool = True,
    **batch_kwargs: Any,
) -> list[CandidateResult]:
    """Load light curves for ``tic_ids`` from MAST, then :func:`run_batch` them.

    Each TIC is loaded via :func:`exopipe.data.loaders.load_tess`. When a target
    cannot be fetched (no network, missing ``lightkurve``, no products) the loader
    raises :class:`~exopipe.data.loaders.DataUnavailable`; if
    ``fallback_to_synthetic`` is ``True`` (default) the driver substitutes a
    synthetic light curve **with a warning** so the run still produces output
    offline, otherwise the target is skipped.

    Parameters
    ----------
    tic_ids:
        Iterable of TESS Input Catalog identifiers.
    sector, author:
        Forwarded to :func:`load_tess`.
    fallback_to_synthetic:
        Replace unavailable targets with a synthetic stand-in (annotated in
        ``meta['synthetic_fallback']``) instead of skipping.
    **batch_kwargs:
        Passed through to :func:`run_batch`.

    Returns
    -------
    list[CandidateResult]
    """
    cfg = config if config is not None else default_config()
    lightcurves = _load_tics(
        list(tic_ids),
        sector=sector,
        author=author,
        fallback_to_synthetic=fallback_to_synthetic,
        flux_column=cfg.data.flux_column,
        quality_bitmask=cfg.data.quality_bitmask,
    )
    if not lightcurves:
        _LOG.warning("No light curves could be loaded for the given TIC ids.")
        return []
    return run_batch(
        lightcurves,
        config=cfg,
        n_jobs=n_jobs,
        models=models,
        out_dir=out_dir,
        make_figures=make_figures,
        top_k_figures=top_k_figures,
        **batch_kwargs,
    )


# --------------------------------------------------------------------------- #
# Parallel / serial map
# --------------------------------------------------------------------------- #
def _map(
    fn: Callable[..., CandidateResult],
    lightcurves: Sequence[LightCurve],
    cfg: Config,
    models: Any | None,
    n_jobs: int,
) -> list[CandidateResult | None]:
    """Map ``fn`` over ``lightcurves`` via joblib (lazy) or a serial fallback.

    Returns a list aligned with ``lightcurves`` (``None`` for a light curve whose
    processing raised — though ``process_lightcurve`` is itself crash-safe, the
    driver still guards the map so one pathological input can never abort the
    batch).
    """
    # Serial fast-path (also the fallback when joblib is unavailable).
    if n_jobs in (0, 1) or len(lightcurves) == 1:
        return [_safe_call(fn, lc, cfg, models) for lc in lightcurves]

    try:
        from joblib import Parallel, delayed  # type: ignore
    except Exception as exc:
        _LOG.info("joblib unavailable (%s); running serially.", exc)
        return [_safe_call(fn, lc, cfg, models) for lc in lightcurves]

    try:
        backend = getattr(cfg.perf, "backend", "loky")
        out = Parallel(n_jobs=n_jobs, backend=backend, batch_size="auto", verbose=0)(
            delayed(_safe_call)(fn, lc, cfg, models) for lc in lightcurves
        )
        return list(out)
    except Exception as exc:  # pragma: no cover - parallel backend hiccup
        _LOG.warning("Parallel map failed (%s); retrying serially.", exc)
        return [_safe_call(fn, lc, cfg, models) for lc in lightcurves]


def _safe_call(
    fn: Callable[..., CandidateResult],
    lc: LightCurve,
    cfg: Config,
    models: Any | None,
) -> CandidateResult | None:
    """Invoke ``fn(lc, config=cfg, models=models)`` swallowing any exception."""
    try:
        return fn(lc, config=cfg, models=models)
    except Exception as exc:  # pragma: no cover - process_lightcurve is crash-safe
        tic = (getattr(lc, "meta", {}) or {}).get("tic_id")
        _LOG.error("process_lightcurve crashed for TIC %s (%s); skipping.", tic, exc)
        return None


def _default_process_fn() -> Callable[..., CandidateResult]:
    """Lazily resolve :func:`exopipe.pipeline.process_lightcurve`.

    Imported on demand (not at module load) so importing the driver does not pull
    in the per-stage optional dependencies.
    """
    from .pipeline import process_lightcurve

    return process_lightcurve


# --------------------------------------------------------------------------- #
# Outputs: catalog + figures
# --------------------------------------------------------------------------- #
def _write_outputs(
    results: Sequence[CandidateResult],
    out_dir: Path,
    make_figures: bool,
    top_k_figures: int,
    catalog_fmt: str,
    manifest: Manifest,
) -> None:
    """Write the catalog, optional vetting sheets, and the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[int, str] = {}
    if make_figures and results:
        figure_paths = _render_top_figures(results, out_dir, top_k_figures)

    _write_catalog(results, out_dir, catalog_fmt, figure_paths)

    try:
        manifest.save(out_dir / "manifest.json")
    except Exception as exc:  # pragma: no cover - non-fatal
        _LOG.warning("Could not save manifest: %s", exc)


def _write_catalog(
    results: Sequence[CandidateResult],
    out_dir: Path,
    catalog_fmt: str,
    figure_paths: dict[int, str],
) -> None:
    """Flatten results to rows and write via ``exopipe.catalog.write_catalog``."""
    rows = []
    for i, result in enumerate(results):
        try:
            row = result.to_row()
        except Exception as exc:  # pragma: no cover - to_row is defensive
            _LOG.warning("to_row failed for result %d (%s); skipping row.", i, exc)
            continue
        if i in figure_paths:
            row["vetting_sheet_path"] = figure_paths[i]
        rows.append(row)

    if not rows:
        _LOG.warning("No catalog rows to write.")
        return

    ext = "parquet" if catalog_fmt == "parquet" else "csv"
    path = out_dir / f"catalog.{ext}"
    try:
        from . import catalog as catalog_mod

        catalog_mod.write_catalog(rows, str(path), fmt=catalog_fmt)
        _LOG.info("Wrote catalog: %s (%d rows).", path, len(rows))
    except Exception as exc:
        _LOG.warning("exopipe.catalog.write_catalog unavailable (%s); writing CSV directly.", exc)
        _write_catalog_fallback(rows, out_dir / "catalog.csv")


def _write_catalog_fallback(rows: list[dict], path: Path) -> None:
    """Minimal pandas CSV writer used when ``exopipe.catalog`` is unavailable."""
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)
        _LOG.info("Wrote fallback catalog: %s (%d rows).", path, len(rows))
    except Exception as exc:  # pragma: no cover - last-resort
        _LOG.error("Failed to write fallback catalog (%s).", exc)


def _render_top_figures(
    results: Sequence[CandidateResult],
    out_dir: Path,
    top_k: int,
) -> dict[int, str]:
    """Render vetting sheets for the top-``top_k`` candidates by SNR/confidence.

    Returns a mapping ``result_index -> figure_path`` so the catalog can link
    each row to its sheet. Figure rendering is best-effort: a failure on one
    candidate (or a missing :mod:`exopipe.viz`) never aborts the batch.
    """
    try:
        from . import viz as viz_mod
    except Exception as exc:
        _LOG.warning("exopipe.viz unavailable (%s); skipping figures.", exc)
        return {}

    fig_dir = out_dir / "vetting_sheets"
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = _rank_indices(results)[: max(int(top_k), 0)]
    paths: dict[int, str] = {}
    for idx in order:
        result = results[idx]
        tic = (getattr(result.lightcurve, "meta", {}) or {}).get("tic_id", idx)
        path = fig_dir / f"vetting_{tic}.png"
        try:
            viz_mod.vetting_sheet(result, save_path=str(path))
            paths[idx] = str(path)
        except Exception as exc:  # pragma: no cover - per-figure failure
            _LOG.warning("vetting_sheet failed for TIC %s (%s).", tic, exc)
    if paths:
        _LOG.info("Rendered %d vetting sheet(s) into %s.", len(paths), fig_dir)
    return paths


def _rank_indices(results: Sequence[CandidateResult]) -> list[int]:
    """Indices of ``results`` sorted by detection SNR then confidence, desc."""

    def key(i: int) -> tuple[float, float]:
        result = results[i]
        snr = _finite(getattr(result.detection, "snr", np.nan))
        sde = _finite(getattr(result.detection, "sde", np.nan))
        conf = _finite(getattr(result.classification, "confidence", np.nan))
        return (max(snr, sde), conf)

    return sorted(range(len(results)), key=key, reverse=True)


# --------------------------------------------------------------------------- #
# Manifest bookkeeping (O(1) skip-completed)
# --------------------------------------------------------------------------- #
def _is_done(lc: LightCurve, manifest: Manifest) -> bool:
    tic = (getattr(lc, "meta", {}) or {}).get("tic_id")
    return tic is not None and manifest.contains(tic)


def _record_completion(
    manifest: Manifest, lc: LightCurve, result: CandidateResult
) -> None:
    """Record a completed candidate in the manifest for O(1) future skipping."""
    meta = getattr(lc, "meta", {}) or {}
    tic = meta.get("tic_id")
    if tic is None:
        return
    try:
        manifest.add(
            tic,
            sector=meta.get("sector"),
            label=getattr(result.classification, "label", None),
            confidence=_finite(getattr(result.classification, "confidence", np.nan)),
            snr=_finite(getattr(result.detection, "snr", np.nan)),
            period=_finite(getattr(result.detection, "period", np.nan)),
            done=True,
        )
    except Exception:  # pragma: no cover - bookkeeping must never break the run
        pass


# --------------------------------------------------------------------------- #
# TIC loading with synthetic fallback
# --------------------------------------------------------------------------- #
def _load_tics(
    tic_ids: list[int | str],
    sector: int | None,
    author: str,
    fallback_to_synthetic: bool,
    flux_column: str,
    quality_bitmask: str,
) -> list[LightCurve]:
    from .data.loaders import DataUnavailable, load_tess

    out: list[LightCurve] = []
    for tic in tic_ids:
        try:
            loaded = load_tess(
                tic,
                sector=sector,
                author=author,
                flux_column=flux_column,
                quality_bitmask=quality_bitmask,
            )
            if isinstance(loaded, list):
                out.extend(loaded)
            else:
                out.append(loaded)
        except DataUnavailable as exc:
            if fallback_to_synthetic:
                _LOG.warning(
                    "TIC %s unavailable (%s); substituting a synthetic light curve.",
                    tic,
                    exc,
                )
                out.append(_synthetic_stub(tic, sector))
            else:
                _LOG.warning("TIC %s unavailable (%s); skipping.", tic, exc)
    return out


def _synthetic_stub(tic: int | str, sector: int | None) -> LightCurve:
    """Make a labelled synthetic light curve standing in for an unfetchable TIC."""
    from .data.synthetic import make_synthetic_lightcurve

    try:
        seed = int(tic) % (2**31 - 1)
    except (TypeError, ValueError):
        seed = abs(hash(str(tic))) % (2**31 - 1)
    lc = make_synthetic_lightcurve(kind="transit", seed=seed)
    lc.meta["tic_id"] = tic
    if sector is not None:
        lc.meta["sector"] = sector
    lc.meta["synthetic_fallback"] = True
    return lc


# --------------------------------------------------------------------------- #
# Summary logging
# --------------------------------------------------------------------------- #
def _log_summary(results: Sequence[CandidateResult]) -> None:
    if not results:
        return
    counts: dict[str, int] = {}
    for result in results:
        label = getattr(result.classification, "label", "other")
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    _LOG.info("Batch complete: %d candidate(s) [%s].", len(results), summary)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0
