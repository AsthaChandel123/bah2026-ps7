"""≤3-page methodology + results report generation for ``exopipe``.

PS7 requirement R8: a **≤3-page report** covering methodology, assumptions,
tools/libraries, and uncertainty estimation. This module renders that report,
mapped 1:1 onto the rubric sections (Objective, Data, Methodology, Assumptions,
Uncertainty estimation, Results, Visualization).

Rendering strategy (graceful degradation)
------------------------------------------
1. Fill the placeholders in :data:`TEMPLATE_PATH` (``report/template.md``) from
   the run catalog + metadata to produce a complete markdown document.
2. If **Quarto** or **Pandoc** is on ``PATH``, render that markdown to a PDF
   (`quarto render` / `pandoc --pdf-engine=...`).
3. Otherwise, fall back to a **matplotlib ``PdfPages``** renderer that lays out a
   text/summary page plus embedded example vetting-sheet figures. This path
   needs only core dependencies, so **a PDF is always produced** even with no
   LaTeX / Quarto / pandoc installed.

The output is kept to ≤3 pages (tight geometry + 9pt font in the Quarto header;
a 1-page text summary + ≤2 figure pages in the fallback).

Public API
----------
``generate_report(results_or_catalog, output_path='report/report.pdf',
                  figures=None, run_meta=None) -> str``
    Returns the path actually written (a ``.pdf`` when rendering succeeds, or a
    ``.md`` if every PDF backend including matplotlib is unavailable).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import catalog as _catalog
from .catalog import CATALOG_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = ["generate_report", "TEMPLATE_PATH"]

# Location of the markdown/Quarto template (report/template.md at the repo root).
TEMPLATE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "report", "template.md")
)

# Human-friendly class display order/names.
_CLASS_ORDER = ("transit", "eclipsing_binary", "blend", "other")
_CLASS_DISPLAY = {
    "transit": "Transit",
    "eclipsing_binary": "Eclipsing binary",
    "blend": "Blend",
    "other": "Other",
}


# --------------------------------------------------------------------------- #
# Input normalisation
# --------------------------------------------------------------------------- #
def _to_dataframe(results_or_catalog: Any) -> tuple[pd.DataFrame, list[Any]]:
    """Coerce the input to ``(catalog_df, candidate_results)``.

    Accepts:
    * a :class:`pandas.DataFrame` (already a catalog) -> no candidate objects;
    * a path to a catalog file -> read via :func:`exopipe.catalog.read_catalog`;
    * an iterable of :class:`~exopipe.types.CandidateResult` (have ``to_row``) ->
      build a catalog DataFrame and keep the objects (for example figures).
    """
    if isinstance(results_or_catalog, pd.DataFrame):
        return results_or_catalog.copy(), []

    if isinstance(results_or_catalog, (str, os.PathLike)):
        return _catalog.read_catalog(str(results_or_catalog)), []

    # Assume an iterable of CandidateResult.
    results = list(results_or_catalog)
    if results and hasattr(results[0], "to_row"):
        rows = [_catalog.result_to_row(r) for r in results]
        df = pd.DataFrame.from_records(rows)
        for col in CATALOG_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df, results
    # Maybe an iterable of row dicts.
    if results and isinstance(results[0], dict):
        return pd.DataFrame.from_records(results), []
    return pd.DataFrame(), list(results)


# --------------------------------------------------------------------------- #
# Section content builders
# --------------------------------------------------------------------------- #
def _fmt_int(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_float(value: Any, fmt: str = "{:.4g}") -> str:
    try:
        v = float(value)
        return fmt.format(v) if np.isfinite(v) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class candidate counts as a small DataFrame (always all 4 classes)."""
    counts = {c: 0 for c in _CLASS_ORDER}
    if "class" in df.columns and len(df):
        observed = df["class"].astype(str).value_counts().to_dict()
        for key, value in observed.items():
            counts[key] = counts.get(key, 0) + int(value)
    out = pd.DataFrame(
        {
            "Class": [_CLASS_DISPLAY.get(c, c) for c in counts],
            "Count": list(counts.values()),
        }
    )
    return out


def _top_candidates(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Top-``n`` candidates by significance, as a compact display table."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    sort_key = "snr" if "snr" in work.columns else (
        "sde" if "sde" in work.columns else None
    )
    if sort_key is not None:
        work = work.sort_values(sort_key, ascending=False, na_position="last")
    work = work.head(n)

    table = pd.DataFrame()
    table["TIC"] = work.get("tic_id", pd.Series(dtype=object)).map(_fmt_int)
    table["Class"] = work.get("class", pd.Series(dtype=object)).astype(str)
    table["Conf"] = work.get("confidence", pd.Series(dtype=float)).map(
        lambda v: _fmt_float(v, "{:.2f}")
    )
    table["P [d]"] = work.get("period", pd.Series(dtype=float)).map(
        lambda v: _fmt_float(v, "{:.4f}")
    )
    depth_ppm = work.get("depth_ppm")
    if depth_ppm is None and "depth" in work.columns:
        depth_ppm = work["depth"] * 1e6
    table["Depth [ppm]"] = (
        depth_ppm.map(lambda v: _fmt_float(v, "{:.0f}"))
        if depth_ppm is not None else "n/a"
    )
    dur = work.get("duration")
    table["Dur [h]"] = (
        (dur * 24.0).map(lambda v: _fmt_float(v, "{:.2f}"))
        if dur is not None else "n/a"
    )
    table["SNR"] = work.get("snr", pd.Series(dtype=float)).map(
        lambda v: _fmt_float(v, "{:.1f}")
    )
    table["SDE"] = work.get("sde", pd.Series(dtype=float)).map(
        lambda v: _fmt_float(v, "{:.1f}")
    )
    return table.reset_index(drop=True)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub-flavoured markdown table (pandas, with a manual fallback)."""
    if df is None or df.empty:
        return "_No candidates._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        # tabulate not installed: hand-roll a simple pipe table.
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [header, sep]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)


def _results_summary_text(df: pd.DataFrame, run_meta: dict | None) -> str:
    """One-paragraph numeric summary of the run."""
    n = len(df)
    n_transit = int((df["class"].astype(str) == "transit").sum()) if "class" in df else 0
    median_conf = (
        _fmt_float(df["confidence"].median(), "{:.2f}") if "confidence" in df and n else "n/a"
    )
    median_snr = (
        _fmt_float(df["snr"].median(), "{:.1f}") if "snr" in df and n else "n/a"
    )
    parts = [
        f"{n} candidate(s) were processed, of which {n_transit} were classified "
        f"as planetary transits.",
        f"Median calibrated confidence {median_conf}; median detection SNR "
        f"{median_snr}.",
    ]
    if run_meta:
        extra = run_meta.get("results_note")
        if extra:
            parts.append(str(extra))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Figure handling
# --------------------------------------------------------------------------- #
def _collect_figures(
    figures: Sequence[str] | None,
    results: Sequence[Any],
    workdir: str,
    max_figs: int = 2,
) -> list[str]:
    """Resolve example vetting-sheet PNG paths, rendering from results if needed.

    Returns a list of existing PNG file paths (≤ ``max_figs``). If explicit
    ``figures`` paths are given they win; otherwise up to ``max_figs`` sheets are
    rendered from the supplied :class:`~exopipe.types.CandidateResult` objects
    into ``workdir`` using :func:`exopipe.viz.vetting_sheet`.
    """
    if figures:
        return [f for f in figures if os.path.exists(f)][:max_figs]

    if not results:
        return []

    # Prefer the highest-significance candidates for the example figures.
    def _score(result: Any) -> float:
        try:
            row = result.to_row()
            value = row.get("snr")
            return float(value) if value is not None and np.isfinite(float(value)) else -1.0
        except Exception:
            return -1.0

    ordered = sorted(results, key=_score, reverse=True)[:max_figs]

    try:
        from .viz import vetting_sheet
    except Exception as exc:  # pragma: no cover - viz import should succeed
        warnings.warn(
            f"could not import viz.vetting_sheet ({exc}); skipping figures.",
            stacklevel=2,
        )
        return []

    paths: list[str] = []
    import matplotlib.pyplot as plt

    for i, result in enumerate(ordered):
        out = os.path.join(workdir, f"example_sheet_{i}.png")
        try:
            fig = vetting_sheet(result, save_path=out, dpi=120)
            plt.close(fig)
            if os.path.exists(out):
                paths.append(out)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"failed to render example sheet {i}: {exc}", stacklevel=2)
    return paths


# --------------------------------------------------------------------------- #
# Markdown assembly
# --------------------------------------------------------------------------- #
def _read_template() -> str:
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, encoding="utf-8") as handle:
            return handle.read()
    # Minimal inline fallback if the template file is missing.
    return (
        "---\ntitle: exopipe report\n---\n\n"
        "# Objective\n{{OBJECTIVE_NOTES}}\n\n# Data\n{{DATA_SUMMARY}}\n\n"
        "# Methodology\n{{METHODOLOGY_METHODS}}\n\n# Assumptions\n{{ASSUMPTIONS_NOTES}}\n\n"
        "# Uncertainty estimation\n{{UNCERTAINTY_NOTES}}\n\n# Results\n"
        "{{RESULTS_SUMMARY}}\n\n{{CLASS_COUNT_TABLE}}\n\n{{TOP_CANDIDATE_TABLE}}\n\n"
        "{{EXAMPLE_FIGURES}}\n\n# Visualization\n{{VISUALIZATION_FIGURE}}\n"
    )


def _fill_template(
    df: pd.DataFrame,
    figure_paths: Sequence[str],
    run_meta: dict | None,
) -> str:
    """Substitute the ``{{PLACEHOLDER}}`` tokens in the template with content."""
    run_meta = run_meta or {}
    template = _read_template()

    data_summary = run_meta.get(
        "data_summary",
        f"{len(df)} candidate light curves "
        f"({'synthetic' if run_meta.get('synthetic', True) else 'observed'}).",
    )

    fig_md_parts = []
    for i, path in enumerate(figure_paths):
        cap = "Representative one-page vetting sheet." if i == 0 else (
            "Additional example vetting sheet."
        )
        fig_md_parts.append(
            f"![{cap}]({os.path.abspath(path)}){{width=95%}}\n"
        )
    figures_md = "\n".join(fig_md_parts) if fig_md_parts else "_No example figures available._"
    viz_md = (
        f"![One-page vetting sheet — the headline visualisation.]"
        f"({os.path.abspath(figure_paths[0])}){{width=95%}}"
        if figure_paths else "_Vetting sheet figure rendered per candidate at run time._"
    )

    import datetime as _dt

    substitutions = {
        "{{DATE}}": run_meta.get("date", _dt.date.today().isoformat()),
        "{{OBJECTIVE_NOTES}}": run_meta.get("objective_notes", ""),
        "{{DATA_SUMMARY}}": data_summary,
        "{{METHODOLOGY_METHODS}}": run_meta.get("methodology_notes", ""),
        "{{ASSUMPTIONS_NOTES}}": run_meta.get("assumptions_notes", ""),
        "{{UNCERTAINTY_NOTES}}": run_meta.get("uncertainty_notes", ""),
        "{{RESULTS_SUMMARY}}": _results_summary_text(df, run_meta),
        "{{CLASS_COUNT_TABLE}}": _df_to_markdown(_class_counts(df)),
        "{{TOP_CANDIDATE_TABLE}}": _df_to_markdown(_top_candidates(df)),
        "{{EXAMPLE_FIGURES}}": figures_md,
        "{{VISUALIZATION_FIGURE}}": viz_md,
    }
    for token, value in substitutions.items():
        template = template.replace(token, str(value))
    return template


# --------------------------------------------------------------------------- #
# PDF backends
# --------------------------------------------------------------------------- #
def _render_with_quarto(markdown_text: str, output_pdf: str) -> bool:
    """Render via ``quarto render`` if available. Returns success."""
    if shutil.which("quarto") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        qmd = os.path.join(tmp, "report.qmd")
        with open(qmd, "w", encoding="utf-8") as handle:
            handle.write(markdown_text)
        try:
            subprocess.run(
                ["quarto", "render", qmd, "--to", "pdf", "--output", os.path.basename(output_pdf)],
                cwd=tmp, check=True, capture_output=True, timeout=300,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        produced = os.path.join(tmp, os.path.basename(output_pdf))
        if os.path.exists(produced):
            shutil.copyfile(produced, output_pdf)
            return True
    return False


def _render_with_pandoc(markdown_text: str, output_pdf: str) -> bool:
    """Render via ``pandoc`` (needs a LaTeX engine) if available. Returns success."""
    if shutil.which("pandoc") is None:
        return False
    engine = next(
        (e for e in ("tectonic", "xelatex", "pdflatex", "lualatex") if shutil.which(e)),
        None,
    )
    if engine is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        md = os.path.join(tmp, "report.md")
        with open(md, "w", encoding="utf-8") as handle:
            handle.write(markdown_text)
        try:
            subprocess.run(
                ["pandoc", md, "-o", output_pdf, f"--pdf-engine={engine}"],
                check=True, capture_output=True, timeout=300,
            )
        except (subprocess.SubprocessError, OSError):
            return False
    return os.path.exists(output_pdf)


def _render_with_matplotlib(
    df: pd.DataFrame,
    figure_paths: Sequence[str],
    output_pdf: str,
    run_meta: dict | None,
) -> bool:
    """Always-available PDF: a text/summary page + embedded vetting sheets.

    Uses matplotlib's :class:`~matplotlib.backends.backend_pdf.PdfPages`, so it
    works with only core dependencies. Limited to ≤3 pages (1 summary + ≤2
    figures).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    run_meta = run_meta or {}
    try:
        with PdfPages(output_pdf) as pdf:
            # ---- Page 1: title + methodology + results summary ------------- #
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.clf()
            y = 0.96

            def line(text: str, size: float = 9.0, weight: str = "normal",
                     dy: float = 0.026, color: str = "k", family: str = "sans-serif") -> None:
                nonlocal y
                fig.text(0.06, y, text, fontsize=size, fontweight=weight, color=color,
                         family=family, va="top", wrap=True)
                y -= dy

            line("exopipe — AI Detection & Classification of Exoplanet Transits in TESS",
                 size=14, weight="bold", dy=0.030)
            line("BAH 2026 — Problem Statement 7   (auto-generated report)",
                 size=9, color="0.35", dy=0.040)

            line("1. Objective", size=11, weight="bold", dy=0.024)
            for txt in _wrap(
                "Detect periodic transit-like dips in noisy TESS light curves; classify each "
                "signal as transit / eclipsing_binary / blend / other; estimate period, depth "
                "and duration with calibrated confidence and detection significance.", 105):
                line(txt, dy=0.020)
            y -= 0.006

            line("2. Data", size=11, weight="bold", dy=0.024)
            data_summary = run_meta.get(
                "data_summary",
                f"{len(df)} candidate light curves "
                f"({'synthetic' if run_meta.get('synthetic', True) else 'observed'}).")
            for txt in _wrap(
                "TESS 2-min SPOC light curves (~20–30k stars/sector); labels from Kepler/K2, "
                "NASA Exoplanet Archive, ExoFOP and EB catalogs; Gaia DR3 + TIC for stellar "
                f"params/blends. This run: {data_summary} Offline synthetic generator enables "
                "zero-network reproduction.", 105):
                line(txt, dy=0.020)
            y -= 0.006

            line("3. Methodology (tools/libraries named)", size=11, weight="bold", dy=0.024)
            for txt in [
                "• Detrend: biweight (wotan) / Savitzky–Golay (scipy) / sigma-clip (astropy).",
                "• Search: BLS (astropy) triage → TLS (transitleastsquares) SDE confirm.",
                "• Significance: transit_snr, CDPP, bootstrap/GEV FAP, SDE.",
                "• Vetting: 15 physics tests (odd/even, secondary, V-shape, centroid, CROWDSAP…).",
                "• Classify: rules + XGBoost/LightGBM/RF (isotonic) + optional CNN; physics veto.",
                "• Fit: trapezoid/LM seed → batman + emcee (16/50/84 CIs, ΔBIC).",
            ]:
                line(txt, dy=0.020)
            y -= 0.006

            line("4. Assumptions", size=11, weight="bold", dy=0.024)
            for txt in _wrap(
                "One dominant signal per pass; PDCSAP systematics mostly pre-corrected; "
                "Mandel–Agol/trapezoid model adequate (grazing → upper limits); detection at "
                "SDE≳7–9; training labels trusted and reconciled by precedence.", 105):
                line(txt, dy=0.020)
            y -= 0.006

            line("5. Uncertainty estimation", size=11, weight="bold", dy=0.024)
            for txt in _wrap(
                "Parameter 16/50/84 credible intervals (asymmetric ±, red-noise-inflated); "
                "detection SNR/SDE + bootstrap FAP; calibrated class probabilities (isotonic / "
                "temperature) validated by a reliability diagram and injection–recovery.", 105):
                line(txt, dy=0.020)
            y -= 0.010

            line("6. Results", size=11, weight="bold", dy=0.024)
            for txt in _wrap(_results_summary_text(df, run_meta), 105):
                line(txt, dy=0.020)

            # Per-class counts + top-candidate tables as matplotlib tables.
            _draw_table(fig, _class_counts(df), bbox=(0.06, 0.20, 0.40, 0.12),
                        title="Per-class counts")
            _draw_table(fig, _top_candidates(df, n=6), bbox=(0.06, 0.03, 0.88, 0.15),
                        title="Top candidates (by SNR)")

            pdf.savefig(fig)
            plt.close(fig)

            # ---- Pages 2..N: embedded example vetting sheets --------------- #
            for path in list(figure_paths)[:2]:
                if not os.path.exists(path):
                    continue
                img = plt.imread(path)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.04, 0.04, 0.92, 0.90])
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Visualization — one-page vetting sheet",
                             fontsize=11, fontweight="bold")
                pdf.savefig(fig)
                plt.close(fig)

            info = pdf.infodict()
            info["Title"] = "exopipe report — BAH 2026 PS7"
            info["Author"] = "Team exopipe"
        return os.path.exists(output_pdf)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"matplotlib PDF fallback failed: {exc}", stacklevel=2)
        return False


def _wrap(text: str, width: int) -> list[str]:
    """Simple greedy word-wrap (avoids a textwrap import surprise on long URLs)."""
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _draw_table(fig, df: pd.DataFrame, bbox: tuple[float, float, float, float],
                title: str) -> None:
    """Render a small DataFrame as a matplotlib table inside ``bbox``."""
    x, y, w, h = bbox
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    if df is None or df.empty:
        ax.text(0, 0.5, "No candidates.", fontsize=8, style="italic", color="0.5")
        return
    tbl = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.2)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_report(
    results_or_catalog: Any,
    output_path: str = "report/report.pdf",
    figures: Sequence[str] | None = None,
    run_meta: dict | None = None,
) -> str:
    """Generate the ≤3-page methodology + results report.

    Parameters
    ----------
    results_or_catalog:
        Either an iterable of :class:`~exopipe.types.CandidateResult`, a
        :class:`pandas.DataFrame` catalog, or a path to a catalog file
        (CSV/JSON/Parquet).
    output_path:
        Desired output path (``.pdf``). The parent directory is created. If no
        PDF backend (Quarto / pandoc+LaTeX / matplotlib) succeeds, a ``.md`` is
        written at the same stem and that path is returned instead.
    figures:
        Optional explicit list of vetting-sheet PNG paths to embed. When omitted,
        up to two example sheets are rendered from the candidate results (if
        ``CandidateResult`` objects were supplied).
    run_meta:
        Optional dict of overrides/notes for the template (e.g. ``data_summary``,
        ``methodology_notes``, ``synthetic``, ``date``).

    Returns
    -------
    str
        The path actually written (``.pdf`` on success, else ``.md``).
    """
    run_meta = dict(run_meta or {})
    df, results = _to_dataframe(results_or_catalog)

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Render example figures into a persistent ``<stem>_figures/`` directory next
    # to the output (not a temp dir): the matplotlib fallback embeds them into the
    # PDF, but the markdown sidecar -- and the Quarto/pandoc path -- reference the
    # files directly, so they must survive past this call.
    stem = os.path.splitext(os.path.abspath(output_path))[0]
    figdir = f"{stem}_figures"
    os.makedirs(figdir, exist_ok=True)

    figure_paths = _collect_figures(figures, results, figdir, max_figs=2)
    markdown_text = _fill_template(df, figure_paths, run_meta)

    # Always write the markdown next to the output for provenance / fallback.
    md_path = stem + ".md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)

    # Try PDF backends in order of fidelity.
    if output_path.lower().endswith(".pdf"):
        for backend in (_render_with_quarto, _render_with_pandoc):
            try:
                if backend(markdown_text, output_path):
                    return output_path
            except Exception:  # pragma: no cover - defensive
                continue
        # Always-available matplotlib fallback.
        if _render_with_matplotlib(df, figure_paths, output_path, run_meta):
            return output_path
        # Last resort: the markdown we already wrote.
        warnings.warn(
            "No PDF backend (quarto / pandoc+LaTeX / matplotlib) succeeded; "
            f"wrote markdown to {md_path!r} instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return md_path

    # Non-PDF requested output: just return the markdown path.
    if output_path.lower().endswith((".md", ".qmd", ".markdown")):
        if os.path.abspath(md_path) != os.path.abspath(output_path):
            shutil.copyfile(md_path, output_path)
        return output_path
    return md_path
