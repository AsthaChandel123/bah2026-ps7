"""exopipe — interactive candidate browser (Streamlit dashboard).

Browse the full per-sector candidate catalog (20–30k rows) and drill into each
candidate's one-page vetting sheet. The dashboard reads **only precomputed
artifacts** — a catalog (CSV / JSON / Parquet) and a directory of per-candidate
vetting-sheet PNGs — which is what keeps it responsive at sector scale (the slow
pipeline is fully decoupled from the instant UI).

Run
---
::

    streamlit run app/dashboard.py -- --catalog <path> --figdir <dir>

``--catalog``  Path to the catalog written by ``exopipe.catalog.write_catalog``
               (``.csv`` / ``.json`` / ``.parquet``).
``--figdir``   Directory containing per-candidate vetting-sheet PNGs. By default
               a file named ``TIC<tic_id>.png`` is looked up; the ``flags`` /
               ``vetting_sheet_path`` columns are honoured when present.

Graceful degradation
---------------------
If Streamlit is not installed the module still **imports cleanly** (so tests and
``python app/dashboard.py`` do not crash) and prints install + usage
instructions instead of launching.
"""

from __future__ import annotations

import argparse
import os
import sys

# --------------------------------------------------------------------------- #
# Optional Streamlit import (graceful fallback) -- must be at the top of file.
# --------------------------------------------------------------------------- #
try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover - exercised only without streamlit
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False

# Okabe--Ito colour-blind-safe class palette (consistent with viz / report).
CLASS_COLORS = {
    "transit": "#0072B2",
    "eclipsing_binary": "#D55E00",
    "blend": "#CC79A7",
    "other": "#999999",
}

# Defaults point at the bundled example results so a freshly deployed container
# (or a bare ``streamlit run app/dashboard.py`` with no ``--`` args) immediately
# renders a populated dashboard. Override via ``-- --catalog <p> --figdir <d>``.
_DEFAULT_CATALOG = "examples/example_catalog.csv"
_DEFAULT_FIGDIR = "examples"


# --------------------------------------------------------------------------- #
# Argument parsing (Streamlit passes user args after a literal ``--``)
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--catalog`` / ``--figdir`` from the post-``--`` argument list."""
    parser = argparse.ArgumentParser(
        prog="exopipe-dashboard",
        description="Interactive exoplanet-candidate browser.",
        add_help=False,
    )
    parser.add_argument("--catalog", default=_DEFAULT_CATALOG,
                        help="Path to the candidate catalog (csv/json/parquet).")
    parser.add_argument("--figdir", default=_DEFAULT_FIGDIR,
                        help="Directory of per-candidate vetting-sheet PNGs.")
    args, _unknown = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    return args


# --------------------------------------------------------------------------- #
# Data access (cached when Streamlit is present)
# --------------------------------------------------------------------------- #
def _load_catalog(path: str):
    """Load a catalog DataFrame using :func:`exopipe.catalog.read_catalog`.

    Falls back to a bare pandas reader if the package is not importable (so the
    dashboard works even from a loose checkout).
    """
    try:
        from exopipe.catalog import read_catalog

        return read_catalog(path)
    except Exception:
        import pandas as pd

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".json":
            return pd.read_json(path, orient="records")
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        raise


if _HAS_STREAMLIT:
    # Decorate with Streamlit's data cache so 20–30k-row catalogs load once.
    load_catalog = st.cache_data(show_spinner=False)(_load_catalog)
else:  # pragma: no cover - import-time only without streamlit
    load_catalog = _load_catalog


def _figure_path(figdir: str, row) -> str | None:
    """Resolve a candidate's vetting-sheet PNG path from the row + figdir."""
    # Honour an explicit path column if the catalog carries one.
    for col in ("vetting_sheet_path", "sheet_path"):
        value = row.get(col) if hasattr(row, "get") else None
        if value and isinstance(value, str) and os.path.exists(value):
            return value
    tic = row.get("tic_id") if hasattr(row, "get") else None
    if tic is None:
        return None
    try:
        tic_int = int(tic)
    except (TypeError, ValueError):
        tic_int = tic
    for name in (f"TIC{tic_int}.png", f"TIC_{tic_int}.png", f"{tic_int}.png"):
        candidate = os.path.join(figdir, name)
        if os.path.exists(candidate):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# The Streamlit app
# --------------------------------------------------------------------------- #
def run_app(argv: list[str] | None = None) -> None:
    """Render the Streamlit dashboard. No-op (with guidance) if Streamlit absent."""
    if not _HAS_STREAMLIT:
        _print_install_instructions()
        return

    import pandas as pd

    args = _parse_args(argv)
    st.set_page_config(page_title="exopipe — Candidate Browser", layout="wide")
    st.title("exopipe — Exoplanet Candidate Browser")
    st.caption("BAH 2026 PS7 · browse detected/classified TESS candidates and their vetting sheets.")

    # ---- Load catalog ----------------------------------------------------- #
    if not os.path.exists(args.catalog):
        st.error(f"Catalog not found: `{args.catalog}`")
        st.info(
            "Pass a catalog with `-- --catalog <path> --figdir <dir>`. "
            "Generate one via `exopipe.catalog.write_catalog(...)`."
        )
        return
    try:
        df = load_catalog(args.catalog)
    except Exception as exc:  # pragma: no cover - runtime UI guard
        st.error(f"Failed to read catalog `{args.catalog}`: {exc}")
        return
    if df is None or len(df) == 0:
        st.warning("Catalog is empty.")
        return

    # ---- Sidebar filters -------------------------------------------------- #
    st.sidebar.header("Filters")

    classes_present = (
        sorted(df["class"].dropna().astype(str).unique()) if "class" in df.columns else []
    )
    selected_classes = st.sidebar.multiselect(
        "Class", classes_present, default=classes_present
    ) if classes_present else []

    mask = pd.Series(True, index=df.index)
    if selected_classes:
        mask &= df["class"].astype(str).isin(selected_classes)

    mask = _slider_filter(st, df, mask, "confidence", "Min confidence", lower_only=True)
    mask = _slider_filter(st, df, mask, "snr", "SNR range")
    mask = _slider_filter(st, df, mask, "sde", "SDE range")

    tic_query = st.sidebar.text_input("Search TIC")
    if tic_query and "tic_id" in df.columns:
        mask &= df["tic_id"].astype(str).str.contains(tic_query, na=False)

    fdf = df[mask].copy()
    sort_col = "snr" if "snr" in fdf.columns else ("sde" if "sde" in fdf.columns else None)
    if sort_col:
        fdf = fdf.sort_values(sort_col, ascending=False, na_position="last")

    st.sidebar.download_button(
        "Download filtered CSV",
        fdf.to_csv(index=False).encode("utf-8"),
        file_name="candidates_filtered.csv",
        mime="text/csv",
    )
    st.sidebar.caption(f"{len(fdf):,} of {len(df):,} candidates match.")

    # ---- Main table ------------------------------------------------------- #
    display_cols = [
        c for c in (
            "tic_id", "sector", "period", "depth_ppm", "duration",
            "snr", "sde", "class", "confidence", "flags",
        ) if c in fdf.columns
    ]
    column_config = {}
    if "confidence" in display_cols:
        column_config["confidence"] = st.column_config.ProgressColumn(
            "confidence", min_value=0.0, max_value=1.0, format="%.2f"
        )

    st.subheader("Candidates")
    try:
        event = st.dataframe(
            fdf[display_cols],
            use_container_width=True,
            height=420,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=column_config,
        )
        selected_rows = event.selection.rows if hasattr(event, "selection") else []
    except TypeError:  # pragma: no cover - older Streamlit without on_select
        st.dataframe(fdf[display_cols], use_container_width=True, height=420)
        selected_rows = []

    # ---- Drill-down ------------------------------------------------------- #
    if selected_rows:
        row = fdf.iloc[selected_rows[0]]
        _render_detail(st, row, args.figdir)
    else:
        st.info("Select a row to view its vetting sheet and parameter detail.")


def _slider_filter(st_mod, df, mask, col, label, lower_only=False):
    """Add a numeric sidebar slider for ``col`` and AND it into ``mask``."""
    import pandas as pd

    if col not in df.columns:
        return mask
    series = pd.to_numeric(df[col], errors="coerce")
    if not series.notna().any():
        return mask
    lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        return mask
    if lower_only:
        thresh = st_mod.sidebar.slider(label, lo, hi, lo)
        return mask & (series >= thresh)
    sel_lo, sel_hi = st_mod.sidebar.slider(label, lo, hi, (lo, hi))
    return mask & series.between(sel_lo, sel_hi)


def _render_detail(st_mod, row, figdir: str) -> None:
    """Render the per-candidate drill-down: metrics, vetting sheet, probabilities."""
    label = str(row.get("class", "n/a"))
    color = CLASS_COLORS.get(label, "#333333")
    tic = row.get("tic_id", "n/a")
    conf = row.get("confidence", float("nan"))
    st_mod.markdown(
        f"<h2 style='color:{color}'>TIC {tic} — {label.upper()} "
        f"(confidence {_fmt(conf, '{:.2f}')})</h2>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st_mod.columns(4)
    period = row.get("period")
    depth_ppm = row.get("depth_ppm")
    if depth_ppm is None and row.get("depth") is not None:
        try:
            depth_ppm = float(row.get("depth")) * 1e6
        except (TypeError, ValueError):
            depth_ppm = None
    duration = row.get("duration")
    c1.metric("Period [d]", _fmt(period, "{:.5f}"))
    c2.metric("Depth [ppm]", _fmt(depth_ppm, "{:.0f}"))
    c3.metric("Duration [h]", _fmt(duration * 24.0 if _isnum(duration) else None, "{:.2f}"))
    c4.metric("SNR / SDE", f"{_fmt(row.get('snr'), '{:.1f}')} / {_fmt(row.get('sde'), '{:.1f}')}")

    left, right = st_mod.columns([1.5, 1])
    with left:
        fig_path = _figure_path(figdir, row)
        if fig_path:
            st_mod.image(fig_path, caption="Vetting sheet", use_container_width=True)
        else:
            st_mod.warning(
                f"No vetting-sheet PNG found for TIC {tic} in `{figdir}` "
                "(expected `TIC<tic_id>.png`)."
            )
    with right:
        st_mod.subheader("Class probabilities")
        probs = {}
        for cls in CLASS_COLORS:
            value = row.get(f"prob_{cls}")
            if _isnum(value):
                probs[cls] = float(value)
        if probs:
            import pandas as pd

            st_mod.bar_chart(pd.Series(probs))
        else:
            st_mod.caption("No probability columns in catalog.")

        st_mod.subheader("Vetting flags")
        flag_cols = {k: bool(row[k]) for k in row.index if str(k).startswith("flag_")}
        if "flags" in row.index and isinstance(row["flags"], str) and row["flags"]:
            st_mod.write("Raised: " + ", ".join(row["flags"].split(";")))
        if flag_cols:
            st_mod.json(flag_cols)


# --------------------------------------------------------------------------- #
# Small formatting helpers (no Streamlit dependency)
# --------------------------------------------------------------------------- #
def _isnum(value) -> bool:
    try:
        import math

        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value, fmt: str = "{:.3g}") -> str:
    return fmt.format(float(value)) if _isnum(value) else "n/a"


def _print_install_instructions() -> None:
    """Tell the user how to install Streamlit and run the dashboard."""
    print(
        "exopipe dashboard\n"
        "-----------------\n"
        "Streamlit is not installed, so the interactive dashboard cannot launch.\n\n"
        "Install it with:\n"
        "    pip install streamlit\n\n"
        "Then run:\n"
        "    streamlit run app/dashboard.py -- --catalog <path> --figdir <dir>\n\n"
        "The dashboard reads a precomputed catalog (csv/json/parquet) and a\n"
        "directory of per-candidate vetting-sheet PNGs."
    )


# When Streamlit executes this file (``streamlit run``) it runs the module body,
# so we launch the app here; under a plain ``python`` invocation without
# Streamlit we print guidance instead of crashing.
if _HAS_STREAMLIT and (st.runtime.exists() if hasattr(st, "runtime") else True):
    run_app()
elif __name__ == "__main__":
    _print_install_instructions()
