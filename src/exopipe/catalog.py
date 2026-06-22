"""Machine-readable candidate catalog I/O for ``exopipe``.

PS7 requires a **machine-readable catalog** (requirement R11) mirroring the
TOI / ExoFOP schema so evaluators recognise it. This module turns
:class:`~exopipe.types.CandidateResult` objects into flat rows and writes them
to CSV / JSON / Parquet (one row per candidate), plus reads them back.

The canonical column order is :data:`CATALOG_COLUMNS`. It is built to **align
exactly** with the keys emitted by ``CandidateResult.to_row()`` (the foundation
contract in :mod:`exopipe.types`) and then extends it with a few convenience
columns the dossier asks for:

* ``depth_ppm`` -- the fractional ``depth`` re-expressed in parts-per-million
  (TOI-style), so the human-readable catalog needs no unit conversion.
* ``prob_transit`` / ``prob_eclipsing_binary`` / ``prob_blend`` / ``prob_other``
  -- the four calibrated class probabilities (already produced by ``to_row``;
  pinned here so the column is always present and ordered).
* ``flags`` -- a single ``;``-joined string of the vetting flags that are set,
  for a compact human-readable summary alongside the per-flag booleans.

Design notes
------------
* **Core-deps only by default.** CSV and JSON use pandas (always available).
  Parquet imports ``pyarrow`` *lazily* and, if it is missing, warns and falls
  back to writing CSV at the same stem -- so a run never crashes for lack of an
  optional dependency.
* The DataFrame is always materialised in :data:`CATALOG_COLUMNS` order; any
  column a particular row is missing is filled with NaN (numeric) / ``''``
  (string) so every catalog has a stable, predictable schema.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import CandidateResult

__all__ = [
    "CATALOG_COLUMNS",
    "result_to_row",
    "write_catalog",
    "read_catalog",
]

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
# These are exactly the keys produced by ``CandidateResult.to_row()`` (see
# exopipe/types.py), in a TOI/ExoFOP-flavoured order, followed by the extra
# convenience columns this module adds (depth_ppm + the four prob_* + flags).
#
# Keeping this list in lock-step with ``to_row()`` means ``result_to_row`` only
# has to *augment* the row, never reshape it -- and a catalog written from raw
# ``to_row()`` dicts is still valid (missing extras are filled).
CATALOG_COLUMNS: list[str] = [
    # --- identifiers -------------------------------------------------------- #
    "tic_id",
    "sector",
    # --- ephemeris / parameters (fractional depth + duration in DAYS, from
    #     to_row(); fit-first with detection fallback) ----------------------- #
    "period",
    "period_err_lo",
    "period_err_hi",
    "t0",
    "t0_err_lo",
    "t0_err_hi",
    "duration",
    "duration_err_lo",
    "duration_err_hi",
    "depth",
    "depth_err_lo",
    "depth_err_hi",
    # --- convenience: depth in ppm (added by result_to_row) ----------------- #
    "depth_ppm",
    # --- significance ------------------------------------------------------- #
    "snr",
    "sde",
    "detection_method",
    # --- classification + calibrated confidence ----------------------------- #
    "class",
    "confidence",
    "classify_method",
    "prob_transit",
    "prob_eclipsing_binary",
    "prob_blend",
    "prob_other",
    # --- vetting flags (booleans from to_row) ------------------------------- #
    "flag_is_eb",
    "flag_secondary_detected",
    "flag_odd_even_mismatch",
    "flag_centroid_shift",
    "flag_is_blend",
    # --- compact joined flag string (added by result_to_row) ---------------- #
    "flags",
    # --- model goodness-of-fit (present when a fit ran) --------------------- #
    "delta_bic",
    "fit_method",
]

# Columns that should be treated as text (filled with '' instead of NaN, kept as
# string dtype on read). Everything else is numeric/boolean.
_STRING_COLUMNS = frozenset(
    {"class", "detection_method", "classify_method", "fit_method", "flags"}
)


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
def _joined_flags(result: CandidateResult) -> str:
    """``;``-joined names of the vetting flags that are truthy (else '')."""
    vetting = getattr(result, "vetting", None)
    flags = getattr(vetting, "flags", None) if vetting is not None else None
    if not flags:
        return ""
    raised = [str(name) for name, value in flags.items() if bool(value)]
    return ";".join(raised)


def result_to_row(result: CandidateResult) -> dict[str, Any]:
    """Flatten a :class:`~exopipe.types.CandidateResult` to a catalog row.

    Wraps ``result.to_row()`` (the foundation contract) and augments it with:

    * ``depth_ppm`` -- ``depth`` (fractional) times 1e6;
    * ``prob_*``    -- the four calibrated class probabilities (already in
      ``to_row``; re-asserted from ``classification.probabilities`` so they are
      always present even if a future ``to_row`` changes);
    * ``flags``     -- the ``;``-joined set vetting-flag names.

    Returns a plain ``dict`` of scalars/strings suitable for
    ``pandas.DataFrame([...])``.
    """
    row: dict[str, Any] = dict(result.to_row())

    # depth_ppm from the (fractional) depth in the row.
    depth = row.get("depth", np.nan)
    try:
        row["depth_ppm"] = float(depth) * 1e6 if np.isfinite(float(depth)) else np.nan
    except (TypeError, ValueError):
        row["depth_ppm"] = np.nan

    # Re-assert the four calibrated class probabilities (idempotent with to_row).
    classification = getattr(result, "classification", None)
    probs = getattr(classification, "probabilities", None) if classification else None
    probs = probs or {}
    for class_name in ("transit", "eclipsing_binary", "blend", "other"):
        key = f"prob_{class_name}"
        if key not in row or not _is_finite(row.get(key)):
            value = probs.get(class_name, np.nan)
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = np.nan

    # Compact joined flag string.
    row["flags"] = _joined_flags(result)

    return row


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _coerce_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Accept ``CandidateResult`` objects and/or pre-built dict rows.

    Anything exposing ``to_row`` (a ``CandidateResult``) is routed through
    :func:`result_to_row` so the extra columns are added; plain mappings are
    passed through unchanged (allowing callers to write hand-built rows).
    """
    out: list[dict[str, Any]] = []
    for item in rows:
        if hasattr(item, "to_row") and callable(item.to_row):
            out.append(result_to_row(item))
        elif isinstance(item, dict):
            out.append(dict(item))
        else:
            raise TypeError(
                "rows must be CandidateResult objects or dicts, "
                f"got {type(item).__name__}"
            )
    return out


def _build_dataframe(rows: Iterable[Any]) -> pd.DataFrame:
    """Build a DataFrame in :data:`CATALOG_COLUMNS` order, filling gaps."""
    records = _coerce_rows(rows)
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

    # Ensure every canonical column exists, with sensible fills, in order.
    for col in CATALOG_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in _STRING_COLUMNS else np.nan

    # Keep any extra columns the caller provided (e.g. truth/debug fields) *after*
    # the canonical ones, so the schema is stable but nothing is silently dropped.
    extra = [c for c in df.columns if c not in CATALOG_COLUMNS]
    df = df[CATALOG_COLUMNS + extra]

    # Normalise string columns so CSV/JSON never carry NaN where '' is expected.
    for col in _STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "").astype(str)
    return df


# --------------------------------------------------------------------------- #
# Writers / readers
# --------------------------------------------------------------------------- #
def _swap_suffix(path: str, new_ext: str) -> str:
    """Return ``path`` with its extension replaced by ``new_ext`` (no dot)."""
    import os

    stem, _ = os.path.splitext(path)
    return f"{stem}.{new_ext}"


def write_catalog(rows: Iterable[Any], path: str, fmt: str = "csv") -> None:
    """Write candidate rows to ``path`` in ``fmt`` (``'csv'|'json'|'parquet'``).

    Parameters
    ----------
    rows:
        An iterable of :class:`~exopipe.types.CandidateResult` objects and/or
        pre-built row dicts. ``CandidateResult`` objects are flattened via
        :func:`result_to_row` (adding ``depth_ppm`` / ``prob_*`` / ``flags``).
    path:
        Output file path.
    fmt:
        Output format. ``'parquet'`` imports :mod:`pyarrow` lazily; if it is
        unavailable a warning is emitted and the catalog is written as CSV at the
        same stem instead (graceful degradation -- core deps always succeed).

    Notes
    -----
    The catalog is always materialised in :data:`CATALOG_COLUMNS` order with
    missing columns filled (NaN / ``''``), so every emitted file has the same,
    predictable schema regardless of which pipeline stages populated the rows.
    """
    fmt = fmt.lower()
    df = _build_dataframe(rows)

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        # ``records`` orientation == a JSON array of per-candidate objects,
        # which mirrors the per-candidate JSON sidecar schema from dossier 06.
        df.to_json(path, orient="records", indent=2)
    elif fmt == "parquet":
        try:
            import pyarrow  # noqa: F401  (presence check only)
        except ImportError:
            fallback = _swap_suffix(path, "csv")
            warnings.warn(
                "pyarrow is not installed; cannot write Parquet. "
                f"Falling back to CSV at {fallback!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            df.to_csv(fallback, index=False)
            return
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        raise ValueError(f"unknown fmt {fmt!r}; expected 'csv', 'json', or 'parquet'")


def read_catalog(path: str) -> pd.DataFrame:
    """Read a catalog written by :func:`write_catalog` back into a DataFrame.

    The format is inferred from the file extension (``.csv`` / ``.json`` /
    ``.parquet``). Parquet uses :mod:`pyarrow` lazily and raises a clear error if
    it is unavailable. String columns are coerced back to ``str`` with empty
    strings for missing values so the schema round-trips.
    """
    import os

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext == "json":
        df = pd.read_json(path, orient="records")
    elif ext in ("parquet", "pq"):
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "pyarrow is required to read Parquet catalogs; "
                "install it or read the CSV/JSON variant."
            ) from exc
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        raise ValueError(
            f"cannot infer catalog format from extension {ext!r}; "
            "expected .csv, .json, or .parquet"
        )

    for col in _STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "").astype(str)
    return df


def to_per_candidate_json(result: CandidateResult, path: str) -> None:
    """Write a single candidate's full row as a JSON object (sidecar).

    A convenience matching the dossier's ``outputs/candidates/TIC<id>.json``
    per-candidate provenance file. Uses :func:`result_to_row` so it carries the
    same fields as the master catalog.
    """
    row = result_to_row(result)
    # Make numpy scalars JSON-serialisable.
    clean = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in row.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2, default=str)
