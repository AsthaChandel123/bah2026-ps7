"""Tests for ``exopipe.catalog`` and ``exopipe.viz``.

* The machine-readable catalog round-trips through CSV and JSON (same rows /
  columns out as in), and a per-candidate JSON sidecar is written.
* ``viz.vetting_sheet`` renders a real (small) :class:`CandidateResult` to a
  non-empty PNG and returns a matplotlib ``Figure``.

The light curve is kept tiny so building the candidate is fast.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from exopipe.catalog import (
    CATALOG_COLUMNS,
    read_catalog,
    to_per_candidate_json,
    write_catalog,
)
from exopipe.config import default_config
from exopipe.data.synthetic import make_synthetic_lightcurve
from exopipe.pipeline import process_lightcurve
from exopipe.types import CandidateResult

_N_DAYS = 16.0
_CADENCE_MIN = 8.0


@pytest.fixture(scope="module")
def candidate() -> CandidateResult:
    cfg = default_config()
    lc = make_synthetic_lightcurve(
        kind="transit", seed=1, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    return process_lightcurve(lc, config=cfg)


def test_catalog_csv_round_trip(candidate, tmp_path):
    rows = [candidate.to_row()]
    path = tmp_path / "catalog.csv"
    write_catalog(rows, str(path), fmt="csv")
    assert path.exists() and path.stat().st_size > 0

    frame = read_catalog(str(path))
    assert len(frame) == 1
    for column in CATALOG_COLUMNS:
        assert column in frame.columns
    # the class survives the round-trip.
    assert frame["class"].iloc[0] == candidate.classification.label


def test_catalog_json_round_trip(candidate, tmp_path):
    rows = [candidate.to_row()]
    path = tmp_path / "catalog.json"
    write_catalog(rows, str(path), fmt="json")
    assert path.exists() and path.stat().st_size > 0

    frame = read_catalog(str(path))
    assert len(frame) == 1
    assert frame["class"].iloc[0] == candidate.classification.label


def test_per_candidate_json_sidecar(candidate, tmp_path):
    path = tmp_path / "cand.json"
    to_per_candidate_json(candidate, str(path))
    assert path.exists() and path.stat().st_size > 0

    obj = json.loads(path.read_text())
    assert isinstance(obj, dict)
    assert "class" in obj and "tic_id" in obj


def test_vetting_sheet_writes_nonempty_png(candidate, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # headless backend for CI
    from exopipe.viz import vetting_sheet

    path = tmp_path / "sheet.png"
    fig = vetting_sheet(candidate, save_path=str(path))

    assert fig is not None
    assert path.exists()
    assert path.stat().st_size > 1000  # a real multi-panel figure, not empty

    import matplotlib.pyplot as plt

    plt.close(fig)
