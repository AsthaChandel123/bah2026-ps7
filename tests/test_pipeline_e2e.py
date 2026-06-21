"""End-to-end tests for the orchestration layer.

* ``pipeline.process_lightcurve`` returns a valid :class:`CandidateResult` for
  *each* of the five synthetic kinds without crashing.
* ``driver.run_batch`` over a small mixed population returns one result per input,
  and ``catalog.write_catalog`` produces a catalog that reads back with every
  ``CATALOG_COLUMNS`` column present.

Small light curves (short baseline, coarse cadence) keep the whole module fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from exopipe.catalog import CATALOG_COLUMNS, read_catalog, write_catalog
from exopipe.config import default_config
from exopipe.data.synthetic import KINDS, make_synthetic_lightcurve
from exopipe.driver import run_batch
from exopipe.pipeline import process_lightcurve
from exopipe.types import (
    CandidateResult,
    Classification,
    DetectionResult,
    TransitFit,
    VettingReport,
)

_CLASSES = ("transit", "eclipsing_binary", "blend", "other")
_N_DAYS = 16.0
_CADENCE_MIN = 8.0


def _assert_valid_result(res: CandidateResult):
    assert isinstance(res, CandidateResult)
    assert isinstance(res.detection, DetectionResult)
    assert isinstance(res.vetting, VettingReport)
    assert isinstance(res.fit, TransitFit)
    assert isinstance(res.classification, Classification)
    assert isinstance(res.features, dict)
    assert res.classification.label in _CLASSES
    # to_row() must flatten without error and carry identifiers + class.
    row = res.to_row()
    assert isinstance(row, dict)
    assert "class" in row and "tic_id" in row


@pytest.mark.parametrize("kind", KINDS)
def test_process_lightcurve_handles_every_kind(kind):
    cfg = default_config()
    lc = make_synthetic_lightcurve(
        kind=kind, seed=2, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    res = process_lightcurve(lc, config=cfg)
    _assert_valid_result(res)


def test_run_batch_and_write_catalog(tmp_path):
    cfg = default_config()
    kinds = ["transit", "eclipsing_binary", "blend", "variable", "noise", "transit"]
    lcs = [
        make_synthetic_lightcurve(
            kind=k, seed=i + 1, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
        )
        for i, k in enumerate(kinds)
    ]

    results = run_batch(lcs, config=cfg, n_jobs=1)
    assert len(results) == len(lcs)
    assert all(isinstance(r, CandidateResult) for r in results)

    catalog_path = tmp_path / "catalog.csv"
    write_catalog([r.to_row() for r in results], str(catalog_path), fmt="csv")
    assert catalog_path.exists() and catalog_path.stat().st_size > 0

    frame = read_catalog(str(catalog_path))
    assert len(frame) == len(lcs)
    for column in CATALOG_COLUMNS:
        assert column in frame.columns, f"catalog missing column {column!r}"
    # every classified label is one of the four canonical classes.
    assert set(frame["class"]).issubset(set(_CLASSES))
