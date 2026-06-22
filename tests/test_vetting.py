"""Tests for ``exopipe.vetting`` (PS7 stage VET) -- false-positive diagnostics.

The physics vetter must raise the *right* flags on the right kinds:

* ``eb_secondary`` fires on an eclipsing binary (deep secondary at phase 0.5),
  but NOT on a clean transiting planet;
* ``blend_contamination`` fires on a blended/diluted signal (low CROWDSAP).

Flags are computed from the detrended light curve + the detection, so the test
runs the front of the real pipeline (detrend -> search -> vet) on small,
fast synthetic light curves.
"""

from __future__ import annotations

import numpy as np

from exopipe.data.synthetic import make_synthetic_lightcurve
from exopipe.detrend import detrend
from exopipe.search import search_two_stage
from exopipe.types import VettingReport
from exopipe.vetting import vet

_N_DAYS = 16.0
_CADENCE_MIN = 8.0


def _vet_kind(kind: str, seed: int) -> VettingReport:
    lc = make_synthetic_lightcurve(
        kind=kind, seed=seed, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    det = search_two_stage(detrend(lc))
    report = vet(detrend(lc), det)
    assert isinstance(report, VettingReport)
    return report


def test_eb_secondary_flagged_on_eclipsing_binary():
    # seed=1 produces an EB whose secondary eclipse is detectable.
    report = _vet_kind("eclipsing_binary", seed=1)
    assert report.flags.get("eb_secondary") is True


def test_eb_secondary_not_flagged_on_clean_transit():
    # A clean transiting planet has no significant secondary -> flag is False.
    report = _vet_kind("transit", seed=1)
    assert report.flags.get("eb_secondary") is False
    # and it is not mistaken for a blend either.
    assert report.flags.get("blend_contamination") is False


def test_blend_contamination_flagged_on_blend():
    # A blended signal carries CROWDSAP < 1 -> the contamination flag fires.
    report = _vet_kind("blend", seed=1)
    assert report.flags.get("blend_contamination") is True


def test_vetting_report_has_metrics_and_flags():
    report = _vet_kind("eclipsing_binary", seed=1)
    assert isinstance(report.metrics, dict) and report.metrics
    assert isinstance(report.flags, dict) and report.flags
    # all flag values are booleans
    assert all(isinstance(v, bool) for v in report.flags.values())
