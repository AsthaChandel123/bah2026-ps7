"""Tests for ``exopipe.search`` (PS7 stage SEARCH) -- injection--recovery.

The detector must recover an injected transit's period and produce a higher
detection significance (SDE) on a transit than on pure noise.

* ``search_two_stage`` (BLS triage -> refine) recovers the true period within a
  few percent on several seeds of ``make_synthetic_lightcurve('transit')``,
  allowing the 2:1 / 1:2 period alias that box searches occasionally lock onto.
* A ``'noise'`` light curve yields a lower SDE than a clearly-recovered transit.

Small light curves (short baseline, coarser cadence) keep each search well under
a second while preserving recovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from exopipe.data.synthetic import make_synthetic_lightcurve
from exopipe.detrend import detrend
from exopipe.search import search, search_two_stage
from exopipe.types import DetectionResult

# Short, coarse light curves: ~2800 cadences -> fast, still recoverable.
_N_DAYS = 16.0
_CADENCE_MIN = 8.0


def _period_alias_distance(recovered: float, truth: float) -> float:
    """Fractional distance to the nearest of {P, 2P, P/2} of the truth."""
    ratios = [recovered / truth, recovered / (2.0 * truth), recovered / (0.5 * truth)]
    return float(min(abs(r - 1.0) for r in ratios))


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_search_two_stage_recovers_injected_period(seed):
    lc = make_synthetic_lightcurve(
        kind="transit", seed=seed, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    truth = float(lc.meta["true_period"])
    assert np.isfinite(truth) and truth > 0

    det = search_two_stage(detrend(lc))
    assert isinstance(det, DetectionResult)
    assert np.isfinite(det.period) and det.period > 0

    # recover the period within ~5% (allowing the 2:1 / 1:2 alias).
    assert _period_alias_distance(det.period, truth) < 0.05
    # a real transit registers a non-trivial significance.
    assert np.isfinite(det.sde) and det.sde > 5.0


def test_search_bls_dispatch_recovers_period():
    # The single-stage ``search(method='bls')`` entry point also recovers it.
    lc = make_synthetic_lightcurve(
        kind="transit", seed=3, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    truth = float(lc.meta["true_period"])
    det = search(detrend(lc), method="bls")
    assert isinstance(det, DetectionResult)
    assert _period_alias_distance(det.period, truth) < 0.05
    # periodogram arrays are populated for the vetting sheet.
    assert det.periods is not None and det.power is not None
    assert np.asarray(det.periods).size > 0


def test_noise_has_lower_sde_than_transit():
    # A clearly-recovered transit must out-score a pure-noise light curve in SDE.
    lc_t = make_synthetic_lightcurve(
        kind="transit", seed=3, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    lc_n = make_synthetic_lightcurve(
        kind="noise", seed=3, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    sde_t = search_two_stage(detrend(lc_t)).sde
    sde_n = search_two_stage(detrend(lc_n)).sde

    assert np.isfinite(sde_t) and np.isfinite(sde_n)
    assert sde_t > sde_n
