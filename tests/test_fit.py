"""Tests for ``exopipe.fit`` (PS7 stage FIT) -- parameter recovery.

``fit_transit(..., method='fast')`` runs the trapezoid/least-squares seed stage
only (no MCMC), which is fast and deterministic. On a clean synthetic transit it
must recover the injected depth and duration within tolerance and return a valid
:class:`TransitFit` exposing the documented parameter keys.
"""

from __future__ import annotations

import numpy as np
import pytest

from exopipe.data.synthetic import make_synthetic_lightcurve
from exopipe.detrend import detrend
from exopipe.fit import fit_transit
from exopipe.search import search_two_stage
from exopipe.types import TransitFit

# A medium baseline gives a tighter depth/duration measurement while staying
# fast because ``method='fast'`` skips the MCMC stage entirely.
_N_DAYS = 24.0
_CADENCE_MIN = 6.0

# Documented parameter keys (ARCHITECTURE.md §6.1 / types.TransitFit).
_EXPECTED_PARAMS = (
    "period",
    "t0",
    "depth",
    "duration",
    "rp_rs",
    "a_rs",
    "b",
    "inclination",
    "u1",
    "u2",
)


@pytest.fixture(scope="module")
def fast_fit():
    lc = make_synthetic_lightcurve(
        kind="transit", seed=1, n_days=_N_DAYS, cadence_min=_CADENCE_MIN
    )
    det = search_two_stage(detrend(lc))
    tf = fit_transit(detrend(lc), det, method="fast")
    return lc, tf


def test_fit_returns_valid_transitfit_with_param_keys(fast_fit):
    _lc, tf = fast_fit
    assert isinstance(tf, TransitFit)
    for key in _EXPECTED_PARAMS:
        assert key in tf.params, f"missing fit param {key!r}"
        triple = np.atleast_1d(np.asarray(tf.params[key], dtype=float))
        assert triple.size >= 1 and np.isfinite(triple[0])
    # transit-vs-flat significance is reported and favours the transit.
    assert np.isfinite(tf.delta_bic)
    assert tf.delta_bic > 0.0
    # 'fast' stays in the seed stage (no MCMC samples required).
    assert tf.method in ("trapezoid", "lsq", "fast")


def test_fit_recovers_depth_within_tolerance(fast_fit):
    lc, tf = fast_fit
    true_depth = float(lc.meta["true_depth"])
    fit_depth = float(np.atleast_1d(tf.params["depth"])[0])
    assert fit_depth > 0
    # within 25% of the injected depth.
    assert abs(fit_depth - true_depth) / true_depth < 0.25


def test_fit_recovers_duration_within_tolerance(fast_fit):
    lc, tf = fast_fit
    true_dur = float(lc.meta["true_duration"])
    fit_dur = float(np.atleast_1d(tf.params["duration"])[0])
    assert fit_dur > 0
    # within 20% of the injected duration.
    assert abs(fit_dur - true_dur) / true_dur < 0.20
