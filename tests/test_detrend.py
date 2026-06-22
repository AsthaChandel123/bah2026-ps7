"""Tests for ``exopipe.detrend`` (PS7 stage DETREND).

The detrender must flatten slow stellar/instrumental variability to ~1.0
*without* eating the transit. These tests assert the three properties the
pipeline relies on:

* a sinusoidal trend is removed (post-detrend baseline scatter << pre-detrend);
* an injected transit is preserved (in-transit median stays below the
  out-of-transit median after detrending);
* no NaNs are introduced where the input was finite.

All tests run with only the core stack (numpy/scipy) -- the biweight default is
pure NumPy, so no ``importorskip`` is needed here.
"""

from __future__ import annotations

import numpy as np

from exopipe.data.synthetic import make_synthetic_lightcurve
from exopipe.detrend import detrend
from exopipe.types import LightCurve


def _box_transit(time: np.ndarray, period: float, t0: float, dur: float, depth: float) -> np.ndarray:
    """Return a flux array (~1.0) with a periodic box transit injected."""
    phase = (((time - t0) / period + 0.5) % 1.0) - 0.5
    in_transit = np.abs(phase) < (0.5 * dur / period)
    flux = np.ones_like(time)
    flux[in_transit] -= depth
    return flux


def test_detrend_removes_sinusoidal_trend():
    # A pure slow sinusoid (period ~ several days) plus white noise must be
    # flattened: the detrended scatter is far smaller than the trend amplitude.
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 20.0, 4000)
    amp = 0.05
    trend = 1.0 + amp * np.sin(2.0 * np.pi * time / 6.0)
    flux = (trend + rng.normal(0.0, 1e-3, time.size)).astype(np.float32)
    lc = LightCurve(time=time, flux=flux, flux_err=np.full(time.size, 1e-3, np.float32))

    out = detrend(lc)

    assert isinstance(out, LightCurve)
    # input is never mutated
    assert np.shares_memory(out.flux, lc.flux) is False
    # the trend (std ~ amp/sqrt(2) ~ 0.035) is gone; residual scatter is tiny
    assert np.nanstd(out.flux) < 0.2 * np.nanstd(lc.flux)
    # baseline restored to ~1.0
    assert abs(np.nanmedian(out.flux) - 1.0) < 0.02


def test_detrend_preserves_injected_transit():
    # Slow trend + an injected periodic transit. After detrending, the
    # in-transit cadences must still sit below the out-of-transit level.
    rng = np.random.default_rng(1)
    time = np.linspace(0.0, 24.0, 6000)
    period, t0, dur, depth = 4.0, 1.0, 0.18, 0.02
    trend = 1.0 + 0.04 * np.sin(2.0 * np.pi * time / 9.0)
    flux = trend * _box_transit(time, period, t0, dur, depth)
    flux = (flux + rng.normal(0.0, 5e-4, time.size)).astype(np.float32)
    lc = LightCurve(time=time, flux=flux, flux_err=np.full(time.size, 5e-4, np.float32))

    out = detrend(lc)
    # detrend may drop outlier/gap cadences, so phase must be computed from the
    # *output* time axis, not the original.
    out_time = np.asarray(out.time, dtype=np.float64)
    f = np.asarray(out.flux, dtype=np.float64)

    phase = (((out_time - t0) / period + 0.5) % 1.0) - 0.5
    in_transit = np.abs(phase) < (0.4 * dur / period)
    out_transit = np.abs(phase) > (2.0 * dur / period)

    in_med = np.nanmedian(f[in_transit])
    out_med = np.nanmedian(f[out_transit])
    # transit survives: in-transit clearly below out-of-transit, by most of the
    # injected depth (allow for some attenuation from the robust filter).
    assert in_med < out_med
    assert (out_med - in_med) > 0.5 * depth


def test_detrend_introduces_no_nans_on_finite_input():
    # On a fully finite synthetic transit light curve the detrended flux must be
    # finite wherever the input flux was finite.
    lc = make_synthetic_lightcurve(kind="transit", seed=4, n_days=16.0, cadence_min=8.0)
    finite_in = np.isfinite(lc.flux)
    assert finite_in.all()  # generator returns finite flux

    out = detrend(lc)
    # detrend may drop a few outlier/gap cadences but must keep the bulk and
    # introduce no NaNs into the flux it returns.
    assert out.flux.shape == out.time.shape
    assert out.flux.size > 0.9 * lc.flux.size
    assert np.isfinite(out.flux).all()
    assert np.isfinite(out.time).all()


def test_detrend_does_not_mutate_input():
    lc = make_synthetic_lightcurve(kind="variable", seed=2, n_days=16.0, cadence_min=8.0)
    flux_before = lc.flux.copy()
    _ = detrend(lc)
    np.testing.assert_array_equal(lc.flux, flux_before)
