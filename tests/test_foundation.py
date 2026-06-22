"""Foundation tests for ``exopipe``.

These exercise the shared contract that every downstream module depends on:
package import, dataclass behaviour, config loading, utilities, and -- most
importantly -- the synthetic light-curve generator. They are written to pass
with **only** numpy/scipy/matplotlib/pandas installed (no science/ML extras).
"""

from __future__ import annotations

import numpy as np
import pytest

import exopipe
from exopipe import (
    CandidateResult,
    Classification,
    Config,
    DetectionResult,
    LightCurve,
    TransitFit,
    VettingReport,
    default_config,
    load_config,
)
from exopipe.data import from_arrays, sigma_clip, stitch
from exopipe.data.synthetic import (
    KINDS,
    make_synthetic_lightcurve,
    make_synthetic_population,
)
from exopipe.utils import (
    Timer,
    nanmad,
    phase_fold,
    robust_std,
    running_median,
    set_seed,
)


# --------------------------------------------------------------------------- #
# Package import & version
# --------------------------------------------------------------------------- #
def test_package_imports_and_version():
    assert exopipe.__version__ == "0.1.0"
    # the core dataclasses are re-exported at the top level
    for name in (
        "LightCurve",
        "DetectionResult",
        "VettingReport",
        "TransitFit",
        "Classification",
        "CandidateResult",
    ):
        assert hasattr(exopipe, name)


def test_lazy_process_lightcurve_is_importable_and_callable():
    # The lazy attribute resolves the real pipeline entry point (PEP 562
    # __getattr__) without having broken `import exopipe`. Now that
    # ``exopipe.pipeline.process_lightcurve`` exists it must be importable
    # both as a lazy top-level attribute and from its module, and be callable.
    fn = exopipe.process_lightcurve
    assert callable(fn)

    from exopipe.pipeline import process_lightcurve as direct

    assert fn is direct


# --------------------------------------------------------------------------- #
# Synthetic generator -- per kind
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_synthetic_kind_dtypes_shapes_and_finiteness(kind):
    lc = make_synthetic_lightcurve(kind=kind, seed=7)
    assert isinstance(lc, LightCurve)

    # canonical dtypes
    assert lc.time.dtype == np.float64
    assert lc.flux.dtype == np.float32
    assert lc.flux_err.dtype == np.float32

    # aligned shapes
    assert lc.time.shape == lc.flux.shape == lc.flux_err.shape
    assert len(lc) > 1000  # ~27 d at 2 min cadence is thousands of points

    # time is sorted ascending
    assert np.all(np.diff(lc.time) >= 0)

    # after cleaning, everything is finite
    clean = lc.remove_nans()
    assert np.all(np.isfinite(clean.time))
    assert np.all(np.isfinite(clean.flux))

    # rich metadata present
    for key in ("tic_id", "sector", "mission", "tmag", "label", "cadence_s"):
        assert key in lc.meta
    assert lc.meta["mission"] == "TESS-sim"
    assert lc.meta["label"] in {"transit", "eclipsing_binary", "blend", "other"}


def test_synthetic_is_reproducible():
    a = make_synthetic_lightcurve("transit", seed=123)
    b = make_synthetic_lightcurve("transit", seed=123)
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.flux, b.flux)
    # a different seed gives different data
    c = make_synthetic_lightcurve("transit", seed=124)
    assert not np.array_equal(a.flux, c.flux)


def test_transit_produces_measurable_dip():
    """The injected transit must actually depress the in-transit flux."""
    lc = make_synthetic_lightcurve("transit", seed=42)
    period = lc.meta["true_period"]
    t0 = lc.meta["true_t0"]
    duration = lc.meta["true_duration"]

    phase = phase_fold(lc.time, period, t0)
    half_phase = 0.5 * duration / period
    in_transit = np.abs(phase) < half_phase
    out_transit = np.abs(phase) > (2.0 * half_phase)

    assert in_transit.sum() > 5, "should capture several in-transit cadences"
    in_med = np.nanmedian(lc.flux[in_transit])
    out_med = np.nanmedian(lc.flux[out_transit])
    assert in_med < out_med, "in-transit flux must be below out-of-transit flux"


def test_eclipsing_binary_has_secondary():
    """EBs must inject a non-trivial secondary eclipse near phase 0.5."""
    lc = make_synthetic_lightcurve("eclipsing_binary", seed=11)
    assert lc.meta["label"] == "eclipsing_binary"
    # the generator records the secondary depth it injected
    assert lc.meta["secondary_depth"] > 0.0

    # and it should be visible in the folded light curve near phase 0.5
    period = lc.meta["true_period"]
    t0 = lc.meta["true_t0"]
    phase = phase_fold(lc.time, period, t0)
    # phase measured from primary; secondary sits near +/-0.5
    near_secondary = np.abs(np.abs(phase) - 0.5) < 0.05
    far = (np.abs(phase) > 0.1) & (np.abs(phase) < 0.4)
    if near_secondary.sum() > 5 and far.sum() > 5:
        assert np.nanmedian(lc.flux[near_secondary]) < np.nanmedian(lc.flux[far])


def test_blend_is_diluted_and_offset():
    lc = make_synthetic_lightcurve("blend", seed=5)
    assert lc.meta["label"] == "blend"
    assert lc.meta["crowdsap"] < 1.0
    assert lc.meta["centroid_offset"] > 0.0


def test_variable_and_noise_have_no_transit_truth():
    for kind in ("variable", "noise"):
        lc = make_synthetic_lightcurve(kind, seed=9)
        assert lc.meta["label"] == "other"
        assert not np.isfinite(lc.meta["true_period"])
        assert not np.isfinite(lc.meta["true_depth"])


def test_depths_span_realistic_range():
    """Across many draws, transit depths should cover a broad, sane range."""
    depths = [
        make_synthetic_lightcurve("transit", seed=s).meta["true_depth"]
        for s in range(40)
    ]
    depths = np.array(depths)
    assert np.all(depths >= 4e-5)
    assert np.all(depths <= 3e-2)
    assert depths.max() / max(depths.min(), 1e-9) > 5  # genuinely varied


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #
def test_population_size_and_labels_balanced_ish():
    pop = make_synthetic_population(60, seed=0)
    assert len(pop) == 60
    labels = [lc.meta["label"] for lc in pop]
    # all four science classes should appear
    assert set(labels) >= {"transit", "eclipsing_binary", "other"}
    # no single class should dominate everything
    from collections import Counter

    counts = Counter(labels)
    assert max(counts.values()) < len(pop)  # not all identical
    assert "transit" in counts and counts["transit"] >= 5


def test_population_custom_fractions_and_reproducible():
    frac = {"transit": 1.0}  # only transits
    pop = make_synthetic_population(10, seed=3, fractions=frac)
    assert len(pop) == 10
    assert all(lc.meta["kind"] == "transit" for lc in pop)

    a = make_synthetic_population(15, seed=1)
    b = make_synthetic_population(15, seed=1)
    assert [lc.meta["tic_id"] for lc in a] == [lc.meta["tic_id"] for lc in b]


def test_population_empty_and_zero_fractions():
    assert make_synthetic_population(0) == []
    with pytest.raises(ValueError):
        make_synthetic_population(5, fractions={"transit": 0.0})


# --------------------------------------------------------------------------- #
# LightCurve methods
# --------------------------------------------------------------------------- #
def test_from_arrays_roundtrip_and_normalisation():
    t = np.linspace(0, 10, 500)
    f = 100.0 + np.random.default_rng(0).normal(0, 1, t.size)  # ~100, not normalised
    lc = from_arrays(t, f)
    assert lc.time.dtype == np.float64
    assert lc.flux.dtype == np.float32
    # normalised to ~1.0
    assert abs(float(np.nanmedian(lc.flux)) - 1.0) < 0.05

    # already-normalised input is left alone
    f2 = 1.0 + np.random.default_rng(1).normal(0, 1e-3, t.size)
    lc2 = from_arrays(t, f2)
    assert abs(float(np.nanmedian(lc2.flux)) - 1.0) < 0.05


def test_from_arrays_sorts_by_time():
    t = np.array([3.0, 1.0, 2.0])
    f = np.array([1.0, 1.0, 1.0])
    lc = from_arrays(t, f)
    assert np.all(np.diff(lc.time) >= 0)


def test_from_arrays_shape_mismatch_raises():
    with pytest.raises(ValueError):
        from_arrays(np.arange(5), np.arange(4))


def test_lightcurve_fold_bin_copy():
    lc = make_synthetic_lightcurve("transit", seed=2)
    phase, flux_sorted = lc.fold(lc.meta["true_period"], lc.meta["true_t0"])
    assert phase.shape == flux_sorted.shape == lc.flux.shape
    assert np.all(np.diff(phase) >= 0)  # sorted by phase
    assert phase.min() >= -0.5 and phase.max() < 0.5

    cx, cy, ce = lc.bin(50)
    assert cx.shape == cy.shape == ce.shape == (50,)

    # copy is independent
    c = lc.copy()
    c.flux[:] = 0.0
    assert not np.array_equal(c.flux, lc.flux)


def test_lightcurve_remove_nans_masks_meta_arrays():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    f = np.array([1.0, np.nan, 1.0, 1.0])
    q = np.array([0, 0, 1, 0])
    lc = LightCurve(time=t, flux=f, flux_err=None, meta={"quality": q})
    clean = lc.remove_nans()
    assert len(clean) == 3
    # the per-cadence quality array was masked in lockstep
    assert clean.meta["quality"].shape == (3,)
    assert np.array_equal(clean.meta["quality"], np.array([0, 1, 0]))


def test_fold_rejects_bad_period():
    lc = make_synthetic_lightcurve("noise", seed=1)
    with pytest.raises(ValueError):
        lc.fold(0.0)
    with pytest.raises(ValueError):
        lc.fold(-3.0)


# --------------------------------------------------------------------------- #
# stitch & sigma_clip
# --------------------------------------------------------------------------- #
def test_stitch_multisector():
    a = make_synthetic_lightcurve("transit", seed=1)
    b = make_synthetic_lightcurve("transit", seed=2)
    combined = stitch([a, b])
    assert len(combined) == len(a) + len(b)
    assert np.all(np.diff(combined.time) >= 0)
    assert combined.meta["n_segments"] == 2
    assert "segment" in combined.meta
    assert combined.meta["segment"].shape == combined.flux.shape


def test_stitch_requires_input():
    with pytest.raises(ValueError):
        stitch([])


def test_sigma_clip_preserves_transit_removes_positive_spikes():
    lc = make_synthetic_lightcurve("transit", seed=8)
    # inject obvious positive cosmic-ray spikes
    flux = np.asarray(lc.flux, dtype=np.float64)
    spike_idx = np.array([100, 500, 1000])
    flux[spike_idx] += 0.5  # huge positive outliers
    lc = LightCurve(lc.time, flux, lc.flux_err, dict(lc.meta))

    n_before = len(lc)
    clipped = sigma_clip(lc, sigma=5.0, asymmetric=True)
    assert len(clipped) < n_before  # some points removed
    # the positive spikes are gone
    assert float(np.nanmax(clipped.flux)) < float(np.nanmax(flux))
    # the in-transit dip survives: minimum flux still well below 1
    assert float(np.nanmin(clipped.flux)) < 1.0 - lc.meta["true_depth"] * 0.3


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
def test_nanmad_and_robust_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    assert np.isclose(nanmad(x), 1.0)  # MAD of 1..5 about median 3 is 1
    assert np.isclose(robust_std(x), 1.4826, atol=1e-3)
    # robust_std of a normal sample approximates its std
    rng = np.random.default_rng(0)
    g = rng.normal(0, 2.0, 10000)
    assert abs(robust_std(g) - 2.0) < 0.1


def test_phase_fold_alignment_and_range():
    t = np.linspace(0, 10, 1000)
    phase = phase_fold(t, period=2.0, t0=0.0)
    assert phase.shape == t.shape  # element-aligned, not sorted
    assert phase.min() >= -0.5 and phase.max() < 0.5


def test_running_median_smooths_and_keeps_length():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 6, 200)) + rng.normal(0, 0.3, 200)
    sm = running_median(x, 11)
    assert sm.shape == x.shape
    assert np.nanstd(sm) < np.nanstd(x)  # smoother than the input
    # small windows are a no-op
    np.testing.assert_array_equal(running_median(x, 1), x)


def test_running_median_handles_nans():
    x = np.array([1.0, np.nan, 1.0, 1.0, 100.0, 1.0, 1.0])
    sm = running_median(x, 3)
    assert sm.shape == x.shape
    assert np.all(np.isfinite(sm))


def test_timer_and_set_seed():
    with Timer("noop") as timer:
        _ = sum(range(1000))
    assert timer.elapsed >= 0.0

    rng = set_seed(99)
    a = rng.random(5)
    rng2 = set_seed(99)
    b = rng2.random(5)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_default_config_structure():
    cfg = default_config()
    assert isinstance(cfg, Config)
    assert cfg.detrend.method == "biweight"
    assert cfg.detrend.window_factor == 3.0
    assert cfg.search.period_min == 0.5
    assert cfg.fit.sampler == "emcee"
    assert cfg.perf.n_jobs == -1
    # round-trips to a plain dict
    d = cfg.to_dict()
    assert d["search"]["methods"] == ["bls", "tls"]


def test_config_merge_overrides_only_given_keys():
    cfg = Config.from_dict({"search": {"period_min": 2.0}, "unknown_key": 1})
    assert cfg.search.period_min == 2.0  # overridden
    assert cfg.search.period_max == 15.0  # default retained
    assert cfg.detrend.method == "biweight"  # untouched sub-config retained


def test_load_config_missing_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, Config)
    assert cfg.seed == 42


def test_load_config_from_yaml_if_available(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"seed": 7, "search": {"min_sde": 9.0}}))
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.search.min_sde == 9.0


def test_default_yaml_file_matches_dataclass():
    """The shipped configs/default.yaml must load cleanly onto Config."""
    pytest.importorskip("yaml")  # skip if pyyaml is not installed
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    default_yaml = repo_root / "configs" / "default.yaml"
    assert default_yaml.exists()
    cfg = load_config(default_yaml)
    # spot-check a few values agree with the dataclass defaults
    assert cfg.seed == 42
    assert cfg.detrend.window_factor == 3.0
    assert cfg.classify.classes == ["transit", "eclipsing_binary", "blend", "other"]


# --------------------------------------------------------------------------- #
# CandidateResult.to_row  -- build from REAL dataclasses with placeholder values
# --------------------------------------------------------------------------- #
def _dummy_candidate() -> CandidateResult:
    lc = make_synthetic_lightcurve("transit", seed=1)
    detection = DetectionResult(
        period=3.21,
        t0=1326.5,
        duration=0.12,
        depth=0.0009,
        sde=12.4,
        snr=15.1,
        method="bls",
    )
    vetting = VettingReport(
        metrics={
            "odd_even_depth_ratio": 1.02,
            "secondary_depth": 0.0,
            "centroid_offset": 0.3,
        },
        flags={
            "is_eb": False,
            "secondary_detected": False,
            "odd_even_mismatch": False,
            "centroid_shift": False,
            "is_blend": False,
        },
    )
    fit = TransitFit(
        params={
            "period": (3.2105, 0.0007, 0.0008),
            "depth": (0.00092, 0.00005, 0.00006),
            "duration": (0.121, 0.004, 0.005),
            "t0": (1326.512, 0.002, 0.002),
            "rp_rs": (0.030, 0.001, 0.001),
        },
        bic_transit=1000.0,
        bic_flat=1100.0,
        delta_bic=-100.0,
        snr=15.5,
        method="emcee",
    )
    classification = Classification(
        label="transit",
        confidence=0.94,
        probabilities={
            "transit": 0.94,
            "eclipsing_binary": 0.03,
            "blend": 0.02,
            "other": 0.01,
        },
        method="xgboost",
    )
    return CandidateResult(
        lightcurve=lc,
        detection=detection,
        vetting=vetting,
        fit=fit,
        classification=classification,
        features={"snr": 15.1},
    )


def test_to_row_flat_dict_for_catalog():
    cand = _dummy_candidate()
    row = cand.to_row()
    assert isinstance(row, dict)

    # identifiers pulled from lightcurve.meta
    assert row["tic_id"] == cand.lightcurve.meta["tic_id"]
    assert row["sector"] == cand.lightcurve.meta["sector"]

    # fit-derived parameters with errors take precedence over detection
    assert np.isclose(row["period"], 3.2105)
    assert np.isclose(row["period_err_lo"], 0.0007)
    assert np.isclose(row["period_err_hi"], 0.0008)
    assert np.isclose(row["depth"], 0.00092)
    assert np.isclose(row["duration"], 0.121)
    assert np.isclose(row["t0"], 1326.512)

    # significance and method from detection / fit
    assert np.isclose(row["snr"], 15.5)
    assert np.isclose(row["sde"], 12.4)
    assert row["detection_method"] == "bls"

    # classification fields
    assert row["class"] == "transit"
    assert np.isclose(row["confidence"], 0.94)
    assert np.isclose(row["prob_transit"], 0.94)
    assert np.isclose(row["prob_other"], 0.01)

    # vetting flags flattened with a flag_ prefix
    assert row["flag_is_eb"] is False
    assert row["flag_secondary_detected"] is False

    # goodness-of-fit
    assert np.isclose(row["delta_bic"], -100.0)
    assert row["fit_method"] == "emcee"

    # every value must be a scalar/str/bool/None -> safe for a DataFrame row
    for key, value in row.items():
        assert isinstance(value, (int, float, str, bool, np.floating, np.integer)) or value is None, (
            f"{key} -> {type(value)} is not catalog-safe"
        )


def test_to_row_falls_back_to_detection_when_fit_missing():
    cand = _dummy_candidate()
    # wipe the fit params -> period/depth/duration should fall back to detection
    cand.fit = TransitFit()  # all-empty
    row = cand.to_row()
    assert np.isclose(row["period"], cand.detection.period)
    assert np.isclose(row["depth"], cand.detection.depth)
    assert np.isclose(row["duration"], cand.detection.duration)
    # snr falls back to detection's snr when fit snr is NaN
    assert np.isclose(row["snr"], cand.detection.snr)


def test_to_row_is_dataframe_ready():
    pd = pytest.importorskip("pandas")
    rows = [_dummy_candidate().to_row() for _ in range(3)]
    df = pd.DataFrame(rows)
    assert len(df) == 3
    assert "period" in df.columns
    assert "class" in df.columns
