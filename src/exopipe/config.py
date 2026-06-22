"""Hierarchical configuration for the ``exopipe`` pipeline.

A nested set of plain dataclasses describes every tunable knob. Configuration is
loaded from a YAML file and *merged onto* the defaults, so a user file only has
to override the keys it cares about. ``pyyaml`` is optional: without it,
:func:`load_config` simply returns the built-in defaults.

The dataclass approach keeps the foundation dependency-free; the research notes
suggest Hydra+pydantic for the full experiment harness, and these dataclasses
are intentionally compatible with that (flat, serialisable, ``to_dict`` round
trips).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DataConfig",
    "DetrendConfig",
    "SearchConfig",
    "VettingConfig",
    "ClassifyConfig",
    "FitConfig",
    "VizConfig",
    "PerfConfig",
    "Config",
    "default_config",
    "load_config",
]


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where light curves come from and how they are pre-cleaned."""

    source: str = "synthetic"  # 'synthetic' | 'mast' | 'local'
    data_dir: str = "data"
    sector: int | None = None
    cadence_min: float = 2.0
    flux_column: str = "pdcsap_flux"
    sigma_clip: float = 5.0
    quality_bitmask: str = "default"  # 'none' | 'default' | 'hard'


@dataclass
class DetrendConfig:
    """Baseline / systematics removal before transit search."""

    method: str = "biweight"  # 'biweight' | 'median' | 'savgol' | 'gp' | 'none'
    window_factor: float = 3.0  # window length as a multiple of expected duration
    window_length: float | None = None  # explicit window (days); overrides factor
    break_tolerance: float = 0.5  # days; split detrending across larger gaps
    edge_cutoff: float = 0.0  # days trimmed from segment edges after detrend


@dataclass
class SearchConfig:
    """Periodic-transit search grid and acceptance threshold."""

    period_min: float = 0.5
    period_max: float = 15.0
    methods: list[str] = field(default_factory=lambda: ["bls", "tls"])
    min_sde: float = 7.0
    min_snr: float = 7.0
    n_durations: int = 8
    duration_min: float = 0.02  # days
    duration_max: float = 0.5  # days
    oversample: int = 5
    n_transits_min: int = 2


@dataclass
class VettingConfig:
    """False-positive vetting thresholds."""

    odd_even_sigma: float = 3.0
    secondary_sigma: float = 3.0
    v_shape_threshold: float = 0.3
    centroid_sigma: float = 3.0
    min_transit_snr: float = 5.0


@dataclass
class ClassifyConfig:
    """Signal classifier selection."""

    method: str = "xgboost"  # 'xgboost' | 'random_forest' | 'cnn' | 'rules'
    model_path: str | None = None
    classes: list[str] = field(
        default_factory=lambda: ["transit", "eclipsing_binary", "blend", "other"]
    )
    threshold: float = 0.5


@dataclass
class FitConfig:
    """Transit-model fitting and uncertainty estimation."""

    sampler: str = "emcee"  # 'emcee' | 'dynesty' | 'least_squares'
    nsteps: int = 2000
    nwalkers: int = 32
    nburn: int = 500
    limb_dark: str = "quadratic"
    fit_eccentricity: bool = False
    progress: bool = False


@dataclass
class VizConfig:
    """Plot/report rendering options."""

    dpi: int = 120
    style: str = "default"
    save_dir: str = "outputs/figures"
    fmt: str = "png"
    n_phase_bins: int = 100
    show: bool = False


@dataclass
class PerfConfig:
    """Parallelism, caching, and compiled-kernel toggles."""

    n_jobs: int = -1
    cache_dir: str = ".cache"
    use_numba: bool = True
    backend: str = "loky"  # joblib backend
    chunk_size: int = 64


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Root configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    vetting: VettingConfig = field(default_factory=VettingConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    seed: int = 42

    # -- (de)serialisation --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Return a nested plain-dict representation (YAML/JSON friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        """Build a :class:`Config`, merging ``data`` onto the defaults.

        Unknown keys are ignored (with no error) so a config file may carry
        extra annotations without breaking older code.
        """
        cfg = cls()
        if not data:
            return cfg
        _merge_dataclass(cfg, data)
        return cfg


# --------------------------------------------------------------------------- #
# Merge helpers
# --------------------------------------------------------------------------- #
def _merge_dataclass(target: Any, data: dict[str, Any]) -> None:
    """Recursively overlay ``data`` onto a dataclass ``target`` in place."""
    if not isinstance(data, dict):
        return
    valid = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(target, key, value)


def default_config() -> Config:
    """Return a fresh :class:`Config` populated entirely with defaults."""
    return Config()


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML ``path`` merged onto the defaults.

    Parameters
    ----------
    path:
        Path to a YAML file. If ``None`` or the file does not exist, the
        built-in defaults are returned. If ``pyyaml`` is not installed, a warning
        is logged and defaults are returned.
    """
    if path is None:
        return default_config()

    path = Path(path)
    if not path.exists():
        from .utils import get_logger

        get_logger().warning("Config file %s not found; using defaults.", path)
        return default_config()

    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - pyyaml is optional
        from .utils import get_logger

        get_logger().warning(
            "pyyaml not installed; ignoring %s and using defaults.", path
        )
        return default_config()

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.from_dict(raw)
