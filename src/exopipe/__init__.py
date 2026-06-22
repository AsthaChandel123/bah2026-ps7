"""exopipe -- AI detection & classification of exoplanet transits in TESS light curves.

BAH 2026 Problem Statement 7. This top-level package exposes the shared
foundation (dataclasses, config, utilities, synthetic data) and *lazily*
re-exports the heavier pipeline entry points so that ``import exopipe`` never
fails just because an optional dependency (astropy, lightkurve, torch, ...) is
missing.

Always-available re-exports
---------------------------
The core dataclasses from :mod:`exopipe.types` and the synthetic-data helpers
are imported eagerly -- they depend only on numpy.

Lazy re-exports
---------------
``process_lightcurve`` (the end-to-end driver) is resolved on first access via
module ``__getattr__`` so importing it pulls in optional deps only when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# -- always-safe, numpy-only re-exports ------------------------------------- #
from .config import Config, default_config, load_config
from .data.synthetic import make_synthetic_lightcurve, make_synthetic_population
from .types import (
    CandidateResult,
    Classification,
    DetectionResult,
    LightCurve,
    TransitFit,
    VettingReport,
)

__all__ = [
    "__version__",
    # dataclasses
    "LightCurve",
    "DetectionResult",
    "VettingReport",
    "TransitFit",
    "Classification",
    "CandidateResult",
    # config
    "Config",
    "default_config",
    "load_config",
    # synthetic data
    "make_synthetic_lightcurve",
    "make_synthetic_population",
    # lazy
    "process_lightcurve",
]

# Names resolved lazily: attribute -> (module, qualname). Keeping these out of
# the eager import path means optional heavy deps load only on first use.
_LAZY: dict[str, tuple[str, str]] = {
    "process_lightcurve": ("exopipe.pipeline", "process_lightcurve"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loader for optional/heavy entry points."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, qualname = target
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on build state
        raise AttributeError(
            f"{name!r} is not available yet: could not import {module_name!r} "
            f"({exc}). The full pipeline modules are under active development."
        ) from exc
    attr = getattr(module, qualname)
    globals()[name] = attr  # cache for subsequent lookups
    return attr


def __dir__() -> list[str]:  # pragma: no cover - introspection nicety
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # pragma: no cover - typing only
    # Help static checkers see the lazy symbol without importing at runtime.
    def process_lightcurve(*args: Any, **kwargs: Any) -> CandidateResult: ...
