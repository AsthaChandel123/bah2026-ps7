"""Data layer for ``exopipe``: light-curve helpers and the synthetic generator.

Submodules:
* :mod:`exopipe.data.lightcurve` -- constructors/operations for ``LightCurve``.
* :mod:`exopipe.data.synthetic` -- physically-motivated TESS-like generator.
"""

from __future__ import annotations

from .lightcurve import from_arrays, quality_mask, sigma_clip, stitch
from .synthetic import (
    KINDS,
    make_synthetic_lightcurve,
    make_synthetic_population,
)

__all__ = [
    "from_arrays",
    "stitch",
    "quality_mask",
    "sigma_clip",
    "make_synthetic_lightcurve",
    "make_synthetic_population",
    "KINDS",
]
