"""Forward transit models for ``exopipe``.

Re-exports the trapezoid and Mandel & Agol transit models together with the
Winn (2010) / Seager & Mallen-Ornelas (2003) geometry helpers so callers can do
``from exopipe.model import transit_model, winn_duration`` directly.
"""

from __future__ import annotations

from .transit import (
    a_rs_from_density,
    a_rs_from_duration,
    density_from_a_rs,
    depth_from_rp_rs,
    impact_from_incl,
    incl_from_impact,
    rp_rearth_from_depth,
    rp_rjup_from_depth,
    rp_rs_from_depth,
    transit_model,
    trapezoid_model,
    winn_duration,
)

__all__ = [
    "trapezoid_model",
    "transit_model",
    "winn_duration",
    "impact_from_incl",
    "incl_from_impact",
    "a_rs_from_duration",
    "density_from_a_rs",
    "a_rs_from_density",
    "rp_rjup_from_depth",
    "rp_rearth_from_depth",
    "depth_from_rp_rs",
    "rp_rs_from_depth",
]
