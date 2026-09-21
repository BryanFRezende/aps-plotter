"""APS-oriented publication plotting utilities."""

from .plotting import (
    APS_COLORS,
    APS_LINESTYLES,
    APS_MARKERS,
    APS_MIN_LINEWIDTH_PT,
    APS_MIN_MARKER_DIAMETER_PT,
    APS_PREFERRED_RASTER_DPI,
    APS_RCPARAMS,
    APS_WIDTHS_IN,
    aps_plot,
)

__all__ = [
    "aps_plot",
    "APS_COLORS",
    "APS_MARKERS",
    "APS_LINESTYLES",
    "APS_WIDTHS_IN",
    "APS_RCPARAMS",
    "APS_MIN_LINEWIDTH_PT",
    "APS_MIN_MARKER_DIAMETER_PT",
    "APS_PREFERRED_RASTER_DPI",
]

__version__ = "0.1.0"
