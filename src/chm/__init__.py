"""
Catchment Health Metrics (CHM)

A lightweight, functions-first toolkit for catchment-scale terrain, vegetation,
hydroclimate and risk profiling workflows.

This package exposes submodules such as:
- chm.topography
- chm.vegetation
- chm.connectivity
- chm.rusle
- chm.hydroclimate_awap
- chm.hydroclimate_awral_historical
- chm.hydroclimate_awral_projection
- chm.landuse
- chm.bushfire
- chm.roads
- chm.monitoring_data
"""
from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "topography",
    "vegetation",
    "connectivity",
    "rusle",
    "hydroclimate_awap",
    "hydroclimate_awral_historical",
    "hydroclimate_awral_projection",
    "landuse",
    "bushfire",
    "roads",
    "monitoring_data",
]

try:
    __version__ = version("catchment-health-metrics")
except PackageNotFoundError:  # during local development
    __version__ = "0.1.0"
