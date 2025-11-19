from importlib.metadata import version, PackageNotFoundError
from .utils.paths import get_project_root, default_paths  # re-export for workspace helpers

try:
    __version__ = version("catchment-health-metrics")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .topography import dem_and_terrain
from .vegetation import veg_indices_and_c_factor
from .connectivity import surface_ground_water_connectivity, process_surface_and_groundwater_connectivity
from .rusle import rusle_and_sdr_rusle
from .hydroclimate_historical import hydroclimate_historical
from .hydroclimate_projection import hydroclimate_projections
from .landuse_2023 import landuse_2023
from .dea_landuse import dea_landuse_change
from .bushfire import historical_bushfire
from .roads import national_roads
from .monitoring_data import monitoring_data
from .generate_report import build_report

__all__ = [
    "get_project_root", "default_paths",
    "dem_and_terrain", "veg_indices_and_c_factor",
    "surface_ground_water_connectivity", "process_surface_and_groundwater_connectivity",
    "rusle_and_sdr_rusle",
    "hydroclimate_historical", "hydroclimate_projections",
    "landuse_2023", "dea_landuse_change",
    "historical_bushfire", "national_roads",
    "monitoring_data", "build_report",
    "__version__",
]
