# Catchment Health Metrics (CHM)

**CHM** is a Python library for assessing **catchment health**.  
It automates the retrieval and processing of topography, hydroclimate, vegetation, bushfire, road, and land use datasets for a given catchment boundary.  
From these datasets it generates groundwater and surface connectivity metrics (e.g., **Topographic Wetness Index (TWI)**, **Sediment Delivery Ratio (SDR)**) and uses them to create **risk profiles** for vegetation (NDVI), bushfire, roads, and land use (e.g., plantation forests, mining, industry, transport, residential, irrigation, cropping, and horticulture).  

The package is built with a **src/** layout for clean packaging and development with `pip install -e .`.  
It is tailored for **Australian conditions and datasets** (e.g., GA DEM, AWRA-L, DEA/AGCD) but the structure is flexible for adaptation elsewhere.

---

## Features

- **Topography**: DEM acquisition (GA WCS) and generation of slope, aspect, TPI, TRI, LS factor  
- **Connectivity**: Flow accumulation, SDR, TWI, sediment connectivity indices  
- **Vegetation & C-factor**: NDVI time series, C-factor derivation, summaries  
- **RUSLE & SDR-RUSLE**: Daily and annual erosion/sediment delivery estimates  
- **Hydroclimate**: Precipitation and temperature (AWAP/AGCD), runoff, ET, and soil moisture (AWRA-L, historical & projections)  
- **Bushfire**: Historical fire severity integration  
- **Roads**: National roads dataset and connectivity overlay  
- **Land Use (2023)**: Summaries for plantation forestry, mining, industry, cropping, horticulture, and more  
- **Pipelines**: Chain modules for reproducible end-to-end workflows  

---

## Installation

```bash
# From a local clone
pip install .

# Editable / development mode
pip install -e .

# Recommended: Use a clean environment
# (Because CHM depends on several heavy geospatial libraries,
#  we recommend installing it into a fresh environment to avoid dependency conflicts)

conda create -n chm python=3.11 -y
conda activate chm
pip install -e .[dev]
pytest -q

```

Python 3.10+ is required.

---

## Dependencies

CHM builds on the scientific Python stack plus geospatial libraries:

- **Core**: `numpy`, `pandas`, `scipy`, `matplotlib`  
- **Geospatial**: `rasterio`, `rioxarray`, `geopandas`, `shapely`, `pyproj`, `pysheds`  
- **I/O & CLI**: `tqdm`, `requests`, `typer`, `pydantic`  
- **STAC**: `pystac-client`, `odc-stac`  

Recommended install:

```bash
pip install -e .[dev]
```

Or a conda-friendly workflow:

```bash
conda create -n chm python=3.11 -y
conda activate chm
pip install numpy pandas scipy matplotlib rasterio rioxarray geopandas shapely pyproj pysheds requests typer pydantic tqdm
```

---

## Data requirements

CHM expects the following inputs (many auto-downloaded via pipelines):

| Data | Source | Usage |
|------|--------|-------|
| Catchment boundary (polygon) | User-provided shapefile | Defines extent and clipping |
| Sites (points) | User-provided | For per-site statistics and local catchments |
| DEM | GA SRTM 1-Sec Hydro-Enforced (DEA/GA WCS) | Terrain metrics |
| Vegetation | DEA Sentinel/Landsat via STAC | NDVI & C-factor |
| Soil erosion factors | K-factor from SLGA (user-provided), others derived | RUSLE inputs |
| Bushfire | National/state fire layers | Risk profile |
| Land use | ABARES/GA land use dataset (2023) | Risk profile |
| Roads | National roads dataset | Risk profile |
| Hydroclimate | AGCD/AWAP (precip, tmin, tmax), AWRA-L historical & projections (runoff, ET, soil moisture) | Hydroclimate metrics |

---

## Output structure (example)

```
Catchment Name (e.g., Coliban River)
├── Catchment Datasets
│   ├── Bushfire
│   ├── Hydrology
│   ├── Landuse
│   ├── National Roads
│   ├── RUSLE and SDR_RUSLE
│   ├── Surface and Groundwater Connectivity
│   ├── Topography
│   └── Vegetation
├── Sites Datasets
│   ├── Site_1
│   ├── Site_2
│   └── ...
├── Catchment Plots and Maps
└── Sites Plots and Maps
    ├── Site_1
    ├── Site_2
    └── ...
```

---

## Project structure

```
catchment-metrics/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  .pre-commit-config.yaml
  .github/workflows/ci.yml
  examples/
    CHM_Quickstart.ipynb
  src/chm/
    __init__.py
    cli.py
    dem_and_terrain.py
    veg_indices_and_c_factor.py
    surface_ground_water_connectivity.py
    rusle_and_sdr_rusle.py
    awap_historical_data.py
    awra_historical_data.py
    awra_projections_data.py
    historical_bushfire_risk_profile.py
    national_roads_risk_profile.py
    landuse_risk_profile.py
    appending_monitoring_data.py
  tests/
    test_imports.py
    Input data/
```

---

## Example usage

```python
from chm import landuse_risk_profile

landuse_risk_profile(
    CHM_Work_Space,
    Catchment_Shapefile_Path
)
```

## Worked Example

A full worked example, showing usage of the high-level interface, is provided in the
[CHM_example notebook](examples/CHM_example.ipynb).

This notebook demonstrates how to run the package end-to-end using example inputs, with step-by-step descriptions before each cell.

---

## License

This package is licensed under the **MIT License**.

---

## Acknowledgements

This project has been developed for **Water Research Australia** by the **Alluvium group**,  
for catchment health assessment in Australian contexts.  

Datasets and services are credited to their respective providers, including:  
- Geoscience Australia (GA DEM)  
- Digital Earth Australia (DEA)  
- Bureau of Meteorology (BoM AWRA-L)  
- Australian Gridded Climate Data (AGCD)  
- ABARES (Land use datasets)  

**Maintainer**: Jabbar Khaledi – [jabbarkhaledi88@gmail.com](mailto:jabbarkhaledi88@gmail.com)
