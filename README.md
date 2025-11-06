# Catchment Health Metrics (CHM)

**CHM** — *Catchment Health Metrics* — is a Python library for assessing **catchment health condition**.  
It automates the **end-to-end workflow** for evaluating catchment environmental condition using geospatial, hydrological, and remote sensing datasets.

CHM can **automatically download, process, and analyse** spatial and temporal data to generate **metrics, indices, and visual outputs** that describe:
- Environmental condition  
- Landscape function  
- Erosion and sediment connectivity risk  

---


## Key Features

| Domain | Description |
|---------|--------------|
| **Topography** | DEM acquisition and processing (GA WCS) for slope, aspect, LS factor, TPI, TRI |
| **Connectivity** | Flow accumulation, Topographic Wetness Index (TWI), Sediment Delivery Ratio (SDR), and hydrological connectivity indices |
| **Vegetation & C-Factor** | Automated NDVI calculation (DEA Landsat/Sentinel), C-factor derivation, riparian NDVI analysis, and annual summaries |
| **RUSLE & SDR-RUSLE** | Annual soil loss and delivered sediment (A = R × K × LS × C × P × SDR) |
| **Hydroclimate (Historical)** | Daily precipitation and temperature (AGCD/AWAP) and daily runoff, ET, and soil moisture (AWRA-L) |
| **Hydroclimate (Projections)** | Future hydrologic projections (Runoff, ET, Soil Moisture) via NCI THREDDS OPeNDAP |
| **Land Use (2023)** | Summaries by land use type, with SDR-based risk profiles |
| **DEA Land Cover Change** | Annual trend analysis using DEA Land Cover mosaics |
| **Bushfire** | Historical fire extent overlay, multi-year exposure window, and AUC metrics (burned area vs. SDR/TWI) |
| **Roads** | Road density and SDR/TWI connectivity exposure profiles with annual AUC metrics |
| **Monitoring Data** | Automated ingestion of per-site Excel files, summary statistics, thresholding, and visualisation |
| **Report Builder** | Automatically compiles all CHM module outputs into a formatted Word report (DOCX) for each catchment |

---

## Installation

```bash
# Standard installation
pip install .

# Editable / development mode
pip install -e .

# Recommended workflow (with conda)
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
Catchment_Name/
├── Catchment Datasets
│   ├── Topography
│   ├── Vegetation
│   │   ├── Satellite data
│   │   ├── Indices
│   │   │   ├── NDVI
│   │   │   └── C Factor
│   │   └── Riparian
│   ├── Surface and Groundwater Connectivity
│   ├── RUSLE and SDR_RUSLE
│   ├── Hydroclimate
│   │   ├── Historical
│   │   └── Projections
│   ├── Landuse
│   │   ├── DEA Landcover and Landuse 2023
│   │   └── Catchment Scale Landuse
│   ├── Historical Bushfire
│   ├── National Roads
│   └── Monitoring Data
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
catchment-health-metrics/
│
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── .editorconfig
├── .pre-commit-config.yaml
├── .github/
│ └── workflows/
│ └── ci.yml
│
├── examples/
│ ├── CHM_Quickstart.ipynb
│ └── CHM_Example_Pipeline.ipynb
│
├── src/
│ └── chm/
│ ├── init.py
│ ├── cli.py
│ ├── utils/
│ │ ├── paths.py
│ │ ├── plots.py
│ │ └── helpers.py
│ │
│ ├── topography.py
│ ├── vegetation.py
│ ├── connectivity.py
│ ├── rusle.py
│ ├── hydroclimate_historical.py
│ ├── hydroclimate_projection.py
│ ├── landuse_2023.py
│ ├── dea_landuse.py
│ ├── bushfire.py
│ ├── roads.py
│ ├── monitoring_data.py
│ ├── generate_report.py
│ └── py.typed
│
└── tests/
├── test_imports.py
├── test_functions.py
├── Input data/
└── sample_configs/
```

---

## Example usage

```python
from chm import veg_indices_and_c_factor, rusle_and_sdr_rusle, build_report
from chm.veg_indices import VegConfig

cfg = VegConfig(
    chm_workspace=r"C:\CHM\Output",
    catchment_path=r"C:\CHM\Input\Catchment\Coliban_River.shp",
    sites_path=r"C:\CHM\Input\Sites\Coliban_Sites.shp",
    datetime_range="2010-01-01/2025-10-25",
    cloud_cover_lt=20,
    riparian_buffer_m=30
)

veg_indices_and_c_factor(cfg)
build_report(r"C:\CHM\Output", r"C:\CHM\Input\Catchment\Coliban_River.shp")
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
