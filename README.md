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

# Installation

To use the Catchment Health Metrics library, you need a suitable Python environment with the required scientific and geospatial libraries.

Set up a Base Python Environment, include scientific Python stack:`numpy`, `pandas`, `jupyter`,`matplotlib`

For **Windows users**, the simplest option is to install **Anaconda**, https://www.anaconda.com/download which provides all of these in one package.

Additional dependencies used by the Catchment Health Metrics library are listed in **`requirements.txt`**.

### Installation steps

**Step 1 — Clone the repository (Git Bash or any terminal):**

```bash
cd "C:\GitHub"
git clone https://github.com/mosaic-insights/Catchment-Health-Metrics.git
```

**Step 2 — Create and activate environment (Anaconda Prompt or VS Code terminal):**

```bash
conda create -n chm python=3.11
conda activate chm
```

**Step 3 — Install the package (editable mode):**

```bash
cd C:\GitHub\Catchment-Health-Metrics
pip install -e .
```

**Step 4 — Install Jupyter kernel (optional but recommended):**

```bash
pip install ipykernel
python -m ipykernel install --user --name chm --display-name "Python (chm)"
```

**Step 5 — Run in Jupyter:**
Open Jupyter Notebook/Lab and select:

**Python (chm)** kernel


### Updating the package

If there are updates in the GitHub repository:

```bash
conda activate chm
cd C:\GitHub\Catchment-Health-Metrics
git pull
pip install -e .
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

## Data Requirements

CHM expects the following input datasets — many of which are **automatically downloaded and processed** through the CHM pipelines.  
User-provided layers (e.g., catchment and site shapefiles) define spatial context for all analyses.

| Data | Source | Purpose / Usage |
|------|---------|----------------|
| **Catchment boundary (polygon)** | User-provided shapefile | Defines spatial extent for all modules and clipping boundaries |
| **Monitoring sites (points/polygons)** | User-provided | Used for per-site analysis, zonal statistics, and report generation |
| **DEM (Digital Elevation Model)** | GA SRTM 1-Sec Hydro-Enforced via DEA WCS | Foundation for slope, aspect, LS factor, flow direction, and flow accumulation |
| **Vegetation indices (NDVI, C-factor)** | DEA Landsat / Sentinel-2 via STAC API | Derivation of NDVI, C-factor, and riparian NDVI metrics |
| **Soil erosion factors (R, K, P)** | K-factor from SLGA or user input; R/P from AWRA-L or empirical rules | Inputs to RUSLE and SDR-RUSLE erosion modelling |
| **Hydroclimate – Historical** | AGCD (AWAP) precipitation & temperature; AWRA-L runoff, ET, soil moisture | Long-term daily catchment and site-scale hydroclimate metrics |
| **Hydroclimate – Projections** | AWRAL hydrologic projections via NCI THREDDS (Historical, RCP4.5, RCP8.5) | Future hydroclimate scenario analysis |
| **Bushfire polygons** | National/state fire history datasets or ArcGIS REST service | Fire exposure profiling and AUC time-series (burned area vs SDR/TWI) |
| **Land use (2023)** | ABARES / DEA Land Cover | SDR-based exposure and temporal trend analysis |
| **DEA Land Cover (annual mosaics)** | Geoscience Australia (DEA) | Multi-year land-cover trend and change analysis |
| **Road networks** | National Roads dataset (ArcGIS REST or local file) | SDR/TWI exposure profiles and connectivity risk assessment |
| **Monitoring data (field Excel files)** | User-provided (e.g., Site_7.xlsx) | Water-quality time-series, exceedance metrics, and per-site statistics |

---

### Notes

- Most datasets are **auto-retrieved** (DEA, AWRA-L, AGCD, NCI THREDDS) through CHM’s internal STAC or REST pipelines.  
- User input is required only for **catchment**, **site**, and **monitoring** layers — ensuring flexibility across projects.  
- All rasters are **harmonised to the DEM grid** for consistent resolution and CRS alignment across modules.

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
from chm.vegetation import VegConfig, veg_indices_and_c_factor

CHM_Work_Space = r"C:\Users\...\Output"
Catchment_Shapefile_Path = r"C:\Users\...\Coliban River.shp"

cfg = VegConfig(
    chm_workspace=CHM_Work_Space,
    catchment_path=Catchment_Shapefile_Path,
    datetime_range="2024-01-01/2025-10-25",
    catchment_crs=None, #"EPSG:3308"
    cloud_cover_lt=20,
    riparian_buffer_m=30.0,
)
veg_indices_and_c_factor(cfg)
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


## 📖 Citation

If you use this package in your research, report, or project, please cite it as:

**APA style:**

Khaledi, J., Nyman, P., & Richards, P. (2026). *Catchment Health Metrics (CHM): A Python package for catchment environmental condition analysis* (Version 0.1.0) [Software]. GitHub. https://github.com/mosaic-insights/Catchment-Health-Metrics

**BibTeX:**
```bibtex
@software{khaledi2026chm,
  author = {Khaledi, Jabbar and Nyman, Petter and Richards, Paul},
  title = {Catchment Health Metrics (CHM): A Python package for catchment-scale environmental analysis},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/mosaic-insights/Catchment-Health-Metrics}
}
