# src/chm/connectivity.py
from __future__ import annotations

# ============================ Imports ============================
# --- stdlib ---
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --- numerical ---
import numpy as np
import pandas as pd

# --- geospatial ---
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.plot import plotting_extent
from pysheds.grid import Grid
from shapely.geometry import Point

# --- plotting ---
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# ============================ Configuration ============================

@dataclass
class ConnectivityConfig:
    """
    Configuration for surface/groundwater connectivity processing.

    Parameters
    ----------
    chm_workspace : str
        Root workspace where the catchment subfolders are/will be created.
    catchment_path : str
        Path to catchment boundary (any vector format geopandas can read).
    sites_path : str
        Path to sites vector. This module assumes point sites in the final
        per-site plots (polygons are used for masking/summary if provided).
    # SDR parameters (Hamel et al.-style logistic transform):
    sdr_max : float
        Maximum SDR value (0..1).
    ic0 : float
        Logistic midpoint for the connectivity index IC.
    k : float
        Logistic steepness parameter for SDR.
    # Stream extraction threshold:
    stream_area_threshold_m2 : float
        Contributing area threshold (m²) to label "stream" pixels.
        (Your previous code used ~13,000 m².)
    """
    chm_workspace: str
    catchment_path: str
    sites_path: str

    sdr_max: float = 0.8
    ic0: float = 0.5
    k: float = 1.0

    stream_area_threshold_m2: float = 1.3e4  # ≈ 13,000 m²


# ============================ Small helpers ============================

def _ensure_dirs(paths: List[str]) -> None:
    """Create the given directories if missing."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _read_dem(catch_datasets: str) -> Tuple[np.ndarray, dict, rio.Affine, float, float, float, str]:
    """
    Load the projected DEM prepared by your terrain step.

    Returns
    -------
    dem : np.ndarray
        DEM as float array with NaNs for nodata.
    meta : dict
        Rasterio metadata (used to write new rasters).
    transform : Affine
        Georeferencing transform.
    xres, yres : float
        Pixel sizes in map units.
    nodata : float
        Nodata value in the on-disk DEM (may be NaN).
    crs_wkt : str
        CRS in WKT/PROJ form compatible with geopandas/rasterio.
    """
    dem_path = os.path.join(catch_datasets, "Topography", "DEM.tif")
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path} (run dem_and_terrain first)")

    with rio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        arr = np.where(arr == nodata, np.nan, arr)
        meta = src.meta.copy()
        transform = src.transform
        xres = transform.a
        yres = abs(transform.e)
        crs = src.crs

    return arr, meta, transform, xres, yres, nodata, crs


def _write_single_band(path: str, data: np.ndarray, template_meta: dict, dtype: str = "float32") -> None:
    """Write a single-band raster using `template_meta` as a template."""
    meta = template_meta.copy()
    meta.update({"count": 1, "dtype": dtype, "nodata": np.nan})
    with rio.open(path, "w", **meta) as dst:
        dst.write(data.astype(dtype), 1)


def _gradient_slope_aspect(dem: np.ndarray, xres: float, yres: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute slope/aspect products.

    Returns
    -------
    slope_ratio : np.ndarray
        |∇z| (rise/run) from dz/dx, dz/dy.
    slope_rad : np.ndarray
        arctan(slope_ratio).
    slope_deg : np.ndarray
        slope in degrees.
    aspect_rad : np.ndarray
        0..2π, 0 ≈ North (using arctan2(dz_dy, -dz_dx)).
    """
    dz_dx, dz_dy = np.gradient(dem, xres, yres)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_rad = np.arctan(slope_ratio)
    slope_deg = np.degrees(slope_rad)

    # aspect = angle of steepest descent (consistent with your earlier code)
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_rad = np.where(aspect_rad < 0, 2 * np.pi + aspect_rad, aspect_rad)
    return slope_ratio, slope_rad, slope_deg, aspect_rad


def _prepare_folders(workspace: str, catchment_name: str) -> Dict[str, str]:
    """Create and return important folder paths."""
    catchment_folder = os.path.join(workspace, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    con_folder = os.path.join(catch_datasets, "Surface and Groundwater Connectivity")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")

    _ensure_dirs([catchment_folder, catch_datasets, con_folder, sites_datasets, sites_plots, catch_plots])
    return {
        "catchment_folder": catchment_folder,
        "catch_datasets": catch_datasets,
        "con_folder": con_folder,
        "sites_datasets": sites_datasets,
        "sites_plots": sites_plots,
        "catch_plots": catch_plots,
    }


def _list_annual_c_rasters(c_dir: str) -> Dict[str, str]:
    """
    Return {year: path} for annual C-Factor rasters in the expected folder.
    Accepts files named like: C_Factor_YYYY.tif
    """
    out: Dict[str, str] = {}
    if not os.path.isdir(c_dir):
        return out
    for f in os.listdir(c_dir):
        if f.lower().endswith(".tif") and "c_factor" in f.lower():
            m = re.search(r"(\d{4})", f)
            if m:
                out[m.group(1)] = os.path.join(c_dir, f)
    return out

# ============================ Core processing ============================
def process_surface_and_groundwater_connectivity(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    Sites_Shapefile_Path: str,
    *,
    sdr_max: float = 0.8,
    ic0: float = 0.5,
    k: float = 1.0,
    stream_area_threshold_m2: float = 1.3e4,
) -> Tuple[str, str, str, str, str]:
    """
    Compute TWI, LS (slope length-gradient factor), SDR (per year from annual C rasters),
    and per-site clipped outputs/plots.

    Returns
    -------
    all_sites_gpkg : str
    ls_path        : str
    twi_path       : str
    sdr_output_dir : str
    sites_datasets : str
    """
    print("Starting processing surface and groundwater connectivity...")

    # -------- Folder structure & inputs --------
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace("_", " ")
    paths = _prepare_folders(CHM_Work_Space, catchment_name)
    catch_datasets = paths["catch_datasets"]
    con_folder = paths["con_folder"]
    catch_plots = paths["catch_plots"]
    sites_datasets = paths["sites_datasets"]
    sites_plots = paths["sites_plots"]

    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation", "Indices", "C Factor", "Annual")

    # -------- DEM & basic derivatives --------
    dem, dem_meta, transform, xres, yres, dem_nodata, dem_crs = _read_dem(catch_datasets)
    pixel_area = xres * yres

    slope_ratio, slope_rad, slope_deg, aspect_rad = _gradient_slope_aspect(dem, xres, yres)

    # -------- Flow direction/accumulation (pysheds) --------
    dem_path = os.path.join(catch_datasets, "Topography", "DEM.tif")
    grid = Grid.from_raster(dem_path)
    dem_grid = grid.read_raster(dem_path)
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.fill_depressions(dem_filled)
    inflated = grid.resolve_flats(dem_filled)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap).astype("float32")
    area_m2 = acc * pixel_area

    # -------- Streams mask (contributing area threshold) --------
    streams_mask = (area_m2 > float(stream_area_threshold_m2)).astype("float32")
    streams_mask = np.where(np.isnan(dem), np.nan, streams_mask)
    streams_path = os.path.join(con_folder, "Streams.tif")
    _write_single_band(streams_path, streams_mask, dem_meta)

    # -------- Thresholded average slope Sth (0.5%..100% cap) --------
    Sth = np.where(slope_ratio < 0.005, 0.005, np.where(slope_ratio <= 1, slope_ratio, 1.0))
    Sth = np.where(np.isnan(dem), np.nan, Sth)
    Sth_path = os.path.join(con_folder, "Average Thresholded Slopes.tif")
    _write_single_band(Sth_path, Sth, dem_meta)

    # -------- TWI = ln(a / tan(β)) --------
    with np.errstate(divide="ignore", invalid="ignore"):
        TWI = np.log(acc / np.tan(slope_rad))
    TWI = np.where(np.isfinite(TWI), TWI, np.nan)
    TWI = np.where(np.isnan(dem), np.nan, TWI)
    TWI_path = os.path.join(con_folder, "Topographic Wetness Index.tif")
    _write_single_band(TWI_path, TWI, dem_meta)

    # -------- LS (slope length-gradient factor) --------
    specific_area = np.sqrt(acc * pixel_area)
    specific_area = np.where(specific_area > 122.0, 122.0, specific_area)  # cap for robustness

    xi = np.abs(np.sin(aspect_rad)) + np.abs(np.cos(aspect_rad))  # aspect term

    slope_pct = slope_ratio * 100.0
    Si = np.zeros_like(slope_pct, dtype="float32")
    mask_lt9 = slope_pct < 9
    Si[mask_lt9] = 10.8 * np.sin(np.radians(slope_deg[mask_lt9])) + 0.03
    Si[~mask_lt9] = 16.8 * np.sin(np.radians(slope_deg[~mask_lt9])) - 0.50

    m = np.zeros_like(slope_pct, dtype="float32")
    m[slope_pct <= 1] = 0.2
    m[(slope_pct > 1) & (slope_pct <= 3.5)] = 0.3
    m[(slope_pct > 3.5) & (slope_pct <= 5)] = 0.4
    m[(slope_pct > 5) & (slope_pct <= 9)] = 0.5
    mask_gt9 = slope_pct > 9
    if np.any(mask_gt9):
        slope_rad_high = np.arctan(slope_pct[mask_gt9] / 100.0)
        beta = (np.sin(slope_rad_high) / 0.0896) / ((3 * (np.sin(slope_rad_high) ** 0.8)) + 0.56)
        m[mask_gt9] = beta / (1 + beta)

    D = xres  # pixel size
    with np.errstate(divide="ignore", invalid="ignore"):
        LSi = Si * (((specific_area + D**2) ** (m + 1)) - (specific_area ** (m + 1))) / (
            (D ** (m + 2)) * (xi ** m) * (22.13 ** m)
        )
    LSi = np.where(np.isnan(dem), np.nan, LSi)
    LS_path = os.path.join(con_folder, "Slope length-gradient factor.tif")
    _write_single_band(LS_path, LSi, dem_meta)

    # -------- Save Flow Accumulation (for reference) --------
    acc_path = os.path.join(con_folder, "Flow Accumulation.tif")
    _write_single_band(acc_path, acc, dem_meta)

    # -------- SDR (per year, using annual C-factor) --------
    sdr_output_dir = os.path.join(con_folder, "SDR")
    _ensure_dirs([sdr_output_dir])

    # Weighted upstream averages for Sth (Av_Sth)
    Sth_raster = grid.read_raster(Sth_path).astype("float32")
    acc_Sth = grid.accumulation(fdir=fdir, weights=Sth_raster)
    acc_no0 = np.where(acc == 0, np.nan, acc)  # avoid divide-by-zero
    Av_Sth = acc_Sth / acc_no0

    # Stream cell indices (for downslope traversal)
    stream_cells = np.where(streams_mask == 1.0)

    # Annual C-factor rasters
    c_years = _list_annual_c_rasters(annual_c_dir)
    if not c_years:
        print(f"[WARN] No annual C-Factor rasters found in: {annual_c_dir}")

    # Per-year SDR & derived rasters
    for year, c_fp in sorted(c_years.items()):
        with rio.open(c_fp) as src_c:
            c_factor = src_c.read(1).astype("float32")
            c_factor = np.where(c_factor == src_c.nodata, np.nan, c_factor)

        # Threshold C to avoid zeros:
        Cth = np.where(c_factor < 0.001, 0.001, c_factor)
        Cth_path = os.path.join(sdr_output_dir, f"Average Thresholded C factor_{year}.tif")
        _write_single_band(Cth_path, Cth, dem_meta)

        # Weighted upstream average for C (Av_Cth)
        Cth_raster = grid.read_raster(Cth_path).astype("float32")
        acc_Cth = grid.accumulation(fdir=fdir, weights=Cth_raster)
        Av_Cth = acc_Cth / acc_no0

        # ---- Downslope path length term (Ddn) and distance to stream ----
        distance_to_stream = np.zeros_like(dem, dtype="float32")
        Ddn = np.zeros_like(dem, dtype="float32")

        st_indices = list(zip(stream_cells[0], stream_cells[1]))

        # 8-neighbour deltas (D8)
        dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
        dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])

        diag = (xres**2 + yres**2) ** 0.5
        grid_lengths = np.array([yres, diag, xres, diag, yres, diag, xres, diag], dtype="float32")

        visited = np.zeros_like(dem, dtype=bool)
        visited[stream_cells] = True

        # BFS from streams following reverse flowdir
        while st_indices:
            row, col = st_indices.pop(0)
            curr_dist = distance_to_stream[row, col]
            for i in range(8):
                nr = row + dy[i]
                nc = col + dx[i]
                if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                    if fdir[nr, nc] == dirmap[(i + 4) % 8]:
                        if not visited[nr, nc]:
                            visited[nr, nc] = True
                            step_len = grid_lengths[i]
                            denom = (Cth[nr, nc] * Sth[nr, nc])
                            add = step_len / denom if (denom is not None and denom > 0) else 0.0
                            Ddn[nr, nc] = Ddn[row, col] + add
                            distance_to_stream[nr, nc] = curr_dist + step_len
                            st_indices.append((nr, nc))

        distance_to_stream = np.where(np.isnan(dem), np.nan, distance_to_stream)
        Ddn = np.where(np.isnan(dem), np.nan, Ddn)

        # ---- Upslope area (Dup) ----
        Dup = Av_Cth * Av_Sth * np.sqrt(area_m2)
        Dup = np.where(np.isnan(dem), np.nan, Dup)

        # ---- Connectivity index & SDR ----
        Ddn_safe = np.where(Ddn == 0, np.nan, Ddn)
        with np.errstate(divide="ignore", invalid="ignore"):
            IC = np.log10(Dup / Ddn_safe)
        IC = np.where(np.isfinite(IC), IC, np.nan)

        SDR = float(sdr_max) / (1.0 + np.exp((float(ic0) - IC) / float(k)))

        # Save yearly products
        for name, data in {
            f"Distance to Stream_{year}.tif": distance_to_stream,
            f"Downslope Path_{year}.tif": Ddn,
            f"Upslope Area_{year}.tif": Dup,
            f"Connectivity Index_{year}.tif": IC,
            f"SDR_{year}.tif": SDR,
        }.items():
            _write_single_band(os.path.join(sdr_output_dir, name), data, dem_meta)

        print(f"[OK] SDR {year} processed.")

        # -------- Per-site clipping & plots for TWI + SDR(year) --------
        if not os.path.exists(all_sites_gpkg):
            print(f"[WARN] Sites dataset not found (expected from terrain step): {all_sites_gpkg}")
            continue

        try:
            sites_gdf = gpd.read_file(all_sites_gpkg)
        except Exception as e:
            print(f"[WARN] Could not read sites GPKG: {e}")
            continue

        sites_pts = None
        try:
            sites_pts = gpd.read_file(Sites_Shapefile_Path)
        except Exception:
            pass  # optional overlay for full-raster plots

        raster_files = [
            TWI_path,
            os.path.join(sdr_output_dir, f"SDR_{year}.tif"),
        ]

        for idx, row in sites_gdf.iterrows():
            try:
                site_id = row.get("id", idx)
                geom = row.geometry
                site_attrs = row.drop(labels="geometry").to_dict()
                site_gdf = gpd.GeoDataFrame([site_attrs], geometry=[geom], crs=sites_gdf.crs)

                site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
                site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
                _ensure_dirs([site_data_dir, site_plot_dir])

                for rpath in raster_files:
                    if not os.path.exists(rpath):
                        print(f"[WARN] Missing raster: {rpath}")
                        continue
                    short = os.path.splitext(os.path.basename(rpath))[0]

                    with rio.open(rpath) as src:
                        # full-extent context plot
                        full = src.read(1).astype("float32")
                        full = np.where(full == src.nodata, np.nan, full)
                        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(full, cmap="viridis", extent=extent, origin="upper")
                        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
                        cbar.set_label(short)

                        if sites_pts is not None:
                            pts = sites_pts
                            if pts.crs != src.crs:
                                pts = pts.to_crs(src.crs)
                            pts.plot(ax=ax, color="red", markersize=15)
                            if "id" in pts.columns:
                                for _, pr in pts.iterrows():
                                    ax.annotate(f"Site {pr['id']}",
                                                (pr.geometry.x, pr.geometry.y),
                                                xytext=(3, 3), textcoords="offset points",
                                                fontsize=7, color="red")
                        ax.set_title(short, fontsize=11, fontweight="bold")
                        ax.set_xlabel("Longitude", fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude", fontsize=10, fontweight="bold")
                        ax.tick_params(axis="both", labelsize=10)
                        plt.tight_layout()
                        plt.savefig(os.path.join(catch_plots, f"{short}.png"), dpi=300)
                        plt.close()

                        # clip to site polygon
                        v = site_gdf.to_crs(src.crs) if (site_gdf.crs != src.crs) else site_gdf
                        out_img, out_tr = rio_mask(src, [v.geometry.iloc[0]], crop=True, filled=False)
                        clipped = out_img[0].filled(np.nan).astype("float32")

                        # save clipped raster
                        meta = src.meta.copy()
                        meta.update({
                            "driver": "GTiff",
                            "height": out_img.shape[1],
                            "width": out_img.shape[2],
                            "transform": out_tr,
                            "crs": src.crs,
                            "nodata": src.nodata
                        })
                        _write_single_band(os.path.join(site_data_dir, f"{short}.tif"), clipped, meta)

                        # plot clipped raster
                        left, top = out_tr[2], out_tr[5]
                        pxw, pxh = out_tr[0], -out_tr[4]
                        width, height = out_img.shape[2], out_img.shape[1]
                        ext_clip = [left, left + pxw * width, top - pxh * height, top]

                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(out_img[0], extent=ext_clip, cmap="viridis", origin="upper")
                        plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85).set_label(short)
                        v.boundary.plot(ax=ax, color="black", linewidth=1.2)
                        ax.set_title(f"{short} - Site {site_id}", fontsize=11, fontweight="bold")
                        ax.set_xlabel("Longitude", fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude", fontsize=10, fontweight="bold")
                        ax.tick_params(axis="both", labelsize=10)
                        plt.tight_layout()
                        plt.savefig(os.path.join(site_plot_dir, f"{short}.png"), dpi=200)
                        plt.close()

                        # ===================== ADD NEW COLUMN(S) LIKE ROADS =====================
                        # if os.path.basename(rpath) == f"SDR_{year}.tif":
                        #     mean_sdr = float(np.nanmean(clipped)) if np.isfinite(clipped).any() else np.nan
                        #     colname = f"Mean_SDR_{year}"
                        #     if colname not in sites_gdf.columns:
                        #         sites_gdf[colname] = np.nan
                        #     sites_gdf.loc[idx, colname] = mean_sdr
                        # =======================================================================

            except Exception as e:
                print(f"[WARN] Error processing site {row.get('id', idx)}: {e}")
                continue

        # === Write-back like roads_hist.py: replace the SAME layer, not the whole file ===
        # try:
        #     import fiona
        #     layers = fiona.listlayers(all_sites_gpkg) if os.path.exists(all_sites_gpkg) else []
        #     target_layer = layers[0] if layers else None  # assumes first layer is sites
        #     if target_layer:
        #         sites_gdf.to_file(all_sites_gpkg, layer=target_layer, driver="GPKG", if_exists="replace")
        #     else:
        #         sites_gdf.to_file(all_sites_gpkg, driver="GPKG")
        #     sites_gdf.drop(columns="geometry").to_csv(
        #         os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv"),
        #         index=False
        #     )
        #     #print(f"[OK] Updated Sites layer written to: {all_sites_gpkg}")
        # except Exception as e:
        #     print(f"[WARN] Could not write updated sites to GPKG/CSV: {e}")

    return all_sites_gpkg, LS_path, TWI_path, sdr_output_dir, sites_datasets

# ============================ NDVI exposure profiles (NDVI vs SDR) ============================
def ndvi_cumulative_risk_profiles(CHM_Work_Space: str, Catchment_Shapefile_Path: str) -> None:
    """
    Create NDVI cumulative exposure profiles vs SDR per year for each site,
    and write AUC time series back to the Sites GPKG.
    """
    print("Starting NDVI cumulative risk profiles by SDR...")

    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace("_", " ")
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")

    _ensure_dirs([catchment_folder, catch_datasets, sites_datasets, sites_plots])

    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    annual_ndvi_dir = os.path.join(catch_datasets, "Vegetation", "Indices", "NDVI", "Annual")
    sdr_output_dir = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")

    if not os.path.exists(all_sites_gpkg):
        print(f"[WARN] Sites GPKG missing: {all_sites_gpkg}")
        return

    try:
        sites_gdf = gpd.read_file(all_sites_gpkg)
    except Exception as e:
        print(f"[WARN] Could not read sites_gdf: {e}")
        return

    ndvi_files = [f for f in os.listdir(annual_ndvi_dir) if f.endswith(".tif") and "NDVI" in f]
    sdr_files = [f for f in os.listdir(sdr_output_dir) if f.endswith(".tif") and "SDR" in f]
    ndvi_dict = {re.search(r"\d{4}", f).group(): f for f in ndvi_files if re.search(r"\d{4}", f)}
    sdr_dict  = {re.search(r"\d{4}", f).group(): f for f in sdr_files if re.search(r"\d{4}", f)}
    common_years = sorted(set(ndvi_dict.keys()) & set(sdr_dict.keys()))

    # global SDR min/max for standardised x-axes
    sdr_min, sdr_max = np.inf, -np.inf
    for f in sdr_files:
        try:
            with rio.open(os.path.join(sdr_output_dir, f)) as src:
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                _mn, _mx = np.nanmin(arr), np.nanmax(arr)
                if np.isfinite(_mn): sdr_min = min(sdr_min, _mn)
                if np.isfinite(_mx): sdr_max = max(sdr_max, _mx)
        except Exception:
            pass
    if not np.isfinite(sdr_min) or not np.isfinite(sdr_max) or sdr_min >= sdr_max:
        sdr_min = sdr_max = None  # let Matplotlib autoscale

    def _auc(x: np.ndarray, y_pct: np.ndarray) -> float:
        if x.size == 0 or y_pct.size == 0:
            return np.nan
        mask = np.isfinite(x) & np.isfinite(y_pct)
        if not np.any(mask):
            return np.nan
        x_in = x[mask]
        y_in = y_pct[mask]
        if x_in.size < 2:
            return np.nan
        return float(np.trapz(y_in, x_in))  # raw AUC

    for i, row in sites_gdf.iterrows():
        site_id = row.get("id", i)
        geom = row.geometry
        site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
        os.makedirs(site_plot_dir, exist_ok=True)

        auc_series: Dict[int, float] = {}

        for y in common_years:
            ndvi_fp = os.path.join(annual_ndvi_dir, ndvi_dict[y])
            sdr_fp  = os.path.join(sdr_output_dir,   sdr_dict[y])

            try:
                with rio.open(ndvi_fp) as ndvi_src:
                    ndvi_img, _ = rio_mask(ndvi_src, [geom], crop=True)
                    ndvi = ndvi_img[0].astype("float32")
                    if ndvi_src.nodata is not None:
                        ndvi = np.where(ndvi == ndvi_src.nodata, np.nan, ndvi)

                with rio.open(sdr_fp) as sdr_src:
                    sdr_img, _ = rio_mask(sdr_src, [geom], crop=True)
                    sdr = sdr_img[0].astype("float32")
                    if sdr_src.nodata is not None:
                        sdr = np.where(sdr == sdr_src.nodata, np.nan, sdr)
            except Exception as e:
                print(f"[WARN] Site {site_id}, year {y}: masking failed: {e}")
                continue

            if sdr.shape != ndvi.shape:
                print(f"[WARN] Site {site_id}, year {y}: shape mismatch NDVI vs SDR.")
                continue

            valid = np.isfinite(ndvi) & np.isfinite(sdr)
            if not np.any(valid):
                continue

            sdr_vals = sdr[valid]
            order = np.argsort(sdr_vals)
            sdr_sorted = sdr_vals[order]
            cum_pct = np.arange(1, sdr_sorted.size + 1) / sdr_sorted.size * 100.0

            # Plot cumulative profile
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sdr_sorted, cum_pct, color="black", linewidth=2)
            ax.set_xlabel("SDR Value", fontsize=10, fontweight="bold")
            ax.set_ylabel("Cumulative NDVI Pixels (%)", fontsize=10, fontweight="bold")
            ax.tick_params(axis="both", labelsize=10)
            ax.set_title(f"Vegetation Exposure Profile (NDVI vs SDR) - {y} (Site {site_id})", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            if sdr_min is not None and sdr_max is not None:
                ax.set_xlim(sdr_min, sdr_max)
            plt.tight_layout()
            plt.savefig(os.path.join(site_plot_dir, f"Site_{site_id}_Exposure_NDVI_SDR_{y}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # AUC (raw) for time series and column write
            try:
                auc_series[int(y)] = _auc(sdr_sorted, cum_pct)
            except Exception:
                pass

        # AUC time series plot + (deactivated) column writes
        if auc_series:
            ys = sorted(auc_series.keys())
            auc_vals = [auc_series[y] for y in ys]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ys, auc_vals, marker="o", color="black", linewidth=1)
            ax.set_xlabel("Year", fontsize=10, fontweight="bold")
            ax.set_ylabel("Area Under Curve", fontsize=10, fontweight="bold")
            ax.set_title(f"Vegetation Exposure Profile (NDVI vs SDR) - Site {site_id}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(site_plot_dir, f"Site_{site_id}_AUC_NDVI_SDR.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # write AUCs back to GPKG (append columns) — DEACTIVATED
            # for y, v in auc_series.items():
            #     col = f"AUC_NDVI_SDR_{y}"
            #     if col not in sites_gdf.columns:
            #         sites_gdf[col] = np.nan
            #     sites_gdf.loc[i, col] = v

    # === Roads-style write-back to GPKG/CSV — DEACTIVATED ===
    # try:
    #     import fiona
    #     layers = fiona.listlayers(all_sites_gpkg) if os.path.exists(all_sites_gpkg) else []
    #     target_layer = layers[0] if layers else None  # assumes first layer is sites
    #     if target_layer:
    #         sites_gdf.to_file(all_sites_gpkg, layer=target_layer, driver="GPKG", if_exists="replace")
    #     else:
    #         sites_gdf.to_file(all_sites_gpkg, driver="GPKG")
    #     # keep CSV mirror in sync (no geometry)
    #     sites_gdf.drop(columns="geometry").to_csv(
    #         os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv"),
    #         index=False
    #     )
    #     #print(f"Updated AUC metrics written to: {all_sites_gpkg}")
    # except Exception as e:
    #     print(f"[WARN] Could not write updated sites GPKG with AUC columns: {e}")

    print("NDVI cumulative exposure profiles by SDR are completed.")

# ============================ Convenience wrapper ============================

def surface_ground_water_connectivity(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    Sites_Shapefile_Path: str,
    *,
    sdr_max: float = 0.8,
    ic0: float = 0.5,
    k: float = 1.0,
    stream_area_threshold_m2: float = 1.3e4,
) -> None:
    """
    High-level convenience wrapper (keeps your two-step flow):
    1) Compute connectivity metrics (TWI, LS, SDR per year) + site products.
    2) Build NDVI–SDR exposure profiles and AUC time series.
    """
    process_surface_and_groundwater_connectivity(
        CHM_Work_Space=CHM_Work_Space,
        Catchment_Shapefile_Path=Catchment_Shapefile_Path,
        Sites_Shapefile_Path=Sites_Shapefile_Path,
        sdr_max=sdr_max,
        ic0=ic0,
        k=k,
        stream_area_threshold_m2=stream_area_threshold_m2,
    )
    ndvi_cumulative_risk_profiles(
        CHM_Work_Space=CHM_Work_Space,
        Catchment_Shapefile_Path=Catchment_Shapefile_Path,
    )