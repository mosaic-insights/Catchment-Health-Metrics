# src/chm/rusle.py
from __future__ import annotations

# =============== stdlib ===============
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =============== numerical / data ===============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# =============== geospatial ===============
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
import fiona


# =============== configuration ===============

@dataclass
class RusleConfig:
    """
    Configuration for the RUSLE and SDR-RUSLE workflow.

    Required paths
    --------------
    chm_workspace : root workspace folder (catchment subfolders live here)
    catchment_path : catchment boundary vector (any format Geopandas can read)
    sites_path : sites dataset (the *combined* sites GPKG is used for per-site stats)
    k_factor_path : path to K raster
    p_factor_path : path to P raster
    r_factor_path : path to R raster (erosivity) — supplied, not computed

    Options
    -------
    buffer_km : float
        Extra buffer added around the catchment bbox when clipping K/P/R before
        reprojecting to the DEM grid. Defaults to 10 km.
    """
    chm_workspace: str
    catchment_path: str
    sites_path: str
    k_factor_path: str
    p_factor_path: str
    r_factor_path: str
    buffer_km: float = 10.0
    catchment_crs: Optional[str] = None


# =============== small helpers ===============

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _read_dem(catch_datasets: str) -> Tuple[str, rio.Affine, float, float]:
    dem_path = os.path.join(catch_datasets, "Topography", "DEM.tif")
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path} (run the terrain step first)")
    with rio.open(dem_path) as src:
        transform = src.transform
        xres, yres = transform.a, abs(transform.e)
    return dem_path, transform, xres, yres


def _bbox_buffered_in_raster_crs(
    vector_path: str,
    raster_path: str,
    buffer_km: float,
    *,
    catchment_crs: Optional[str] = None,  # <-- NEW (keyword-only)
) -> Tuple[float, float, float, float]:
    gdf = gpd.read_file(vector_path)

    # If a user-specified CRS is provided, respect it:
    # - If the file has no CRS, assign it.
    # - If the file has a CRS and it differs, reproject first.
    if catchment_crs:
        if gdf.crs is None:
            gdf = gdf.set_crs(catchment_crs)
        elif str(gdf.crs).upper() != str(catchment_crs).upper():
            gdf = gdf.to_crs(catchment_crs)

    with rio.open(raster_path) as src:
        r_crs = src.crs

    # Work in the raster CRS for a correct bounding box in raster space
    gdf_r = gdf.to_crs(r_crs)
    minx, miny, maxx, maxy = gdf_r.total_bounds

    # Approx degrees per km (good enough for small buffers & non-polar AOIs)
    deg = buffer_km / 111.0
    return (minx - deg, miny - deg, maxx + deg, maxy + deg)


def _clip_by_bbox(src_path: str, out_path: str, bbox: Tuple[float, float, float, float]) -> None:
    with rio.open(src_path) as src:
        win = from_bounds(*bbox, transform=src.transform)
        if win.height <= 0 or win.width <= 0:
            raise ValueError(f"Empty clip for {src_path} with bbox={bbox}")
        arr = src.read(1, window=win, masked=True)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": win.height,
            "width": win.width,
            "transform": src.window_transform(win)
        })
        with rio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)


def _align_to_dem_grid(src_raster: str, dem_path: str, out_path: str, resampling: Resampling = Resampling.nearest) -> None:
    with rio.open(dem_path) as dem:
        dem_crs = dem.crs
        dem_transform = dem.transform
        dem_w, dem_h = dem.width, dem.height
        dem_nd = dem.nodata

        dst = np.full((dem_h, dem_w), dem_nd if dem_nd is not None else np.nan, dtype=np.float32)

        with rio.open(src_raster) as src:
            out_meta = src.meta.copy()
            reproject(
                source=rio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=resampling,
            )
            # Mask where DEM is nodata
            dem_mask = dem.read(1, masked=True).mask  # True where DEM nodata
            dst = np.where(dem_mask, np.nan, dst)

        out_meta.update({
            "crs": dem_crs,
            "transform": dem_transform,
            "width": dem_w,
            "height": dem_h,
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": np.nan
        })
        with rio.open(out_path, "w", **out_meta) as dst_ds:
            dst_ds.write(dst, 1)


def _read_factor(path: str) -> np.ndarray:
    with rio.open(path) as src:
        arr = src.read(1).astype("float64")
        nd = src.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return arr


def _extract_year(p: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", os.path.basename(p))
    return int(m.group(0)) if m else None


def _nearest_year(target: int, available: List[int]) -> Optional[int]:
    if not available:
        return None
    return min(available, key=lambda y: (abs(y - target), -y))  # tie -> later year


# =============== main entrypoint ===============
def rusle_and_sdr_rusle(cfg: RusleConfig) -> Tuple[str, str]:
    """
    Compute RUSLE (t/ha/yr) and SDR-RUSLE (t/ha/yr) using a *provided* R-factor raster.

    Also:
      • Writes catchment annual CSV of means/totals.
      • (DEACTIVATED) Append annual totals into catchment layer.
      • Writes per-site annual CSVs.
      • (DEACTIVATED) Add per-year total/mean columns to the existing sites layer.
      • (DEACTIVATED) Write a new 'RUSLE_SDR_Annual_Sites' layer (site×year).

    Returns
    -------
    all_sites_gpkg : str
    sites_datasets : str
    """
    print("Starting processing RUSLE (with supplied R-factor)...")

    # ---- folders / fixed inputs prepared by other modules ----
    catchment_name = os.path.splitext(os.path.basename(cfg.catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(cfg.chm_workspace, catchment_name)
    catch_datasets  = os.path.join(catchment_folder, "Catchment Datasets")
    rusle_folder    = os.path.join(catch_datasets, "RUSLE and SDR_RUSLE")
    sites_datasets  = os.path.join(catchment_folder, "Sites Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    _ensure_dirs([catchment_folder, catch_datasets, rusle_folder, sites_datasets, catch_plots, sites_plots])

    catch_gpkg      = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg  = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")

    dem_path, dem_transform, xres, yres = _read_dem(catch_datasets)
    LS_path         = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Slope length-gradient factor.tif")
    annual_c_dir    = os.path.join(catch_datasets, "Vegetation", "Indices", "C Factor", "Annual")
    sdr_output_dir  = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")

    # ==== 1) clip + align K/P/R to DEM grid ====
    tmp_k_clip = os.path.join(rusle_folder, "_k_clip.tif")
    tmp_p_clip = os.path.join(rusle_folder, "_p_clip.tif")
    tmp_r_clip = os.path.join(rusle_folder, "_r_clip.tif")
    for src, out in [(cfg.k_factor_path, tmp_k_clip),
                     (cfg.p_factor_path, tmp_p_clip),
                     (cfg.r_factor_path, tmp_r_clip)]:
        bbox = _bbox_buffered_in_raster_crs(
            cfg.catchment_path, src, cfg.buffer_km,
            catchment_crs=cfg.catchment_crs,  # <-- NEW
        )
        _clip_by_bbox(src, out, bbox)

    k_factor = os.path.join(rusle_folder, "k_factor.tif")
    p_factor = os.path.join(rusle_folder, "p_factor.tif")
    r_factor = os.path.join(rusle_folder, "r_factor.tif")
    _align_to_dem_grid(tmp_k_clip, dem_path, k_factor)
    _align_to_dem_grid(tmp_p_clip, dem_path, p_factor)
    _align_to_dem_grid(tmp_r_clip, dem_path, r_factor)
    for p in (tmp_k_clip, tmp_p_clip, tmp_r_clip):
        try: os.remove(p)
        except Exception: pass

    # ==== 2) read aligned LS & factors ====
    with rio.open(LS_path) as ls_ds:
        ls_band = ls_ds.read(1)
        ls_array = np.where(ls_band == ls_ds.nodata, np.nan, ls_band).astype("float64")
        ls_transform = ls_ds.transform
    cell_area_m2 = abs(ls_transform.a) * abs(ls_transform.e)
    cell_area_ha = cell_area_m2 / 10000.0

    k_array = _read_factor(k_factor)
    p_array = _read_factor(p_factor)
    r_array = _read_factor(r_factor)

    # ==== 3) collect C & SDR annual rasters ====
    c_paths   = sorted(glob.glob(os.path.join(annual_c_dir, "C_Factor_*.tif")))
    sdr_paths = sorted(glob.glob(os.path.join(sdr_output_dir, "SDR_*.tif")))
    c_by_year   = {y: p for p in c_paths   if (y := _extract_year(p)) is not None}
    sdr_by_year = {y: p for p in sdr_paths if (y := _extract_year(p)) is not None}
    c_years, sdr_years = sorted(c_by_year), sorted(sdr_by_year)

    # ==== 4) sites & catchment mask on DEM grid ====
    dem_xr = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)
    tform, height, width = dem_xr.rio.transform(), dem_xr.rio.height, dem_xr.rio.width

    # Sites (combined GPKG from prior modules)
    sites_gdf = gpd.read_file(all_sites_gpkg)
    if cfg.catchment_crs:
        # If the layer lacks a CRS, assign; otherwise reproject if different
        if sites_gdf.crs is None:
            sites_gdf = sites_gdf.set_crs(cfg.catchment_crs)
        elif str(sites_gdf.crs).upper() != str(cfg.catchment_crs).upper():
            sites_gdf = sites_gdf.to_crs(cfg.catchment_crs)
    # Finally, cast to DEM CRS for raster alignment
    sites_gdf = sites_gdf.to_crs(dem_xr.rio.crs)
    
    # Catchment boundary
    catch_gdf = gpd.read_file(cfg.catchment_path)
    if cfg.catchment_crs:
        if catch_gdf.crs is None:
            catch_gdf = catch_gdf.set_crs(cfg.catchment_crs)
        elif str(catch_gdf.crs).upper() != str(cfg.catchment_crs).upper():
            catch_gdf = catch_gdf.to_crs(cfg.catchment_crs)
    catch_gdf = catch_gdf.to_crs(dem_xr.rio.crs)

    catch_union = catch_gdf.unary_union
    catch_mask = geometry_mask([mapping(catch_union)], (height, width), tform, invert=True)

    # ==== 5) iterate years, compute RUSLE & SDR-RUSLE ====
    target_years = sorted(set(c_years) | set(sdr_years))
    if not target_years:
        print("[WARN] No annual C or SDR rasters found. Nothing to compute.")
        return all_sites_gpkg, sites_datasets

    site_tables: Dict[object, List[dict]] = {row["Site_id"]: [] for _, row in sites_gdf.iterrows()}
    catch_rows: List[dict] = []
    site_gpkg_rows: List[dict] = []

    for year in target_years:
        cy = year if year in c_years else _nearest_year(year, c_years)
        sy = year if year in sdr_years else _nearest_year(year, sdr_years)
        if cy is None or sy is None:
            print(f"[INFO] Skipping {year}: C or SDR not available (nearest-year lookup failed).")
            continue

        with rio.open(c_by_year[cy]) as c_ds:
            c_arr = c_ds.read(1).astype("float64")
            if c_ds.nodata is not None:
                c_arr = np.where(c_arr == c_ds.nodata, np.nan, c_arr)

        with rio.open(sdr_by_year[sy]) as sdr_ds:
            sdr_arr = sdr_ds.read(1).astype("float64")
            if sdr_ds.nodata is not None:
                sdr_arr = np.where(sdr_arr == sdr_ds.nodata, np.nan, sdr_arr)

        # RUSLE & SDR-RUSLE (t/ha/yr)
        rusle_year = r_array * k_array * ls_array * c_arr * p_array
        sdr_rusle_year = rusle_year * sdr_arr

        # catchment stats
        valid = catch_mask & np.isfinite(rusle_year)
        n_valid = int(valid.sum())
        if n_valid > 0:
            rusle_mean = float(np.nanmean(rusle_year[valid]))
            sdr_rusle_mean = float(np.nanmean(sdr_rusle_year[valid]))
            area_ha = n_valid * float(cell_area_ha)
            rusle_total = float(np.nansum(rusle_year[valid])) * float(cell_area_ha)
            sdr_rusle_total = float(np.nansum(sdr_rusle_year[valid])) * float(cell_area_ha)
        else:
            rusle_mean = sdr_rusle_mean = np.nan
            area_ha = rusle_total = sdr_rusle_total = 0.0

        catch_rows.append({
            "Year": year,
            "C_year_used": cy,
            "SDR_year_used": sy,
            "RUSLE mean (t/ha/yr)": rusle_mean,
            "SDR-RUSLE mean (t/ha/yr)": sdr_rusle_mean,
            "Catchment area (ha)": area_ha,
            "RUSLE total (t/yr)": rusle_total,
            "SDR-RUSLE total (t/yr)": sdr_rusle_total,
        })

        # per-site stats + (DEACTIVATED writes later; we still compute CSVs/plots)
        for _, site in sites_gdf.iterrows():
            site_id = site.get("Site_id")
            geom = site.geometry
            mask_np = geometry_mask([mapping(geom)], (height, width), tform, invert=True)
            valid_s = mask_np & np.isfinite(rusle_year)

            n_vs = int(valid_s.sum())
            if n_vs > 0:
                r_mean = float(np.nanmean(rusle_year[valid_s]))
                sr_mean = float(np.nanmean(sdr_rusle_year[valid_s]))
                site_area_ha = n_vs * float(cell_area_ha)
                r_total = float(np.nansum(rusle_year[valid_s])) * float(cell_area_ha)
                sr_total = float(np.nansum(sdr_rusle_year[valid_s])) * float(cell_area_ha)
            else:
                r_mean = sr_mean = np.nan
                site_area_ha = 0.0
                r_total = sr_total = 0.0

            site_tables[site_id].append({
                "Year": year,
                "C_year_used": cy,
                "SDR_year_used": sy,
                "RUSLE (t/ha/yr)": r_mean,
                "SDR-RUSLE (t/ha/yr)": sr_mean,
                "Site area (ha)": site_area_ha,
                "RUSLE total (t/yr)": r_total,
                "SDR-RUSLE total (t/yr)": sr_total,
            })

            # kept for possible future layer creation; disabled later:
            site_gpkg_rows.append({
                "id": site_id,
                "Year": year,
                "C_year_used": cy,
                "SDR_year_used": sy,
                "RUSLE (t/ha/yr)": r_mean,
                "SDR-RUSLE (t/ha/yr)": sr_mean,
                "Site area (ha)": site_area_ha,
                "RUSLE total (t/yr)": r_total,
                "SDR-RUSLE total (t/yr)": sr_total,
                "geometry": geom,
            })

    # ==== 6) outputs ====
    # 6a) Catchment CSV
    catch_df = pd.DataFrame(catch_rows).sort_values("Year")
    out_catch_csv = os.path.join(rusle_folder, f"{catchment_name}_RUSLE_SDR-RUSLE_Annual.csv")
    catch_df.to_csv(out_catch_csv, index=False)
    print(f"[OK] Catchment CSV written: {out_catch_csv}")

    # 6b) Append annual totals into catchment layer (DEACTIVATED: no GPKG writes)
    # try:
    #     target_years_vec = sorted(set(catch_df["Year"].astype(int))) if not catch_df.empty else []
    #     totals_df = (catch_df[["Year", "RUSLE total (t/yr)", "SDR-RUSLE total (t/yr)"]]
    #                  .drop_duplicates("Year").copy())
    #     totals_df["Year"] = pd.to_numeric(totals_df["Year"], errors="coerce").astype("Int64")
    #     catch_layer = f"{catchment_name} Data"
    #     base_gdf = gpd.read_file(catch_gpkg, layer=catch_layer)
    #     # ... original logic omitted intentionally ...
    # except Exception as e:
    #     print(f"[ERROR] Appending RUSLE totals failed: {e}")

    # 6c) per-site CSVs (kept)
    for site_id, rows in site_tables.items():
        df = pd.DataFrame(rows).sort_values("Year")
        site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
        _ensure_dirs([site_folder])
        out_csv = os.path.join(site_folder, f"Site {site_id} - Annual RUSLE and SDR-RUSLE.csv")
        df.to_csv(out_csv, index=False)

    # 6d) Roads-style: add per-year columns to the *primary* sites layer (DEACTIVATED)
    # try:
    #     tidy_rows = []
    #     for site_id, rows in site_tables.items():
    #         for r in rows:
    #             tidy_rows.append({
    #                 "id": site_id,
    #                 "Year": int(r["Year"]),
    #                 "RUSLE total (t/yr)": float(r["RUSLE total (t/yr)"]),
    #                 "SDR-RUSLE total (t/yr)": float(r["SDR-RUSLE total (t/yr)"]),
    #                 "RUSLE mean (t/ha/yr)": float(r["RUSLE (t/ha/yr)"]),
    #                 "SDR-RUSLE mean (t/ha/yr)": float(r["SDR-RUSLE (t/ha/yr)"]),
    #             })
    #     if tidy_rows:
    #         tidy_df = pd.DataFrame(tidy_rows)
    #         # ... original pivot/merge/write logic omitted intentionally ...
    # except Exception as e:
    #     print(f"[WARN] Could not augment sites layer with yearly totals/means: {e}")

    # 6e) tidy per-site×year layer for analysis/joins (DEACTIVATED: no GPKG writes)
    # try:
    #     if site_gpkg_rows:
    #         site_year_gdf = gpd.GeoDataFrame(site_gpkg_rows, geometry="geometry", crs=sites_gdf.crs)
    #         site_year_gdf.to_file(all_sites_gpkg, layer="RUSLE_SDR_Annual_Sites", driver="GPKG", if_exists="replace")
    #         print(f"[OK] Wrote 'RUSLE_SDR_Annual_Sites' layer to {all_sites_gpkg}")
    #     else:
    #         print("[INFO] No per-site annual rows to write.")
    # except Exception as e:
    #     print(f"[WARN] Could not write 'RUSLE_SDR_Annual_Sites' layer: {e}")

    # 6f) PLOTTING (unchanged except consistent fonts)
    try:
        if not catch_df.empty:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
            axes[0].plot(catch_df["Year"], catch_df["RUSLE total (t/yr)"], color="black", marker="o", linewidth=1.3)
            axes[0].set_ylabel("Total RUSLE (t/yr)", fontsize=10, fontweight="bold")
            axes[0].tick_params(axis="both", labelsize=10)
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].plot(catch_df["Year"], catch_df["SDR-RUSLE total (t/yr)"], color="black", marker="o", linewidth=1.3)
            axes[1].set_ylabel("Total SDR-RUSLE (t/yr)", fontsize=10, fontweight="bold")
            axes[1].tick_params(axis="both", labelsize=10)
            axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
            axes[1].grid(True, linestyle="--", alpha=0.5)
            fig.suptitle(f"{catchment_name} – Catchment Totals", fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            out_plot_catch = os.path.join(catch_plots, f"{catchment_name}_RUSLE_SDR-RUSLE_Totals.png")
            plt.savefig(out_plot_catch, dpi=300)
            plt.close(fig)
            print(f"[OK] Catchment totals plot saved")
        else:
            print("[INFO] No catchment data available for plotting.")

        for site_id, rows in site_tables.items():
            df = pd.DataFrame(rows).sort_values("Year")
            if df.empty:
                continue

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
            axes[0].plot(df["Year"], df["RUSLE total (t/yr)"], color="black", marker="o", linewidth=1.3)
            axes[0].set_ylabel("Total RUSLE (t/yr)", fontsize=10, fontweight="bold")
            axes[0].tick_params(axis="both", labelsize=10)
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].plot(df["Year"], df["SDR-RUSLE total (t/yr)"], color="black", marker="o", linewidth=1.3)
            axes[1].set_ylabel("Total SDR-RUSLE (t/yr)", fontsize=10, fontweight="bold")
            axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
            axes[1].tick_params(axis="both", labelsize=10)
            axes[1].grid(True, linestyle="--", alpha=0.5)
            fig.suptitle(f"Site {site_id} – Annual RUSLE & SDR-RUSLE", fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            site_folder = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_folder, exist_ok=True)
            out_plot_site = os.path.join(site_folder, f"Site_{site_id}_RUSLE_SDR-RUSLE_Totals.png")
            plt.savefig(out_plot_site, dpi=300)
            plt.close(fig)
            print(f"[OK] Site {site_id} totals plot saved")

    except Exception as e:
        print(f"[WARN] Plotting RUSLE/SDR-RUSLE totals failed: {e}")

    print("Completed RUSLE using provided R-factor.")
    return all_sites_gpkg, sites_datasets
