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
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds
from shapely.geometry import mapping, Point
import fiona

# =============== plotting helpers ===============
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============== configuration ===============

@dataclass
class RusleConfig:
    """
    Configuration for the RUSLE and SDR-RUSLE workflow.

    Required paths
    --------------
    chm_workspace : root workspace folder (catchment subfolders live here)
    catchment_path : catchment boundary vector (any format Geopandas can read)
    k_factor_path : path to K raster
    p_factor_path : path to P raster
    r_factor_path : path to R raster (erosivity) — supplied, not computed

    Optional paths
    --------------
    sites_path : optional sites dataset path. If not provided, the module still runs
        for catchment-level processing and skips site-based outputs/plots.

    Options
    -------
    buffer_km : float
        Extra buffer added around the catchment bbox when clipping K/P/R before
        reprojecting to the DEM grid. Defaults to 10 km.
    """
    chm_workspace: str
    catchment_path: str
    k_factor_path: str
    p_factor_path: str
    r_factor_path: str
    sites_path: Optional[str] = None
    buffer_km: float = 10.0
    catchment_crs: Optional[str] = None


# =============== small helpers ===============

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _log(status: str, message: str) -> None:
    """Simple consistent console logger."""
    print(f"[{status}] {message}")


def _folder_label(path: str) -> str:
    """Return the last folder/file label for cleaner logging."""
    return os.path.basename(os.path.normpath(path))


def _log_folder_summary(
    catchment_folder: str,
    catch_datasets: str,
    catch_plots: str,
    rusle_folder: str,
    sites_datasets: str,
    sites_plots: str,
) -> None:
    """Print full output folder structure once at the start."""
    _log("OK", "Output folders are ready.")
    _log("INFO", "Created/using output folders:")
    _log("INFO", f"  Catchment folder       : {catchment_folder}")
    _log("INFO", f"  Catchment datasets     : {catch_datasets}")
    _log("INFO", f"  Catchment plots/maps   : {catch_plots}")
    _log("INFO", f"  RUSLE                  : {rusle_folder}")
    _log("INFO", f"  Sites datasets         : {sites_datasets}")
    _log("INFO", f"  Sites plots/maps       : {sites_plots}")


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
    catchment_crs: Optional[str] = None,
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

    # NOTE:
    # This preserves your original logic. I am not changing it even though
    # the name suggests raster CRS and the buffer is approximated.
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


def _write_single_band(path: str, data: np.ndarray, template_meta: dict, dtype: str = "float32") -> None:
    """Write a single-band raster using `template_meta` as a template."""
    meta = template_meta.copy()
    meta.update({"count": 1, "dtype": dtype, "nodata": np.nan})
    with rio.open(path, "w", **meta) as dst:
        dst.write(data.astype(dtype), 1)


def _add_matched_colorbar(fig, ax, im, label: str, size: str = "4.5%", pad: float = 0.08):
    """Add a colorbar whose height matches the plotted axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=9)
    return cbar


def _reproject_array_for_plotting(
    data: np.ndarray,
    src_meta: dict,
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.nearest
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Reproject a raster array to another CRS for plotting only.
    Returns reprojected array and extent as (xmin, xmax, ymin, ymax).
    """
    src_crs = src_meta["crs"]
    src_transform = src_meta["transform"]
    src_width = src_meta["width"]
    src_height = src_meta["height"]

    left, bottom, right, top = array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, left, bottom, right, top
    )

    dst_array = np.full((dst_height, dst_width), np.nan, dtype="float32")

    reproject(
        source=data.astype("float32"),
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    xmin, ymin, xmax, ymax = array_bounds(dst_height, dst_width, dst_transform)
    return dst_array, (xmin, xmax, ymin, ymax)


def _style_map_axes(ax, xlabel: str = "Longitude (°)", ylabel: str = "Latitude (°)", nbins: int = 4) -> None:
    """Apply consistent map styling across all plots."""
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.6)


def _apply_small_plot_padding(ax, extent: Tuple[float, float, float, float], pad_fraction: float = 0.02) -> None:
    """Apply a small padding around the plotted extent."""
    xmin, xmax, ymin, ymax = extent
    xpad = (xmax - xmin) * pad_fraction
    ypad = (ymax - ymin) * pad_fraction
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def _plot_raster_context_in_degrees(
    data: np.ndarray,
    src_meta: dict,
    out_png: str,
    title: str,
    catchment_gdf: Optional[gpd.GeoDataFrame] = None,
    site_points: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Plot full raster in latitude/longitude degrees for context."""
    plot_data, extent = _reproject_array_for_plotting(
        data,
        src_meta,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )

    catch_plot = catchment_gdf.to_crs(epsg=4326) if catchment_gdf is not None else None
    pts_plot = site_points.to_crs(epsg=4326) if site_points is not None else None

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(plot_data, cmap="viridis", extent=extent, origin="upper", interpolation="nearest")
    _add_matched_colorbar(fig, ax, im, title)

    if catch_plot is not None and not catch_plot.empty:
        catch_plot.boundary.plot(ax=ax, color="black", linewidth=1.2)

    if pts_plot is not None and not pts_plot.empty:
        pts_plot.plot(ax=ax, color="red", markersize=15)
        if "Site_id" in pts_plot.columns:
            for _, pr in pts_plot.iterrows():
                geom = pr.geometry
                x = geom.x if geom.geom_type == "Point" else geom.centroid.x
                y = geom.y if geom.geom_type == "Point" else geom.centroid.y
                ax.annotate(
                    f"{pr['Site_id']}",
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="red",
                )

    ax.set_title(title, fontsize=11, fontweight="bold")
    _apply_small_plot_padding(ax, extent, pad_fraction=0.02)
    _style_map_axes(ax, nbins=4)

    fig.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_site_points_from_all_sites_gpkg(all_sites_gpkg: str, target_crs):
    """
    Read site polygons from all_sites_gpkg and derive point locations.

    Priority
    --------
    1. Use X_site / Y_site if available
    2. Otherwise use polygon centroid
    """
    if not os.path.exists(all_sites_gpkg):
        return None

    try:
        sites = gpd.read_file(all_sites_gpkg)
        if sites.empty:
            return None

        if sites.crs is None:
            sites = sites.set_crs(target_crs)
        elif sites.crs != target_crs:
            sites = sites.to_crs(target_crs)

        pts = sites.copy()

        if {"X_site", "Y_site"}.issubset(pts.columns):
            valid_xy = pts["X_site"].notna() & pts["Y_site"].notna()
            if valid_xy.any():
                pts = pts.loc[valid_xy].copy()
                pts["geometry"] = [Point(x, y) for x, y in zip(pts["X_site"], pts["Y_site"])]
            else:
                pts["geometry"] = pts.geometry.centroid
        else:
            pts["geometry"] = pts.geometry.centroid

        return pts

    except Exception as e:
        _log("WARN", f"Could not derive site points from all_sites_gpkg: {e}")
        return None


def _get_adaptive_figsize_from_bounds(
    bounds: Tuple[float, float, float, float],
    base_size: float = 5.0,
    min_size: float = 4.5,
    max_size: float = 9.0
) -> Tuple[float, float]:
    """
    Compute an adaptive figure size from geometry bounds.
    """
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-9)
    height = max(maxy - miny, 1e-9)
    aspect = width / height

    if aspect >= 1:
        fig_w = base_size * aspect
        fig_h = base_size
    else:
        fig_w = base_size
        fig_h = base_size / aspect

    fig_w = min(max(fig_w, min_size), max_size)
    fig_h = min(max(fig_h, min_size), max_size)
    return fig_w, fig_h


def _set_axis_limits_from_gdf(ax, gdf: gpd.GeoDataFrame, pad_fraction: float = 0.02) -> None:
    """Set axis limits from GeoDataFrame bounds with small padding."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = (maxx - minx) * pad_fraction
    ypad = (maxy - miny) * pad_fraction
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)


def _plot_clipped_raster_in_degrees(
    data: np.ndarray,
    src_meta: dict,
    out_png: str,
    title: str,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    site_point_gdf: Optional[gpd.GeoDataFrame] = None,
    site_id: Optional[object] = None,
) -> None:
    """Plot clipped raster in latitude/longitude degrees."""
    plot_data, extent = _reproject_array_for_plotting(
        data,
        src_meta,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )

    boundary_plot = boundary_gdf.to_crs(epsg=4326) if boundary_gdf is not None else None
    site_plot = site_point_gdf.to_crs(epsg=4326) if site_point_gdf is not None else None

    if boundary_plot is not None and not boundary_plot.empty:
        plot_bounds = tuple(boundary_plot.total_bounds)
    else:
        plot_bounds = (extent[0], extent[2], extent[1], extent[3])

    fig_w, fig_h = _get_adaptive_figsize_from_bounds(
        plot_bounds,
        base_size=5.0,
        min_size=4.5,
        max_size=9.0
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        plot_data,
        extent=extent,
        cmap="viridis",
        origin="upper",
        interpolation="nearest"
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(title, fontsize=9)

    if site_plot is not None and not site_plot.empty:
        site_plot.plot(
            ax=ax,
            color="red",
            markersize=20,
            marker="o",
            label="Site location"
        )

        if site_id is not None:
            geom = site_plot.geometry.iloc[0]
            x = geom.x if geom.geom_type == "Point" else geom.centroid.x
            y = geom.y if geom.geom_type == "Point" else geom.centroid.y
            ax.annotate(
                f"{site_id}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="black"
            )

    ax.set_title(title, fontsize=11, fontweight="bold")

    if boundary_plot is not None and not boundary_plot.empty:
        _set_axis_limits_from_gdf(ax, boundary_plot, pad_fraction=0.02)
    else:
        _apply_small_plot_padding(ax, extent, pad_fraction=0.02)

    _style_map_axes(ax, xlabel="Longitude (°)", ylabel="Latitude (°)", nbins=4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_waterbodies_mask(catch_datasets: str, template_meta: dict) -> Optional[np.ndarray]:
    wb_path = os.path.join(catch_datasets, "Topography", "DEA_Waterbodies.gpkg")

    if not os.path.exists(wb_path):
        return None

    try:
        wb = gpd.read_file(wb_path)
        if wb.empty:
            return None

        if wb.crs != template_meta["crs"]:
            wb = wb.to_crs(template_meta["crs"])

        mask = geometry_mask(
            [mapping(g) for g in wb.geometry if g is not None and not g.is_empty],
            (template_meta["height"], template_meta["width"]),
            template_meta["transform"],
            invert=True,
        )

        return mask

    except Exception as e:
        _log("WARN", f"Waterbodies mask failed: {e}")
        return None


# =============== main entrypoint ===============
def rusle_and_sdr_rusle(cfg: RusleConfig) -> Tuple[str, str]:
    """
    Compute RUSLE (t/ha/yr) and SDR-RUSLE (t/ha/yr) using a *provided* R-factor raster.

    Also:
      • Writes catchment annual CSV of means/totals.
      • Writes annual RUSLE and SDR-RUSLE rasters.
      • Writes catchment annual map plots in degrees.
      • Writes per-site annual CSVs.
      • Writes per-site annual clipped rasters and map plots in degrees.
      • (DEACTIVATED) Append annual totals into catchment layer.
      • (DEACTIVATED) Add per-year total/mean columns to the existing sites layer.
      • (DEACTIVATED) Write a new 'RUSLE_SDR_Annual_Sites' layer (site×year).

    Returns
    -------
    all_sites_gpkg : str
    sites_datasets : str
    """
    _log("INFO", "Starting RUSLE processing (with supplied R-factor)...")

    # ---- folders / fixed inputs prepared by other modules ----
    catchment_name = os.path.splitext(os.path.basename(cfg.catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(cfg.chm_workspace, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    rusle_folder = os.path.join(catch_datasets, "RUSLE and SDR_RUSLE")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    _ensure_dirs([catchment_folder, catch_datasets, rusle_folder, sites_datasets, catch_plots, sites_plots])

    _log_folder_summary(
        catchment_folder,
        catch_datasets,
        catch_plots,
        rusle_folder,
        sites_datasets,
        sites_plots,
    )

    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")

    dem_path, dem_transform, xres, yres = _read_dem(catch_datasets)
    _log("OK", f"DEM loaded from {dem_path}")
    _log("INFO", f"DEM resolution: {xres:.2f} x {yres:.2f} meters")

    LS_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Slope length-gradient factor.tif")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation", "Indices", "C Factor", "Annual")
    sdr_output_dir = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")

    # ---- optional catchment and site layers for plotting/processing ----
    catch_gdf = None
    try:
        catch_gdf = gpd.read_file(cfg.catchment_path)
        if cfg.catchment_crs:
            if catch_gdf.crs is None:
                catch_gdf = catch_gdf.set_crs(cfg.catchment_crs)
            elif str(catch_gdf.crs).upper() != str(cfg.catchment_crs).upper():
                catch_gdf = catch_gdf.to_crs(cfg.catchment_crs)
        _log("OK", f"Catchment loaded: {cfg.catchment_path}")
    except Exception as e:
        _log("WARN", f"Catchment boundary could not be loaded for plotting overlays: {e}")

    sites_pts_user = None

    # ==== 1) clip + align K/P/R to DEM grid ====
    tmp_k_clip = os.path.join(rusle_folder, "_k_clip.tif")
    tmp_p_clip = os.path.join(rusle_folder, "_p_clip.tif")
    tmp_r_clip = os.path.join(rusle_folder, "_r_clip.tif")
    for src, out in [
        (cfg.k_factor_path, tmp_k_clip),
        (cfg.p_factor_path, tmp_p_clip),
        (cfg.r_factor_path, tmp_r_clip),
    ]:
        bbox = _bbox_buffered_in_raster_crs(
            cfg.catchment_path, src, cfg.buffer_km,
            catchment_crs=cfg.catchment_crs,
        )
        _clip_by_bbox(src, out, bbox)
        _log("OK", f"Clipped raster: {os.path.basename(src)} → {os.path.basename(out)}")

    k_factor = os.path.join(rusle_folder, "k_factor.tif")
    p_factor = os.path.join(rusle_folder, "p_factor.tif")
    r_factor = os.path.join(rusle_folder, "r_factor.tif")

    _align_to_dem_grid(tmp_k_clip, dem_path, k_factor)
    _log("OK", f"K aligned → {k_factor}")

    _align_to_dem_grid(tmp_p_clip, dem_path, p_factor)
    _log("OK", f"P aligned → {p_factor}")

    _align_to_dem_grid(tmp_r_clip, dem_path, r_factor)
    _log("OK", f"R aligned → {r_factor}")

    for p in (tmp_k_clip, tmp_p_clip, tmp_r_clip):
        try:
            os.remove(p)
        except Exception:
            pass
    _log("OK", f"K, P and R rasters aligned to DEM grid and saved in {_folder_label(rusle_folder)}")

    # ==== 2) read aligned LS & factors ====
    with rio.open(LS_path) as ls_ds:
        ls_band = ls_ds.read(1)
        ls_array = np.where(ls_band == ls_ds.nodata, np.nan, ls_band).astype("float64")
        ls_transform = ls_ds.transform
        ls_meta = ls_ds.meta.copy()
    _log("OK", f"LS factor loaded: {LS_path}")

    cell_area_m2 = abs(ls_transform.a) * abs(ls_transform.e)
    cell_area_ha = cell_area_m2 / 10000.0

    k_array = _read_factor(k_factor)
    p_array = _read_factor(p_factor)
    r_array = _read_factor(r_factor)
    _log("INFO", "K, P, R and LS factors loaded into memory")

    # ==== 3) collect C & SDR annual rasters ====
    c_paths = sorted(glob.glob(os.path.join(annual_c_dir, "C_Factor_*.tif")))
    sdr_paths = sorted(glob.glob(os.path.join(sdr_output_dir, "SDR_*.tif")))
    c_by_year = {y: p for p in c_paths if (y := _extract_year(p)) is not None}
    sdr_by_year = {y: p for p in sdr_paths if (y := _extract_year(p)) is not None}
    c_years, sdr_years = sorted(c_by_year), sorted(sdr_by_year)

    _log("INFO", f"Annual C rasters found: {len(c_years)}")
    _log("INFO", f"Annual SDR rasters found: {len(sdr_years)}")

    # ==== 4) sites & catchment mask on DEM grid ====
    dem_xr = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)
    sites_pts_user = _get_site_points_from_all_sites_gpkg(all_sites_gpkg, dem_xr.rio.crs)
    if sites_pts_user is not None:
        _log("OK", f"Site points prepared: {len(sites_pts_user)} locations")
    else:
        _log("WARN", "No site points available")

    water_mask = _get_waterbodies_mask(catch_datasets, ls_meta)

    if water_mask is not None:
        _log("OK", "Waterbodies mask loaded and will be applied to RUSLE.")
    else:
        _log("WARN", "No waterbodies mask found. RUSLE will not be masked.")

    tform, height, width = dem_xr.rio.transform(), dem_xr.rio.height, dem_xr.rio.width

    sites_gdf = None
    if os.path.exists(all_sites_gpkg):
        try:
            sites_gdf = gpd.read_file(all_sites_gpkg)
            if cfg.catchment_crs:
                if sites_gdf.crs is None:
                    sites_gdf = sites_gdf.set_crs(cfg.catchment_crs)
                elif str(sites_gdf.crs).upper() != str(cfg.catchment_crs).upper():
                    sites_gdf = sites_gdf.to_crs(cfg.catchment_crs)
            sites_gdf = sites_gdf.to_crs(dem_xr.rio.crs)
            _log("OK", f"Site polygons loaded: {all_sites_gpkg}")
            _log("INFO", f"Number of site polygons: {len(sites_gdf)}")
        except Exception as e:
            sites_gdf = None
            _log("WARN", f"Sites dataset could not be read and site-based outputs will be skipped: {e}")
    else:
        _log("WARN", f"No site polygons found at {all_sites_gpkg}. Site-based outputs will be skipped.")

    if catch_gdf is not None:
        catch_gdf = catch_gdf.to_crs(dem_xr.rio.crs)
        catch_union = catch_gdf.unary_union
        catch_mask = geometry_mask([mapping(catch_union)], (height, width), tform, invert=True)
        _log("OK", "Catchment mask prepared on DEM grid")
    else:
        catch_mask = np.isfinite(ls_array)
        _log("WARN", "Catchment boundary was unavailable; using finite LS pixels as catchment mask")

    # ==== 5) iterate years, compute RUSLE & SDR-RUSLE ====
    target_years = sorted(set(c_years) | set(sdr_years))
    if not target_years:
        _log("WARN", "No annual C or SDR rasters found. Nothing to compute.")
        return all_sites_gpkg, sites_datasets

    _log("INFO", f"Years to process: {target_years}")

    site_tables: Dict[object, List[dict]] = {}
    if sites_gdf is not None and not sites_gdf.empty:
        site_tables = {row["Site_id"]: [] for _, row in sites_gdf.iterrows()}
        _log("OK", "Initialized site summary tables")

    catch_rows: List[dict] = []
    site_gpkg_rows: List[dict] = []

    for year in target_years:
        _log("INFO", f"Processing year: {year}")

        cy = year if year in c_years else _nearest_year(year, c_years)
        sy = year if year in sdr_years else _nearest_year(year, sdr_years)
        if cy is None or sy is None:
            _log("INFO", f"Skipping {year}: C or SDR not available (nearest-year lookup failed).")
            continue

        _log("INFO", f"Using C year: {cy}, SDR year: {sy}")

        with rio.open(c_by_year[cy]) as c_ds:
            c_arr = c_ds.read(1).astype("float64")
            if c_ds.nodata is not None:
                c_arr = np.where(c_arr == c_ds.nodata, np.nan, c_arr)
        _log("OK", f"C-factor raster loaded for processing: {os.path.basename(c_by_year[cy])}")

        with rio.open(sdr_by_year[sy]) as sdr_ds:
            sdr_arr = sdr_ds.read(1).astype("float64")
            if sdr_ds.nodata is not None:
                sdr_arr = np.where(sdr_arr == sdr_ds.nodata, np.nan, sdr_arr)
        _log("OK", f"SDR raster loaded for processing: {os.path.basename(sdr_by_year[sy])}")

        # RUSLE & SDR-RUSLE (t/ha/yr)
        rusle_year = r_array * k_array * ls_array * c_arr * p_array
        sdr_rusle_year = rusle_year * sdr_arr
        sdr_rusle_year = np.where(np.isnan(ls_array), np.nan, sdr_rusle_year)

        # Apply DEM mask
        rusle_year = np.where(np.isnan(ls_array), np.nan, rusle_year)
        sdr_rusle_year = np.where(np.isnan(ls_array), np.nan, sdr_rusle_year)

        # Apply waterbody mask
        if water_mask is not None:
            rusle_year = np.where(water_mask, np.nan, rusle_year)
            sdr_rusle_year = np.where(water_mask, np.nan, sdr_rusle_year)

        _log("INFO", f"RUSLE calculations completed for {year}")

        # ---- save annual rasters ----
        rusle_year_path = os.path.join(rusle_folder, f"RUSLE_{year}.tif")
        sdr_rusle_year_path = os.path.join(rusle_folder, f"SDR_RUSLE_{year}.tif")
        _write_single_band(rusle_year_path, rusle_year, ls_meta)
        _log("OK", f"Raster saved: {os.path.basename(rusle_year_path)}")

        _write_single_band(sdr_rusle_year_path, sdr_rusle_year, ls_meta)
        _log("OK", f"Raster saved: {os.path.basename(sdr_rusle_year_path)}")

        # ---- catchment annual map plots in degrees ----
        try:
            if catch_gdf is not None:
                pts_for_plot = sites_pts_user.to_crs(dem_xr.rio.crs) if sites_pts_user is not None else None

                _plot_raster_context_in_degrees(
                    data=rusle_year,
                    src_meta=ls_meta,
                    out_png=os.path.join(catch_plots, f"RUSLE_{year}.png"),
                    title=f"RUSLE (t/ha/yr) - {year}",
                    catchment_gdf=catch_gdf,
                    site_points=pts_for_plot,
                )
                _plot_raster_context_in_degrees(
                    data=sdr_rusle_year,
                    src_meta=ls_meta,
                    out_png=os.path.join(catch_plots, f"SDR_RUSLE_{year}.png"),
                    title=f"SDR-RUSLE (t/ha/yr) - {year}",
                    catchment_gdf=catch_gdf,
                    site_points=pts_for_plot,
                )
                _log("OK", f"Catchment plots saved for {year}")
        except Exception as e:
            _log("WARN", f"Catchment annual maps for {year} failed: {e}")

        # ---- catchment stats ----
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
        _log("INFO", f"Catchment stats computed for {year}")

        # ---- per-site stats and annual raster plots ----
        if sites_gdf is not None and not sites_gdf.empty:
            for _, site in sites_gdf.iterrows():
                site_id = site.get("Site_id")
                _log("INFO", f"Processing Site {site_id} for year {year}")

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
                _log("OK", f"Site {site_id} stats computed for year {year}")

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

                # ---- save clipped site rasters and site plots ----
                try:
                    site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
                    site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
                    _ensure_dirs([site_data_dir, site_plot_dir])

                    site_poly = gpd.GeoDataFrame([site.drop(labels="geometry").to_dict()], geometry=[geom], crs=sites_gdf.crs)

                    one_site_point = None
                    if sites_pts_user is not None and not sites_pts_user.empty and "Site_id" in sites_pts_user.columns:
                        pts_match = sites_pts_user[sites_pts_user["Site_id"] == site_id]
                        if len(pts_match):
                            one_site_point = pts_match.iloc[[0]].to_crs(dem_xr.rio.crs)

                    for short_name, arr, out_raster_name, out_plot_name in [
                        (f"RUSLE_{year}", rusle_year, f"RUSLE_{year}.tif", f"RUSLE_{year}.png"),
                        (f"SDR_RUSLE_{year}", sdr_rusle_year, f"SDR_RUSLE_{year}.tif", f"SDR_RUSLE_{year}.png"),
                    ]:
                        with rio.open(dem_path) as dem_src:
                            v = site_poly.to_crs(ls_meta["crs"]) if site_poly.crs != ls_meta["crs"] else site_poly

                            tmp_meta = ls_meta.copy()
                            tmp_meta.update({
                                "driver": "GTiff",
                                "height": arr.shape[0],
                                "width": arr.shape[1],
                                "transform": ls_meta["transform"],
                                "crs": ls_meta["crs"],
                                "nodata": np.nan,
                                "count": 1,
                                "dtype": "float32",
                            })

                            with MemoryFile() as memfile:
                                with memfile.open(**tmp_meta) as tmp_ds:
                                    tmp_ds.write(arr.astype("float32"), 1)

                                    out_img, out_tr = rio_mask(
                                        tmp_ds,
                                        [mapping(v.geometry.iloc[0])],
                                        crop=True,
                                        filled=False
                                    )

                                    clipped = out_img[0].filled(np.nan).astype("float32")

                                    clipped_meta = tmp_ds.meta.copy()
                                    clipped_meta.update({
                                        "driver": "GTiff",
                                        "height": out_img.shape[1],
                                        "width": out_img.shape[2],
                                        "transform": out_tr,
                                        "crs": tmp_ds.crs,
                                        "nodata": np.nan,
                                        "count": 1,
                                        "dtype": "float32",
                                    })

                                    _write_single_band(
                                        os.path.join(site_data_dir, out_raster_name),
                                        clipped,
                                        clipped_meta,
                                    )
                                    _log("OK", f"{out_raster_name} raster saved for Site {site_id}")

                                    _plot_clipped_raster_in_degrees(
                                        data=clipped,
                                        src_meta=clipped_meta,
                                        out_png=os.path.join(site_plot_dir, out_plot_name),
                                        title=f"{short_name} (t/ha/yr) - Site {site_id}",
                                        boundary_gdf=v,
                                        site_point_gdf=one_site_point,
                                        site_id=site_id,
                                    )
                                    _log("OK", f"{out_plot_name} plot saved for Site {site_id}")

                except Exception as e:
                    _log("WARN", f"Site {site_id} annual raster outputs for {year} failed: {e}")

            _log("INFO", f"Completed all site processing for year {year}")

        _log("OK", f"RUSLE and SDR-RUSLE {year} processed.")

    # ==== 6) outputs ====
    # 6a) Catchment CSV
    catch_df = pd.DataFrame(catch_rows).sort_values("Year")
    out_catch_csv = os.path.join(rusle_folder, f"{catchment_name}_RUSLE_SDR-RUSLE_Annual.csv")
    catch_df.to_csv(out_catch_csv, index=False)

    out_catch_csv_ = os.path.join(catch_plots, f"{catchment_name}_RUSLE_SDR-RUSLE_Annual.csv")
    catch_df.to_csv(out_catch_csv_, index=False)
    _log("OK", f"Catchment CSV written: {out_catch_csv}")
    _log("OK", "Catchment CSV saved in datasets and plots folders")

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
    if site_tables:
        for site_id, rows in site_tables.items():
            df = pd.DataFrame(rows).sort_values("Year")
            site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
            site_plot_folder = os.path.join(sites_plots, f"Site_{site_id}")
            _ensure_dirs([site_folder, site_plot_folder])

            out_csv = os.path.join(site_folder, f"Site {site_id} - Annual RUSLE and SDR-RUSLE.csv")
            df.to_csv(out_csv, index=False)

            out_csv_ = os.path.join(site_plot_folder, f"Site {site_id} - Annual RUSLE and SDR-RUSLE.csv")
            df.to_csv(out_csv_, index=False)
            _log("OK", f"Site {site_id} CSV saved in datasets and plots folders")
    else:
        _log("WARN", "No site tables were created. Site-based CSV outputs were skipped.")

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

    # 6f) TIME-SERIES PLOTTING
    try:
        if not catch_df.empty:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

            axes[0].plot(catch_df["Year"], catch_df["RUSLE mean (t/ha/yr)"], color="black", marker="o", linewidth=1.3)
            axes[0].set_ylabel("RUSLE (t/ha/yr)", fontsize=10, fontweight="bold")
            axes[0].tick_params(axis="both", labelsize=10)
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].plot(catch_df["Year"], catch_df["SDR-RUSLE mean (t/ha/yr)"], color="black", marker="o", linewidth=1.3)
            axes[1].set_ylabel("SDR-RUSLE (t/ha/yr)", fontsize=10, fontweight="bold")
            axes[1].tick_params(axis="both", labelsize=10)
            axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
            axes[1].grid(True, linestyle="--", alpha=0.5)

            fig.suptitle(f"{catchment_name} – Catchment Soil Loss", fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])

            out_plot_catch = os.path.join(catch_plots, f"{catchment_name}_RUSLE_SDR-RUSLE_TimeSeries.png")
            plt.savefig(out_plot_catch, dpi=300, bbox_inches="tight")
            plt.close(fig)
            _log("OK", f"Catchment time-series plot saved: {_folder_label(out_plot_catch)}")
        else:
            _log("INFO", "No catchment data available for plotting.")

        if site_tables:
            for site_id, rows in site_tables.items():
                df = pd.DataFrame(rows).sort_values("Year")
                if df.empty:
                    continue

                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
                axes[0].plot(df["Year"], df["RUSLE (t/ha/yr)"], color="black", marker="o", linewidth=1.3)
                axes[0].set_ylabel("RUSLE (t/ha/yr)", fontsize=10, fontweight="bold")
                axes[0].tick_params(axis="both", labelsize=10)
                axes[0].grid(True, linestyle="--", alpha=0.5)

                axes[1].plot(df["Year"], df["SDR-RUSLE (t/ha/yr)"], color="black", marker="o", linewidth=1.3)
                axes[1].set_ylabel("SDR-RUSLE (t/ha/yr)", fontsize=10, fontweight="bold")
                axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
                axes[1].tick_params(axis="both", labelsize=10)
                axes[1].grid(True, linestyle="--", alpha=0.5)

                fig.suptitle(f"Site {site_id} – Annual RUSLE & SDR-RUSLE", fontsize=11, fontweight="bold")
                fig.tight_layout(rect=[0, 0.02, 1, 0.96])

                site_folder = os.path.join(sites_plots, f"Site_{site_id}")
                os.makedirs(site_folder, exist_ok=True)
                out_plot_site = os.path.join(site_folder, f"Site_{site_id}_RUSLE_SDR-RUSLE_TimeSeries.png")
                plt.savefig(out_plot_site, dpi=300, bbox_inches="tight")
                plt.close(fig)
                _log("OK", f"Site {site_id} time-series plot saved")
        else:
            _log("WARN", "No site-based totals plots were created.")

    except Exception as e:
        _log("WARN", f"Plotting RUSLE/SDR-RUSLE totals failed: {e}")

    _log("INFO", f"Total years processed: {len(target_years)}")
    _log("INFO", f"Outputs stored in: {_folder_label(catchment_folder)}")
    _log("OK", "Completed RUSLE using provided R-factor.")
    return all_sites_gpkg, sites_datasets