# src/chm_veg/veg_indices.py
# -*- coding: utf-8 -*-
"""
Vegetation indices (NDVI) and C-factor from DEA STAC; annual summaries,
riparian stats, per-site stats & plots.

Dependencies: geopandas, rasterio, rioxarray, xarray, numpy, pandas,
pystac_client, odc.stac, shapely, matplotlib, fiona (optional for layer replace)
"""

from __future__ import annotations

# --- Stdlib ---
import os
import re
import gc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

# --- Scientific ---
import numpy as np
import pandas as pd
import xarray as xr

# --- Geo ---
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from rasterio.enums import Resampling
from shapely.geometry import Point
import fiona  # only used for fiona.remove(...)

# --- STAC / DEA ---
import pystac_client
import odc.stac

# --- Plotting ---
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================== Utilities ==============================

def _ensure_dirs(paths: Iterable[str]) -> None:
    """Create folders if they do not exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _log(status: str, message: str) -> None:
    """Simple consistent console logger."""
    print(f"[{status}] {message}")


def _folder_label(path: str) -> str:
    """Return the last folder/file label for cleaner logging."""
    return os.path.basename(os.path.normpath(path))


def _existing_or_none(path: str) -> Optional[str]:
    """Return path only if it exists on disk, otherwise None."""
    return path if os.path.exists(path) else None


def _log_folder_summary(
    catchment_folder: str,
    catch_datasets: str,
    catch_plots: str,
    veg_folder: str,
    sites_datasets: str,
    sites_plots: str,
) -> None:
    """Print full output folder structure once at the start."""
    _log("OK", "Output folders are ready.")
    _log("INFO", "Created/using output folders:")
    _log("INFO", f"  Catchment folder       : {catchment_folder}")
    _log("INFO", f"  Catchment datasets     : {catch_datasets}")
    _log("INFO", f"  Catchment plots/maps   : {catch_plots}")
    _log("INFO", f"  Vegetation             : {veg_folder}")
    _log("INFO", f"  Sites datasets         : {sites_datasets}")
    _log("INFO", f"  Sites plots/maps       : {sites_plots}")


def _add_matched_colorbar(fig, ax, im, label: str, size: str = "4.5%", pad: float = 0.08):
    """Add a colorbar whose height matches the plotted axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=9)
    return cbar


def _years_from_interval(datetime_str: str) -> List[int]:
    """'YYYY-MM-DD/YYYY-MM-DD' -> list of years inclusive."""
    dt0, dt1 = datetime_str.split("/")
    y0, y1 = pd.Timestamp(dt0).year, pd.Timestamp(dt1).year
    return list(range(y0, y1 + 1))


def _both_exist(a: str, b: str) -> bool:
    return os.path.exists(a) and os.path.exists(b)


def _all_annual_outputs_exist(annual_ndvi_dir: str, annual_c_dir: str, years: List[int]) -> bool:
    for y in years:
        if not _both_exist(
            os.path.join(annual_ndvi_dir, f"NDVI_{y}.tif"),
            os.path.join(annual_c_dir, f"C_Factor_{y}.tif")
        ):
            return False
    return True

def _clean_index_array(arr: np.ndarray, index_name: str = "") -> np.ndarray:
    """
    Clean NDVI / C-factor arrays before plotting or saving.

    NDVI should be between -1 and 1.
    C-factor should be between 0 and 1.
    Very large values such as 1e38 are usually NoData/fill values.
    """
    arr = arr.astype("float32", copy=True)

    # Remove non-finite values and common huge raster fill values
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.where(np.abs(arr) > 1e6, np.nan, arr)

    name = index_name.lower()

    if "ndvi" in name:
        arr = np.where((arr >= -1.0) & (arr <= 1.0), arr, np.nan)

    if "c_factor" in name or "c factor" in name:
        arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)

    return arr

def _zonal_mean(raster_path: str, geom, target_crs) -> float:
    """Mean over geometry; returns np.nan if file missing or clipped region has no valid raster cells."""
    if not os.path.exists(raster_path):
        return np.nan

    try:
        with rio.open(raster_path) as src:
            geom_proj = gpd.GeoSeries([geom], crs=target_crs).to_crs(src.crs).geometry.iloc[0]
            out, _ = rio_mask(src, [geom_proj], crop=True, filled=False)

            band = out[0]
            data = band.filled(np.nan).astype(float) if np.ma.isMaskedArray(band) else band.astype(float)

            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

            data = np.where(np.isfinite(data), data, np.nan)

            if not np.isfinite(data).any():
                return np.nan

            return float(np.nanmean(data))

    except Exception:
        return np.nan


def _masked_mean(src: rio.DatasetReader, geom) -> float:
    """Mean of masked region; returns np.nan if region has no valid raster cells."""
    try:
        out_img, _ = rio_mask(src, [geom], crop=True, filled=False)
        band = out_img[0]

        arr = band.filled(np.nan).astype(float) if np.ma.isMaskedArray(band) else band.astype(float)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        arr = np.where(np.isfinite(arr), arr, np.nan)

        if not np.isfinite(arr).any():
            return np.nan

        return float(np.nanmean(arr))

    except Exception:
        return np.nan

def _get_adaptive_figsize_from_bounds(
    bounds,
    base_size=6.0,
    min_size=4.0,
    max_size=12.0,
    scale_factor=0.00005  # controls how much extent affects size
):
    """
    Adaptive figure size based on both shape AND spatial extent.
    """

    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-9)
    height = max(maxy - miny, 1e-9)

    aspect = width / height

    # 🔥 NEW: scale by spatial size
    scale = max(width, height) * scale_factor

    if aspect >= 1:
        fig_w = base_size * aspect * (1 + scale)
        fig_h = base_size * (1 + scale)
    else:
        fig_w = base_size * (1 + scale)
        fig_h = base_size / aspect * (1 + scale)

    fig_w = min(max(fig_w, min_size), max_size)
    fig_h = min(max(fig_h, min_size), max_size)

    return fig_w, fig_h

def _plot_valid_raster_outline(
    ax,
    data: np.ndarray,
    extent: Tuple[float, float, float, float],
    color: str = "black",
    linewidth: float = 1.2
) -> None:
    """
    Plot the outline of valid (non-NaN) raster cells so the boundary
    matches the displayed raster pixels more accurately.
    """
    valid_mask = np.isfinite(data).astype(np.uint8)

    if valid_mask.max() == 0:
        return

    xmin, xmax, ymin, ymax = extent
    nrows, ncols = valid_mask.shape

    xres = (xmax - xmin) / ncols
    yres = (ymax - ymin) / nrows

    xs = np.linspace(xmin + xres / 2, xmax - xres / 2, ncols)
    ys = np.linspace(ymax - yres / 2, ymin + yres / 2, nrows)  # origin='upper'

    ax.contour(xs, ys, valid_mask, levels=[0.5], colors=color, linewidths=linewidth)


def _set_axis_limits_from_gdf(ax, gdf: gpd.GeoDataFrame, pad_fraction: float = 0.02) -> None:
    """Set axis limits from GeoDataFrame bounds with small padding."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = (maxx - minx) * pad_fraction
    ypad = (maxy - miny) * pad_fraction
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)


def _reproject_array_for_plotting(
    data: np.ndarray,
    src_meta: dict,
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.nearest
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Reproject a raster array to another CRS for plotting only.

    Returns
    -------
    dst_array : np.ndarray
        Reprojected raster array.
    dst_extent : tuple
        Extent as (xmin, xmax, ymin, ymax) in target CRS.
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
    dst_array = np.where(np.isfinite(dst_array), dst_array, np.nan)
    dst_array = np.where(np.abs(dst_array) > 1e6, np.nan, dst_array)
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
) -> None:
    """Plot full raster in latitude/longitude degrees for context."""
    plot_data, extent = _reproject_array_for_plotting(
        data,
        src_meta,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )

    catch_plot = catchment_gdf.to_crs(epsg=4326) if catchment_gdf is not None else None

    if catch_plot is not None and not catch_plot.empty:
        plot_bounds = tuple(catch_plot.total_bounds)
    else:
        plot_bounds = (extent[0], extent[2], extent[1], extent[3])

    fig_w, fig_h = _get_adaptive_figsize_from_bounds(
        plot_bounds,
        base_size=6.0,
        min_size=5.0,
        max_size=11.0
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_data = np.ma.masked_invalid(plot_data)
    im = ax.imshow(plot_data, cmap="viridis", extent=extent, origin="upper", interpolation="nearest")
    _add_matched_colorbar(fig, ax, im, title)

    #_plot_valid_raster_outline(ax, plot_data, extent, color="black", linewidth=1.0)

    if catch_plot is not None and not catch_plot.empty:
        catch_plot.boundary.plot(ax=ax, color="black", linewidth=1.2)
        _set_axis_limits_from_gdf(ax, catch_plot, pad_fraction=0.02)
    else:
        _apply_small_plot_padding(ax, extent, pad_fraction=0.02)

    ax.set_title(title, fontsize=11, fontweight="bold")
    _style_map_axes(ax, nbins=4)

    fig.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    plot_data = np.ma.masked_invalid(plot_data)
    im = ax.imshow(plot_data, extent=extent, cmap="viridis", origin="upper", interpolation="nearest")
    _add_matched_colorbar(fig, ax, im, title)

    #_plot_valid_raster_outline(ax, plot_data, extent, color="black", linewidth=1.0)

    if boundary_plot is not None and not boundary_plot.empty:
        #boundary_plot.boundary.plot(ax=ax, color="black", linewidth=1.2)
        _set_axis_limits_from_gdf(ax, boundary_plot, pad_fraction=0.02)
    else:
        _apply_small_plot_padding(ax, extent, pad_fraction=0.02)

    if site_plot is not None and not site_plot.empty:
        site_plot.plot(ax=ax, markersize=15, color="red", marker="o")
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
                color="black",
            )

    ax.set_title(title, fontsize=11, fontweight="bold")
    _style_map_axes(ax, nbins=4)

    fig.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_riparian_map_for_year(
    ST_gdf: gpd.GeoDataFrame,
    catch_gdf: gpd.GeoDataFrame,
    year: int,
    out_png: str,
):
    """
    Plot riparian NDVI by stream segment for a single year, with fixed bins and legend.
    Expects a column on ST_gdf named 'Riparian NDVI - {year}'.

    Plot is made in EPSG:4326 so axes are longitude/latitude in degrees,
    consistent with the topography plotting style.
    """
    col = f"Riparian NDVI - {year}"
    if col not in ST_gdf.columns:
        _log("WARN", f"{_plot_riparian_map_for_year.__name__}: '{col}' not found on streams.")
        return

    breaks = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
    colors = ["#d7191c", "#fdae61", "#ffea00", "#a6d96a", "#1a9641", "#00441b"]
    labels = ["≤ 0.0", "0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "> 0.8"]

    plot_crs = 4326
    streams_plot = ST_gdf.to_crs(epsg=plot_crs)
    catch_plot = catch_gdf.to_crs(epsg=plot_crs)

    vals = pd.to_numeric(streams_plot[col], errors="coerce")
    cats = pd.cut(vals, breaks, right=True, labels=labels, include_lowest=True)
    streams_plot = streams_plot.assign(__class=cats)

    fig_w, fig_h = _get_adaptive_figsize_from_bounds(
        tuple(catch_plot.total_bounds),
        base_size=6.5,
        min_size=5.0,
        max_size=11.0
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    xmin, ymin, xmax, ymax = catch_plot.total_bounds
    xpad = (xmax - xmin) * 0.02
    ypad = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    catch_plot.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=10)

    for lab, col_hex in zip(labels, colors):
        seg = streams_plot[streams_plot["__class"] == lab]
        if len(seg):
            seg.plot(ax=ax, color=col_hex, linewidth=3.0, zorder=20)

    legend_lines = [Line2D([0], [0], color="black", lw=6, label="Catchment boundary")]
    for lab, col_hex in zip(labels, colors):
        legend_lines.append(Line2D([0], [0], color=col_hex, lw=6, label=lab))

    ax.legend(
        handles=legend_lines,
        title="Riparian NDVI",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    ax.set_title(f"Riparian NDVI by Stream Order — {year}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latitude (°)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _mask_raster_by_waterbodies(
    da: xr.DataArray,
    waterbodies_gpkg: str,
    waterbodies_layer: str = "DEA_Waterbodies",
) -> xr.DataArray:
    """
    Mask raster cells overlapping DEA waterbodies.

    Waterbody-overlapped cells are set to NaN.
    If the waterbodies file/layer is missing or empty, the original raster is returned.
    """
    if not os.path.exists(waterbodies_gpkg):
        _log("WARN", f"DEA Waterbodies file not found. NDVI water mask skipped: {waterbodies_gpkg}")
        return da

    try:
        try:
            water_gdf = gpd.read_file(waterbodies_gpkg, layer=waterbodies_layer)
        except Exception:
            # Fallback if the GPKG has no named layer or layer name differs
            water_gdf = gpd.read_file(waterbodies_gpkg)

        if water_gdf.empty:
            _log("WARN", "DEA Waterbodies layer is empty. NDVI water mask skipped.")
            return da

        if da.rio.crs is None:
            _log("WARN", "NDVI raster has no CRS. NDVI water mask skipped.")
            return da

        if water_gdf.crs is None:
            _log("WARN", "DEA Waterbodies has no CRS. NDVI water mask skipped.")
            return da

        water_gdf = water_gdf.to_crs(da.rio.crs)
        water_gdf = water_gdf[water_gdf.geometry.notna() & ~water_gdf.geometry.is_empty]

        if water_gdf.empty:
            _log("WARN", "DEA Waterbodies has no valid geometries. NDVI water mask skipped.")
            return da

        # rioxarray.clip with invert=True masks inside the supplied geometry.
        masked = da.rio.clip(
            water_gdf.geometry,
            water_gdf.crs,
            drop=False,
            invert=True,
            all_touched=True,
        )

        masked = masked.where(np.isfinite(masked))
        masked.rio.write_crs(da.rio.crs, inplace=True)
        masked.rio.write_nodata(np.nan, inplace=True)

        #_log("OK", "NDVI masked by DEA Waterbodies.")
        return masked

    except Exception as e:
        _log("WARN", f"NDVI waterbody masking failed and was skipped: {e}")
        return da
    
@dataclass
class VegConfig:
    """Inputs & knobs for vegetation processing."""
    chm_workspace: str
    catchment_path: str
    datetime_range: str  # e.g., "2000-01-01/2024-12-31"

    # Optional sites input.
    # If provided and valid, site-based vegetation outputs are produced.
    # If missing/None/invalid, the module continues with catchment + riparian outputs only.
    sites_path: Optional[str] = None

    catchment_crs: Optional[object] = None

    # STAC/DEA parameters
    stac_url: str = "https://explorer.dea.ga.gov.au/stac"
    cloud_cover_lt: int = 20
    filter_query: Optional[dict] = None
    split_date: str = "2016-01-01"
    bands_ls7: Tuple[str, str] = ("nbart_red", "nbart_nir")
    bands_s2: Tuple[str, str] = ("nbart_red", "nbart_nir_1")
    riparian_buffer_m: float = 30.0


def _write_catchment_annual_context_plots(
    indices_output: str,
    catch_plots: str,
    catchment_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Write catchment-level annual raster plots for Annual NDVI and Annual C Factor.
    This is independent of site polygons and ensures catchment plots are produced
    even when the all-sites GPKG is missing.
    """
    try:
        annual_targets = [
            ("NDVI", os.path.join(indices_output, "NDVI", "Annual")),
            ("C Factor", os.path.join(indices_output, "C Factor", "Annual")),
        ]

        for group_name, annual_dir in annual_targets:
            if not os.path.isdir(annual_dir):
                _log("WARN", f"Annual folder not found for {group_name}: {annual_dir}")
                continue

            for file in sorted(os.listdir(annual_dir)):
                if not file.lower().endswith(".tif"):
                    continue

                raster_path = os.path.join(annual_dir, file)
                short_name = os.path.splitext(file)[0]
                out_png = os.path.join(catch_plots, f"{short_name}.png")

                try:
                    with rio.open(raster_path) as src:
                        arr = src.read(1, masked=True)
                        arr = arr.filled(np.nan).astype("float32")
                        arr = _clean_index_array(arr, short_name)

                        src_meta = {
                            "crs": src.crs,
                            "transform": src.transform,
                            "width": src.width,
                            "height": src.height,
                        }

                    _plot_raster_context_in_degrees(
                        data=arr,
                        src_meta=src_meta,
                        out_png=out_png,
                        title=short_name,
                        catchment_gdf=catchment_gdf,
                    )
                except Exception as e:
                    _log("WARN", f"Catchment plot failed for {raster_path}: {e}")

        _log("OK", "Catchment annual NDVI/C-factor plots checked/written.")
    except Exception as e:
        _log("WARN", f"Catchment annual plotting block failed: {e}")


def veg_indices_and_c_factor(cfg: VegConfig):
    """
    End-to-end workflow:
      - Query DEA STAC for LS7 (pre-split) and S2 (post-split), load as xarray
      - Compute timestep NDVI & C; write rasters (cached)
      - Compute annual NDVI & C; write rasters (cached)
      - Append annual NDVI & C means to catchment GPKG layer
      - Compute riparian NDVI:
          • per stream segment (buffered), per year (columns on Streams layer)
          • per order × year summary (CSV + plot)
          • per-year maps by stream segment (classified color legend)
      - Per-site summaries & plots for all annual rasters under /Indices/**/Annual
      - Save per-site GPKG + CSV and combined all-sites files

    Returns
    -------
    all_sites_gpkg, dem_crs, sites_datasets, indices_output, annual_ndvi_dir, annual_c_dir
    """
    _log("INFO", "Starting vegetation indices and C-factor processing...")

    # ---------- Folders ----------
    catchment_name = os.path.splitext(os.path.basename(cfg.catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(cfg.chm_workspace, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    veg_folder = os.path.join(catch_datasets, "Vegetation")
    satellite_output = os.path.join(veg_folder, "Satellite data")
    indices_output = os.path.join(veg_folder, "Indices")
    riparian_output = os.path.join(veg_folder, "Riparian")
    ndvi_output = os.path.join(indices_output, "NDVI")
    c_factor_output = os.path.join(indices_output, "C Factor")

    for d in [
        catchment_folder, catch_datasets, catch_plots, sites_datasets, sites_plots,
        veg_folder, satellite_output, indices_output, ndvi_output, c_factor_output, riparian_output
    ]:
        _ensure_dirs([d])

    _log_folder_summary(catchment_folder, catch_datasets, catch_plots, veg_folder, sites_datasets, sites_plots)

    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    catch_layer = f"{catchment_name} Data"
    all_sites_gpkg = os.path.join(
        sites_datasets,
        f"{catchment_name} Sites Data.gpkg"
    )

    all_sites_gpkg_or_none = _existing_or_none(all_sites_gpkg)

    if all_sites_gpkg_or_none is None:
        _log(
            "WARN",
            f"Topography-generated site polygons not found: {all_sites_gpkg}. "
            "Site-based vegetation outputs will be skipped."
        )
    else:
        _log("OK", f"Using topography-generated site polygons: {all_sites_gpkg}")

    annual_ndvi_dir = os.path.join(ndvi_output, "Annual")
    annual_c_dir = os.path.join(c_factor_output, "Annual")
    _ensure_dirs([annual_ndvi_dir, annual_c_dir])

    # ---------- Catchment & DEM ----------
    gdf = gpd.read_file(cfg.catchment_path)
    if gdf.empty:
        raise ValueError("Catchment file has no features.")
    _log("OK", f"Catchment loaded: {cfg.catchment_path}")

    if cfg.catchment_crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(cfg.catchment_crs)
        elif gdf.crs != cfg.catchment_crs:
            gdf = gdf.to_crs(cfg.catchment_crs)

    gdf_wgs84 = gdf.to_crs(epsg=4326)
    bbox = gdf_wgs84.total_bounds

    topo_folder = os.path.join(catch_datasets, "Topography")
    dem_projected_file = os.path.join(topo_folder, "DEM.tif")
    if not os.path.exists(dem_projected_file):
        raise FileNotFoundError(f"DEM missing at {dem_projected_file}. Run dem_and_terrain first.")

    with rio.open(dem_projected_file) as dem_src:
        dem_crs = dem_src.crs
        dem_resolution = abs(dem_src.transform.a)

    _log("OK", f"Using DEM from {_folder_label(topo_folder)}: {dem_projected_file}")
    stream_network = os.path.join(topo_folder, "Stream_Network.gpkg")
    waterbodies_gpkg = os.path.join(topo_folder, "DEA_Waterbodies.gpkg")

    # ---------- STAC: cache gate ----------
    try:
        requested_years = _years_from_interval(cfg.datetime_range)
    except Exception:
        requested_years = sorted({
            int(m.group(1)) for f in os.listdir(annual_ndvi_dir)
            if (m := re.match(r"NDVI_(\d{4})\.tif$", f))
        })

    if requested_years and _all_annual_outputs_exist(annual_ndvi_dir, annual_c_dir, requested_years):
        _log("INFO", "All annual NDVI and C-factor rasters already exist for requested years; skipping STAC load.")
    else:
        catalog = pystac_client.Client.open(cfg.stac_url)
        odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

        SPLIT_DATE = pd.Timestamp(cfg.split_date)
        dt0, dt1 = cfg.datetime_range.split("/")
        dt_start, dt_end = pd.Timestamp(dt0), pd.Timestamp(dt1)

        ts_ndvi_output = os.path.join(ndvi_output, "Time step")
        ts_c_factor_out = os.path.join(c_factor_output, "Time step")
        _ensure_dirs([ts_ndvi_output, ts_c_factor_out])

        dem_ref = rxr.open_rasterio(dem_projected_file, masked=True).squeeze().copy()

        default_cql2 = {
            "op": "<",
            "args": [
                {"property": "eo:cloud_cover"},
                int(cfg.cloud_cover_lt),
            ],
        }

        for year in requested_years:
            year_start = pd.Timestamp(year=year, month=1, day=1)
            year_end = pd.Timestamp(year=year, month=12, day=31)

            y0 = max(year_start, dt_start)
            y1 = min(year_end, dt_end)
            if y0 > y1:
                continue

            ndvi_fp_y = os.path.join(annual_ndvi_dir, f"NDVI_{year}.tif")
            c_fp_y = os.path.join(annual_c_dir, f"C_Factor_{year}.tif")
            if _both_exist(ndvi_fp_y, c_fp_y):
                _log("INFO", f"Annual NDVI/C for {year} already on disk; skipping STAC load for this year.")
                continue

            subranges: List[Tuple[pd.Timestamp, pd.Timestamp, List[str], List[str]]] = []
            if y0 < SPLIT_DATE:
                subranges.append((
                    y0, min(y1, SPLIT_DATE - pd.Timedelta(seconds=1)),
                    ["ga_ls7e_ard_3"], list(cfg.bands_ls7)
                ))
            if y1 >= SPLIT_DATE:
                subranges.append((
                    max(y0, SPLIT_DATE), y1,
                    ["ga_s2am_ard_3", "ga_s2bm_ard_3"], list(cfg.bands_s2)
                ))

            ds_year_parts: List[xr.Dataset] = []

            for d0, d1, collections_sel, bands_sel in subranges:
                if d0 > d1:
                    continue
                dt_str = f"{d0:%Y-%m-%d}/{d1:%Y-%m-%d}"

                try:
                    search = catalog.search(
                        bbox=bbox,
                        collections=collections_sel,
                        datetime=dt_str,
                        filter=(cfg.filter_query if cfg.filter_query is not None else default_cql2),
                        filter_lang="cql2-json",
                    )
                    items = list(search.items())
                except Exception:
                    search = catalog.search(
                        bbox=bbox,
                        collections=collections_sel,
                        datetime=dt_str,
                        query={"eo:cloud_cover": {"lt": int(cfg.cloud_cover_lt)}},
                    )
                    items = list(search.items())

                if not items:
                    _log("WARN", f"No items for {collections_sel} in {dt_str}")
                    continue

                ds_part = odc.stac.load(
                    items,
                    bands=bands_sel,
                    crs=dem_crs,
                    resolution=dem_resolution,
                    groupby="solar_day",
                    bbox=bbox,
                    dtype="float32",
                    chunks={"time": 1, "x": 1024, "y": 1024},
                    fail_on_error=False,
                    progress=False,
                )

                if "nbart_nir_1" in ds_part.data_vars and "nbart_nir" not in ds_part.data_vars:
                    ds_part = ds_part.rename({"nbart_nir_1": "nbart_nir"})

                if ds_part.sizes.get("time", 0) == 0:
                    continue

                ds_year_parts.append(ds_part)

            if not ds_year_parts:
                _log("WARN", f"No imagery loaded for {year}; skipping this year.")
                continue

            ds_year = xr.concat(ds_year_parts, dim="time").sortby("time")
            if ds_year.sizes.get("time", 0) == 0:
                _log("WARN", f"Year {year} concat produced 0 timesteps; skipping.")
                del ds_year_parts, ds_year
                gc.collect()
                continue

            for t_idx, ts in enumerate(ds_year.time.values):
                stamp = pd.to_datetime(ts).strftime("%Y%m%d")
                ndvi_path = os.path.join(ts_ndvi_output, f"NDVI_{stamp}.tif")
                c_path = os.path.join(ts_c_factor_out, f"C_Factor_{stamp}.tif")
                if _both_exist(ndvi_path, c_path):
                    continue

                red = ds_year["nbart_red"].isel(time=t_idx)
                nir = ds_year["nbart_nir"].isel(time=t_idx)
                ndvi_ts = (nir - red) / (nir + red)

                ndvi_clip = ndvi_ts.rio.clip(gdf.geometry, gdf.crs, drop=True)
                ndvi_aln = ndvi_clip.rio.reproject_match(dem_ref)
                ndvi_aln.rio.write_nodata(np.nan, inplace=True)

                # Mask NDVI where it overlaps DEA Waterbodies.
                # Because C-factor is calculated from NDVI, this also masks C-factor.
                ndvi_aln = _mask_raster_by_waterbodies(
                    ndvi_aln,
                    waterbodies_gpkg=waterbodies_gpkg,
                )

                c_factor = np.clip(np.exp(-2 * ndvi_aln), 0, 1)
                cf_da = xr.DataArray(c_factor, coords=ndvi_aln.coords, dims=ndvi_aln.dims)
                cf_da = cf_da.where(np.isfinite(ndvi_aln))
                cf_da.rio.write_crs(ndvi_aln.rio.crs, inplace=True)
                cf_da.rio.write_nodata(np.nan, inplace=True)

            # ---------- Annual NDVI and C-factor ----------
            # ---------- Annual NDVI and C-factor ----------
            red_all = ds_year["nbart_red"].astype("float32")
            nir_all = ds_year["nbart_nir"].astype("float32")
            # DEA optical NoData is commonly 0. Remove it BEFORE NDVI.
            valid = (
                np.isfinite(red_all) &
                np.isfinite(nir_all) &
                (red_all > 0) &
                (nir_all > 0)
            )
            denom_all = nir_all + red_all
            ndvi_all = xr.where(
                valid & (denom_all > 0),
                (nir_all - red_all) / denom_all,
                np.nan
            )
            # Keep physical NDVI range only
            ndvi_all = ndvi_all.where(
                np.isfinite(ndvi_all) &
                (ndvi_all >= -1.0) &
                (ndvi_all <= 1.0)
            )
            valid_pixel_count = ndvi_all.count(dim="time")
            ndvi_med_year = ndvi_all.median(dim="time", skipna=True)
            # Remove pixels with too few valid observations
            ndvi_med_year = ndvi_med_year.where(valid_pixel_count >= 1)
            # Make CRS / nodata explicit BEFORE clip/reproject
            ndvi_med_year.rio.write_crs(dem_crs, inplace=True)
            ndvi_med_year.rio.write_nodata(np.nan, inplace=True)
            # Clip and align to DEM
            ndvi_year = ndvi_med_year.rio.clip(
                gdf.geometry,
                gdf.crs,
                drop=True,
                all_touched=True
            )
            ndvi_year.rio.write_crs(dem_crs, inplace=True)
            ndvi_year.rio.write_nodata(np.nan, inplace=True)
            ndvi_year = ndvi_year.rio.reproject_match(
                dem_ref,
                resampling=Resampling.bilinear,
                nodata=np.nan
            )
            # Clean AFTER reprojection
            ndvi_year = ndvi_year.where(
                np.isfinite(ndvi_year) &
                (ndvi_year >= -1.0) &
                (ndvi_year <= 1.0)
            )
            ndvi_year.rio.write_crs(dem_ref.rio.crs, inplace=True)
            ndvi_year.rio.write_nodata(np.nan, inplace=True)
            # Optional waterbody mask
            ndvi_year = _mask_raster_by_waterbodies(
                ndvi_year,
                waterbodies_gpkg=waterbodies_gpkg,
            )
            # C-factor only from valid NDVI
            c_year = xr.where(
                np.isfinite(ndvi_year),
                np.clip(np.exp(-2.0 * ndvi_year), 0.0, 1.0),
                np.nan
            )
            c_year.rio.write_crs(dem_ref.rio.crs, inplace=True)
            c_year.rio.write_nodata(np.nan, inplace=True)
            ndvi_year.rio.to_raster(ndvi_fp_y, dtype="float32", nodata=np.nan)
            c_year.rio.to_raster(c_fp_y, dtype="float32", nodata=np.nan)
            _log("OK", f"Wrote annual NDVI/C rasters for {year}")
            del ds_year_parts, ds_year, ndvi_all
            gc.collect()

        _log("OK", "Saved timestep and annual NDVI/C rasters (processed year-by-year).")

    # ================= Riparian NDVI (per segment; per order × year) =================
    ST_gdf = gpd.read_file(stream_network)
    if ST_gdf.crs is None:
        ST_gdf = ST_gdf.set_crs(dem_crs)
    if ST_gdf.crs != dem_crs:
        ST_gdf = ST_gdf.to_crs(dem_crs)

    riparian_buf = ST_gdf.buffer(cfg.riparian_buffer_m).unary_union
    riparian_gdf = gpd.GeoDataFrame(geometry=[riparian_buf], crs=dem_crs)

    years_list = sorted({
        int(m.group(1)) for f in os.listdir(annual_ndvi_dir)
        if (m := re.match(r"NDVI_(\d{4})\.tif$", f))
    })

    for yy in years_list:
        colname = f"Riparian NDVI - {yy}"
        if colname not in ST_gdf.columns:
            ST_gdf[colname] = np.nan
        ndvi_path = os.path.join(annual_ndvi_dir, f"NDVI_{yy}.tif")
        if not os.path.exists(ndvi_path):
            _log("WARN", f"Missing annual NDVI for {yy}: {ndvi_path}")
            continue

        with rio.open(ndvi_path) as src:
            st_proj = ST_gdf.to_crs(src.crs) if ST_gdf.crs != src.crs else ST_gdf
            vals = []
            for geom in st_proj.geometry:
                if geom is None or geom.is_empty:
                    vals.append(np.nan)
                    continue
                buf = geom.buffer(cfg.riparian_buffer_m)
                vals.append(_masked_mean(src, buf))

            ST_gdf[colname] = (
                pd.Series(vals, index=st_proj.index)
                .reindex(ST_gdf.index)
                .astype(float)
            )

        try:
            out_png_year = os.path.join(catch_plots, f"Riparian_NDVI_by_StreamSegment_{yy}.png")
            _plot_riparian_map_for_year(
                ST_gdf=ST_gdf,
                catch_gdf=gdf,
                year=int(yy),
                out_png=out_png_year,
            )
            _log("OK", f"Saved riparian map for {yy}")
        except Exception as e:
            _log("WARN", f"Riparian map for {yy} failed: {e}")

        try:
            with rio.open(ndvi_path) as src_union:
                rip_proj = riparian_gdf.to_crs(src_union.crs)
                out_img, out_tr = rio_mask(src_union, rip_proj.geometry, crop=True, filled=False)
                out_meta = src_union.meta.copy()
                out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_tr})
                rip_fp = os.path.join(riparian_output, f"Reparian_{yy}.tif")
                _ensure_dirs([os.path.dirname(rip_fp)])
                with rio.open(rip_fp, "w", **out_meta) as dst:
                    dst.write(out_img)
        except Exception as e:
            _log("WARN", f"Could not write union riparian raster for {yy}: {e}")

    try:
        riparian_year_cols = [c for c in ST_gdf.columns if isinstance(c, str) and c.startswith("Riparian NDVI - ")]
        if riparian_year_cols:
            years_sorted = sorted(int(c.split(" - ")[1]) for c in riparian_year_cols)
            order_col = "order" if "order" in ST_gdf.columns else None

            if order_col is None:
                _log("WARN", "Stream order column not found on stream network; riparian order summary skipped.")
            else:
                rows = []
                for yy in years_sorted:
                    col = f"Riparian NDVI - {yy}"
                    tmp = ST_gdf[[order_col, col]].copy()
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                    tmp = tmp.dropna(subset=[order_col, col])
                    if tmp.empty:
                        continue
                    grp = tmp.groupby(order_col, dropna=True)[col].mean().reset_index()
                    grp["Year"] = int(yy)
                    grp = grp.rename(columns={order_col: "Order", col: "Riparian_NDVI"})
                    rows.append(grp)

                if rows:
                    riparian_summary_df = pd.concat(rows, ignore_index=True)
                    riparian_summary_csv = os.path.join(riparian_output, "Riparian_NDVI_Mean_by_StreamOrder.csv")
                    riparian_summary_df.to_csv(riparian_summary_csv, index=False)
                    _log("OK", f"Saved riparian NDVI order summary CSV: {riparian_summary_csv}")

                    pivot_df = riparian_summary_df.pivot(index="Year", columns="Order", values="Riparian_NDVI").sort_index()

                    plt.figure(figsize=(8, 4.5))
                    ax = plt.gca()
                    for order in pivot_df.columns:
                        ax.plot(
                            pivot_df.index,
                            pivot_df[order],
                            marker="o",
                            linewidth=1.6,
                            label=f"Order {int(order)}"
                        )
                    ax.set_xlabel("Year", fontsize=10, fontweight="bold")
                    ax.set_ylabel("Mean Riparian NDVI", fontsize=10, fontweight="bold")
                    ax.set_title("Riparian NDVI Mean by Stream Order", fontsize=11, fontweight="bold")
                    ax.tick_params(axis="both", labelsize=10)
                    ax.grid(True, which="both", linestyle="--", alpha=0.6)
                    ax.legend(
                        title="Stream Order",
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        ncols=1,
                        fontsize=9,
                        title_fontsize=9,
                        frameon=True,
                        borderaxespad=0.0,
                    )
                    fig = plt.gcf()
                    fig.tight_layout(rect=[0, 0, 0.80, 1])
                    out_png = os.path.join(catch_plots, "Riparian_NDVI_Mean_by_StreamOrder_Timeseries.png")
                    plt.savefig(out_png, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    _log("OK", f"Saved riparian NDVI time-series plot: {out_png}")

                    def _ordinal_word(n: int) -> str:
                        mapping = {
                            1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth",
                            6: "Sixth", 7: "Seventh", 8: "Eighth", 9: "Ninth", 10: "Tenth"
                        }
                        return mapping.get(int(n), f"{int(n)}th")

                    wide = pivot_df.reset_index().rename(columns={"Year": "Date"})
                    rename_map = {
                        c: f"Riparian NDVI- {_ordinal_word(int(c))} Stream Order"
                        for c in wide.columns if c != "Date" and not pd.isna(c)
                    }
                    wide = wide.rename(columns=rename_map)
                    wide["Date"] = pd.to_numeric(wide["Date"], errors="coerce").astype("Int64")

                    try:
                        base_gdf = gpd.read_file(catch_gpkg, layer=catch_layer)
                        merged = base_gdf.merge(wide, on="Date", how="left")
                        try:
                            fiona.remove(catch_gpkg, layer=catch_layer)
                        except Exception:
                            pass
                        gpd.GeoDataFrame(merged, geometry="geometry", crs=base_gdf.crs).to_file(
                            catch_gpkg, layer=catch_layer, driver="GPKG"
                        )
                        _log("OK", f"Appended per-order riparian NDVI columns to {_folder_label(catch_gpkg)} ({catch_layer})")
                    except Exception as e:
                        _log("ERROR", f"Appending per-order riparian NDVI failed: {e}")
        else:
            _log("WARN", "No 'Riparian NDVI - YYYY' columns found on streams.")
    except Exception as e:
        _log("ERROR", f"Riparian NDVI order-timeseries failed: {e}")

    try:
        ST_gdf.to_file(stream_network, layer="Streams", driver="GPKG", if_exists="replace")
        _log("OK", f"Wrote stream riparian NDVI columns to {_folder_label(stream_network)} (Streams)")
    except Exception as e:
        _log("ERROR", f"Writing riparian NDVI columns failed: {e}")

    # ================= Catchment annual plots (independent of site polygons) =================
    _write_catchment_annual_context_plots(
        indices_output=indices_output,
        catch_plots=catch_plots,
        catchment_gdf=gdf,
    )

    # ================= Per-site summaries & plots (Annual rasters only) =================
    sites_poly_gdf = None

    if all_sites_gpkg_or_none is not None:
        try:
            sites_poly_gdf = gpd.read_file(all_sites_gpkg_or_none)

            if sites_poly_gdf.empty:
                _log("WARN", "Site polygons file is empty. Site-based summaries will be skipped.")
                sites_poly_gdf = None
            else:
                if sites_poly_gdf.crs is None:
                    raise ValueError(
                        "Sites file has no CRS. Please define its CRS before running vegetation processing."
                    )

                if sites_poly_gdf.crs != dem_crs:
                    sites_poly_gdf = sites_poly_gdf.to_crs(dem_crs)

                _log("OK", f"Site polygons loaded and projected to DEM CRS: {all_sites_gpkg_or_none}")

        except Exception as e:
            _log("WARN", f"Could not load site polygons and site-based summaries will be skipped: {e}")
            sites_poly_gdf = None
    else:
        _log("WARN", "No all-sites GPKG found. Catchment and riparian outputs will still be produced, but site-based summaries will be skipped.")

    WH_rows = []
    if sites_poly_gdf is not None and not sites_poly_gdf.empty:
        for idx, row in sites_poly_gdf.iterrows():
            try:
                site_id = row.get("Site_id", idx)
                site_geom = row.geometry
                attrs = row.drop(labels="geometry").to_dict()
                site_gdf = gpd.GeoDataFrame([attrs], geometry=[site_geom], crs=sites_poly_gdf.crs)
                site_vals: Dict[str, float] = {}

                site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
                site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
                _ensure_dirs([site_data_dir, site_plot_dir])

                for set1 in os.listdir(indices_output):
                    set1_path = os.path.join(indices_output, set1)
                    if not os.path.isdir(set1_path):
                        continue
                    set2 = "Annual"
                    set2_path = os.path.join(set1_path, set2)
                    if not os.path.isdir(set2_path):
                        continue

                    for file in os.listdir(set2_path):
                        if not file.lower().endswith(".tif"):
                            continue
                        raster_path = os.path.join(set2_path, file)
                        short_name = os.path.splitext(file)[0]

                        with rio.open(raster_path) as src:
                            # Stats over this site's polygon
                            v = site_gdf.to_crs(src.crs) if site_gdf.crs != src.crs else site_gdf
                            geom = [v.geometry.iloc[0]]
                            out_image, out_transform = rio_mask(src, geom, crop=True, filled=False)
                            masked_band = out_image[0]
                            data = masked_band.filled(np.nan).astype(float)
                            site_vals[f"{short_name} (mean)"] = round(float(np.nanmean(data)), 2) if data.size else np.nan
                            site_vals[f"{short_name} (median)"] = round(float(np.nanmedian(data)), 2) if data.size else np.nan

                            # Save clipped raster
                            clipped_meta = src.meta.copy()
                            clipped_meta.update({
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform,
                                "crs": src.crs,
                                "nodata": np.nan,
                                "count": 1,
                                "dtype": "float32",
                            })
                            with rio.open(os.path.join(site_data_dir, f"{short_name}.tif"), "w", **clipped_meta) as dest:
                                dest.write(out_image[0].filled(np.nan).astype("float32"), 1)

                            # Plot clipped raster in degrees
                            try:
                                plot_arr = out_image[0]
                                if np.ma.isMaskedArray(plot_arr):
                                    plot_arr = plot_arr.filled(np.nan)
                                plot_arr = plot_arr.astype("float32")

                                clip_meta = {
                                    "crs": src.crs,
                                    "transform": out_transform,
                                    "width": out_image.shape[2],
                                    "height": out_image.shape[1],
                                }

                                site_point_gdf = None
                                if "X_site" in site_gdf.columns and "Y_site" in site_gdf.columns:
                                    sx, sy = site_gdf["X_site"].iloc[0], site_gdf["Y_site"].iloc[0]
                                    site_point_gdf = gpd.GeoDataFrame(
                                        {"Site_id": [site_id]},
                                        geometry=[Point(sx, sy)],
                                        crs=v.crs,
                                    )

                                _plot_clipped_raster_in_degrees(
                                    data=plot_arr,
                                    src_meta=clip_meta,
                                    out_png=os.path.join(site_plot_dir, f"{short_name}.png"),
                                    title=f"{short_name} - Site {site_id}",
                                    boundary_gdf=v,
                                    site_point_gdf=site_point_gdf,
                                    site_id=site_id,
                                )
                            except Exception as e:
                                _log("WARN", f"Plotting {short_name} for site {site_id} failed: {e}")

                # === Per-site mean of Reparian_{year}.tif (union riparian) ===
                if os.path.isdir(riparian_output):
                    for rf in os.listdir(riparian_output):
                        if not (rf.lower().endswith(".tif") and rf.startswith("Reparian_")):
                            continue
                        year_str = os.path.splitext(rf)[0].split("_")[-1]
                        rip_fp = os.path.join(riparian_output, rf)
                        with rio.open(rip_fp) as src_rip:
                            v_rip = site_gdf.to_crs(src_rip.crs) if site_gdf.crs != src_rip.crs else site_gdf
                            geom = [v_rip.geometry.iloc[0]]
                            rip_img, _ = rio_mask(src_rip, geom, crop=True, filled=False)
                            rip_band = rip_img[0]
                            site_vals[f"Reparian_ndvi_{year_str}"] = (
                                np.nan if (rip_band.size == 0 or np.ma.count(rip_band) == 0)
                                else round(float(np.ma.mean(rip_band)), 4)
                            )

                rip_pairs = []
                for k, val in site_vals.items():
                    if k.startswith("Reparian_ndvi_"):
                        try:
                            yr = int(k.split("_")[-1])
                            rip_pairs.append((yr, float(val)))
                        except Exception:
                            pass

                rip_pairs.sort(key=lambda t: t[0])
                years = [y for y, _v in rip_pairs]
                vals = [v for _y, v in rip_pairs if np.isfinite(v)]

                if len(years) >= 2 and len(vals) >= 2:
                    plt.figure(figsize=(7, 4))
                    plt.plot(years, [vv for vv in vals], marker="o", color="black", linewidth=1.2)
                    plt.xlabel("Year", fontsize=10, fontweight="bold")
                    plt.ylabel("Riparian NDVI (mean)", fontsize=10, fontweight="bold")
                    plt.title(f"Riparian Area mean NDVI - Site {site_id}", fontsize=11, fontweight="bold")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    first, last = years[0], years[-1]
                    tick_years = list(range(first, last + 1, 4))
                    plt.xlim(first, last)
                    plt.xticks(tick_years, [str(y) for y in tick_years], fontsize=9)
                    ax = plt.gca()
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                    ax.tick_params(axis="both", labelsize=10)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(site_plot_dir, f"Site_{site_id}_Riparian_NDVI_Timeseries.png"),
                        dpi=300,
                        bbox_inches="tight"
                    )
                    plt.close()

                site_gdf = site_gdf.assign(**site_vals)
                WH_rows.append(site_gdf)
                site_gpkg = os.path.join(site_data_dir, f"Site_{site_id}.gpkg")
                site_csv = os.path.join(site_data_dir, f"Site_{site_id}.csv")
                site_gdf.to_file(site_gpkg, driver="GPKG")
                site_gdf.drop(columns="geometry").to_csv(site_csv, index=False)
                _log("OK", f"Site {site_id} vegetation data and plots saved.")
            except Exception as e:
                _log("ERROR", f"Site {row.get('Site_id', idx)} failed: {e}")
                continue
    else:
        _log("WARN", "No valid site polygons available. Per-site vegetation summaries and plots were skipped.")

    # =========================================================================
    # DEACTIVATED: Combined "all sites" update/write to all_sites_gpkg & CSV
    # -------------------------------------------------------------------------
    # all_gdf = gpd.GeoDataFrame(pd.concat(WH_rows, ignore_index=True), crs=dem_crs)
    # all_sites_csv = os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv")
    # all_gdf.to_file(all_sites_gpkg, driver="GPKG")
    # all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)
    # =========================================================================
    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del gdf, gdf_wgs84
    except Exception:
        pass

    try:
        del ST_gdf, riparian_gdf, riparian_buf
    except Exception:
        pass

    try:
        del sites_poly_gdf, WH_rows
    except Exception:
        pass

    try:
        del site_gdf, site_vals, site_geom, site_point_gdf
    except Exception:
        pass

    try:
        del dem_ref
    except Exception:
        pass

    try:
        del ds_year_parts, ds_year, ds_part
    except Exception:
        pass

    try:
        del ndvi_all, ndvi_med_year, ndvi_year, c_year
    except Exception:
        pass

    try:
        del red, nir, ndvi_ts, ndvi_clip, ndvi_aln, c_factor, cf_da
    except Exception:
        pass

    try:
        del out_image, out_img, rip_img, data, plot_arr
    except Exception:
        pass

    try:
        del riparian_summary_df, pivot_df, wide, merged, base_gdf
    except Exception:
        pass

    gc.collect()

    _log("INFO", "Memory cleanup completed.")
    _log("OK", "Vegetation indices and C-factor processing completed.")

    return
