# src/chm/connectivity.py
from __future__ import annotations

# ============================ Imports ============================
# --- stdlib ---
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# --- numerical ---
import numpy as np
import pandas as pd

# --- geospatial ---
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import plotting_extent
from pysheds.grid import Grid
from shapely.geometry import Point

# --- plotting ---
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc

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
    """
    chm_workspace: str
    catchment_path: str
    catchment_crs: Optional[str] = None
    sdr_max: float = 0.9
    ic0: float = 0.5
    k: float = 3
    stream_area_threshold_m2: float = 1.3e4  # ≈ 13,000 m²


# ============================ Small helpers ============================

def _ensure_dirs(paths: List[str]) -> None:
    """Create the given directories if missing."""
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
    con_folder: str,
    sites_datasets: str,
    sites_plots: str,
) -> None:
    """Print full output folder structure once at the start."""
    _log("OK", "Output folders are ready.")
    _log("INFO", "Created/using output folders:")
    _log("INFO", f"  Catchment folder       : {catchment_folder}")
    _log("INFO", f"  Catchment datasets     : {catch_datasets}")
    _log("INFO", f"  Catchment plots/maps   : {catch_plots}")
    _log("INFO", f"  Connectivity           : {con_folder}")
    _log("INFO", f"  Sites datasets         : {sites_datasets}")
    _log("INFO", f"  Sites plots/maps       : {sites_plots}")


def _read_dem(catch_datasets: str) -> Tuple[np.ndarray, dict, rio.Affine, float, float, float, str]:
    """
    Load the projected DEM prepared by your terrain step.
    """
    dem_path = os.path.join(catch_datasets, "Topography", "DEM.tif")
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path} (run dem_and_terrain first)")

    with rio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
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
    """
    dz_dx, dz_dy = np.gradient(dem, xres, yres)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_rad = np.arctan(slope_ratio)
    slope_deg = np.degrees(slope_rad)

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

def _get_waterbodies_mask(catch_datasets: str, template_meta: dict) -> Optional[np.ndarray]:
    """
    Read DEA Waterbodies polygons from Topography folder, rasterize them to the DEM grid,
    and return a boolean mask where True means waterbody.
    """
    waterbodies_path = os.path.join(catch_datasets, "Topography", "DEA_Waterbodies.gpkg")

    if not os.path.exists(waterbodies_path):
        return None

    try:
        waterbodies_gdf = gpd.read_file(waterbodies_path)
        if waterbodies_gdf.empty:
            return None

        if waterbodies_gdf.crs is None:
            return None

        if waterbodies_gdf.crs != template_meta["crs"]:
            waterbodies_gdf = waterbodies_gdf.to_crs(template_meta["crs"])

        shapes_to_burn = [(geom, 1) for geom in waterbodies_gdf.geometry if geom is not None and not geom.is_empty]
        if not shapes_to_burn:
            return None

        mask = rasterize(
            shapes=shapes_to_burn,
            out_shape=(template_meta["height"], template_meta["width"]),
            transform=template_meta["transform"],
            fill=0,
            default_value=1,
            all_touched=True,
            dtype="uint8",
        )

        return mask.astype(bool)

    except Exception as e:
        _log("WARN", f"Could not build waterbodies mask: {e}")
        return None
    
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


def _get_adaptive_figsize_from_bounds(
    bounds: Tuple[float, float, float, float],
    base_size: float = 6.0,
    min_size: float = 4.5,
    max_size: float = 12.0
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
    ys = np.linspace(ymax - yres / 2, ymin + yres / 2, nrows)

    ax.contour(xs, ys, valid_mask, levels=[0.5], colors=color, linewidths=linewidth)


def _set_axis_limits_from_gdf(ax, gdf: gpd.GeoDataFrame, pad_fraction: float = 0.02) -> None:
    """Set axis limits from GeoDataFrame bounds with small padding."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = (maxx - minx) * pad_fraction
    ypad = (maxy - miny) * pad_fraction
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)


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


def _add_matched_colorbar(fig, ax, im, label: str, size: str = "4.5%", pad: float = 0.08):
    """Add a colorbar whose height matches the plotted axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=9)
    return cbar


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
    im = ax.imshow(plot_data, cmap="viridis", extent=extent, origin="upper", interpolation="nearest")
    _add_matched_colorbar(fig, ax, im, title)

    #_plot_valid_raster_outline(ax, plot_data, extent, color="black", linewidth=1.2)

    if catch_plot is not None and not catch_plot.empty:
        catch_plot.boundary.plot(ax=ax, color="black", linewidth=1.2)
        _set_axis_limits_from_gdf(ax, catch_plot, pad_fraction=0.02)
    else:
        _apply_small_plot_padding(ax, extent, pad_fraction=0.02)

    if pts_plot is not None and not pts_plot.empty:
        pts_plot.plot(ax=ax, color="red", markersize=22, marker="o")
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
                    fontsize=8,
                    color="black",
                    weight="bold",
                )

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
    im = ax.imshow(plot_data, cmap="viridis", extent=extent, origin="upper", interpolation="nearest")
    _add_matched_colorbar(fig, ax, im, title)

    #_plot_valid_raster_outline(ax, plot_data, extent, color="black", linewidth=1.2)

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


def _build_reverse_flow_traversal(
    fdir: np.ndarray,
    streams_mask: np.ndarray,
    dem: np.ndarray,
    xres: float,
    yres: float,
    dirmap: Tuple[int, ...],
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
    """
    Build:
      1) distance_to_stream (constant across years)
      2) traversal edges for yearly Ddn calculation
    """
    distance_to_stream = np.zeros_like(dem, dtype="float32")
    visited = np.zeros_like(dem, dtype=bool)

    stream_cells = np.where(streams_mask == 1.0)
    visited[stream_cells] = True

    queue = deque(zip(stream_cells[0], stream_cells[1]))
    traversal_edges: List[Tuple[int, int, int, int, float]] = []

    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    diag = float((xres**2 + yres**2) ** 0.5)
    grid_lengths = np.array([yres, diag, xres, diag, yres, diag, xres, diag], dtype="float32")

    while queue:
        row, col = queue.popleft()
        curr_dist = distance_to_stream[row, col]

        for i in range(8):
            nr = row + dy[i]
            nc = col + dx[i]

            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                if np.isnan(dem[nr, nc]):
                    continue

                if fdir[nr, nc] == dirmap[(i + 4) % 8]:
                    if not visited[nr, nc]:
                        visited[nr, nc] = True
                        step_len = float(grid_lengths[i])
                        distance_to_stream[nr, nc] = curr_dist + step_len
                        traversal_edges.append((nr, nc, row, col, step_len))
                        queue.append((nr, nc))

    distance_to_stream = np.where(np.isnan(dem), np.nan, distance_to_stream)
    return distance_to_stream, traversal_edges


def _get_site_polygons_and_points(all_sites_gpkg: str, dem_crs):
    """
    Read site polygons from combined GPKG and derive site points.

    Priority
    --------
    1. Use X_site / Y_site if available
    2. Otherwise fall back to polygon centroid

    Returns
    -------
    (site_polygons, site_points) or (None, None) if missing/unreadable.
    """
    if not os.path.exists(all_sites_gpkg):
        return None, None

    try:
        sites_gdf = gpd.read_file(all_sites_gpkg)
        if sites_gdf.empty:
            return None, None

        if sites_gdf.crs is None:
            sites_gdf = sites_gdf.set_crs(dem_crs)
        elif sites_gdf.crs != dem_crs:
            sites_gdf = sites_gdf.to_crs(dem_crs)

        pts = sites_gdf.copy()

        # Use true site coordinates if available
        if {"X_site", "Y_site"}.issubset(pts.columns):
            valid_xy = pts["X_site"].notna() & pts["Y_site"].notna()

            if valid_xy.any():
                pts = pts.loc[valid_xy].copy()
                pts["geometry"] = [
                    Point(x, y) for x, y in zip(pts["X_site"], pts["Y_site"])
                ]

                # X_site / Y_site are assumed to already be in the same CRS as sites_gdf
                # If that assumption is wrong, then this part would need reprojection logic.
            else:
                pts["geometry"] = pts.geometry.centroid
        else:
            # fallback only if no true site coordinates exist
            pts["geometry"] = pts.geometry.centroid

        return sites_gdf, pts

    except Exception as e:
        _log("WARN", f"Could not read site polygons/points from all_sites_gpkg: {e}")
        return None, None


# ============================ Core processing ============================
def process_surface_and_groundwater_connectivity(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    *,
    catchment_crs: Optional[str] = None,
    sdr_max: float = 0.8,
    ic0: float = 0.5,
    k: float = 1.0,
    stream_area_threshold_m2: float = 1.3e4,
) -> Tuple[str, str, str, str, str]:
    """
    Compute TWI, LS, SDR (per year from annual C rasters),
    and per-site clipped outputs/plots when all_sites_gpkg exists.
    """
    _log("INFO", "Starting surface and groundwater connectivity processing...")

    # -------- Folder structure & inputs --------
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace("_", " ")
    paths = _prepare_folders(CHM_Work_Space, catchment_name)
    catch_datasets = paths["catch_datasets"]
    con_folder = paths["con_folder"]
    catch_plots = paths["catch_plots"]
    sites_datasets = paths["sites_datasets"]
    sites_plots = paths["sites_plots"]

    _log_folder_summary(
        paths["catchment_folder"],
        catch_datasets,
        catch_plots,
        con_folder,
        sites_datasets,
        sites_plots,
    )

    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation", "Indices", "C Factor", "Annual")

    # -------- Apply user CRS to catchment --------
    try:
        _catch_gdf = gpd.read_file(Catchment_Shapefile_Path)
        if catchment_crs:
            if _catch_gdf.crs is None:
                _catch_gdf = _catch_gdf.set_crs(catchment_crs)
            elif str(_catch_gdf.crs).upper() != str(catchment_crs).upper():
                _catch_gdf = _catch_gdf.to_crs(catchment_crs)
        _log("OK", f"Catchment loaded: {Catchment_Shapefile_Path}")
    except Exception as e:
        _catch_gdf = None
        _log("WARN", f"Catchment boundary could not be loaded for plotting overlays: {e}")

    # -------- DEM & basic derivatives --------
    dem, dem_meta, transform, xres, yres, dem_nodata, dem_crs = _read_dem(catch_datasets)
    waterbodies_mask = _get_waterbodies_mask(catch_datasets, dem_meta)
    if waterbodies_mask is not None:
        _log("OK", "Waterbodies mask loaded from Topography and will be applied to SDR.")
    else:
        _log("WARN", "No waterbodies mask found. SDR will be saved without waterbody masking.")
    pixel_area = xres * yres
    _log("OK", f"Using DEM from {_folder_label(os.path.join(catch_datasets, 'Topography'))}: {os.path.join(catch_datasets, 'Topography', 'DEM.tif')}")

    slope_ratio, slope_rad, slope_deg, aspect_rad = _gradient_slope_aspect(dem, xres, yres)

    # -------- Site polygons/points from all_sites_gpkg only --------
    sites_gdf, sites_pts = _get_site_polygons_and_points(all_sites_gpkg, dem_crs)
    if sites_gdf is None:
        _log("WARN", "No all-sites GPKG found. Catchment-level connectivity outputs will still be produced, but site-based outputs and plots will be skipped.")
    else:
        _log("OK", f"Site polygons loaded: {all_sites_gpkg}")

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

    # -------- Streams mask --------
    streams_mask = (area_m2 > float(stream_area_threshold_m2)).astype("float32")
    streams_mask = np.where(np.isnan(dem), np.nan, streams_mask)
    streams_path = os.path.join(con_folder, "Streams.tif")
    _write_single_band(streams_path, streams_mask, dem_meta)
    _log("OK", f"Streams raster saved in {_folder_label(con_folder)}")

    # -------- Thresholded average slope Sth --------
    Sth = np.where(slope_ratio < 0.005, 0.005, np.where(slope_ratio <= 1, slope_ratio, 1.0))
    Sth = np.where(np.isnan(dem), np.nan, Sth)
    Sth_path = os.path.join(con_folder, "Average Thresholded Slopes.tif")
    _write_single_band(Sth_path, Sth, dem_meta)
    _log("OK", f"Average thresholded slopes saved in {_folder_label(con_folder)}")

    # -------- TWI --------
    with np.errstate(divide="ignore", invalid="ignore"):
        TWI = np.log(acc / np.tan(slope_rad))
    TWI = np.where(np.isfinite(TWI), TWI, np.nan)
    TWI = np.where(np.isnan(dem), np.nan, TWI)
    TWI_path = os.path.join(con_folder, "Topographic Wetness Index.tif")
    _write_single_band(TWI_path, TWI, dem_meta)
    _log("OK", f"Topographic Wetness Index saved in {_folder_label(con_folder)}")

    # -------- LS --------
    specific_area = np.sqrt(acc * pixel_area)
    specific_area = np.where(specific_area > 122.0, 122.0, specific_area)

    xi = np.abs(np.sin(aspect_rad)) + np.abs(np.cos(aspect_rad))

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

    D = xres
    with np.errstate(divide="ignore", invalid="ignore"):
        LSi = Si * (((specific_area + D**2) ** (m + 1)) - (specific_area ** (m + 1))) / (
            (D ** (m + 2)) * (xi ** m) * (22.13 ** m)
        )
    LSi = np.where(np.isnan(dem), np.nan, LSi)
    LS_path = os.path.join(con_folder, "Slope length-gradient factor.tif")
    _write_single_band(LS_path, LSi, dem_meta)
    _log("OK", f"Slope length-gradient factor saved in {_folder_label(con_folder)}")

    # -------- Save Flow Accumulation --------
    acc_path = os.path.join(con_folder, "Flow Accumulation.tif")
    _write_single_band(acc_path, acc, dem_meta)
    _log("OK", f"Flow accumulation saved in {_folder_label(con_folder)}")

    # -------- Catchment context plots for base layers --------
    base_context_layers = [
        (TWI_path, "Topographic Wetness Index"),
        (LS_path, "Slope length-gradient factor"),
        (streams_path, "Streams"),
        (acc_path, "Flow Accumulation"),
    ]

    for rpath, title in base_context_layers:
        if not os.path.exists(rpath):
            continue
        out_context_png = os.path.join(catch_plots, f"{os.path.splitext(os.path.basename(rpath))[0]}.png")
        if not os.path.exists(out_context_png):
            try:
                with rio.open(rpath) as src:
                    full = src.read(1).astype("float32")
                    if src.nodata is not None:
                        full = np.where(full == src.nodata, np.nan, full)

                    _plot_raster_context_in_degrees(
                        data=full,
                        src_meta=src.meta.copy(),
                        out_png=out_context_png,
                        title=title,
                        catchment_gdf=_catch_gdf,
                        site_points=sites_pts,
                    )
            except Exception as e:
                _log("WARN", f"Context plot for {title} failed: {e}")

    # -------- SDR --------
    sdr_output_dir = os.path.join(con_folder, "SDR")
    _ensure_dirs([sdr_output_dir])

    Sth_raster = grid.read_raster(Sth_path).astype("float32")
    acc_Sth = grid.accumulation(fdir=fdir, weights=Sth_raster)
    acc_no0 = np.where(acc == 0, np.nan, acc)
    Av_Sth = acc_Sth / acc_no0

    distance_to_stream, traversal_edges = _build_reverse_flow_traversal(
        fdir=fdir,
        streams_mask=streams_mask,
        dem=dem,
        xres=xres,
        yres=yres,
        dirmap=dirmap,
    )
    distance_to_stream_path = os.path.join(sdr_output_dir, "Distance to Stream.tif")
    _write_single_band(distance_to_stream_path, distance_to_stream, dem_meta)
    _log("OK", f"Distance to Stream saved once in {_folder_label(sdr_output_dir)}")

    c_years = _list_annual_c_rasters(annual_c_dir)
    if not c_years:
        _log("WARN", f"No annual C-Factor rasters found in: {annual_c_dir}")

    for year, c_fp in sorted(c_years.items()):
        with rio.open(c_fp) as src_c:
            c_factor = src_c.read(1).astype("float32")
            if src_c.nodata is not None:
                c_factor = np.where(c_factor == src_c.nodata, np.nan, c_factor)

        Cth = np.where(c_factor < 0.001, 0.001, c_factor)
        Cth_path = os.path.join(sdr_output_dir, f"Average Thresholded C factor_{year}.tif")
        _write_single_band(Cth_path, Cth, dem_meta)

        Cth_raster = grid.read_raster(Cth_path).astype("float32")
        acc_Cth = grid.accumulation(fdir=fdir, weights=Cth_raster)
        Av_Cth = acc_Cth / acc_no0

        Ddn = np.zeros_like(dem, dtype="float32")
        for up_row, up_col, down_row, down_col, step_len in traversal_edges:
            denom = Cth[up_row, up_col] * Sth[up_row, up_col]
            add = (step_len / denom) if (np.isfinite(denom) and denom > 0) else 0.0
            Ddn[up_row, up_col] = Ddn[down_row, down_col] + add
        Ddn = np.where(np.isnan(dem), np.nan, Ddn)

        Dup = Av_Cth * Av_Sth * np.sqrt(area_m2)
        Dup = np.where(np.isnan(dem), np.nan, Dup)

        null_mask = np.isnan(dem)

        Dup = Dup.astype("float32", copy=False)
        Ddn = Ddn.astype("float32", copy=False)

        Dup[null_mask] = np.nan
        Ddn[null_mask] = np.nan

        EPS_DDN = 1.0
        EPS_DUP = 1e-6

        Ddn_safe = np.where(np.isfinite(Ddn) & (Ddn > 0), Ddn, EPS_DDN)
        Dup_safe = np.where(np.isfinite(Dup) & (Dup > 0), Dup, EPS_DUP)

        IC = np.log10(Dup_safe / Ddn_safe)
        IC[null_mask] = np.nan
        IC = np.where(np.isfinite(IC), IC, np.nan)

        SDR = float(sdr_max) / (1.0 + np.exp((float(ic0) - IC) / float(k)))
        SDR = np.where(np.isnan(dem), np.nan, SDR)

        if waterbodies_mask is not None:
            SDR = np.where(waterbodies_mask, np.nan, SDR)

        yearly_products = {
            f"Downslope Path_{year}.tif": Ddn,
            f"Upslope Area_{year}.tif": Dup,
            f"Connectivity Index_{year}.tif": IC,
            f"SDR_{year}.tif": SDR,
        }

        for name, data in yearly_products.items():
            _write_single_band(os.path.join(sdr_output_dir, name), data, dem_meta)

        _log("OK", f"SDR {year} processed.")
        # Clean yearly SDR arrays after each year
        try:
            del c_factor, Cth, Cth_raster, acc_Cth, Av_Cth
        except Exception:
            pass

        try:
            del Ddn, Dup, Ddn_safe, Dup_safe, IC, SDR, yearly_products
        except Exception:
            pass

        gc.collect()

        # -------- Catchment context plots for yearly layers --------
        yearly_context_layers = [
            os.path.join(sdr_output_dir, f"SDR_{year}.tif"),
            os.path.join(sdr_output_dir, f"Connectivity Index_{year}.tif"),
        ]

        for rpath in yearly_context_layers:
            if not os.path.exists(rpath):
                continue

            short = os.path.splitext(os.path.basename(rpath))[0]
            out_context_png = os.path.join(catch_plots, f"{short}.png")

            if not os.path.exists(out_context_png):
                try:
                    with rio.open(rpath) as src:
                        full = src.read(1).astype("float32")
                        if src.nodata is not None:
                            full = np.where(full == src.nodata, np.nan, full)

                        _plot_raster_context_in_degrees(
                            data=full,
                            src_meta=src.meta.copy(),
                            out_png=out_context_png,
                            title=short,
                            catchment_gdf=_catch_gdf,
                            site_points=sites_pts,
                        )
                except Exception as e:
                    _log("WARN", f"Context plot for {short} failed: {e}")

        # -------- Per-site clipping & plots --------
        if sites_gdf is None or sites_gdf.empty:
            continue

        raster_files = [
            TWI_path,
            os.path.join(sdr_output_dir, f"SDR_{year}.tif"),
        ]

        for idx, row in sites_gdf.iterrows():
            try:
                site_id = row.get("Site_id", idx)
                geom = row.geometry
                site_attrs = row.drop(labels="geometry").to_dict()
                site_gdf = gpd.GeoDataFrame([site_attrs], geometry=[geom], crs=sites_gdf.crs)

                site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
                site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
                _ensure_dirs([site_data_dir, site_plot_dir])

                one_site_point = None
                if sites_pts is not None and not sites_pts.empty and "Site_id" in sites_pts.columns:
                    matched = sites_pts[sites_pts["Site_id"] == site_id]
                    if len(matched):
                        one_site_point = matched.iloc[[0]]

                for rpath in raster_files:
                    if not os.path.exists(rpath):
                        _log("WARN", f"Missing raster: {rpath}")
                        continue

                    short = os.path.splitext(os.path.basename(rpath))[0]

                    with rio.open(rpath) as src:
                        v = site_gdf.to_crs(src.crs) if (site_gdf.crs != src.crs) else site_gdf
                        out_img, out_tr = rio_mask(src, [v.geometry.iloc[0]], crop=True, filled=False)
                        clipped = out_img[0].filled(np.nan).astype("float32")

                        meta = src.meta.copy()
                        meta.update({
                            "driver": "GTiff",
                            "height": out_img.shape[1],
                            "width": out_img.shape[2],
                            "transform": out_tr,
                            "crs": src.crs,
                            "nodata": np.nan,
                            "count": 1,
                            "dtype": "float32",
                        })

                        _write_single_band(os.path.join(site_data_dir, f"{short}.tif"), clipped, meta)

                        try:
                            point_for_plot = None
                            if one_site_point is not None:
                                point_for_plot = one_site_point.to_crs(src.crs) if one_site_point.crs != src.crs else one_site_point

                            _plot_clipped_raster_in_degrees(
                                data=clipped,
                                src_meta=meta,
                                out_png=os.path.join(site_plot_dir, f"{short}.png"),
                                title=f"{short} - Site {site_id}",
                                boundary_gdf=v,
                                site_point_gdf=point_for_plot,
                                site_id=site_id,
                            )
                        except Exception as e:
                            _log("WARN", f"Plotting {short} for site {site_id} failed: {e}")

                _log("OK", f"Site {site_id} connectivity data and plots saved.")

            except Exception as e:
                _log("WARN", f"Error processing site {row.get('Site_id', idx)}: {e}")
                continue

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del dem, dem_grid, dem_filled, inflated
    except Exception:
        pass

    try:
        del slope_ratio, slope_rad, slope_deg, aspect_rad
    except Exception:
        pass

    try:
        del fdir, acc, area_m2, streams_mask
    except Exception:
        pass

    try:
        del Sth, Sth_raster, acc_Sth, acc_no0, Av_Sth
    except Exception:
        pass

    try:
        del TWI, specific_area, xi, slope_pct, Si, m, LSi
    except Exception:
        pass

    try:
        del Cth, Cth_raster, acc_Cth, Av_Cth
    except Exception:
        pass

    try:
        del Ddn, Dup, Ddn_safe, Dup_safe, IC, SDR, yearly_products
    except Exception:
        pass

    try:
        del c_factor, waterbodies_mask, distance_to_stream, traversal_edges
    except Exception:
        pass

    try:
        del sites_gdf, sites_pts, site_gdf, one_site_point
    except Exception:
        pass

    try:
        del _catch_gdf, full, clipped, out_img
    except Exception:
        pass

    try:
        del grid
    except Exception:
        pass

    gc.collect()
    _log("INFO", "Memory cleanup completed.")

    return all_sites_gpkg, LS_path, TWI_path, sdr_output_dir, sites_datasets


# ============================ NDVI exposure profiles ============================
def ndvi_cumulative_risk_profiles(CHM_Work_Space: str, Catchment_Shapefile_Path: str) -> None:
    """
    Create NDVI cumulative exposure profiles vs SDR per year for each site,
    with one combined plot per site containing all years.
    """
    _log("INFO", "Starting NDVI cumulative risk profiles by SDR...")

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
        _log("WARN", f"Sites GPKG missing: {all_sites_gpkg}. NDVI-SDR exposure plots will be skipped.")
        return

    try:
        sites_gdf = gpd.read_file(all_sites_gpkg)
    except Exception as e:
        _log("WARN", f"Could not read sites_gdf: {e}")
        return

    ndvi_files = [f for f in os.listdir(annual_ndvi_dir) if f.endswith(".tif") and "NDVI" in f]
    sdr_files = [f for f in os.listdir(sdr_output_dir) if f.endswith(".tif") and re.match(r"SDR_\d{4}\.tif$", f)]
    ndvi_dict = {re.search(r"\d{4}", f).group(): f for f in ndvi_files if re.search(r"\d{4}", f)}
    sdr_dict = {re.search(r"\d{4}", f).group(): f for f in sdr_files if re.search(r"\d{4}", f)}
    common_years = sorted(set(ndvi_dict.keys()) & set(sdr_dict.keys()))

    sdr_min, sdr_max = np.inf, -np.inf
    for f in sdr_files:
        try:
            with rio.open(os.path.join(sdr_output_dir, f)) as src:
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                _mn, _mx = np.nanmin(arr), np.nanmax(arr)
                if np.isfinite(_mn):
                    sdr_min = min(sdr_min, _mn)
                if np.isfinite(_mx):
                    sdr_max = max(sdr_max, _mx)
        except Exception:
            pass

    if not np.isfinite(sdr_min) or not np.isfinite(sdr_max) or sdr_min >= sdr_max:
        sdr_min = sdr_max = None

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
        return float(np.trapz(y_in, x_in))

    for i, row in sites_gdf.iterrows():
        site_id = row.get("Site_id", i)
        geom = row.geometry
        site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
        os.makedirs(site_plot_dir, exist_ok=True)

        auc_series: Dict[int, float] = {}
        line_profiles: List[Tuple[int, np.ndarray, np.ndarray]] = []

        for y in common_years:
            ndvi_fp = os.path.join(annual_ndvi_dir, ndvi_dict[y])
            sdr_fp = os.path.join(sdr_output_dir, sdr_dict[y])

            try:
                with rio.open(ndvi_fp) as ndvi_src:
                    v_geom = gpd.GeoSeries([geom], crs=sites_gdf.crs)
                    if sites_gdf.crs != ndvi_src.crs:
                        v_geom = v_geom.to_crs(ndvi_src.crs)
                    ndvi_img, _ = rio_mask(ndvi_src, [v_geom.iloc[0]], crop=True)
                    ndvi = ndvi_img[0].astype("float32")
                    if ndvi_src.nodata is not None:
                        ndvi = np.where(ndvi == ndvi_src.nodata, np.nan, ndvi)

                with rio.open(sdr_fp) as sdr_src:
                    v_geom2 = gpd.GeoSeries([geom], crs=sites_gdf.crs)
                    if sites_gdf.crs != sdr_src.crs:
                        v_geom2 = v_geom2.to_crs(sdr_src.crs)
                    sdr_img, _ = rio_mask(sdr_src, [v_geom2.iloc[0]], crop=True)
                    sdr = sdr_img[0].astype("float32")
                    if sdr_src.nodata is not None:
                        sdr = np.where(sdr == sdr_src.nodata, np.nan, sdr)
            except Exception as e:
                _log("WARN", f"Site {site_id}, year {y}: masking failed: {e}")
                continue

            if sdr.shape != ndvi.shape:
                _log("WARN", f"Site {site_id}, year {y}: shape mismatch NDVI vs SDR.")
                continue

            valid = np.isfinite(ndvi) & np.isfinite(sdr)
            if not np.any(valid):
                continue

            sdr_vals = sdr[valid]
            order = np.argsort(sdr_vals)
            sdr_sorted = sdr_vals[order]
            cum_pct = np.arange(1, sdr_sorted.size + 1) / sdr_sorted.size * 100.0
            cum_frac = cum_pct / 100.0

            line_profiles.append((int(y), sdr_sorted, cum_pct))

            try:
                auc_val = _auc(sdr_sorted, cum_frac)
                if np.isfinite(auc_val) and sdr_min is not None and sdr_max is not None and sdr_max > sdr_min:
                    auc_series[int(y)] = 1.0 - (auc_val / (sdr_max - sdr_min))
                else:
                    auc_series[int(y)] = np.nan
            except Exception:
                pass

        if line_profiles:
            fig, ax = plt.subplots(figsize=(8, 5))

            cmap = plt.cm.get_cmap("tab20", max(len(line_profiles), 1))

            x_pad = 0.0
            if sdr_min is not None and sdr_max is not None:
                x_pad = (sdr_max - sdr_min) * 0.01

            for j, (year_int, xvals, yvals) in enumerate(sorted(line_profiles, key=lambda t: t[0])):
                ax.plot(
                    xvals,
                    yvals,
                    linewidth=1.4,
                    color=cmap(j),
                    label=str(year_int)
                )

            ax.set_xlabel("SDR Value", fontsize=10, fontweight="bold")
            ax.set_ylabel("Cumulative NDVI Pixels (%)", fontsize=10, fontweight="bold")
            ax.set_title(f"Vegetation Exposure Profile (NDVI vs SDR) - Site {site_id}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)

            if sdr_min is not None and sdr_max is not None:
                ax.set_xlim(sdr_min - x_pad, sdr_max + x_pad)

            ax.set_ylim(0, 100)

            ax.legend(
                title="Year",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize=9,
                title_fontsize=9,
                ncol=1
            )

            fig.tight_layout(rect=[0, 0, 0.84, 1])
            plt.savefig(
                os.path.join(site_plot_dir, f"Site_{site_id}_Exposure_NDVI_SDR.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

        if auc_series:
            ys = sorted(auc_series.keys())
            auc_vals = [auc_series[y] for y in ys]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ys, auc_vals, marker="o", color="black", linewidth=1)
            ax.set_xlabel("Year", fontsize=10, fontweight="bold")
            ax.set_ylabel("Exposure (1 − AUC)", fontsize=10, fontweight="bold")
            ax.set_title(f"Vegetation Exposure Profile (NDVI vs SDR) - Site {site_id}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(site_plot_dir, f"Site_{site_id}_AUC_NDVI_SDR.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    _log("OK", "NDVI cumulative exposure profiles by SDR are completed.")
    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del sites_gdf
    except Exception:
        pass

    try:
        del ndvi, sdr, ndvi_img, sdr_img
    except Exception:
        pass

    try:
        del sdr_vals, sdr_sorted, cum_pct, cum_frac
    except Exception:
        pass

    try:
        del line_profiles, auc_series, auc_vals
    except Exception:
        pass

    try:
        del arr
    except Exception:
        pass

    gc.collect()
    _log("INFO", "Memory cleanup completed.")

# ============================ Convenience wrapper ============================

def surface_ground_water_connectivity(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    *,
    catchment_crs: Optional[str] = None,
    sdr_max: float = 0.8,
    ic0: float = 0.5,
    k: float = 1.0,
    stream_area_threshold_m2: float = 1.3e4,
) -> None:
    """
    High-level convenience wrapper:
    1) Compute connectivity metrics (TWI, LS, SDR per year) + site products.
    2) Build NDVI–SDR exposure profiles and AUC time series.
    """
    process_surface_and_groundwater_connectivity(
        CHM_Work_Space=CHM_Work_Space,
        Catchment_Shapefile_Path=Catchment_Shapefile_Path,
        catchment_crs=catchment_crs,
        sdr_max=sdr_max,
        ic0=ic0,
        k=k,
        stream_area_threshold_m2=stream_area_threshold_m2,
    )
    ndvi_cumulative_risk_profiles(
        CHM_Work_Space=CHM_Work_Space,
        Catchment_Shapefile_Path=Catchment_Shapefile_Path,
    )