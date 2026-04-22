# dem_and_terrain.py
# ===================== IMPORTS =====================
from __future__ import annotations

# --- Standard library ---
import os
import gc
import copy
import tempfile
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Union, Any

# --- Arrays & Data ---
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter

# --- Geo ---
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import plotting_extent
from rasterio.merge import merge as rio_merge
from shapely.geometry import Point, LineString, shape as shp_shape

# --- Hydro / RS ---
from pysheds.grid import Grid
from rasterstats import zonal_stats

# --- Viz ---
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Misc ---
import requests

# Optional basemap (kept graceful)
try:
    import contextily as ctx
except Exception:
    ctx = None  # basemap will be skipped gracefully


# ===================== CONFIG & HELPERS =====================

@dataclass
class DemTerrainConfig:
    """Configuration for DEM & terrain processing.

    Attributes
    ----------
    chm_workspace : str
        Root output folder for the catchment package.
    catchment_path : str
        Path to the catchment boundary (vector – any format Geopandas can read).
    sites_path : Optional[str]
        Optional path to sites vector. Points => delineate catchments; polygons => use as subcatchments.
        If not provided, the module still runs for catchment-only processing.
    dem_url : Optional[str]
        WCS URL to download DEM if local DEM is not supplied.
    dem_local_path : Optional[str]
        Existing DEM in any supported raster format. If set and exists, it is used instead of WCS.
    target_res_m : float
        Target DEM pixel size (m) when reprojecting to catchment CRS.
    stream_target_area_ha : float
        Target upstream contributing area (ha) to define the stream threshold.
    """
    chm_workspace: str
    catchment_path: str
    sites_path: Optional[str] = None
    dem_url: Optional[str] = None
    dem_local_path: Optional[str] = None
    target_res_m: float = 30.0
    stream_target_area_ha: float = 60.0


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


def _log_folder_summary(
    catchment_folder: str,
    catch_datasets: str,
    catch_plots: str,
    topo_folder: str,
    sites_datasets: str,
    sites_plots: str,
) -> None:
    """Print full output folder structure once at the start."""
    _log("OK", "Output folders are ready.")
    _log("INFO", "Created/using output folders:")
    _log("INFO", f"  Catchment folder       : {catchment_folder}")
    _log("INFO", f"  Catchment datasets     : {catch_datasets}")
    _log("INFO", f"  Catchment plots/maps   : {catch_plots}")
    _log("INFO", f"  Topography             : {topo_folder}")
    _log("INFO", f"  Sites datasets         : {sites_datasets}")
    _log("INFO", f"  Sites plots/maps       : {sites_plots}")


def _crs_from_user_input(crs_like) -> Optional[rio.crs.CRS]:
    """
    Accepts int (e.g., 3308), str (e.g., 'EPSG:3308'), dict/WKT/PROJJSON.
    Returns a rasterio CRS or None.
    """
    if crs_like is None:
        return None
    try:
        return rio.crs.CRS.from_user_input(crs_like)
    except Exception as e:
        raise ValueError(f"Invalid CRS provided for 'catchment_crs': {crs_like!r} ({e})")


def _read_catchment(
    catchment_path: str,
    catchment_crs: Optional[Union[int, str, dict]] = None
) -> gpd.GeoDataFrame:
    """
    Read & dissolve catchment; ensure CRS according to catchment_crs; add date column.

    Notes
    -----
    - If the source has no CRS and 'catchment_crs' is provided, the CRS is assigned (no reprojection).
    - If the source has a CRS and 'catchment_crs' is provided, the layer is reprojected to it.
    """
    if not os.path.exists(catchment_path):
        raise FileNotFoundError(f"Catchment file not found: {catchment_path}")

    gdf = gpd.read_file(catchment_path)
    if gdf.empty:
        raise ValueError("Catchment file has no features.")

    desired_crs = _crs_from_user_input(catchment_crs)

    if gdf.crs is None:
        if desired_crs is None:
            raise ValueError(
                "Catchment file has no CRS. Please provide 'catchment_crs' "
                "(e.g., 3308 or 'EPSG:3308')."
            )
        gdf = gdf.set_crs(desired_crs, allow_override=True)
    else:
        if desired_crs is not None and rio.crs.CRS.from_user_input(gdf.crs) != desired_crs:
            gdf = gdf.to_crs(desired_crs)

    # Dissolve to a single catchment polygon
    gdf = gdf.dissolve()

    # Safety: ensure projected CRS (metres) for area-related calculations
    crs_obj = rio.crs.CRS.from_user_input(gdf.crs)
    if crs_obj.is_geographic:
        raise ValueError(
            f"Catchment CRS is geographic ({gdf.crs}). Please use a projected CRS in metres "
            f"via 'catchment_crs'."
        )

    gdf["Date"] = pd.Series(pd.NA, index=gdf.index, dtype="Int64")
    return gdf


def _bbox_wgs84(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return minx, miny, maxx, maxy in EPSG:4326."""
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    return tuple(gdf_wgs84.total_bounds)


def _download_dem_wcs(
    dem_url: str,
    bbox: Tuple[float, float, float, float],
    out_tif: str,
    *,
    tile_when_px_over: int = 4000,
    tile_cols_rows: Tuple[int, int] = (2, 2),
) -> None:
    """
    Download DEM from a WCS endpoint to GeoTIFF clipped to bbox (EPSG:4326).

    Behavior:
      - Small bbox -> single request
      - Large bbox -> tiled requests then mosaic
    """
    minx, miny, maxx, maxy = bbox

    # 1 arc-second in degrees
    resolution_deg = 1 / 3600.0
    full_w = max(1, int(round((maxx - minx) / resolution_deg)))
    full_h = max(1, int(round((maxy - miny) / resolution_deg)))

    def _single_request(_bbox, _out_path):
        _minx, _miny, _maxx, _maxy = _bbox
        _width = max(1, int(round((_maxx - _minx) / resolution_deg)))
        _height = max(1, int(round((_maxy - _miny) / resolution_deg)))

        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "1",
            "crs": "EPSG:4326",
            "bbox": f"{_minx},{_miny},{_maxx},{_maxy}",
            "width": _width,
            "height": _height,
            "format": "GeoTIFF",
        }

        r = requests.get(dem_url, params=params, stream=True, timeout=180)
        ctype = (r.headers.get("Content-Type") or "").lower()

        if r.status_code != 200 or ("tiff" not in ctype and "geotiff" not in ctype):
            snippet = ""
            try:
                snippet = "\n".join((r.text or "").splitlines()[:8])
            except Exception:
                pass

            raise RuntimeError(
                f"DEM WCS download failed (status={r.status_code}, ctype={ctype}).\n"
                f"URL: {r.url}\n"
                f"Server said:\n{snippet}"
            )

        os.makedirs(os.path.dirname(_out_path), exist_ok=True)
        with open(_out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    # Single request
    if full_w <= tile_when_px_over and full_h <= tile_when_px_over:
        _single_request(bbox, out_tif)
        return

    # Tiled requests
    ncols, nrows = tile_cols_rows
    eps = 1e-9  # small overlap to avoid seam artifacts

    xs = np.linspace(minx, maxx, ncols + 1)
    ys = np.linspace(miny, maxy, nrows + 1)

    tmp_files: list[str] = []
    srcs: list[rio.DatasetReader] = []

    try:
        for r in range(nrows):
            for c in range(ncols):
                t_minx = xs[c]
                t_maxx = xs[c + 1]
                t_miny = ys[r]
                t_maxy = ys[r + 1]

                if c > 0:
                    t_minx -= eps
                if c < ncols - 1:
                    t_maxx += eps
                if r > 0:
                    t_miny -= eps
                if r < nrows - 1:
                    t_maxy += eps

                tmp_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                _single_request((t_minx, t_miny, t_maxx, t_maxy), tmp_path)
                tmp_files.append(tmp_path)

        for p in tmp_files:
            srcs.append(rio.open(p))

        mosaic, out_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "count": mosaic.shape[0],
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        })

        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        with rio.open(out_tif, "w", **meta) as dst:
            dst.write(mosaic)

    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


def _clip_to_catchment_wgs84(src_path: str, catchment_wgs84: gpd.GeoDataFrame, out_path: str) -> None:
    """Clip raster to catchment geometry in the raster CRS; save WGS84-clipped DEM."""
    with rxr.open_rasterio(src_path, masked=True) as dem:
        clip_geom = catchment_wgs84.geometry

        if dem.rio.crs is None:
            raise ValueError("DEM has no CRS.")

        if dem.rio.crs != catchment_wgs84.crs:
            clip_geom = catchment_wgs84.to_crs(dem.rio.crs).geometry

        clipped = dem.rio.clip(clip_geom, dem.rio.crs, drop=True, all_touched=True)
        clipped = clipped.where(clipped != clipped.rio.nodata, np.nan)
        clipped.rio.to_raster(out_path)


def _reproject_dem_to_catchment(
    dem_wgs84_path: str,
    target_crs,
    target_res_m: float,
    out_path: str
) -> None:
    """Reproject DEM to catchment CRS and target resolution (m)."""
    with rio.open(dem_wgs84_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_res_m, target_res_m)
        )

        meta = src.meta.copy()
        meta.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": np.nan,
        })

        with rio.open(out_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,  # keep bilinear for DEM
                )


def _download_and_save_waterbodies(
    catchment_gdf: gpd.GeoDataFrame,
    out_gpkg: str,
    layer_name: str = "DEA_Waterbodies",
) -> None:
    """
    Download DEA Waterbodies intersecting the catchment, reproject to the
    catchment CRS, clip to the catchment, and save to GeoPackage.
    """
    if catchment_gdf is None or catchment_gdf.empty:
        _log("WARN", "Catchment is empty. DEA Waterbodies download skipped.")
        return

    wfs_url = "https://geoserver.dea.ga.gov.au/geoserver/ows"
    bbox_gdf = catchment_gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    try:
        cap = requests.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetCapabilities",
            },
            timeout=120,
        )
        cap.raise_for_status()
        cap_text = cap.text

        layer_name_found = None
        layer_matches = re.findall(r"<Name>([^<]*waterbod[^<]*)</Name>", cap_text, flags=re.IGNORECASE)
        if layer_matches:
            layer_name_found = layer_matches[0]

        if layer_name_found is None:
            _log("WARN", "DEA Waterbodies layer was not found in WFS capabilities. Skipping.")
            return

        r = requests.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": layer_name_found,
                "outputFormat": "application/json",
                "srsName": "EPSG:4326",
                "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
            },
            timeout=300,
        )
        r.raise_for_status()

        if not r.text.strip():
            _log("WARN", "DEA Waterbodies request returned no content. Skipping.")
            return

        tmp_geojson = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False).name
        with open(tmp_geojson, "w", encoding="utf-8") as f:
            f.write(r.text)

        try:
            waterbodies = gpd.read_file(tmp_geojson)
        finally:
            try:
                os.remove(tmp_geojson)
            except Exception:
                pass

        if waterbodies.empty:
            _log("WARN", "No DEA Waterbodies intersected the catchment bbox.")
            return

        if waterbodies.crs is None:
            waterbodies = waterbodies.set_crs(epsg=4326, allow_override=True)

        if waterbodies.crs != catchment_gdf.crs:
            waterbodies = waterbodies.to_crs(catchment_gdf.crs)

        waterbodies = gpd.clip(waterbodies, catchment_gdf)

        if waterbodies.empty:
            _log("WARN", "No DEA Waterbodies intersected the catchment.")
            return

        for p in (out_gpkg, out_gpkg + "-wal", out_gpkg + "-shm"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        waterbodies.to_file(out_gpkg, layer=layer_name, driver="GPKG")
        _log("OK", f"DEA Waterbodies saved: {out_gpkg}")

    except Exception as e:
        _log("WARN", f"DEA Waterbodies download failed and will be skipped: {e}")


def _compute_slope_aspect(dem_data: np.ndarray, xres: float, yres: float) -> Dict[str, np.ndarray]:
    """Compute slope (deg, rad, %) and aspect (deg, rad)."""
    dem = dem_data.astype("float32", copy=True)

    # DEM/slope-like continuous calculations
    dz_dx, dz_dy = np.gradient(dem, xres, yres)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_rad = np.arctan(slope_ratio)
    slope_deg = np.degrees(slope_rad)
    slope_pct = slope_ratio * 100.0

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_rad = np.where(aspect_rad < 0, 2 * np.pi + aspect_rad, aspect_rad)
    aspect_deg = np.degrees(aspect_rad)

    return {
        "Slope in degree.tif": slope_deg,
        "Slope in radian.tif": slope_rad,
        "Slope in percent.tif": slope_pct,
        "Aspect in radian.tif": aspect_rad,
        "Aspect in degree.tif": aspect_deg,
    }


def _compute_tpi_tri(dem_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute TPI/TRI using a 3x3 neighborhood.
    - TPI: center - mean(neighbors)
    - TRI: mean(|neighbors - center|)
    NaN-safe for both neighbors and center.
    """
    def tpi_func(window):
        center = window[len(window) // 2]
        if not np.isfinite(center):
            return np.nan
        neighbors = np.delete(window, len(window) // 2)
        finite = np.isfinite(neighbors)
        if not finite.any():
            return np.nan
        return center - np.nanmean(neighbors[finite])

    def tri_func(window):
        center = window[len(window) // 2]
        if not np.isfinite(center):
            return np.nan
        neighbors = np.delete(window, len(window) // 2)
        finite = np.isfinite(neighbors)
        if not finite.any():
            return np.nan
        diffs = neighbors[finite] - center
        finite2 = np.isfinite(diffs)
        if not finite2.any():
            return np.nan
        return np.nanmean(np.abs(diffs[finite2]))

    footprint = np.ones((3, 3), dtype=bool)
    tpi = generic_filter(dem_data, tpi_func, footprint=footprint, mode="constant", cval=np.nan)
    tri = generic_filter(dem_data, tri_func, footprint=footprint, mode="constant", cval=np.nan)

    return {
        "Topographic Position Index.tif": tpi.astype("float32"),
        "Terrain Ruggedness Index.tif": tri.astype("float32"),
    }


def _write_raster(out_path: str, data: np.ndarray, template_meta: dict) -> None:
    """
    Write a single-band float32 GeoTIFF with safe nodata handling.
    """
    meta = template_meta.copy()
    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan
    })

    data = data.astype("float32", copy=False)
    data = np.where(np.isfinite(data), data, np.nan)

    with rio.open(out_path, "w", **meta) as dst:
        dst.write(data, 1)


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
    resampling: Resampling = Resampling.bilinear
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Reproject a raster array to another CRS for plotting only.

    Parameters
    ----------
    data : np.ndarray
        2D raster array.
    src_meta : dict
        Raster metadata containing crs, transform, width, height, dtype, nodata.
    dst_crs : str
        Target CRS for plotting. Default is EPSG:4326.
    resampling : Resampling
        Resampling method. Kept bilinear for DEM and slope-like continuous rasters.

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

    left, bottom, right, top = rio.transform.array_bounds(src_height, src_width, src_transform)

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
        dst_nodata=np.nan
    )

    xmin, ymin, xmax, ymax = rio.transform.array_bounds(dst_height, dst_width, dst_transform)
    return dst_array, (xmin, xmax, ymin, ymax)


def _get_adaptive_figsize_from_bounds(
    bounds: Tuple[float, float, float, float],
    base_size: float = 6.0,
    min_size: float = 4.5,
    max_size: float = 12.0
) -> Tuple[float, float]:
    """
    Compute an adaptive figure size from geometry bounds.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy)
    base_size : float
        Base size used for the shorter dimension logic.
    min_size : float
        Minimum allowed width/height.
    max_size : float
        Maximum allowed width/height.

    Returns
    -------
    (fig_w, fig_h) : tuple[float, float]
        Adaptive matplotlib figure size.
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


def _style_map_axes(ax, xlabel: str = "Longitude (°)", ylabel: str = "Latitude (°)", nbins: int = 4) -> None:
    """Apply consistent map styling across all plots."""
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.6)


def _apply_small_plot_padding(ax, extent: Tuple[float, float, float, float], pad_fraction: float = 0.03) -> None:
    """Apply a small padding around the plotted extent to reduce empty frame space."""
    xmin, xmax, ymin, ymax = extent
    xpad = (xmax - xmin) * pad_fraction
    ypad = (ymax - ymin) * pad_fraction
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def _plot_raster_with_overlays(
    data: np.ndarray,
    src_meta: dict,
    catchment_gdf: gpd.GeoDataFrame,
    out_png: str,
    title: str,
    site_points: Optional[gpd.GeoDataFrame] = None,
    annotate_ids: bool = False,
    title_kwargs: Optional[Dict[str, Any]] = None,
    colorbar_label: Optional[str] = None,
    plot_in_geographic_coords: bool = True,
) -> None:
    """
    Plot raster with catchment and optional site overlays.

    Notes
    -----
    - Processing remains in projected CRS.
    - Plotting can be done in geographic coordinates for readability.
    """
    # Reproject to geographic coordinates for plotting only
    if plot_in_geographic_coords:
        plot_data, extent = _reproject_array_for_plotting(
            data, src_meta, dst_crs="EPSG:4326", resampling=Resampling.nearest
        )
        catchment_plot = catchment_gdf.to_crs(epsg=4326) if catchment_gdf is not None else None
        site_points_plot = site_points.to_crs(epsg=4326) if site_points is not None else None
        xlabel = "Longitude (°)"
        ylabel = "Latitude (°)"
    else:
        plot_data = data
        with rio.MemoryFile() as memfile:
            with memfile.open(**src_meta) as ds:
                extent = plotting_extent(ds)
        catchment_plot = catchment_gdf
        site_points_plot = site_points
        xlabel = "Longitude"
        ylabel = "Latitude"

    # Adaptive figure size based on catchment shape
    if catchment_plot is not None and not catchment_plot.empty:
        plot_bounds = tuple(catchment_plot.total_bounds)
    else:
        plot_bounds = (extent[0], extent[2], extent[1], extent[3])

    fig_w, fig_h = _get_adaptive_figsize_from_bounds(
        plot_bounds,
        base_size=6.0,
        min_size=5.0,
        max_size=11.0
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    try:
        im = ax.imshow(
            plot_data,
            cmap="viridis",
            extent=extent,
            origin="upper",
            interpolation="nearest"
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(colorbar_label or "Value", fontsize=9)

        ax.set_title(title, **(title_kwargs or {"fontsize": 11, "fontweight": "bold"}))

        _plot_valid_raster_outline(ax, plot_data, extent, color="black", linewidth=1.2)

        if site_points_plot is not None and not site_points_plot.empty:
            site_points_plot.plot(ax=ax, color="red", markersize=22, marker="o")
            if annotate_ids:
                for _i, _r in site_points_plot.iterrows():
                    sid = _r.get("Site_id", _i)
                    geom = _r.geometry
                    x, y = (geom.x, geom.y) if geom.geom_type == "Point" else (geom.centroid.x, geom.centroid.y)
                    ax.annotate(
                        str(sid),
                        (x, y),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        weight="bold"
                    )

        _apply_small_plot_padding(ax, extent, pad_fraction=0.02)
        _style_map_axes(ax, xlabel=xlabel, ylabel=ylabel, nbins=4)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _detect_sites_role(sites_path: str, target_crs) -> Tuple[gpd.GeoDataFrame, bool]:
    """Load sites and decide if they are points (delineate) or polygons (use directly)."""
    if not os.path.exists(sites_path):
        raise FileNotFoundError(f"Sites file not found: {sites_path}")

    gdf = gpd.read_file(sites_path)
    if gdf.empty:
        raise ValueError("Sites file has no features.")

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    geom0 = next((g for g in gdf.geometry if g is not None and not g.is_empty), None)
    if geom0 is None:
        raise ValueError("Sites file has no valid geometries.")

    if geom0.geom_type in ("Point", "MultiPoint"):
        return gdf, False
    if geom0.geom_type in ("Polygon", "MultiPolygon"):
        return gdf, True

    raise ValueError(f"Unsupported geometry type: {geom0.geom_type}")


def _extract_stream_network(
    dem_path: str,
    stream_target_area_ha: float,
    out_gpkg: str,
    layer_name: str = "Streams",
) -> gpd.GeoDataFrame:
    """Derive flow dir/accumulation, threshold by target area, compute Strahler order, vectorize and save."""
    with rio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        pixel_area_m2 = abs(dem_transform.a) * abs(dem_transform.e)

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    target_m2 = max(1.0, stream_target_area_ha * 10_000.0)
    stream_thresh_cells = max(1, int(np.ceil(target_m2 / pixel_area_m2)))
    stream_mask = acc > stream_thresh_cells

    try:
        order_r = grid.stream_order(fdir, mask=stream_mask, dirmap=dirmap, method="strahler")
    except TypeError:
        order_r = grid.stream_order(fdir, mask=stream_mask, dirmap=dirmap)

    order_arr = np.where(np.isfinite(order_r), order_r, 0).astype(np.uint8)
    order_tif = os.path.splitext(out_gpkg)[0] + "_Strahler_Order.tif"

    meta = {
        "driver": "GTiff",
        "height": order_arr.shape[0],
        "width": order_arr.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": dem_crs,
        "transform": dem_transform,
        "nodata": 0,
        "compress": "LZW",
    }

    with rio.open(order_tif, "w", **meta) as dst:
        dst.write(order_arr, 1)

    # Vectorize streams
    branches = grid.extract_river_network(fdir, stream_mask, dirmap=dirmap)
    lines, lengths = [], []
    for feat in branches.get("features", []):
        coords = feat["geometry"]["coordinates"]
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append(line)
            lengths.append(float(line.length))

    streams = gpd.GeoDataFrame({"length": lengths}, geometry=lines, crs=dem_crs)

    # Assign order via thin buffer majority over order raster
    px_w = abs(dem_transform.a)
    px_h = abs(dem_transform.e)
    buf_w = 0.75 * max(px_w, px_h)
    streams_eval = streams.copy()
    streams_eval["__buf"] = streams_eval.geometry.buffer(buf_w)

    zs = zonal_stats(
        streams_eval["__buf"],
        order_tif,
        nodata=0,
        categorical=True,
        all_touched=True,
    )

    def _majority(d: dict) -> int:
        clean = {int(k): v for k, v in d.items() if int(k) > 0}
        if not clean:
            return 1
        return max(clean.items(), key=lambda kv: kv[1])[0]

    streams["order"] = [_majority(d) for d in zs]

    for p in (out_gpkg, out_gpkg + "-wal", out_gpkg + "-shm"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    streams.to_file(out_gpkg, layer=layer_name, driver="GPKG")
    return streams


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

    Notes
    -----
    contour() expects coordinates at cell centres, while extent gives
    outer raster bounds. So we shift to pixel centres before contouring.
    """
    valid_mask = np.isfinite(data).astype(np.uint8)

    if valid_mask.max() == 0:
        return

    xmin, xmax, ymin, ymax = extent
    nrows, ncols = valid_mask.shape

    xres = (xmax - xmin) / ncols
    yres = (ymax - ymin) / nrows

    # Cell centres, not outer edges
    xs = np.linspace(xmin + xres / 2, xmax - xres / 2, ncols)
    ys = np.linspace(ymax - yres / 2, ymin + yres / 2, nrows)  # origin='upper'

    ax.contour(xs, ys, valid_mask, levels=[0.5], colors=color, linewidths=linewidth)


def _vectorized_catchment_for_point(
    grid: Grid,
    fdir: np.ndarray,
    acc: np.ndarray,
    dirmap,
    pt_x: float,
    pt_y: float,
    dem_transform
):
    """Delineate and vectorize a point catchment."""
    grid_local = copy.deepcopy(grid)

    # Snap to a higher accumulation cell before delineation
    x_snap, y_snap = grid_local.snap_to_mask(acc > 25, (pt_x, pt_y))
    mask = grid_local.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate")
    grid_local.clip_to(mask)
    clipped = np.array(grid_local.view(mask), dtype=np.int16)

    min_x, min_y, max_x, max_y = grid_local.extent
    px_x = dem_transform.a
    px_y = -dem_transform.e
    tr = from_origin(min_x, max_y, px_x, px_y)

    geoms = [shp_shape(g) for g, val in shapes(clipped, transform=tr) if val == 1]
    if not geoms:
        return None

    gs = gpd.GeoSeries(geoms)
    if hasattr(gs, "union_all"):
        return gs.union_all()
    return gs.unary_union


# ===================== MAIN ENTRYPOINT =====================

def dem_and_terrain(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    Sites_Shapefile_Path: Optional[str] = None,
    DEM_Input_Path: Optional[str] = None,
    DEM_url: Optional[str] = None,
    target_res_m: float = 30.0,
    stream_target_area_ha: float = 60.0,
    catchment_crs: Optional[Union[int, str, dict]] = None,
    wcs_tile_cols_rows: Tuple[int, int] = (2, 2),
    plot_in_geographic_coords: bool = True,
    site_id_field: str = "Site_id",
    stream_basemap: bool = False,
) -> Tuple[str, str, str, str]:
    """
    End-to-end pipeline: acquire DEM (WCS or local), clip to catchment, reproject,
    compute terrain rasters, derive stream network, summarize/plot per-site,
    and download DEA Waterbodies.
    """
    _log("INFO", "Starting DEM and topography processing...")

    # --- Folder structure ---
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace("_", " ")
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    topo_folder = os.path.join(catch_datasets, "Topography")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    _ensure_dirs([catchment_folder, catch_datasets, catch_plots, topo_folder, sites_datasets, sites_plots])
    _log("OK", "Output folders are ready.")

    # --- Catchment layer ---
    gdf_catch = _read_catchment(Catchment_Shapefile_Path, catchment_crs=catchment_crs)
    catch_crs = gdf_catch.crs
    if catch_crs is None:
        raise ValueError("Catchment file has no CRS. Please assign a projected CRS (metres).")

    try:
        gdf_catch["Area_ha"] = round(gdf_catch.geometry.area.values[0] / 10_000.0, 1)
    except Exception:
        gdf_catch["Area_ha"] = np.nan

    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    gdf_catch.to_file(catch_gpkg, layer=f"{catchment_name} Data", driver="GPKG")
    _log("OK", f"Catchment loaded and saved: {catch_gpkg}")

    # --- DEM acquisition ---
    dem_temp = DEM_Input_Path if (DEM_Input_Path and os.path.exists(DEM_Input_Path)) else None
    downloaded = False

    if dem_temp is None:
        if not DEM_url:
            raise ValueError("No local DEM supplied and no DEM_url provided for WCS download.")

        _log("INFO", "Requesting DEM from WCS...")
        dem_temp = os.path.join(topo_folder, "DEM_temp.tif")
        _download_dem_wcs(
            DEM_url,
            _bbox_wgs84(gdf_catch),
            dem_temp,
            tile_cols_rows=wcs_tile_cols_rows
        )
        downloaded = True
        _log("OK", f"DEM downloaded: {dem_temp}")
    else:
        _log("OK", f"Using provided DEM: {dem_temp}")

    # --- Clip and reproject DEM ---
    dem_clipped_wgs84 = os.path.join(topo_folder, "DEM_WGS84.tif")
    gdf_wgs84 = gdf_catch.to_crs(epsg=4326)
    _clip_to_catchment_wgs84(dem_temp, gdf_wgs84, dem_clipped_wgs84)
    _log("OK", f"DEM clipped to catchment: {dem_clipped_wgs84}")

    dem_projected_file = os.path.join(topo_folder, "DEM.tif")
    _reproject_dem_to_catchment(dem_clipped_wgs84, catch_crs, target_res_m, dem_projected_file)
    _log("OK", f"DEM reprojected to catchment CRS and target resolution: {dem_projected_file}")

    # --- Load DEM for downstream processing ---
    with rio.open(dem_projected_file) as src:
        dem_data = src.read(1).astype("float32")
        dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
        dem_meta = src.meta.copy()
        dem_extent = plotting_extent(src)
        dem_crs = src.crs
        dem_transform = src.transform

    # --- Optional sites loading ---
    sites_gdf = None
    use_subcatch = False
    site_points_for_plots = None

    if Sites_Shapefile_Path and str(Sites_Shapefile_Path).strip():
        try:
            sites_gdf, use_subcatch = _detect_sites_role(Sites_Shapefile_Path, dem_crs)
            site_points_for_plots = sites_gdf if not use_subcatch else None
            _log(
                "OK",
                f"Sites loaded successfully as {'subcatchments (polygons)' if use_subcatch else 'points'}."
            )
        except Exception as e:
            _log("WARN", f"Sites file could not be processed and will be skipped: {e}")
            sites_gdf = None
            use_subcatch = False
            site_points_for_plots = None
    else:
        _log("WARN", "No Sites_Shapefile_Path provided. Site-based processing will be skipped.")

    # --- Terrain layers ---
    terrain_layers: Dict[str, np.ndarray] = {"DEM.tif": dem_data}

    xres = dem_transform.a
    yres = abs(dem_transform.e)
    terrain_layers.update(_compute_slope_aspect(dem_data, xres, yres))
    terrain_layers.update(_compute_tpi_tri(dem_data))

    # --- Save terrain rasters and catchment plots ---
    for name, arr in terrain_layers.items():
        out_tif = os.path.join(topo_folder, name)
        try:
            _write_raster(out_tif, arr, dem_meta)
            _log("OK", f"Raster written: {out_tif}")
        except Exception as e:
            _log("ERROR", f"Failed to write raster {out_tif}: {e}")
            continue

        plot_png = os.path.join(catch_plots, os.path.splitext(name)[0] + ".png")
        try:
            _plot_raster_with_overlays(
                data=arr,
                src_meta=dem_meta,
                catchment_gdf=gdf_catch,
                out_png=plot_png,
                title=f"{catchment_name} — {os.path.splitext(name)[0]}",
                site_points=site_points_for_plots,
                annotate_ids=True,
                title_kwargs={"fontsize": 11, "fontweight": "bold"},
                colorbar_label=os.path.splitext(name)[0],
                plot_in_geographic_coords=plot_in_geographic_coords,
            )
            _log("OK", f"Plot saved: {plot_png}")
        except Exception as e:
            _log("ERROR", f"Failed to save plot {plot_png}: {e}")

    # --- Stream network ---
    streams_gpkg = os.path.join(topo_folder, "Stream_Network.gpkg")
    streams_gdf = _extract_stream_network(
        dem_projected_file,
        stream_target_area_ha,
        out_gpkg=streams_gpkg,
        layer_name="Streams"
    )
    _log("OK", f"Stream network written: {streams_gpkg} (layer='Streams')")

    # --- DEA Waterbodies ---
    waterbodies_gpkg = os.path.join(topo_folder, "DEA_Waterbodies.gpkg")
    _download_and_save_waterbodies(
        catchment_gdf=gdf_catch,
        out_gpkg=waterbodies_gpkg,
        layer_name="DEA_Waterbodies"
    )

    # Stream network map
    try:
        catch_plot_gdf = gdf_catch.copy()
        if catch_plot_gdf.crs != streams_gdf.crs:
            catch_plot_gdf = catch_plot_gdf.to_crs(streams_gdf.crs)

        streams_plot = streams_gdf
        catch_plot = catch_plot_gdf
        sites_plot = site_points_for_plots

        if plot_in_geographic_coords:
            streams_plot = streams_gdf.to_crs(epsg=4326)
            catch_plot = catch_plot_gdf.to_crs(epsg=4326)
            if sites_plot is not None:
                sites_plot = sites_plot.to_crs(epsg=4326)
            xlabel = "Longitude (°)"
            ylabel = "Latitude (°)"
        else:
            xlabel = "Longitude"
            ylabel = "Latitude"

        fig_w, fig_h = _get_adaptive_figsize_from_bounds(
            tuple(catch_plot.total_bounds),
            base_size=6.5,
            min_size=5.0,
            max_size=11.0
        )

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(f"{catchment_name} - Stream Network", fontsize=11, fontweight="bold")

        xmin, ymin, xmax, ymax = catch_plot.total_bounds
        stream_map_saved = False

        # Optional basemap while keeping axes in degrees when plotting in geographic coords
        if ctx is not None and stream_basemap and plot_in_geographic_coords:
            try:
                plt.close(fig)  # close the first empty figure before opening a new one
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                ax.set_title(f"{catchment_name} - Stream Network", fontsize=11, fontweight="bold")

                xmin, ymin, xmax, ymax = catch_plot.total_bounds
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                # Keep plot in EPSG:4326, and let contextily warp the basemap to this CRS
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs="EPSG:4326",
                    zoom="auto",
                    alpha=1.0
                )

                catch_plot.plot(ax=ax, color="none", edgecolor="black", linewidth=1.5, zorder=10)

                if "order" in streams_plot.columns and streams_plot["order"].notna().any():
                    for o in sorted(streams_plot["order"].dropna().unique()):
                        seg = streams_plot[streams_plot["order"] == int(o)]
                        lw = max(1.0, 0.8 + 0.6 * float(o))
                        seg.plot(ax=ax, linewidth=lw, color="#1f78b4", label=f"Order {int(o)}", zorder=20)
                    handles, labels = ax.get_legend_handles_labels()
                    bylabel = dict(zip(labels, handles))
                    ax.legend(bylabel.values(), bylabel.keys(), loc="upper left", frameon=True)
                else:
                    streams_plot.plot(ax=ax, linewidth=1.5, color="#1f78b4", label="Streams", zorder=20)
                    ax.legend(loc="upper left", frameon=True)

                if sites_plot is not None and not use_subcatch and not sites_plot.empty:
                    sites_plot.plot(ax=ax, color="red", markersize=40, marker="o", zorder=30)
                    for i, r in sites_plot.iterrows():
                        sid = r.get(site_id_field, i)
                        ax.annotate(
                            str(sid),
                            (r.geometry.x, r.geometry.y),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=11,
                            color="black",
                            weight="bold"
                        )

                _apply_small_plot_padding(ax, (xmin, xmax, ymin, ymax), pad_fraction=0.02)
                _style_map_axes(ax, xlabel="Longitude (°)", ylabel="Latitude (°)", nbins=4)

                stream_png = os.path.join(catch_plots, "Catchment_Stream_Network.png")
                plt.tight_layout()
                plt.savefig(stream_png, dpi=300, bbox_inches="tight")
                plt.close(fig)
                _log("OK", f"Stream network map saved: {stream_png}")
                stream_map_saved = True
            except Exception as e:
                _log("WARN", f"Basemap stream plotting failed, fallback plain map will be used: {e}")
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                ax.set_title(f"{catchment_name} - Stream Network", fontsize=11, fontweight="bold")

        # Plain stream map
        if not stream_map_saved:
            catch_plot.plot(ax=ax, color="none", edgecolor="black", linewidth=1.5, zorder=10)

            if "order" in streams_plot.columns and streams_plot["order"].notna().any():
                for o in sorted(streams_plot["order"].dropna().unique()):
                    seg = streams_plot[streams_plot["order"] == int(o)]
                    lw = max(1.0, 0.8 + 0.6 * float(o))
                    seg.plot(ax=ax, linewidth=lw, color="#1f78b4", label=f"Order {int(o)}", zorder=20)

                handles, labels = ax.get_legend_handles_labels()
                bylabel = dict(zip(labels, handles))
                ax.legend(bylabel.values(), bylabel.keys(), loc="upper left", frameon=True)
            else:
                streams_plot.plot(ax=ax, linewidth=1.5, color="#1f78b4", label="Streams", zorder=20)
                ax.legend(loc="upper left", frameon=True)

            if sites_plot is not None and not use_subcatch and not sites_plot.empty:
                sites_plot.plot(ax=ax, color="red", markersize=40, marker="o", zorder=30)
                for i, r in sites_plot.iterrows():
                    sid = r.get(site_id_field, i)
                    x = r.geometry.x
                    y = r.geometry.y
                    ax.annotate(
                        str(sid),
                        (x, y),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=11,
                        color="black",
                        weight="bold"
                    )

            _apply_small_plot_padding(ax, (xmin, xmax, ymin, ymax), pad_fraction=0.02)
            _style_map_axes(ax, xlabel=xlabel, ylabel=ylabel, nbins=4)

            stream_png = os.path.join(catch_plots, "Catchment_Stream_Network.png")
            plt.tight_layout()
            plt.savefig(stream_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            _log("OK", f"Stream network map saved: {stream_png}")

    except Exception as e:
        _log("WARN", f"Stream network plotting failed: {e}")

    # --- Flow accumulation ---
    grid = Grid.from_raster(dem_projected_file)
    dem = grid.read_raster(dem_projected_file)
    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    acc_f32 = np.where(np.isfinite(acc), acc.astype("float32"), np.nan)
    acc_tif = os.path.join(topo_folder, "Flow_accumulation.tif")
    _write_raster(acc_tif, acc_f32, dem_meta)
    _log("OK", f"Flow accumulation raster written: {acc_tif}")

    # --- Per-site summaries and plots ---
    site_rows = []

    if sites_gdf is not None and not sites_gdf.empty:
        for idx, row in sites_gdf.iterrows():
            try:
                attrs = row.drop(labels="geometry").to_dict()

                if use_subcatch:
                    combined_geom = row.geometry
                    x_site = y_site = None
                else:
                    if row.geometry is None or row.geometry.is_empty:
                        _log("WARN", f"Skipping site {row.get(site_id_field, idx)}: empty geometry.")
                        continue

                    x_site, y_site = row.geometry.x, row.geometry.y
                    combined_geom = _vectorized_catchment_for_point(
                        grid, fdir, acc, dirmap, x_site, y_site, dem_transform
                    )
                    if combined_geom is None:
                        _log("WARN", f"No delineated catchment for site {row.get(site_id_field, idx)}.")
                        continue

                site_gdf = gpd.GeoDataFrame([attrs], geometry=[combined_geom], crs=dem_crs)

                if not use_subcatch:
                    site_gdf["X_site"] = x_site
                    site_gdf["Y_site"] = y_site

                site_gdf["Area_m2"] = round(site_gdf.geometry.area.values[0], 0)
                site_gdf["Area_ha"] = round(site_gdf["Area_m2"] / 10_000.0, 1)

                site_id = row.get(site_id_field, idx)
                site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
                site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
                _ensure_dirs([site_data_dir, site_plot_dir])

                raster_names = [
                    "Aspect in degree.tif",
                    "DEM.tif",
                    "Slope in degree.tif",
                    "Terrain Ruggedness Index.tif",
                    "Topographic Position Index.tif",
                ]

                for rname in raster_names:
                    rpath = os.path.join(topo_folder, rname)
                    if not os.path.exists(rpath):
                        _log("WARN", f"{rname} not found — skipping.")
                        continue

                    short = os.path.splitext(rname)[0]
                    site_gdf[f"{short} (mean)"] = np.nan
                    site_gdf[f"{short} (median)"] = np.nan
                    if not use_subcatch:
                        site_gdf[f"{short} (at site)"] = np.nan

                    with rio.open(rpath) as src:
                        v = site_gdf.to_crs(src.crs)
                        geom = [v.geometry.iloc[0]]
                        out_image, out_transform = rio_mask(src, geom, crop=True)
                        data = out_image[0]
                        data = np.where(data == src.nodata, np.nan, data)

                        site_gdf.at[0, f"{short} (mean)"] = round(np.nanmean(data), 2)
                        site_gdf.at[0, f"{short} (median)"] = round(np.nanmedian(data), 2)

                        if not use_subcatch and x_site is not None and y_site is not None:
                            r_idx = src.index(x_site, y_site)
                            val = src.read(1)[r_idx[0], r_idx[1]]
                            if val == src.nodata:
                                val = np.nan
                            site_gdf.at[0, f"{short} (at site)"] = round(float(val), 2) if not np.isnan(val) else np.nan

                        # Save clipped raster
                        clipped_meta = src.meta.copy()
                        clipped_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "crs": src.crs,
                            "nodata": src.nodata
                        })

                        clipped_tif = os.path.join(site_data_dir, f"{short}.tif")
                        with rio.open(clipped_tif, "w", **clipped_meta) as dst:
                            dst.write(out_image)

                        # Plot clipped raster
                        try:
                            if plot_in_geographic_coords:
                                plot_data, extent = _reproject_array_for_plotting(
                                    out_image[0],
                                    clipped_meta,
                                    dst_crs="EPSG:4326",
                                    resampling=Resampling.nearest
                                )
                                v_plot = v.to_crs(epsg=4326)
                                xlabel = "Longitude (°)"
                                ylabel = "Latitude (°)"
                            else:
                                plot_data = out_image[0]
                                left = out_transform[2]
                                top = out_transform[5]
                                pxw = out_transform[0]
                                pxh = -out_transform[4]
                                right = left + pxw * out_image.shape[2]
                                bottom = top - pxh * out_image.shape[1]
                                extent = [left, right, bottom, top]
                                v_plot = v
                                xlabel = "Longitude"
                                ylabel = "Latitude"

                            fig_w, fig_h = _get_adaptive_figsize_from_bounds(
                                tuple(v_plot.total_bounds),
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
                            cbar.set_label(short, fontsize=9)

                            if not use_subcatch and x_site is not None and y_site is not None:
                                if plot_in_geographic_coords:
                                    site_pt_plot = gpd.GeoSeries([Point(x_site, y_site)], crs=v.crs).to_crs(epsg=4326)
                                else:
                                    site_pt_plot = gpd.GeoSeries([Point(x_site, y_site)], crs=v.crs)

                                site_pt_plot.plot(
                                    ax=ax,
                                    color="red",
                                    markersize=20,
                                    marker="o",
                                    label="Site location"
                                )

                                pt = site_pt_plot.iloc[0]
                                ax.annotate(
                                    f"{site_id}",
                                    (pt.x, pt.y),
                                    xytext=(5, 5),
                                    textcoords="offset points",
                                    fontsize=7,
                                    color="black"
                                )

                            ax.set_title(f"{short} - Site {site_id}", fontsize=11, fontweight="bold")
                            _set_axis_limits_from_gdf(ax, v_plot, pad_fraction=0.02)
                            _style_map_axes(ax, xlabel=xlabel, ylabel=ylabel, nbins=4)

                            plt.tight_layout()
                            plt.savefig(os.path.join(site_plot_dir, f"{short}.png"), dpi=300, bbox_inches="tight")
                            plt.close(fig)
                        except Exception as e:
                            _log("WARN", f"Error plotting {short} for site {site_id}: {e}")

                site_rows.append(site_gdf)

                gpkg_path = os.path.join(site_data_dir, f"Site_{site_id}.gpkg")
                csv_path = os.path.join(site_data_dir, f"Site_{site_id}.csv")

                site_gdf.to_file(gpkg_path, driver="GPKG")
                site_gdf.drop(columns="geometry").to_csv(csv_path, index=False)
                _log("OK", f"Site {site_id} topographic data and plots saved.")

            except Exception as e:
                _log("ERROR", f"Error processing site {row.get(site_id_field, idx)}: {e}")
                continue
    else:
        _log("WARN", "No valid sites available. Per-site summaries and plots were skipped.")

    # --- Combined site outputs ---
    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    all_sites_csv = os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv")

    for p in (all_sites_gpkg, all_sites_gpkg + "-wal", all_sites_gpkg + "-shm", all_sites_csv):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    if site_rows:
        all_gdf = gpd.GeoDataFrame(pd.concat(site_rows, ignore_index=True), crs=dem_crs)
        all_gdf.to_file(all_sites_gpkg, layer=f"{catchment_name} Sites", driver="GPKG")
        all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)
        _log("OK", f"All sites summary saved in {sites_datasets}")
    else:
        # safer placeholder output
        empty_df = pd.DataFrame()
        empty_df.to_csv(all_sites_csv, index=False)
        _log("WARN", f"No site summaries were created. CSV placeholder written to {sites_datasets}")

    # --- Cleanup ---
    if downloaded and dem_temp and os.path.exists(dem_temp):
        try:
            os.remove(dem_temp)
            _log("OK", f"Temporary downloaded DEM removed: {dem_temp}")
        except Exception as e:
            _log("WARN", f"Could not remove temporary DEM: {e}")

    gc.collect()
    _log("OK", "DEM and topography processing completed.")

    return dem_projected_file, all_sites_gpkg, sites_datasets, sites_plots