# dem_and_terrain.py
# ===================== IMPORTS =====================
from __future__ import annotations

# --- Standard library ---
import os
import io
import gc
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

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
from shapely.geometry import Point, LineString, shape as shp_shape

# --- Hydro / RS ---
from pysheds.grid import Grid
from rasterstats import zonal_stats

# --- Viz ---
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
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
    sites_path : str
        Path to sites vector. Points => delineate catchments; polygons => use as subcatchments.
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
    sites_path: str
    dem_url: Optional[str] = None
    dem_local_path: Optional[str] = None
    target_res_m: float = 30.0
    stream_target_area_ha: float = 60.0


def _ensure_dirs(paths: Iterable[str]) -> None:
    """Create folders if they do not exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _read_catchment(catchment_path: str) -> gpd.GeoDataFrame:
    """Read & dissolve catchment; add area and placeholder date column."""
    if not os.path.exists(catchment_path):
        raise FileNotFoundError(f"Catchment file not found: {catchment_path}")
    gdf = gpd.read_file(catchment_path)
    if gdf.empty:
        raise ValueError("Catchment file has no features.")
    gdf = gdf.dissolve()
    # Area in hectares in native CRS units (assumes projected metres; if geographic, user should reproject)
    gdf["Area_ha"] = round(gdf.geometry.area.values[0] / 10_000.0, 0)
    gdf["Date"] = pd.Series(pd.NA, index=gdf.index, dtype="Int64")
    return gdf


def _bbox_wgs84(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return minx, miny, maxx, maxy in EPSG:4326."""
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    return tuple(gdf_wgs84.total_bounds)


def _download_dem_wcs(dem_url: str, bbox: Tuple[float, float, float, float], out_tif: str) -> None:
    """Download DEM from a WCS endpoint to GeoTIFF clipped to bbox (EPSG:4326)."""
    minx, miny, maxx, maxy = bbox
    # 1 arc-sec ≈ 0.000277778 deg
    resolution_deg = 1 / 3600.0
    width = max(1, int((maxx - minx) / resolution_deg))
    height = max(1, int((maxy - miny) / resolution_deg))

    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": "1",           # may vary per service
        "crs": "EPSG:4326",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": width,
        "height": height,
        "format": "GeoTIFF"
    }
    r = requests.get(dem_url, params=params, stream=True, timeout=120)
    if r.status_code != 200 or "image/tiff" not in r.headers.get("Content-Type", ""):
        # Show a helpful snippet for debugging
        snippet = ""
        try:
            snippet = r.content[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"DEM WCS download failed (status={r.status_code}, ctype={r.headers.get('Content-Type')}). "
            f"Server said (first 500 bytes): {snippet!r}"
        )

    with open(out_tif, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _clip_to_catchment_wgs84(src_path: str, catchment_wgs84: gpd.GeoDataFrame, out_path: str) -> None:
    """Clip raster to catchment geometry in the *raster’s* CRS; save WGS84-clipped DEM."""
    with rxr.open_rasterio(src_path, masked=True) as dem:
        clip_geom = catchment_wgs84.geometry
        if dem.rio.crs is None:
            raise ValueError("DEM has no CRS.")
        if dem.rio.crs != catchment_wgs84.crs:
            # reproject the polygon to the DEM CRS before clipping
            clip_geom = catchment_wgs84.to_crs(dem.rio.crs).geometry
        clipped = dem.rio.clip(clip_geom, dem.rio.crs, drop=True, all_touched=True)
        # Convert nodata to NaN
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
            src.crs, target_crs, src.width, src.height, *src.bounds,
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
                    resampling=Resampling.bilinear,   # smoother than nearest for DEM
                )


def _compute_slope_aspect(dem_data: np.ndarray, xres: float, yres: float) -> Dict[str, np.ndarray]:
    """Compute slope (deg, rad, %) and aspect (deg, rad)."""
    # Replace NoData with NaN for stable numeric ops
    dem = dem_data.astype("float32", copy=True)
    # Gradient
    dz_dx, dz_dy = np.gradient(dem, xres, yres)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_rad = np.arctan(slope_ratio)
    slope_deg = np.degrees(slope_rad)
    slope_pct = slope_ratio * 100.0

    # Aspect: arctan2(dz_dy, -dz_dx) so that 0 rad ~ North (as in your code)
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

    Ensures:
      - Float32 dtype (for consistency across modules)
      - NaN values are preserved (not replaced with finite sentinel)
      - No rioxarray/rasterio dtype mismatch warnings
    """
    meta = template_meta.copy()
    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan  # explicitly declare NaN nodata
    })

    # Ensure the array is float32 and clean of inf values
    data = data.astype("float32", copy=False)
    data = np.where(np.isfinite(data), data, np.nan)

    with rio.open(out_path, "w", **meta) as dst:
        dst.write(data, 1)



from typing import Optional, Tuple, Dict, Any
def _plot_raster_with_overlays(
    data: np.ndarray,
    extent: Tuple[float, float, float, float],
    catchment_gdf: gpd.GeoDataFrame,
    out_png: str,
    title: str,                                   # plot title only
    site_points: Optional[gpd.GeoDataFrame] = None,
    annotate_ids: bool = False,
    title_kwargs: Optional[Dict[str, Any]] = None,
    colorbar_label: Optional[str] = None,         # <— NEW: label for the colorbar
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        im = ax.imshow(data, cmap="viridis", extent=extent, origin="upper")
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # colorbar label defaults to something sensible if not given
        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(colorbar_label or "Value", fontsize=9)

        # title with external styling
        ax.set_title(title, **(title_kwargs or {"fontsize": 11, "fontweight": "bold"}))

        if (catchment_gdf is not None) and (not catchment_gdf.empty):
            catchment_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

        if site_points is not None and not site_points.empty:
            site_points.plot(ax=ax, color="red", markersize=22, marker="o")
            if annotate_ids:
                for _i, _r in site_points.iterrows():
                    sid = _r.get("id", _i)
                    geom = _r.geometry
                    x, y = (geom.x, geom.y) if geom.geom_type == "Point" else (geom.centroid.x, geom.centroid.y)
                    ax.annotate(str(sid), (x, y), xytext=(3, 3), textcoords="offset points",
                                fontsize=8, color="black", weight="bold")

        ax.set_xlabel("Longitude", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, linestyle="--", alpha=0.6)

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
        return gdf, False     # points => delineate catchment
    if geom0.geom_type in ("Polygon", "MultiPolygon"):
        return gdf, True      # polygons => subcatchments
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

    # Convert area (ha) → cells
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
    streams.to_file(out_gpkg, layer=layer_name, driver="GPKG", if_exists="replace")
    return streams


def _vectorized_catchment_for_point(
    grid: Grid, fdir: np.ndarray, acc: np.ndarray, dirmap, pt_x: float, pt_y: float, dem_transform
):
    """Delineate and vectorize a point catchment."""
    grid_local = copy.deepcopy(grid)
    # Snap to high accumulation cell
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
    return gpd.GeoSeries(geoms).unary_union


# ===================== MAIN ENTRYPOINT =====================

def dem_and_terrain(
    CHM_Work_Space: str,
    Catchment_Shapefile_Path: str,
    Sites_Shapefile_Path: str,
    DEM_Input_Path: Optional[str] = None,
    DEM_url: Optional[str] = None,
    target_res_m: float = 30.0,
    stream_target_area_ha: float = 60.0,
) -> Tuple[str, str, str, str]:
    """
    End-to-end pipeline: acquire DEM (WCS or local), clip to catchment, reproject,
    compute terrain rasters, derive stream network, and summarize/plot per-site.

    Parameters
    ----------
    CHM_Work_Space : str
        Root output folder.
    Catchment_Shapefile_Path : str
        Path to catchment boundary.
    Sites_Shapefile_Path : str
        Path to sites (points => delineate catchments; polygons => subcatchments).
    DEM_Input_Path : str, optional
        Local DEM path. If provided and exists, used instead of WCS.
    DEM_url : str, optional
        WCS endpoint for DEM download (used when DEM_Input_Path is None or missing).
    target_res_m : float
        Target DEM resolution in metres when reprojecting to catchment CRS.
    stream_target_area_ha : float
        Target upstream area threshold (ha) to define streams.

    Returns
    -------
    dem_projected_file : str
    all_sites_gpkg : str
    sites_datasets : str
    sites_plots : str
    """
    print("Starting processes DEM and topography ...")

    # --- Folder structure (preserved) ---
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace("_", " ")
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    topo_folder = os.path.join(catch_datasets, "Topography")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    _ensure_dirs([catchment_folder, catch_datasets, catch_plots, topo_folder, sites_datasets, sites_plots])

    # --- Catchment layer & metadata store ---
    gdf_catch = _read_catchment(Catchment_Shapefile_Path)
    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    gdf_catch.to_file(catch_gpkg, layer=f"{catchment_name} Data", driver="GPKG")

    catch_crs = gdf_catch.crs
    if catch_crs is None:
        raise ValueError("Catchment file has no CRS. Please assign a projected CRS (metres).")

    # --- DEM acquisition (local vs WCS) ---
    dem_temp = DEM_Input_Path if (DEM_Input_Path and os.path.exists(DEM_Input_Path)) else None
    downloaded = False
    if dem_temp is None:
        if not DEM_url:
            raise ValueError("No local DEM supplied and no DEM_url provided for WCS download.")
        print("Requesting DEM from WCS ...")
        dem_temp = os.path.join(topo_folder, "DEM_temp.tif")
        _download_dem_wcs(DEM_url, _bbox_wgs84(gdf_catch), dem_temp)
        downloaded = True
        print(f"DEM downloaded: {dem_temp}")
    else:
        print(f"Using provided DEM: {dem_temp}")

    # --- Clip to catchment in WGS84 space ---
    dem_clipped_wgs84 = os.path.join(topo_folder, "DEM_WGS84.tif")
    gdf_wgs84 = gdf_catch.to_crs(epsg=4326)
    _clip_to_catchment_wgs84(dem_temp, gdf_wgs84, dem_clipped_wgs84)

    # --- Reproject to catchment CRS & target res ---
    dem_projected_file = os.path.join(topo_folder, "DEM.tif")
    _reproject_dem_to_catchment(dem_clipped_wgs84, catch_crs, target_res_m, dem_projected_file)

    # --- Load DEM & basic info for layers/plots ---
    with rio.open(dem_projected_file) as src:
        dem_data = src.read(1).astype("float32")
        dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
        dem_meta = src.meta.copy()
        dem_extent = plotting_extent(src)
        dem_crs = src.crs
        dem_transform = src.transform

    sites_gdf, use_subcatch = _detect_sites_role(Sites_Shapefile_Path, dem_crs)
    site_points_for_plots = sites_gdf if not use_subcatch else None
    terrain_layers: Dict[str, np.ndarray] = {"DEM.tif": dem_data}

    # --- Slope/aspect + TPI/TRI ---
    xres = dem_transform.a
    yres = abs(dem_transform.e)
    terrain_layers.update(_compute_slope_aspect(dem_data, xres, yres))
    terrain_layers.update(_compute_tpi_tri(dem_data))

    # --- Save terrain rasters & plots ---
    for name, arr in terrain_layers.items():
        out_tif = os.path.join(topo_folder, name)
        _write_raster(out_tif, arr, dem_meta)
    
        plot_png = os.path.join(catch_plots, os.path.splitext(name)[0] + ".png")
        _plot_raster_with_overlays(
            data=arr,
            extent=dem_extent,
            catchment_gdf=gdf_catch,
            out_png=plot_png,
            title=f"{catchment_name} — {os.path.splitext(name)[0]}",
            site_points=site_points_for_plots,
            annotate_ids=True,
            title_kwargs={"fontsize": 11, "fontweight": "bold"},
            colorbar_label=os.path.splitext(name)[0]  # e.g., "Aspect in radian"
        )


    # --- Stream network (Strahler), saved to GPKG + a catchment map with tiles & labels ---
    streams_gpkg = os.path.join(topo_folder, "Stream_Network.gpkg")
    streams_gdf = _extract_stream_network(
        dem_projected_file, stream_target_area_ha, out_gpkg=streams_gpkg, layer_name="Streams"
    )
    print(f"Stream network written: {streams_gpkg} (layer='Streams')")
    
    # Assumes: streams_gdf, catch_plots, Catchment_Shapefile_Path, use_subcatch, site_points_for_plots are defined.
    try:
        # Reproject catchment to streams CRS (then to Web Mercator for basemap/tiles)
        catch_plot_gdf = gpd.read_file(Catchment_Shapefile_Path)
        if catch_plot_gdf.crs != streams_gdf.crs:
            catch_plot_gdf = catch_plot_gdf.to_crs(streams_gdf.crs)
    
        streams_web = streams_gdf.to_crs(epsg=3857)
        catch_web   = catch_plot_gdf.to_crs(epsg=3857)
    
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{catchment_name} - stream network", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude",fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude",fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        # Set extent first so any basemap call requests tiles only for AOI
        xmin, ymin, xmax, ymax = catch_web.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
        # Basemap (optional)
        if ctx is not None:
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto", alpha=1.0)
            except Exception as e:
                print(f"[WARN] Basemap not added: {e}")
    
        # Catchment outline on top of basemap
        catch_web.plot(ax=ax, color="none", edgecolor="black", linewidth=1.5, zorder=10)
    
        # Streams by Strahler order (thicker lines for higher order)
        if "order" in streams_web.columns and streams_web["order"].notna().any():
            for o in sorted(streams_web["order"].dropna().unique()):
                seg = streams_web[streams_web["order"] == int(o)]
                lw = max(1.0, 0.8 + 0.6 * float(o))
                seg.plot(ax=ax, linewidth=lw, color="#1f78b4", label=f"Order {int(o)}", zorder=20)
            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            bylabel = dict(zip(labels, handles))
            ax.legend(bylabel.values(), bylabel.keys(), loc="upper left", frameon=True)
        else:
            streams_web.plot(ax=ax, linewidth=1.5, color="#1f78b4", label="Streams", zorder=20)
            ax.legend(loc="upper left", frameon=True)
    
        # Overlay site points (only if sites are points)
        if not use_subcatch and site_points_for_plots is not None and not site_points_for_plots.empty:
            sites_web = site_points_for_plots.to_crs(epsg=3857)
            sites_web.plot(ax=ax, color="red", markersize=40, marker="o", zorder=30)
            for i, r in sites_web.iterrows():
                sid = r.get("id", i)
                ax.annotate(str(sid), (r.geometry.x, r.geometry.y),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=11, color="black", weight="bold")
    
        stream_png = os.path.join(catch_plots, "Catchment_Stream_Network.png")
        plt.tight_layout()
        plt.savefig(stream_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Stream network map saved: {stream_png}")
    except Exception as e:
        print(f"Stream network plotting failed: {e}")


    # Reuse grid/fdir/acc for delineations
    grid = Grid.from_raster(dem_projected_file)
    dem = grid.read_raster(dem_projected_file)
    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # --- Per-site summaries & plots ---
    site_rows = []
    for idx, row in sites_gdf.iterrows():
        try:
            attrs = row.drop(labels="geometry").to_dict()
            if use_subcatch:
                combined_geom = row.geometry
                x_site = y_site = None
            else:
                if row.geometry is None or row.geometry.is_empty:
                    print(f"Skipping site {row.get('id', idx)}: empty geometry.")
                    continue
                x_site, y_site = row.geometry.x, row.geometry.y
                combined_geom = _vectorized_catchment_for_point(grid, fdir, acc, dirmap, x_site, y_site, dem_transform)
                if combined_geom is None:
                    print(f"No delineated catchment for site {row.get('id', idx)}.")
                    continue

            site_gdf = gpd.GeoDataFrame([attrs], geometry=[combined_geom], crs=dem_crs)
            if not use_subcatch:
                site_gdf["X_site"] = x_site
                site_gdf["Y_site"] = y_site
            site_gdf["Area_m2"] = round(site_gdf.geometry.area.values[0], 0)
            site_gdf["Area_ha"] = round(site_gdf["Area_m2"] / 10_000.0, 1)

            site_id = row.get("id", idx)
            site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
            site_plot_dir = os.path.join(sites_plots, f"Site_{site_id}")
            _ensure_dirs([site_data_dir, site_plot_dir])

            # Rasters to summarise
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
                    print(f"{rname} not found — skipping.")
                    continue
                short = os.path.splitext(rname)[0]
                site_gdf[f"{short} (mean)"] = np.nan
                site_gdf[f"{short} (median)"] = np.nan
                if not use_subcatch:
                    site_gdf[f"{short} (at site)"] = np.nan

                with rio.open(rpath) as src:
                    # Align CRS for masking
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

                    # Save clipped raster for the site
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

                    # Plot clipped raster with site polygon outline (+ point for point sites)
                    try:
                        if out_transform:
                            left = out_transform[2]; top = out_transform[5]
                            pxw = out_transform[0]; pxh = -out_transform[4]
                            right = left + pxw * out_image.shape[2]
                            bottom = top - pxh * out_image.shape[1]
                            extent = [left, right, bottom, top]
                        else:
                            extent = [0, out_image.shape[2], 0, out_image.shape[1]]

                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(out_image[0], extent=extent, cmap="viridis", origin="upper")
                        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85)
                        cbar.set_label(short, fontsize=9)
                        v.boundary.plot(ax=ax, color="black", linewidth=1.2)
                        if not use_subcatch and x_site is not None and y_site is not None:
                            gpd.GeoSeries([Point(x_site, y_site)], crs=v.crs).plot(
                                ax=ax, color="red", markersize=20, marker="o", label="Site location"
                            )
                            ax.annotate(f"{site_id}", (x_site, y_site),
                                        xytext=(5, 5), textcoords="offset points",
                                        fontsize=7, color="black")
                        ax.set_title(f"{short} - Site {site_id}", fontsize=11, fontweight="bold")
                        ax.set_xlabel("Longitude",fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude",fontsize=10, fontweight="bold")
                        ax.tick_params(axis="both", labelsize=10)
                        # Limit number of ticks
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                
                        # Use scientific notation (e.g., 2.44×10⁶)
                        #fmt = ScalarFormatter(useMathText=True)
                        #fmt.set_powerlimits((-3, 3))
                        #ax.xaxis.set_major_formatter(fmt)
                        #ax.yaxis.set_major_formatter(fmt)
                
                        # Add grid for clarity (optional)
                        ax.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(os.path.join(site_plot_dir, f"{short}.png"), dpi=300)
                        plt.close("all")
                    except Exception as e:
                        print(f"Error plotting {short} for site {site_id}: {e}")

            # Persist per-site outputs
            site_rows.append(site_gdf)
            gpkg_path = os.path.join(site_data_dir, f"Site_{site_id}.gpkg")
            csv_path = os.path.join(site_data_dir, f"Site_{site_id}.csv")
            site_gdf.to_file(gpkg_path, driver="GPKG")
            site_gdf.drop(columns="geometry").to_csv(csv_path, index=False)
            print(f"Site {site_id} topographic data and plots saved.")

        except Exception as e:
            print(f"Error processing site {row.get('id', idx)}: {e}")
            continue

    # --- Combined outputs (same naming) ---
    all_gdf = gpd.GeoDataFrame(pd.concat(site_rows, ignore_index=True), crs=dem_crs)
    all_sites_gpkg = os.path.join(sites_datasets, f"{catchment_name} Sites.gpkg")    
    all_sites_gpkg_data = os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg") 
    all_sites_csv = os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv")

    # Clean previous (and SQLite sidecars) then write fresh
    for p in (all_sites_gpkg, all_sites_gpkg + "-wal", all_sites_gpkg + "-shm", all_sites_csv):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    all_gdf.to_file(all_sites_gpkg, layer=f"{catchment_name} Sites", driver="GPKG")
    all_gdf.to_file(all_sites_gpkg_data, layer=f"{catchment_name} Sites", driver="GPKG")
    #all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)

    # Remove downloaded temp DEM (never delete user-supplied DEM)
    if downloaded and dem_temp and os.path.exists(dem_temp):
        try:
            os.remove(dem_temp)
        except Exception:
            pass

    gc.collect()
    print(f"All sites summary saved in {sites_datasets}")
    return dem_projected_file, all_sites_gpkg, sites_datasets, sites_plots