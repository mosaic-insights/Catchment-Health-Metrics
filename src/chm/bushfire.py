from __future__ import annotations

# ============== stdlib ==============
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ============== third-party ==============
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from shapely.ops import unary_union
import gc

# ========================= Config =========================

@dataclass
class BushfireConfig:
    """
    Historical bushfire risk profiles (burnt area vs SDR / TWI).

    Required
    --------
    chm_workspace: str           -> project workspace root
    catchment_path: str          -> catchment boundary (vector; readable by GeoPandas)

    Optional (IO)
    -------------
    sites_gpkg: Optional[str]    -> existing sites GPKG (defaults to <Catchment>/Sites Datasets/<name> Sites Data.gpkg)
    bushfire_input_path: Optional[str]
        If provided, read bushfire polygons locally (any GeoDataFrame-readable file).
        If None, fetch via ArcGIS REST at `bushfire_url`.
    bushfire_url: str
        ArcGIS REST layer query endpoint returning GeoJSON features (when no local file provided).

    Optional (processing)
    ---------------------
    sdr_dirname: str             -> relative to <Catchment Datasets> (default "Surface and Groundwater Connectivity/SDR")
    twi_path: str                -> absolute path to TWI raster, or relative under <Catchment Datasets>
    window_years: int            -> rolling window size for "recent fires" (default 5)
    make_plots: bool             -> create PNGs (default True)
    requests_timeout: int        -> seconds per HTTP request (default 120)
    max_arcgis_page: int         -> max features per ArcGIS page (default 2000)
    """
    chm_workspace: str
    catchment_path: str

    sites_gpkg: Optional[str] = None
    bushfire_input_path: Optional[str] = None
    bushfire_url: str = (
        "https://services-ap1.arcgis.com/ypkPEy1AmwPKGNNv/arcgis/rest/services/"
        "Bushfire_Boundaries_Historic_Dec_view/FeatureServer/0/query"
    )

    sdr_dirname: str = os.path.join("Surface and Groundwater Connectivity", "SDR")
    twi_relpath: str = os.path.join("Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")
    window_years: int = 5
    make_plots: bool = True
    requests_timeout: int = 120
    max_arcgis_page: int = 2000  # ArcGIS servers usually cap at ~2000/32000 per page


# ========================= Helpers =========================

def _log(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str) -> Tuple[str, str, str, str, str, str]:
    """Return common folders and ensure they exist."""
    catchment_name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(workspace, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    bushfire_output = os.path.join(catch_datasets, "Historical Bushfire")
    _ensure_dirs([catchment_folder, catch_datasets, catch_plots, sites_datasets, sites_plots, bushfire_output])
    return catchment_name, catchment_folder, catch_datasets, catch_plots, sites_datasets, sites_plots


def _arcgis_paginated_geojson(url: str, bbox: Tuple[float, float, float, float],
                              out_fields: str = "*", timeout: int = 120, pagesize: int = 2000) -> gpd.GeoDataFrame:
    """
    Pull all features intersecting bbox from an ArcGIS REST /query endpoint, handling pagination.
    Returns a GeoDataFrame in EPSG:4326.
    """
    minx, miny, maxx, maxy = bbox
    params_base = {
        "f": "geojson",
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true",
        "resultOffset": 0,
        "resultRecordCount": pagesize,
    }

    frames: List[pd.DataFrame] = []
    while True:
        resp = requests.get(url, params=params_base, timeout=timeout)
        resp.raise_for_status()
        feats = resp.json().get("features", [])
        if not feats:
            break
        gdf_page = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        frames.append(gdf_page)
        if len(feats) < pagesize:
            break
        params_base["resultOffset"] += pagesize

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return pd.concat(frames, ignore_index=True)


def _read_or_download_bushfire(cfg: BushfireConfig, gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if cfg.bushfire_input_path and os.path.exists(cfg.bushfire_input_path):
        g = gpd.read_file(cfg.bushfire_input_path)
        if g.crs is None:
            g = g.set_crs(gdf_wgs84.crs, allow_override=True)
        elif g.crs != gdf_wgs84.crs:
            g = g.to_crs(gdf_wgs84.crs)
        return g

    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    return _arcgis_paginated_geojson(
        cfg.bushfire_url, (minx, miny, maxx, maxy), timeout=cfg.requests_timeout, pagesize=cfg.max_arcgis_page
    )


def _coerce_ignition_date(df: pd.DataFrame) -> pd.Series:
    """
    Try to coerce ignition date column to pandas datetime.
    Handles common arcgis epoch (ms) or ISO strings.
    """
    cand = [c for c in df.columns if c.lower() in ("ignition_date", "ignitiondate", "date", "start_date")]
    if not cand:
        cand = [c for c in df.columns if df[c].dtype.kind in "iu" or df[c].dtype.kind == "f"]
        if cand:
            s = pd.to_datetime(df[cand[0]], errors="coerce", unit="ms")
            return s
        return pd.to_datetime(pd.Series([pd.NaT] * len(df)))

    col = cand[0]
    s = df[col]
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, errors="coerce", unit="ms")
    return pd.to_datetime(s, errors="coerce")


def _scan_sdr_range(sdr_dir: str, catch_gdf: gpd.GeoDataFrame) -> Tuple[float, float, List[int]]:
    """Return (min, max, years[]) of SDR across all rasters within the catchment."""
    smin, smax = np.inf, -np.inf
    years: List[int] = []
    if not os.path.isdir(sdr_dir):
        return 0.0, 1.0, years

    for fname in os.listdir(sdr_dir):
        if not (fname.endswith(".tif") and re.match(r"SDR_\d{4}\.tif$", fname)):
            continue
        m = re.search(r"\d{4}", fname)
        if m:
            try:
                years.append(int(m.group(0)))
            except Exception:
                pass
        fpath = os.path.join(sdr_dir, fname)
        try:
            with rio.open(fpath) as src:
                img, _ = rio_mask(src, catch_gdf.geometry, crop=True)
                arr = img[0].astype(float)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                mn, mx = np.nanmin(arr), np.nanmax(arr)
                if np.isfinite(mn):
                    smin = min(smin, float(mn))
                if np.isfinite(mx):
                    smax = max(smax, float(mx))
        except Exception:
            pass

    if not np.isfinite(smin) or not np.isfinite(smax) or smin >= smax:
        return 0.0, 1.0, sorted(set(years))
    return smin, smax, sorted(set(years))


def _twi_range(twi_path: str, catch_gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    try:
        with rio.open(twi_path) as src:
            img, _ = rio_mask(src, catch_gdf.geometry, crop=True)
            arr = img[0].astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            mn = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
            mx = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
            if mn >= mx:
                return 0.0, 1.0
            return mn, mx
    except Exception:
        return 0.0, 1.0


def _auc_normalised(x_sorted: np.ndarray, y_fraction: np.ndarray, xmin: float, xmax: float) -> float:
    if x_sorted.size == 0 or xmax <= xmin:
        return 0.0
    mask = (x_sorted >= xmin) & (x_sorted <= xmax)
    if not np.any(mask):
        return 0.0
    xr = x_sorted[mask]
    yr = y_fraction[mask]
    return float(np.trapz(yr, xr) / (xmax - xmin))


# ========================= Main API =========================

def historical_bushfire(cfg: BushfireConfig) -> Tuple[str, str]:
    """
    Build historical bushfire “exposure” profiles:
      • source bushfire polygons (local or ArcGIS REST), clip to catchment & per-site
      • for each SDR raster year Y, take fires in [Y - window + 1, Y] and plot/compute
        cumulative burnt-area vs SDR; normalise AUC to fixed catchment SDR range
      • repeat for TWI (single raster), iterating over the SDR years but using TWI values
      • export per-site plots and GPKG of site-clipped bushfires

    Returns
    -------
    (all_sites_gpkg, sites_datasets)
    """
    _log("INFO", "Starting historical bushfire processing...")

    # ---- folders / names ----
    (catchment_name, _, catch_datasets, catch_plots, sites_datasets, sites_plots) = \
        _scaffold(cfg.chm_workspace, cfg.catchment_path)
    bushfire_output = os.path.join(catch_datasets, "Historical Bushfire")

    _log("OK", f"Workspace folders prepared for {catchment_name}")
    _log("INFO", f"Bushfire output folder: {bushfire_output}")

    # ---- paths ----
    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    sdr_dir = os.path.join(catch_datasets, cfg.sdr_dirname)
    twi_path = cfg.twi_relpath if os.path.isabs(cfg.twi_relpath) else os.path.join(catch_datasets, cfg.twi_relpath)

    # ---- catchment geometry (WGS84 & project CRS) ----
    catch_gdf = gpd.read_file(cfg.catchment_path)
    catch_crs = catch_gdf.crs
    catch_wgs = catch_gdf.to_crs(epsg=4326)
    _log("OK", f"Catchment loaded: {cfg.catchment_path}")

    # ---- bushfire source (local or ArcGIS REST) ----
    fire_all = _read_or_download_bushfire(cfg, catch_wgs)
    if fire_all.empty:
        _log("INFO", "No bushfire features downloaded/found. Exiting early.")
        return all_sites_gpkg, sites_datasets

    _log("OK", f"Bushfire source loaded. Features: {len(fire_all)}")

    # clip to catchment polygon (in WGS84), then back to project CRS
    fire_clip = gpd.clip(fire_all, catch_wgs).to_crs(catch_crs)
    _log("OK", f"Bushfire features clipped to catchment. Features: {len(fire_clip)}")

    # ignition date coercion
    fire_clip["ignition_date"] = _coerce_ignition_date(fire_clip)
    _log("OK", "Ignition dates prepared")

    # save full catchment bushfire layers
    fire_gpkg = os.path.join(bushfire_output, "Historical_Bushfires_Boundary.gpkg")
    fire_csv = os.path.join(bushfire_output, "Historical_Bushfires_Attributes.csv")
    fire_clip.to_file(fire_gpkg, driver="GPKG")
    fire_clip.drop(columns="geometry").to_csv(fire_csv, index=False)
    _log("OK", f"Catchment bushfire GPKG saved: {fire_gpkg}")
    _log("OK", f"Catchment bushfire CSV saved: {fire_csv}")

    # ---- SDR/TWI ranges for fixed x-axes ----
    sdr_min, sdr_max, sdr_years = _scan_sdr_range(sdr_dir, catch_gdf)
    twi_min, twi_max = _twi_range(twi_path, catch_gdf)
    _log("INFO", f"Catchment SDR range: {sdr_min} to {sdr_max}")
    _log("INFO", f"Catchment TWI range: {twi_min} to {twi_max}")
    _log("INFO", f"SDR years found: {sdr_years}")

    # ---- load sites (optional) ----
    sites = None
    if os.path.exists(all_sites_gpkg):
        try:
            sites = gpd.read_file(all_sites_gpkg)
            _log("OK", f"Sites loaded: {all_sites_gpkg}")
            _log("INFO", f"Number of sites: {len(sites)}")
        except Exception as e:
            sites = None
            _log("WARN", f"Could not read sites GPKG, so site-based outputs will be skipped: {e}")
    else:
        _log("WARN", f"Sites GPKG does not exist, so site-based outputs will be skipped: {all_sites_gpkg}")

    if sites is None or sites.empty:
        _log("OK", "Catchment-level bushfire tasks completed. No site-specific processing was run.")
        return all_sites_gpkg, sites_datasets

    # ================= per-site processing =================
    for idx, site in sites.iterrows():
        try:
            sid = site.get("Site_id", idx)
            _log("INFO", f"Processing Site {sid}...")

            site_geom = gpd.GeoDataFrame(geometry=[site.geometry], crs=sites.crs)

            site_dir = os.path.join(sites_datasets, f"Site_{sid}")
            plot_dir = os.path.join(sites_plots, f"Site_{sid}")
            _ensure_dirs([site_dir, plot_dir])

            # site-level bushfire polygons
            fire_site = gpd.clip(fire_clip, site_geom)
            if not fire_site.empty:
                site_fire_gpkg = os.path.join(site_dir, f"Site_{sid}_Bushfires.gpkg")
                fire_site.to_file(site_fire_gpkg, driver="GPKG")
                _log("OK", f"Site {sid} bushfire GPKG saved")
            else:
                _log("INFO", f"Site {sid} has no intersecting bushfire polygons")

            sdr_auc_ts: Dict[int, float] = {}
            twi_auc_ts: Dict[int, float] = {}
            sdr_line_profiles: List[Tuple[int, np.ndarray, np.ndarray]] = []
            twi_line_profiles: List[Tuple[int, np.ndarray, np.ndarray]] = []

            # ---------------- SDR loop (per-year rasters) ----------------
            for fname in os.listdir(sdr_dir):
                if not (fname.endswith(".tif") and re.match(r"SDR_\d{4}\.tif$", fname)):
                    continue

                fpath = os.path.join(sdr_dir, fname)
                m = re.search(r"\d{4}", fname)
                year = int(m.group(0)) if m else None

                with rio.open(fpath) as src:
                    img, transform = rio_mask(src, site_geom.geometry, crop=True)
                    sdr = img[0].astype(float)
                    if src.nodata is not None:
                        sdr[sdr == src.nodata] = np.nan
                    valid = ~np.isnan(sdr)

                if fire_site.empty:
                    continue

                # window filter on fires by ignition_date around the SDR year
                if year is not None:
                    y0 = year - (cfg.window_years - 1)
                    win = fire_site[
                        (fire_site["ignition_date"].dt.year >= y0) &
                        (fire_site["ignition_date"].dt.year <= year)
                    ]
                    y_label = f"{year}"
                else:
                    win = fire_site.copy()
                    y_label = "Unknown"

                if win.empty:
                    continue

                # rasterize burned polygons to SDR grid
                burn = rasterize(
                    [(geom, 1) for geom in win.geometry if geom is not None and not geom.is_empty],
                    out_shape=sdr.shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8"
                ).astype(bool)

                burned_sdr = sdr[burn & valid]
                if burned_sdr.size == 0:
                    continue

                sorted_sdr = np.sort(burned_sdr)
                cum_counts = np.arange(1, len(sorted_sdr) + 1, dtype=float)
                n_site = int(valid.sum())
                if n_site == 0:
                    continue
                cum_pct_site = (cum_counts / n_site) * 100.0
                cum_frac_site = cum_pct_site / 100.0

                if year is not None:
                    auc_norm = _auc_normalised(sorted_sdr, cum_frac_site, sdr_min, sdr_max)
                    exposure_val = 1.0 - auc_norm
                    sites.loc[idx, f"AUC_SDR_{y_label}"] = auc_norm
                    sites.loc[idx, f"Exposure_SDR_{y_label}"] = exposure_val
                    sdr_auc_ts[year] = exposure_val
                    sdr_line_profiles.append((year, sorted_sdr, cum_pct_site))
                    _log("OK", f"Site {sid}: Bushfire SDR profile processed for {year}")

            # ---- Combined SDR plot: all years on one figure ----
            if cfg.make_plots and sdr_line_profiles:
                fig, ax = plt.subplots(figsize=(8, 5))
                try:
                    cmap = plt.cm.get_cmap("tab20", max(len(sdr_line_profiles), 1))
                    ymax = 0.0

                    for j, (year_int, xvals, yvals) in enumerate(sorted(sdr_line_profiles, key=lambda t: t[0])):
                        ax.plot(
                            xvals,
                            yvals,
                            linewidth=1.4,
                            color=cmap(j),
                            label=str(year_int)
                        )
                        if len(yvals):
                            ymax = max(ymax, float(np.nanmax(yvals)))

                    ax.set_xlabel("SDR Value", fontsize=10, fontweight="bold")
                    ax.set_ylabel("Cumulative burned area (% of site)", fontsize=10, fontweight="bold")
                    ax.set_title(f"Bushfire Exposure Profile (Burned vs SDR) - Site {sid}", fontsize=11, fontweight="bold")
                    ax.tick_params(axis="both", labelsize=10)
                    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.set_xlim(sdr_min, sdr_max)

                    if ymax > 0:
                        ax.set_ylim(0, ymax * 1.05)

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
                    out_sdr_plot = os.path.join(plot_dir, f"Site_{sid}_Bushfire_Risk_SDR.png")
                    fig.savefig(out_sdr_plot, dpi=300, bbox_inches="tight")
                    _log("OK", f"Site {sid}: combined SDR risk plot saved")
                finally:
                    plt.close(fig)

            # ---------------- TWI loop (iterate over SDR years) ----------------
            if os.path.exists(twi_path) and sdr_years:
                with rio.open(twi_path) as src_twi:
                    timg, ttransform = rio_mask(src_twi, site_geom.geometry, crop=True)
                    twi = timg[0].astype(float)
                    if src_twi.nodata is not None:
                        twi[twi == src_twi.nodata] = np.nan
                    valid_twi = ~np.isnan(twi)

                if not fire_site.empty:
                    for y in sdr_years:
                        win = fire_site[
                            (fire_site["ignition_date"].dt.year >= (y - (cfg.window_years - 1))) &
                            (fire_site["ignition_date"].dt.year <= y)
                        ]
                        if win.empty:
                            continue

                        burn_twi = rasterize(
                            [(geom, 1) for geom in win.geometry if geom is not None and not geom.is_empty],
                            out_shape=twi.shape,
                            transform=ttransform,
                            fill=0,
                            dtype="uint8"
                        ).astype(bool)

                        twi_burn = twi[burn_twi & valid_twi]
                        if twi_burn.size == 0:
                            continue

                        sorted_twi = np.sort(twi_burn)
                        cum_counts = np.arange(1, len(sorted_twi) + 1, dtype=float)
                        n_site = int(valid_twi.sum())
                        if n_site == 0:
                            continue
                        cum_pct_site = (cum_counts / n_site) * 100.0
                        cum_frac_site = cum_pct_site / 100.0

                        auc_norm_twi = _auc_normalised(sorted_twi, cum_frac_site, twi_min, twi_max)
                        exposure_val_twi = 1.0 - auc_norm_twi
                        sites.loc[idx, f"AUC_TWI_{y}"] = auc_norm_twi
                        sites.loc[idx, f"Exposure_TWI_{y}"] = exposure_val_twi
                        twi_auc_ts[y] = exposure_val_twi
                        twi_line_profiles.append((y, sorted_twi, cum_pct_site))
                        _log("OK", f"Site {sid}: Bushfire TWI profile processed for {y}")

            # ---- Combined TWI plot: all years on one figure ----
            if cfg.make_plots and twi_line_profiles:
                fig, ax = plt.subplots(figsize=(8, 5))
                try:
                    cmap = plt.cm.get_cmap("tab20", max(len(twi_line_profiles), 1))
                    ymax = 0.0

                    for j, (year_int, xvals, yvals) in enumerate(sorted(twi_line_profiles, key=lambda t: t[0])):
                        ax.plot(
                            xvals,
                            yvals,
                            linewidth=1.4,
                            color=cmap(j),
                            label=str(year_int)
                        )
                        if len(yvals):
                            ymax = max(ymax, float(np.nanmax(yvals)))

                    ax.set_xlabel("Topographic Wetness Index", fontsize=10, fontweight="bold")
                    ax.set_ylabel("Cumulative burned area (% of site)", fontsize=10, fontweight="bold")
                    ax.set_title(f"Bushfire Exposure Profile (Burned vs TWI) - Site {sid}", fontsize=11, fontweight="bold")
                    ax.tick_params(axis="both", labelsize=10)
                    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.set_xlim(twi_min, twi_max)

                    if ymax > 0:
                        ax.set_ylim(0, ymax * 1.05)

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
                    out_twi_plot = os.path.join(plot_dir, f"Site_{sid}_Bushfire_Risk_TWI.png")
                    fig.savefig(out_twi_plot, dpi=300, bbox_inches="tight")
                    _log("OK", f"Site {sid}: combined TWI risk plot saved")
                finally:
                    plt.close(fig)

            # ---------- per-site Exposure time-series summary ----------
            if cfg.make_plots and (sdr_auc_ts or twi_auc_ts):
                years_all = sorted(set(list(sdr_auc_ts.keys()) + list(twi_auc_ts.keys())))
                sdr_y = [sdr_auc_ts.get(y, np.nan) for y in years_all]
                twi_y = [twi_auc_ts.get(y, np.nan) for y in years_all]

                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
                try:
                    axes[0].plot(years_all, sdr_y, marker="o", color="black", linewidth=1.3)
                    axes[0].set_ylabel("Exposure (1 − AUC)", fontsize=10, fontweight="bold")
                    axes[0].set_title(f"Bushfire Exposure (rolling {cfg.window_years}-yr window) — Site {sid}", fontsize=11, fontweight="bold")
                    axes[0].tick_params(axis="both", labelsize=10)
                    axes[0].ticklabel_format(style="plain", axis="both", useOffset=False)
                    axes[0].grid(True, linestyle="--", alpha=0.6)
                    axes[0].text(0.01, 0.92, "SDR", transform=axes[0].transAxes, fontsize=10, fontweight="bold")

                    axes[1].plot(years_all, twi_y, marker="o", color="black", linewidth=1.3)
                    axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
                    axes[1].set_ylabel("Exposure (1 − AUC)", fontsize=10, fontweight="bold")
                    axes[1].tick_params(axis="both", labelsize=10)
                    axes[1].ticklabel_format(style="plain", axis="both", useOffset=False)
                    axes[1].grid(True, linestyle="--", alpha=0.6)
                    axes[1].text(0.01, 0.92, "TWI", transform=axes[1].transAxes, fontsize=10, fontweight="bold")

                    plt.tight_layout()
                    out_auc_plot = os.path.join(plot_dir, f"Site_{sid}_Exposure_Bushfire.png")
                    plt.savefig(out_auc_plot, dpi=300, bbox_inches="tight")
                    _log("OK", f"Site {sid}: exposure time-series plot saved")
                finally:
                    plt.close(fig)

            _log("OK", f"Site {sid} completed")

        except Exception as e:
            _log("WARN", f"Site {site.get('Site_id', idx)} failed: {e}")

    """
    # ---- write back amended sites GPKG (AUC columns) ----
    try:
        import fiona
        layers = fiona.listlayers(all_sites_gpkg)
        layer = layers[0] if layers else None
        if layer:
            sites.to_file(all_sites_gpkg, layer=layer, driver="GPKG", if_exists="replace")
        else:
            sites.to_file(all_sites_gpkg, driver="GPKG")
        _log("OK", f"Updated exposure metrics written to: {all_sites_gpkg}")
    except Exception as e:
        _log("WARN", f"Could not write updated sites GPKG: {e}")
    """

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del catch_gdf, catch_wgs
    except Exception:
        pass

    try:
        del fire_all, fire_clip, fire_site, win
    except Exception:
        pass

    try:
        del sites, site_geom
    except Exception:
        pass

    try:
        del sdr, twi, burn, burn_twi
    except Exception:
        pass

    try:
        del burned_sdr, twi_burn, sorted_sdr, sorted_twi
    except Exception:
        pass

    try:
        del cum_counts, cum_pct_site, cum_frac_site
    except Exception:
        pass

    try:
        del sdr_auc_ts, twi_auc_ts
    except Exception:
        pass

    try:
        del sdr_line_profiles, twi_line_profiles
    except Exception:
        pass

    try:
        del years_all, sdr_y, twi_y
    except Exception:
        pass

    gc.collect()

    _log("INFO", "Memory cleanup completed.")
    _log("OK", "Historical bushfire processing completed")

    return