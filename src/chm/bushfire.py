# src/chm/bushfire_hist.py
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
        data = resp.json()
        feats = data.get("features", [])
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
        # default to WGS84 if source had no CRS
        if g.crs is None:
            g = g.set_crs(gdf_wgs84.crs, allow_override=True)
        elif g.crs != gdf_wgs84.crs:
            g = g.to_crs(gdf_wgs84.crs)
        return g
    # download via ArcGIS REST
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    return _arcgis_paginated_geojson(
        cfg.bushfire_url, (minx, miny, maxx, maxy), timeout=cfg.requests_timeout, pagesize=cfg.max_arcgis_page
    )


def _coerce_ignition_date(df: pd.DataFrame) -> pd.Series:
    """
    Try to coerce ignition date column to pandas datetime.
    Handles common arcgis epoch (ms) or ISO strings.
    """
    # try common column names
    cand = [c for c in df.columns if c.lower() in ("ignition_date", "ignitiondate", "date", "start_date")]
    if not cand:
        # fallback: try to detect an epoch-like numeric column
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
        if not (fname.endswith(".tif") and "SDR" in fname):
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
      • append AUC columns back into the sites GPKG
      • export per-site plots and GPKG of site-clipped bushfires

    Returns
    -------
    (all_sites_gpkg, sites_datasets)
    """
    print("Processing bushfire data...")

    # ---- folders / names ----
    (catchment_name, _, catch_datasets, catch_plots, sites_datasets, sites_plots) = \
        _scaffold(cfg.chm_workspace, cfg.catchment_path)
    bushfire_output = os.path.join(catch_datasets, "Historical Bushfire")

    # ---- paths ----
    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")
    sdr_dir = os.path.join(catch_datasets, cfg.sdr_dirname)
    twi_path = cfg.twi_relpath if os.path.isabs(cfg.twi_relpath) else os.path.join(catch_datasets, cfg.twi_relpath)

    # ---- catchment geometry (WGS84 & project CRS) ----
    catch_gdf = gpd.read_file(cfg.catchment_path)
    catch_crs = catch_gdf.crs
    catch_wgs = catch_gdf.to_crs(epsg=4326)

    # ---- bushfire source (local or ArcGIS REST) ----
    fire_all = _read_or_download_bushfire(cfg, catch_wgs)
    if fire_all.empty:
        print("[INFO] No bushfire features downloaded/found. Exiting early.")
        return all_sites_gpkg, sites_datasets

    # clip to catchment polygon (in WGS84), then back to project CRS
    fire_clip = gpd.clip(fire_all, catch_wgs).to_crs(catch_crs)

    # ignition date coercion (handles epoch ms or ISO)
    fire_clip["ignition_date"] = _coerce_ignition_date(fire_clip)

    # save full catchment bushfire layers
    fire_gpkg = os.path.join(bushfire_output, "Historical_Bushfires_Boundary.gpkg")
    fire_clip.to_file(fire_gpkg, driver="GPKG")
    fire_clip.drop(columns="geometry").to_csv(os.path.join(bushfire_output, "Historical_Bushfires_Attributes.csv"),
                                              index=False)

    # ---- SDR/TWI ranges for fixed x-axes ----
    sdr_min, sdr_max, sdr_years = _scan_sdr_range(sdr_dir, catch_gdf)
    twi_min, twi_max = _twi_range(twi_path, catch_gdf)

    # ---- load sites ----
    try:
        sites = gpd.read_file(all_sites_gpkg)
    except Exception as e:
        print(f"[WARN] Could not read sites GPKG: {e}")
        return all_sites_gpkg, sites_datasets

    # ================= per-site processing =================
    for idx, site in sites.iterrows():
        try:
            sid = site.get("Site_id", idx)
            site_geom = gpd.GeoDataFrame(geometry=[site.geometry], crs=sites.crs)

            site_dir = os.path.join(sites_datasets, f"Site_{sid}")
            plot_dir = os.path.join(sites_plots, f"Site_{sid}")
            _ensure_dirs([site_dir, plot_dir])
            # site-level bushfire polygons
            fire_site = gpd.clip(fire_clip, site_geom)
            if not fire_site.empty:
                fire_site.to_file(os.path.join(site_dir, f"Site_{sid}_Bushfires.gpkg"), driver="GPKG")

            sdr_auc_ts: Dict[int, float] = {}
            twi_auc_ts: Dict[int, float] = {}

            # ---------------- SDR loop (per-year rasters) ----------------
            for fname in os.listdir(sdr_dir):
                if not (fname.endswith(".tif") and "SDR" in fname):
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

                # cumulative % of SITE area (not just burned area)
                sorted_sdr = np.sort(burned_sdr)
                cum_counts = np.arange(1, len(sorted_sdr) + 1, dtype=float)
                N_site = int(valid.sum())
                if N_site == 0:
                    continue
                cum_pct_site = (cum_counts / N_site) * 100.0
                cum_frac_site = cum_counts / N_site

                if cfg.make_plots:
                    plt.figure(figsize=(6, 4))
                    plt.plot(sorted_sdr, cum_pct_site, color="black", linewidth=1.3)
                    plt.xlabel("SDR value", fontsize=10, fontweight="bold")
                    plt.ylabel("Cumulative burned area (% of site)", fontsize=10, fontweight="bold")
                    plt.title(f"Bushfire Exposure: Burned vs SDR ({y_label}) — Site {sid}", fontsize=11, fontweight="bold")
                    plt.xlim(sdr_min, sdr_max)
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"Site_{sid}_Bushfire_Risk_{os.path.splitext(fname)[0]}.png"), dpi=300)
                    plt.close()

                # AUC normalised to fixed [sdr_min, sdr_max]
                auc_norm = _auc_normalised(sorted_sdr, cum_frac_site, sdr_min, sdr_max)
                sites.loc[idx, f"AUC_SDR_{y_label}"] = auc_norm
                if year is not None:
                    sdr_auc_ts[year] = auc_norm

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
                        N_site = int(valid_twi.sum())
                        if N_site == 0:
                            continue
                        cum_pct_site = (cum_counts / N_site) * 100.0
                        cum_frac_site = cum_counts / N_site

                        if cfg.make_plots:
                            plt.figure(figsize=(6, 4))
                            plt.plot(sorted_twi, cum_pct_site,color="black", linewidth=1.3)
                            plt.xlabel("Topographic Wetness Index", fontsize=10, fontweight="bold")
                            plt.ylabel("Cumulative burned area (% of site)", fontsize=10, fontweight="bold")
                            plt.title(f"Bushfire Exposure: Burned vs TWI ({y}) — Site {sid}", fontsize=11, fontweight="bold")
                            plt.xlim(twi_min, twi_max)
                            plt.grid(True, linestyle="--", alpha=0.6)
                            plt.tight_layout()
                            plt.savefig(os.path.join(plot_dir, f"Site_{sid}_Bushfire_Risk_TWI_{y}.png"), dpi=300)
                            plt.close()

                        auc_norm_twi = _auc_normalised(sorted_twi, cum_frac_site, twi_min, twi_max)
                        sites.loc[idx, f"AUC_TWI_{y}"] = auc_norm_twi
                        twi_auc_ts[y] = auc_norm_twi

            # ---------- per-site AUC time-series summary ----------
            if cfg.make_plots and (sdr_auc_ts or twi_auc_ts):
                years_all = sorted(set(list(sdr_auc_ts.keys()) + list(twi_auc_ts.keys())))
                sdr_y = [sdr_auc_ts.get(y, np.nan) for y in years_all]
                twi_y = [twi_auc_ts.get(y, np.nan) for y in years_all]
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
                axes[0].plot(years_all, sdr_y, marker="o", color="black",linewidth=1.3)
                axes[0].set_ylabel("AUC (normalised)", fontsize=10, fontweight="bold")
                axes[0].set_title(f"Bushfire Exposure (rolling {cfg.window_years}-yr window) — Site {sid}", fontsize=11, fontweight="bold")
                axes[0].grid(True, linestyle="--", alpha=0.6)
                axes[0].text(0.01, 0.92, "SDR", transform=axes[0].transAxes, fontsize=10, fontweight="bold")

                axes[1].plot(years_all, twi_y, marker="o", color="black",linewidth=1.3)
                axes[1].set_xlabel("Year", fontsize=10, fontweight="bold")
                axes[1].set_ylabel("AUC (normalised)", fontsize=10, fontweight="bold")
                axes[1].grid(True, linestyle="--", alpha=0.6)
                axes[1].text(0.01, 0.92, "TWI", transform=axes[1].transAxes, fontsize=10, fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"Site_{sid}_AUC_Bushfire.png"), dpi=300, bbox_inches="tight")
                plt.close()
            print(f"Site {sid} is processed")
        except Exception as e:
            print(f"[WARN] Site {site.get('Site_id', idx)} failed: {e}")
    """
    # ---- write back amended sites GPKG (AUC columns) ----
    try:
        # choose the original layer name if present
        import fiona
        layers = fiona.listlayers(all_sites_gpkg)
        layer = layers[0] if layers else None
        if layer:
            sites.to_file(all_sites_gpkg, layer=layer, driver="GPKG", if_exists="replace")
        else:
            sites.to_file(all_sites_gpkg, driver="GPKG")
        print(f"[OK] Updated AUC metrics written to: {all_sites_gpkg}")
    except Exception as e:
        print(f"[WARN] Could not write updated sites GPKG: {e}")
    """
    print("Done!")
    return all_sites_gpkg, sites_datasets