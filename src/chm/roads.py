# src/chm/roads_hist.py
from __future__ import annotations

# ---------- stdlib ----------
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- third-party ----------
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt


# ========================= Config =========================

@dataclass
class RoadsConfig:
    chm_workspace: str
    catchment_path: str

    # IO
    sites_gpkg: Optional[str] = None
    road_input_path: Optional[str] = None
    road_url: str = (
        "https://services-ap1.arcgis.com/ypkPEy1AmwPKGNNv/ArcGIS/rest/services/"
        "National_Roads/FeatureServer/0/query"
    )

    # Dependencies inside the catchment datasets
    sdr_dirname: str = os.path.join("Surface and Groundwater Connectivity", "SDR")
    twi_relpath: str = os.path.join("Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")

    # Behaviour
    make_plots: bool = True
    requests_timeout: int = 120
    max_arcgis_page: int = 2000  # server-side pagination page size


# ========================= Helpers =========================

def _ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str) -> Tuple[str, str, str, str, str]:
    name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    base = os.path.join(workspace, name)
    ds   = os.path.join(base, "Catchment Datasets")
    sites_ds = os.path.join(base, "Sites Datasets")
    plots = os.path.join(base, "Sites Plots and Maps")
    roads_out = os.path.join(ds, "National Roads")
    _ensure_dirs([base, ds, sites_ds, plots, roads_out])
    return name, ds, sites_ds, plots, roads_out


def _arcgis_paginated_geojson(url: str, bbox4326: Tuple[float, float, float, float],
                              timeout: int, pagesize: int) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox4326
    params = {
        "f": "geojson",
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": 0,
        "resultRecordCount": pagesize,
    }
    frames: List[gpd.GeoDataFrame] = []
    while True:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        if len(feats) < pagesize:  # last page
            break
        params["resultOffset"] += pagesize
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def _scan_sdr_range(sdr_dir: str, catch_gdf: gpd.GeoDataFrame) -> Tuple[float, float, List[int]]:
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

def national_roads(cfg: RoadsConfig) -> Tuple[str, str]:
    """
    For each site:
      • Clips national roads to site
      • For each SDR year, computes cumulative road area (as % of site) vs SDR,
        normalises AUC to fixed catchment SDR range, and plots
      • For TWI (single raster), plots cumulative vs TWI
      • Writes AUC columns back to sites GPKG
    Returns (all_sites_gpkg, sites_datasets)
    """
    print("Processing national road data...")

    # Folders / paths
    name, catch_ds, sites_ds, plots_ds, roads_out = _scaffold(cfg.chm_workspace, cfg.catchment_path)
    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_ds, f"{name} Sites Data.gpkg")
    sdr_dir = os.path.join(catch_ds, cfg.sdr_dirname)
    twi_path = cfg.twi_relpath if os.path.isabs(cfg.twi_relpath) else os.path.join(catch_ds, cfg.twi_relpath)

    # Catchment (project + WGS84)
    catch = gpd.read_file(cfg.catchment_path)
    catch_wgs = catch.to_crs(epsg=4326)

    # Roads source (local or ArcGIS)
    if cfg.road_input_path and os.path.exists(cfg.road_input_path):
        roads = gpd.read_file(cfg.road_input_path)
        if roads.crs is None:
            roads = roads.set_crs(catch_wgs.crs, allow_override=True)
        elif roads.crs != catch_wgs.crs:
            roads = roads.to_crs(catch_wgs.crs)
    else:
        minx, miny, maxx, maxy = catch_wgs.total_bounds
        roads = _arcgis_paginated_geojson(cfg.road_url, (minx, miny, maxx, maxy),
                                          timeout=cfg.requests_timeout, pagesize=cfg.max_arcgis_page)

    # Clip to catchment & save
    roads_clip = gpd.clip(roads, catch_wgs).to_crs(catch.crs)
    roads_clip.to_file(os.path.join(roads_out, "National_Roads_in_Catchment.gpkg"), driver="GPKG")
    roads_clip.drop(columns="geometry").to_csv(os.path.join(roads_out, "National_Roads_Attributes.csv"), index=False)

    # Fixed axes ranges + SDR years
    sdr_min, sdr_max, sdr_years = _scan_sdr_range(sdr_dir, catch)
    twi_min, twi_max = _twi_range(twi_path, catch)

    # Sites
    try:
        sites = gpd.read_file(all_sites_gpkg)
    except Exception as e:
        print(f"[WARN] Could not read sites GPKG: {e}")
        return all_sites_gpkg, sites_ds

    # Per-site
    for i, site in sites.iterrows():
        try:
            sid = site.get("Site_id", i)
            site_gdf = gpd.GeoDataFrame(geometry=[site.geometry], crs=sites.crs)
            site_dir = os.path.join(sites_ds, f"Site_{sid}")
            plot_dir = os.path.join(plots_ds, f"Site_{sid}")
            _ensure_dirs([site_dir, plot_dir])

            site_roads = gpd.clip(roads_clip, site_gdf)
            sdr_auc_ts: Dict[int, float] = {}

            # ---- SDR loop ----
            for fname in os.listdir(sdr_dir):
                if not (fname.endswith(".tif") and "SDR" in fname):
                    continue
                fpath = os.path.join(sdr_dir, fname)
                with rio.open(fpath) as src:
                    img, transform = rio_mask(src, site_gdf.geometry, crop=True)
                    sdr = img[0].astype(float)
                    if src.nodata is not None:
                        sdr[sdr == src.nodata] = np.nan
                    valid = ~np.isnan(sdr)

                if site_roads.empty or valid.sum() == 0:
                    continue

                road_ras = rasterize(
                    [(geom, 1) for geom in site_roads.geometry if geom is not None and not geom.is_empty],
                    out_shape=sdr.shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8"
                ).astype(bool)

                sdr_on_road = sdr[road_ras & valid]
                if sdr_on_road.size == 0:
                    continue

                sorted_vals = np.sort(sdr_on_road)
                cum_counts = np.arange(1, len(sorted_vals) + 1, dtype=float)
                N_site = int(valid.sum())
                cum_pct_site = (cum_counts / N_site) * 100.0
                cum_frac_site = cum_counts / N_site

                m = re.search(r"\d{4}", fname)
                y_label = m.group(0) if m else "Unknown"
                if m:
                    sdr_auc_ts[int(y_label)] = _auc_normalised(sorted_vals, cum_frac_site, sdr_min, sdr_max)
                    sites.loc[i, f"AUC_Road_SDR_{y_label}"] = sdr_auc_ts[int(y_label)]

                if cfg.make_plots:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    try:
                        ax.plot(sorted_vals, cum_pct_site, color="black", linewidth=1.3)
                        ax.set_xlabel("SDR value", fontsize=10, fontweight="bold")
                        ax.set_ylabel("Cumulative road area (% of site)", fontsize=10, fontweight="bold")
                        ax.set_title(f"Road Exposure vs SDR ({y_label}) — Site {sid}", fontsize=11, fontweight="bold")
                        ax.tick_params(axis="both", labelsize=10)
                        ax.set_xlim(sdr_min, sdr_max)
                        ax.grid(True, linestyle="--", alpha=0.6)
                        fig.tight_layout()
                        fig.savefig(os.path.join(plot_dir, f"Site_{sid}_Roads_Risk_SDR_{y_label}.png"), dpi=300)
                    finally:
                        plt.close(fig)

            # ---- TWI (single raster) ----
            if os.path.exists(twi_path):
                with rio.open(twi_path) as src:
                    img, transform = rio_mask(src, site_gdf.geometry, crop=True)
                    twi = img[0].astype(float)
                    if src.nodata is not None:
                        twi[twi == src.nodata] = np.nan
                    valid_twi = ~np.isnan(twi)

                if not site_roads.empty and valid_twi.sum() > 0:
                    road_twi = rasterize(
                        [(geom, 1) for geom in site_roads.geometry if geom is not None and not geom.is_empty],
                        out_shape=twi.shape,
                        transform=transform,
                        fill=0,
                        dtype="uint8"
                    ).astype(bool)

                    vals = twi[road_twi & valid_twi]
                    if vals.size > 0:
                        sorted_vals = np.sort(vals)
                        cum_counts = np.arange(1, len(sorted_vals) + 1, dtype=float)
                        N_site = int(valid_twi.sum())
                        cum_pct_site = (cum_counts / N_site) * 100.0

                        if cfg.make_plots:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            try:
                                ax.plot(sorted_vals, cum_pct_site, color="black", linewidth=1.3)
                                ax.set_xlabel("Topographic Wetness Index", fontsize=10, fontweight="bold")
                                ax.set_ylabel("Cumulative road area (% of site)", fontsize=10, fontweight="bold")
                                ax.set_title(f"Road Exposure vs TWI — Site {sid}", fontsize=11, fontweight="bold")
                                ax.tick_params(axis="both", labelsize=10)
                                ax.set_xlim(sdr_min, sdr_max)
                                ax.grid(True, linestyle="--", alpha=0.6)
                                fig.tight_layout()
                                fig.savefig(os.path.join(plot_dir, f"Site_{sid}_Roads_Risk_TWI_{y_label}.png"), dpi=300)
                            finally:
                                plt.close(fig)

            # ---- AUC time-series plot ----
            if cfg.make_plots and sdr_auc_ts:
                years = sorted(sdr_auc_ts.keys())
                series = [sdr_auc_ts[y] for y in years]
                plt.figure(figsize=(6, 4))
                plt.plot(years, series, marker="o", color="black", linewidth=1.3)
                plt.xlabel("Year", fontsize=10, fontweight="bold")
                plt.ylabel("AUC (normalised)", fontsize=10, fontweight="bold")
                plt.title(f"Road Exposure — AUC time series (SDR) — Site {sid}", fontsize=11, fontweight="bold")
                ax.tick_params(axis="both", labelsize=10)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"Site_{sid}_Road_AUC_TimeSeries.png"), dpi=300)
                plt.close()
            print(f"Site {sid} is processed")
        except Exception as e:
            print(f"[WARN] Site {site.get('Site_id', i)} failed: {e}")
    """
    # Write back amended sites GPKG
    try:
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
    return all_sites_gpkg, sites_ds