from __future__ import annotations

# ---------- stdlib ----------
import os
import re
import gc
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

def _log(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str) -> Tuple[str, str, str, str, str]:
    name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    base = os.path.join(workspace, name)
    ds = os.path.join(base, "Catchment Datasets")
    sites_ds = os.path.join(base, "Sites Datasets")
    plots = os.path.join(base, "Sites Plots and Maps")
    roads_out = os.path.join(ds, "National Roads")
    _ensure_dirs([base, ds, sites_ds, plots, roads_out])
    return name, ds, sites_ds, plots, roads_out


def _arcgis_paginated_geojson(
    url: str,
    bbox4326: Tuple[float, float, float, float],
    timeout: int,
    pagesize: int
) -> gpd.GeoDataFrame:
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
        if len(feats) < pagesize:
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

def national_roads(cfg: RoadsConfig) -> Tuple[str, str]:
    """
    For each site (when sites GPKG exists):
      • Clips national roads to site
      • For each SDR year, computes cumulative road area (as % of site) vs SDR
      • Plots all SDR years on one combined Roads_Risk_SDR plot per site
      • Computes normalised AUC over fixed catchment SDR range
      • Plots Exposure (1 − AUC) time series per site
      • For TWI (single raster), plots cumulative vs TWI

    If all_sites_gpkg does not exist, the module still runs:
      • Roads are downloaded/read
      • Roads are clipped to catchment and saved
      • Catchment-wide ranges are scanned
      • Site-specific outputs are skipped safely

    Returns
    -------
    all_sites_gpkg : str
    sites_datasets : str
    """
    _log("INFO", "Starting national roads processing...")

    # Folders / paths
    name, catch_ds, sites_ds, plots_ds, roads_out = _scaffold(cfg.chm_workspace, cfg.catchment_path)
    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_ds, f"{name} Sites Data.gpkg")
    sdr_dir = os.path.join(catch_ds, cfg.sdr_dirname)
    twi_path = cfg.twi_relpath if os.path.isabs(cfg.twi_relpath) else os.path.join(catch_ds, cfg.twi_relpath)

    _log("OK", f"Workspace ready for catchment: {name}")
    _log("INFO", f"Road outputs folder: {roads_out}")

    # Catchment (project + WGS84)
    catch = gpd.read_file(cfg.catchment_path)
    catch_wgs = catch.to_crs(epsg=4326)
    _log("OK", f"Catchment loaded: {cfg.catchment_path}")

    # Roads source (local or ArcGIS)
    if cfg.road_input_path and os.path.exists(cfg.road_input_path):
        roads = gpd.read_file(cfg.road_input_path)
        if roads.crs is None:
            roads = roads.set_crs(catch_wgs.crs, allow_override=True)
        elif roads.crs != catch_wgs.crs:
            roads = roads.to_crs(catch_wgs.crs)
        _log("OK", f"Roads loaded from local file: {cfg.road_input_path}")
    else:
        minx, miny, maxx, maxy = catch_wgs.total_bounds
        roads = _arcgis_paginated_geojson(
            cfg.road_url,
            (minx, miny, maxx, maxy),
            timeout=cfg.requests_timeout,
            pagesize=cfg.max_arcgis_page
        )
        _log("OK", f"Roads downloaded from ArcGIS service. Features retrieved: {len(roads)}")

    # Clip to catchment & save
    roads_clip = gpd.clip(roads, catch_wgs).to_crs(catch.crs)
    roads_clip_gpkg = os.path.join(roads_out, "National_Roads_in_Catchment.gpkg")
    roads_clip_csv = os.path.join(roads_out, "National_Roads_Attributes.csv")
    roads_clip.to_file(roads_clip_gpkg, driver="GPKG")
    roads_clip.drop(columns="geometry").to_csv(roads_clip_csv, index=False)
    _log("OK", f"Catchment-clipped roads saved: {roads_clip_gpkg}")
    _log("OK", f"Road attributes CSV saved: {roads_clip_csv}")

    # Fixed axes ranges + SDR years
    sdr_min, sdr_max, sdr_years = _scan_sdr_range(sdr_dir, catch)
    twi_min, twi_max = _twi_range(twi_path, catch)
    _log("INFO", f"Catchment SDR range: {sdr_min:.6f} to {sdr_max:.6f}")
    _log("INFO", f"Catchment TWI range: {twi_min:.6f} to {twi_max:.6f}")
    _log("INFO", f"SDR years found: {sdr_years}")

    # Sites (optional)
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
        _log("OK", "National roads catchment tasks completed. No site-specific processing was run.")
        return all_sites_gpkg, sites_ds

    # Per-site
    for i, site in sites.iterrows():
        try:
            sid = site.get("Site_id", i)
            _log("INFO", f"Processing Site {sid}...")

            site_gdf = gpd.GeoDataFrame(geometry=[site.geometry], crs=sites.crs)
            site_dir = os.path.join(sites_ds, f"Site_{sid}")
            plot_dir = os.path.join(plots_ds, f"Site_{sid}")
            _ensure_dirs([site_dir, plot_dir])

            site_roads = gpd.clip(roads_clip, site_gdf)
            _log("OK", f"Site {sid}: roads clipped")

            sdr_auc_ts: Dict[int, float] = {}
            exposure_ts: Dict[int, float] = {}
            sdr_line_profiles: List[Tuple[int, np.ndarray, np.ndarray]] = []

            # ---- SDR loop ----
            for fname in os.listdir(sdr_dir):
                if not (fname.endswith(".tif") and re.match(r"SDR_\d{4}\.tif$", fname)):
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
                cum_frac_site = cum_pct_site / 100.0

                m = re.search(r"\d{4}", fname)
                if m:
                    y_label = int(m.group(0))

                    if np.isfinite(sdr_min) and np.isfinite(sdr_max) and sdr_max > sdr_min:
                        auc_val = _auc_normalised(sorted_vals, cum_frac_site, sdr_min, sdr_max)
                        exposure_val = 1.0 - auc_val
                    else:
                        auc_val = np.nan
                        exposure_val = np.nan

                    sdr_auc_ts[y_label] = auc_val
                    exposure_ts[y_label] = exposure_val
                    sdr_line_profiles.append((y_label, sorted_vals, cum_pct_site))

                    # Optional attributes if user later writes them back
                    sites.loc[i, f"AUC_Road_SDR_{y_label}"] = auc_val
                    sites.loc[i, f"Exposure_Road_SDR_{y_label}"] = exposure_val

                    _log("OK", f"Site {sid}: SDR profile processed for year {y_label}")
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
                    ax.set_ylabel("Cumulative road area (% of site)", fontsize=10, fontweight="bold")
                    ax.set_title(f"Road Exposure Profile (Roads vs SDR) - Site {sid}", fontsize=11, fontweight="bold")
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
                    out_sdr_plot = os.path.join(plot_dir, f"Site_{sid}_Roads_Risk_SDR.png")
                    fig.savefig(out_sdr_plot, dpi=300, bbox_inches="tight")
                    _log("OK", f"Site {sid}: combined SDR risk plot saved")
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
                    if vals.size > 0 and cfg.make_plots:
                        sorted_vals = np.sort(vals)
                        cum_counts = np.arange(1, len(sorted_vals) + 1, dtype=float)
                        n_site = int(valid_twi.sum())
                        cum_pct_site = (cum_counts / n_site) * 100.0

                        fig, ax = plt.subplots(figsize=(6, 4))
                        try:
                            ax.plot(sorted_vals, cum_pct_site, color="black", linewidth=1.3)
                            ax.set_xlabel("Topographic Wetness Index", fontsize=10, fontweight="bold")
                            ax.set_ylabel("Cumulative road area (% of site)", fontsize=10, fontweight="bold")
                            ax.set_title(f"Road Exposure vs TWI — Site {sid}", fontsize=11, fontweight="bold")
                            ax.tick_params(axis="both", labelsize=10)
                            ax.set_xlim(twi_min, twi_max)
                            ax.set_ylim(0, 100)
                            ax.grid(True, linestyle="--", alpha=0.6)
                            fig.tight_layout()

                            out_twi_plot = os.path.join(plot_dir, f"Site_{sid}_Roads_Risk_TWI.png")
                            fig.savefig(out_twi_plot, dpi=300)
                            _log("OK", f"Site {sid}: TWI risk plot saved")
                        finally:
                            plt.close(fig)

            # ---- Exposure time-series plot: Exposure = 1 - AUC ----
            if cfg.make_plots and exposure_ts:
                years = sorted(exposure_ts.keys())
                series = [exposure_ts[y] for y in years]

                fig, ax = plt.subplots(figsize=(6, 4))
                try:
                    ax.plot(years, series, marker="o", color="black", linewidth=1.3)
                    ax.set_xlabel("Year", fontsize=10, fontweight="bold")
                    ax.set_ylabel("Exposure (1 − AUC)", fontsize=10, fontweight="bold")
                    ax.set_title(f"Road Exposure — Time Series (SDR) — Site {sid}", fontsize=11, fontweight="bold")
                    ax.tick_params(axis="both", labelsize=10)
                    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    fig.tight_layout()

                    out_auc_plot = os.path.join(plot_dir, f"Site_{sid}_Road_Exposure_TimeSeries.png")
                    fig.savefig(out_auc_plot, dpi=300)
                    _log("OK", f"Site {sid}: exposure time-series plot saved")
                finally:
                    plt.close(fig)
            # Clean per-site arrays/objects
            plt.close("all")

            try:
                del site_gdf, site_roads
            except Exception:
                pass

            try:
                del sdr, twi, road_ras, road_twi
            except Exception:
                pass

            try:
                del sorted_vals, cum_counts, cum_pct_site, cum_frac_site
            except Exception:
                pass

            try:
                del sdr_auc_ts, exposure_ts, sdr_line_profiles
            except Exception:
                pass

            gc.collect()
            _log("OK", f"Site {sid} completed")

        except Exception as e:
            _log("WARN", f"Site {site.get('Site_id', i)} failed: {e}")

    # Optional write-back kept deactivated by design
    """
    try:
        import fiona
        layers = fiona.listlayers(all_sites_gpkg)
        layer = layers[0] if layers else None
        if layer:
            sites.to_file(all_sites_gpkg, layer=layer, driver="GPKG", if_exists="replace")
        else:
            sites.to_file(all_sites_gpkg, driver="GPKG")
        _log("OK", f"Updated site metrics written to: {all_sites_gpkg}")
    except Exception as e:
        _log("WARN", f"Could not write updated sites GPKG: {e}")
    """

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del catch, catch_wgs, roads, roads_clip
    except Exception:
        pass

    try:
        del sites, site_gdf, site_roads
    except Exception:
        pass

    try:
        del sdr, twi, road_ras, road_twi
    except Exception:
        pass

    try:
        del sorted_vals, cum_counts, cum_pct_site, cum_frac_site
    except Exception:
        pass

    try:
        del sdr_auc_ts, exposure_ts, sdr_line_profiles
    except Exception:
        pass

    try:
        del vals, series, years
    except Exception:
        pass

    gc.collect()

    _log("INFO", "Memory cleanup completed.")
    _log("OK", "National roads processing completed")

    return