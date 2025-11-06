# src/chm/awral_projections.py
from __future__ import annotations

# ============== stdlib ==============
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============== scientific ==============
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np

# rioxarray just needs to be imported to activate the .rio accessor
import rioxarray  # noqa: F401
from shapely.geometry import mapping
from shapely.ops import unary_union


# ========================= Config =========================

@dataclass
class AwralProjConfig:
    """
    AWRAL projections (historical/rcp45/rcp85) downloader + summariser.

    Required
    --------
    chm_workspace : str  -> root output directory of the project
    catchment_path: str  -> path to catchment boundary (any GeoPandas-readable vector)

    Optional
    --------
    sites_gpkg     : path to existing sites GPKG; if None, use <Catchment>/Sites Datasets/<name> Sites Data.gpkg
    make_site_csvs : export per-site CSVs (default True)
    out_subdir     : subfolder under <Catchment Datasets>/Hydroclimate (default "Future AWRAL")
    projections    : STAC-like key -> pretty name (historical/rcp45/rcp85)
    variables      : AWRAL variable -> pretty name (qtot, etot, s0, sd by default)
    thredds_root   : Root OPeNDAP path *up to* the experiment folder (model bits configurable)
    model_path     : Path segment of the downscaled model to append under thredds_root
    """

    chm_workspace: str
    catchment_path: str

    sites_gpkg: Optional[str] = None
    make_site_csvs: bool = True
    out_subdir: str = "Future AWRAL"

    projections: Dict[str, str] = field(default_factory=lambda: {
        "historical": "Historical",
        "rcp45": "RCP4.5",
        "rcp85": "RCP8.5",
    })

    variables: Dict[str, str] = field(default_factory=lambda: {
        "qtot": "Runoff",
        "etot": "Actual ET",
        "s0":   "Upper Soil Moisture",
        "sd":   "Deeper Soil Moisture",
    })

    thredds_root: str = (
        "https://thredds.nci.org.au/thredds/dodsC/"
        "iu04/australian-water-outlook/hydrologic-projections/hydrologic-output-variables/output/AUS-5/BoM"
    )
    model_path: str = (
        "AWRALv6-1-CNRM-CERFACS-CNRM-CM5/"
        "{proj}/r1i1p1/CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP/latest/day/{var}/"
        "AWRALv6-1-CNRM-CERFACS-CNRM-CM5_CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP_"
        "{proj}_r1i1p1_{var}_AUS-5_day_v1_{timestr}.nc"
    )

    # time windows for each projection
    times: Dict[str, Tuple[str, str, str]] = field(default_factory=lambda: {
        "historical": ("1960-01-01", "2005-12-31", "19600101-20051231"),
        "rcp45":      ("2006-01-01", "2099-12-31", "20060101-20991231"),
        "rcp85":      ("2006-01-01", "2099-12-31", "20060101-20991231"),
    })


# ========================= Helpers =========================

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str) -> Tuple[str, str, str, str, str]:
    """Return common folders and ensure they exist."""
    catchment_name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(workspace, catchment_name)
    catch_datasets   = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder     = os.path.join(catch_datasets, "Hydroclimate")
    sites_datasets   = os.path.join(catchment_folder, "Sites Datasets")
    _ensure_dirs([catchment_folder, catch_datasets, hydro_folder, sites_datasets])
    return catchment_name, catchment_folder, catch_datasets, hydro_folder, sites_datasets


def _url_for(cfg: AwralProjConfig, proj: str, var: str) -> str:
    """Build the OPeNDAP URL for a given projection & variable."""
    _, _, timestr = cfg.times[proj]
    return f"{cfg.thredds_root}/{cfg.model_path.format(proj=proj, var=var, timestr=timestr)}"


def _subset_to_bbox_time(ds: xr.Dataset, var: str,
                         bbox: Tuple[float, float, float, float],
                         t0: str, t1: str) -> xr.Dataset:
    """Subset DS to lon/lat bbox + time; robust to coord names and lat order."""
    minx, miny, maxx, maxy = bbox
    # coord names
    latn = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lonn = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if latn is None or lonn is None:
        raise ValueError("Dataset missing lat/lon coordinates.")

    lat_desc = bool(ds[latn][0] > ds[latn][-1])
    lat_slice = slice(maxy, miny) if lat_desc else slice(miny, maxy)
    lon_slice = slice(minx, maxx)

    out = ds.sel({lonn: lon_slice, latn: lat_slice, "time": slice(t0, t1)})
    # normalize to rioxarray expectations
    out = out.rename({latn: "y", lonn: "x"}).rio.write_crs("EPSG:4326")
    return out[[var]]


def _safe_mean_clip(da: xr.DataArray, geom: dict, fallback_pt: Tuple[float, float]) -> xr.DataArray:
    """
    Clip by polygon; if empty (tiny polygon vs grid), fall back to nearest grid cell.
    If a 'depth' dim exists, reduce it first.
    """
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    if "depth" in da.dims:
        da = da.mean(dim="depth", skipna=True)

    try:
        clipped = da.rio.clip([geom], all_touched=True, drop=True)
        if clipped.sizes.get("y", 0) == 0 or clipped.sizes.get("x", 0) == 0:
            raise ValueError("empty clip")
        return clipped.mean(dim=["y", "x"], skipna=True)
    except Exception:
        y, x = fallback_pt[1], fallback_pt[0]
        return da.sel(y=y, x=x, method="nearest")


# ========================= Main API =========================

def hydroclimate_projections(cfg: AwralProjConfig) -> Tuple[str, str]:
    """
    Downloads AWRAL projection variables, clips to catchment + (optionally) sites, and outputs:
      - Clipped NetCDF files per (projection × variable) into <Hydroclimate>/<cfg.out_subdir>
      - Catchment-wide merged daily CSV across all projections/variables
      - (Optional) Per-site daily CSVs (one per site) merging all projections/variables

    Returns
    -------
    (all_sites_gpkg, sites_datasets)
    """
    print("Start downloading projected hydrological data...")

    # ---- folders / paths ----
    (catchment_name, _catchment_folder, catch_datasets, hydro_folder, sites_datasets) = \
        _scaffold(cfg.chm_workspace, cfg.catchment_path)

    out_folder = os.path.join(hydro_folder, cfg.out_subdir)
    _ensure_dirs([out_folder])

    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")

    # ---- catchment geom/bbox in WGS84 ----
    catch_gdf = gpd.read_file(cfg.catchment_path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = catch_gdf.total_bounds
    bbox = (minx, miny, maxx, maxy)
    catch_union = unary_union(catch_gdf.geometry)
    catch_geom = mapping(catch_union)
    catch_cent = (catch_union.centroid.x, catch_union.centroid.y)

    # ---- (optional) load sites (WGS84) ----
    sites = None
    if cfg.make_site_csvs and os.path.exists(all_sites_gpkg):
        try:
            sites = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)
        except Exception as e:
            print(f"[WARN] Could not read sites GPKG: {e}. Proceeding without site CSVs.")
            sites = None
    if cfg.make_site_csvs and sites is None:
        cfg.make_site_csvs = False

    # ---- accumulators ----
    catch_merged: Optional[pd.DataFrame] = None
    site_frames: Dict[str, pd.DataFrame] = {str(row.get("id")): pd.DataFrame()
                                            for _, row in (sites.iterrows() if sites is not None else [])}

    # ========== main loops ==========
    for proj_key, proj_name in cfg.projections.items():
        t0, t1, _timestr = cfg.times[proj_key]

        for var, pretty in cfg.variables.items():
            print(f"Processing {proj_key.upper()} → {var} ({pretty})")

            # build URL and open remote
            try:
                url = _url_for(cfg, proj_key, var)
                ds = xr.open_dataset(url)
            except Exception as e:
                print(f"[ERR] open failed for {proj_key} {var}: {e}")
                continue

            # subset to bbox + time; normalise coords
            try:
                ds_sub = _subset_to_bbox_time(ds, var, bbox, t0, t1)
            except Exception as e:
                print(f"[ERR] subset failed for {proj_key} {var}: {e}")
                try: ds.close()
                except Exception: pass
                continue
            finally:
                try: ds.close()
                except Exception: pass

            # persist clipped NetCDF (cache friendly)
            out_nc = os.path.join(out_folder, f"{proj_key}_{var}.nc")
            try:
                # overwrite on purpose (small clip files). if you prefer strict caching, guard with os.path.exists.
                ds_sub.to_netcdf(out_nc)
            except Exception as e:
                print(f"[WARN] could not write {out_nc}: {e}")

            # ---------- A) Catchment daily mean ----------
            try:
                da = ds_sub[var]
                daily_catch = _safe_mean_clip(da, catch_geom, catch_cent)
                df_c = daily_catch.to_dataframe(name=f"{pretty} - {proj_name}").reset_index()
                df_c["Date"] = pd.to_datetime(df_c["time"]).dt.date
                df_c.drop(columns=["time"], inplace=True)
                # merge into catch_merged
                if catch_merged is None:
                    catch_merged = df_c
                else:
                    catch_merged = pd.merge(catch_merged, df_c, on="Date", how="outer")
            except Exception as e:
                print(f"[WARN] catchment mean failed for {proj_key} {var}: {e}")

            # ---------- B) Site daily mean (optional) ----------
            if cfg.make_site_csvs and sites is not None:
                for _, row in sites.iterrows():
                    sid = str(row.get("id"))
                    try:
                        daily_site = _safe_mean_clip(da, mapping(row.geometry), (row.geometry.centroid.x, row.geometry.centroid.y))
                        df_s = daily_site.to_dataframe(name=f"{pretty} - {proj_name}").reset_index()
                        df_s["Date"] = pd.to_datetime(df_s["time"]).dt.date
                        df_s.drop(columns=["time"], inplace=True)
                        if site_frames[sid].empty:
                            site_frames[sid] = df_s
                        else:
                            site_frames[sid] = pd.merge(site_frames[sid], df_s, on="Date", how="outer")
                    except Exception as e:
                        print(f"[WARN] site {sid} failed for {proj_key} {var}: {e}")

            # close subset dataset
            try: ds_sub.close()
            except Exception: pass

    # ========== write outputs ==========
    # Catchment CSV (merged)
    if catch_merged is not None and not catch_merged.empty:
        catch_merged.sort_values("Date", inplace=True)
        # optional rounding
        for c in [c for c in catch_merged.columns if c != "Date"]:
            catch_merged[c] = pd.to_numeric(catch_merged[c], errors="coerce").round(2)
        out_catch_csv = os.path.join(out_folder, f"{catchment_name}_AWRAL_projections_daily.csv")
        catch_merged.to_csv(out_catch_csv, index=False)
        print(f"[OK] Catchment projections CSV: {out_catch_csv}")
    else:
        print("[INFO] No catchment projections CSV produced.")

    # Per-site CSVs
    if cfg.make_site_csvs and sites is not None:
        for sid, df in site_frames.items():
            if df.empty:
                print(f"[INFO] site {sid}: no data")
                continue
            df.sort_values("Date", inplace=True)
            site_folder = os.path.join(sites_datasets, f"Site_{sid}")
            _ensure_dirs([site_folder])
            out_csv = os.path.join(site_folder, f"Site {sid} Projected Hydrological Data.csv")
            df.to_csv(out_csv, index=False)

    print("Finished projected AWRAL processing.")
    return (all_sites_gpkg, sites_datasets)
