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
    AWRAL projections (historical / rcp45 / rcp85) downloader + summariser.

    Required
    --------
    chm_workspace : str
        Root output directory of the project.
    catchment_path : str
        Path to catchment boundary (any GeoPandas-readable vector).

    Optional
    --------
    sites_path : Optional[str]
        Path to the sites layer used for clipping/summarising sites.
        This can be a GPKG, SHP, or any GeoPandas-readable vector.
        If None, site-level CSVs are skipped unless a default CHM sites file exists.

    make_site_csvs : bool
        Export per-site CSVs (default True).

    out_subdir : str
        Subfolder under <Catchment Datasets>/Hydroclimate (default "Future AWRAL").

    historical_start_date / historical_end_date : Optional[str]
        Requested historical analysis window.

    projection_start_date / projection_end_date : Optional[str]
        Requested future projection analysis window.

    Notes
    -----
    - Historical uses the historical_* date range.
    - rcp45 / rcp85 use the projection_* date range.
    - If a requested time window does not overlap a projection period,
      that projection is skipped.
    """

    chm_workspace: str
    catchment_path: str

    sites_path: Optional[str] = None
    make_site_csvs: bool = True
    out_subdir: str = "Future AWRAL"

    # Separate time windows
    historical_start_date: Optional[str] = None
    historical_end_date: Optional[str] = None
    projection_start_date: Optional[str] = None
    projection_end_date: Optional[str] = None

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

    times: Dict[str, Tuple[str, str, str]] = field(default_factory=lambda: {
        "historical": ("1960-01-01", "2005-12-31", "19600101-20051231"),
        "rcp45":      ("2006-01-01", "2099-12-31", "20060101-20991231"),
        "rcp85":      ("2006-01-01", "2099-12-31", "20060101-20991231"),
    })


def _requested_window_for_projection(
    cfg: AwralProjConfig,
    proj_key: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the requested date window for a given projection type.
    """
    if proj_key == "historical":
        return cfg.historical_start_date, cfg.historical_end_date
    return cfg.projection_start_date, cfg.projection_end_date


# ========================= Helpers =========================

def _ensure_dirs(paths: List[str]) -> None:
    """Create folders if they do not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str) -> Tuple[str, str, str, str, str]:
    """
    Return common folders and ensure they exist.

    Returns
    -------
    catchment_name, catchment_folder, catch_datasets, hydro_folder, sites_datasets
    """
    catchment_name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(workspace, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder = os.path.join(catch_datasets, "Hydroclimate")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    _ensure_dirs([catchment_folder, catch_datasets, hydro_folder, sites_datasets])
    return catchment_name, catchment_folder, catch_datasets, hydro_folder, sites_datasets


def _default_sites_path(sites_datasets: str, catchment_name: str) -> str:
    """
    Default CHM sites file path, matching your usual folder structure.
    """
    return os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")


def _url_for(cfg: AwralProjConfig, proj: str, var: str) -> str:
    """Build the OPeNDAP URL for a given projection and variable."""
    _, _, timestr = cfg.times[proj]
    return f"{cfg.thredds_root}/{cfg.model_path.format(proj=proj, var=var, timestr=timestr)}"


def _resolve_time_window(
    proj_t0: str,
    proj_t1: str,
    user_t0: Optional[str],
    user_t1: Optional[str],
) -> Optional[Tuple[str, str]]:
    """
    Intersect the projection's valid period with the user-requested period.

    Returns
    -------
    (start_date, end_date) in YYYY-MM-DD format, or None if there is no overlap.
    """
    proj_start = pd.Timestamp(proj_t0)
    proj_end = pd.Timestamp(proj_t1)

    req_start = pd.Timestamp(user_t0) if user_t0 else proj_start
    req_end = pd.Timestamp(user_t1) if user_t1 else proj_end

    start = max(proj_start, req_start)
    end = min(proj_end, req_end)

    if start > end:
        return None

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _subset_to_bbox_time(
    ds: xr.Dataset,
    var: str,
    bbox: Tuple[float, float, float, float],
    t0: str,
    t1: str,
) -> xr.Dataset:
    """
    Subset dataset to lon/lat bbox + time; robust to coord names and latitude order.

    Notes
    -----
    - This subsets to the catchment bounding box, not the exact polygon.
    - Exact polygon means are computed later from this bbox subset.
    """
    minx, miny, maxx, maxy = bbox

    latn = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lonn = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if latn is None or lonn is None:
        raise ValueError("Dataset missing lat/lon coordinates.")

    lat_desc = bool(ds[latn][0] > ds[latn][-1])
    lat_slice = slice(maxy, miny) if lat_desc else slice(miny, maxy)
    lon_slice = slice(minx, maxx)

    out = ds.sel({lonn: lon_slice, latn: lat_slice, "time": slice(t0, t1)})
    out = out.rename({latn: "y", lonn: "x"}).rio.write_crs("EPSG:4326")
    return out[[var]]


def _safe_mean_clip(da: xr.DataArray, geom: dict, fallback_pt: Tuple[float, float]) -> xr.DataArray:
    """
    Clip by polygon and return spatial mean through time.

    If the clip is empty (for example, tiny polygon vs coarse grid), fall back to
    the nearest grid cell at the supplied fallback point.
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


def _open_remote_dataset(url: str) -> xr.Dataset:
    """
    Open a remote NetCDF / OPeNDAP dataset with an explicit backend.
    """
    return xr.open_dataset(url, engine="netcdf4")


# ========================= Main API =========================

def hydroclimate_projections(cfg: AwralProjConfig) -> Tuple[Optional[str], str]:
    """
    Download AWRAL projection variables, subset them to the catchment bbox and
    requested time window, then summarise for the catchment and (optionally) sites.

    Outputs
    -------
    1) BBOX-clipped NetCDF files per (projection × variable × time-window) into:
       <Catchment Datasets>/Hydroclimate/<cfg.out_subdir>

    2) Catchment-wide merged daily CSV across all processed projections/variables

    3) Optional per-site daily CSVs (one per site), merging all processed
       projections/variables

    Returns
    -------
    (sites_path_used, sites_datasets)
    """
    print("Start downloading projected hydrological data...")

    # ---- folders / paths ----
    (
        catchment_name,
        _catchment_folder,
        catch_datasets,
        hydro_folder,
        sites_datasets,
    ) = _scaffold(cfg.chm_workspace, cfg.catchment_path)

    out_folder = os.path.join(hydro_folder, cfg.out_subdir)
    _ensure_dirs([out_folder])

    # ---- catchment geometry / bbox in WGS84 ----
    catch_gdf = gpd.read_file(cfg.catchment_path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = catch_gdf.total_bounds
    bbox = (minx, miny, maxx, maxy)

    catch_union = unary_union(catch_gdf.geometry)
    catch_geom = mapping(catch_union)
    catch_cent = (catch_union.centroid.x, catch_union.centroid.y)

    # ---- optional sites layer (input only; not modified) ----
    sites = None
    sites_path_used = cfg.sites_path or _default_sites_path(sites_datasets, catchment_name)

    if cfg.make_site_csvs and os.path.exists(sites_path_used):
        try:
            sites = gpd.read_file(sites_path_used).to_crs(epsg=4326)
        except Exception as e:
            print(f"[WARN] Could not read sites layer: {e}. Proceeding without site CSVs.")
            sites = None
    else:
        if cfg.make_site_csvs:
            print(f"[WARN] Sites layer not found: {sites_path_used}. Proceeding without site CSVs.")

    if cfg.make_site_csvs and sites is None:
        cfg.make_site_csvs = False

    # ---- accumulators ----
    catch_merged: Optional[pd.DataFrame] = None

    site_frames: Dict[str, pd.DataFrame] = {
        str(row.get("Site_id", i)): pd.DataFrame()
        for i, (_, row) in enumerate(sites.iterrows() if sites is not None else [])
    }

    # ========== main loops ==========
    for proj_key, proj_name in cfg.projections.items():
        proj_t0, proj_t1, _timestr = cfg.times[proj_key]

        user_t0, user_t1 = _requested_window_for_projection(cfg, proj_key)
        resolved_window = _resolve_time_window(
            proj_t0=proj_t0,
            proj_t1=proj_t1,
            user_t0=user_t0,
            user_t1=user_t1,
        )

        if resolved_window is None:
            print(f"[INFO] Skipping {proj_key.upper()}: no overlap with requested time window.")
            continue

        t0, t1 = resolved_window
        t0_tag = t0.replace("-", "")
        t1_tag = t1.replace("-", "")

        for var, pretty in cfg.variables.items():
            print(f"Processing {proj_key.upper()} → {var} ({pretty}) from {t0} to {t1}")

            # ---- open remote dataset ----
            try:
                url = _url_for(cfg, proj_key, var)
                ds = _open_remote_dataset(url)
            except Exception as e:
                print(f"[ERR] open failed for {proj_key} {var}: {e}")
                continue

            # ---- subset to bbox + time ----
            try:
                ds_sub = _subset_to_bbox_time(ds, var, bbox, t0, t1)
            except Exception as e:
                print(f"[ERR] subset failed for {proj_key} {var}: {e}")
                try:
                    ds.close()
                except Exception:
                    pass
                continue
            finally:
                try:
                    ds.close()
                except Exception:
                    pass

            # ---- save bbox-clipped NetCDF ----
            out_nc = os.path.join(out_folder, f"{proj_key}_{var}_{t0_tag}_{t1_tag}.nc")
            try:
                ds_sub.to_netcdf(out_nc)
            except Exception as e:
                print(f"[WARN] could not write {out_nc}: {e}")

            # ---------- A) Catchment daily mean ----------
            try:
                da = ds_sub[var]
                daily_catch = _safe_mean_clip(da, catch_geom, catch_cent)

                col_name = f"{pretty} - {proj_name}"
                df_c = daily_catch.to_dataframe(name=col_name).reset_index()

                # Keep only the time/value columns to avoid merge problems from spatial_ref and other coords
                df_c["Date"] = pd.to_datetime(df_c["time"]).dt.date
                df_c = df_c[["Date", col_name]].copy()

                if catch_merged is None:
                    catch_merged = df_c
                else:
                    catch_merged = pd.merge(catch_merged, df_c, on="Date", how="outer")
            except Exception as e:
                print(f"[WARN] catchment mean failed for {proj_key} {var}: {e}")

            # ---------- B) Site daily mean ----------
            if cfg.make_site_csvs and sites is not None:
                for i, (_, row) in enumerate(sites.iterrows()):
                    sid = str(row.get("Site_id", i))
                    try:
                        daily_site = _safe_mean_clip(
                            da,
                            mapping(row.geometry),
                            (row.geometry.centroid.x, row.geometry.centroid.y),
                        )

                        col_name = f"{pretty} - {proj_name}"
                        df_s = daily_site.to_dataframe(name=col_name).reset_index()

                        # Keep only the time/value columns to avoid merge problems from spatial_ref and other coords
                        df_s["Date"] = pd.to_datetime(df_s["time"]).dt.date
                        df_s = df_s[["Date", col_name]].copy()

                        if site_frames[sid].empty:
                            site_frames[sid] = df_s
                        else:
                            site_frames[sid] = pd.merge(site_frames[sid], df_s, on="Date", how="outer")
                    except Exception as e:
                        print(f"[WARN] site {sid} failed for {proj_key} {var}: {e}")

            try:
                ds_sub.close()
            except Exception:
                pass

    # ========== write outputs ==========
    # ---- Catchment CSV ----
    if catch_merged is not None and not catch_merged.empty:
        catch_merged.sort_values("Date", inplace=True)

        for c in [c for c in catch_merged.columns if c != "Date"]:
            catch_merged[c] = pd.to_numeric(catch_merged[c], errors="coerce").round(2)

        out_catch_csv = os.path.join(
            out_folder,
            f"{catchment_name}_AWRAL_projections_daily.csv",
        )
        catch_merged.to_csv(out_catch_csv, index=False)
        print(f"[OK] Catchment projections CSV: {out_catch_csv}")
    else:
        print("[INFO] No catchment projections CSV produced.")

    # ---- Per-site CSVs only ----
    if cfg.make_site_csvs and sites is not None:
        for sid, df in site_frames.items():
            if df.empty:
                print(f"[INFO] site {sid}: no data")
                continue

            df.sort_values("Date", inplace=True)

            for c in [c for c in df.columns if c != "Date"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

            site_folder = os.path.join(sites_datasets, f"Site_{sid}")
            _ensure_dirs([site_folder])

            out_csv = os.path.join(
                site_folder,
                f"Site {sid} Projected Hydrological Data.csv",
            )
            df.to_csv(out_csv, index=False)

    print("Finished projected AWRAL processing.")
    return sites_path_used, sites_datasets