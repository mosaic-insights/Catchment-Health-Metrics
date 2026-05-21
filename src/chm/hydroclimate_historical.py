from __future__ import annotations

# ============== stdlib ==============
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ============== scientific ==============
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import gc

# rioxarray activates .rio accessors (CRS, transform, clip)
import rioxarray  # noqa: F401
from rasterio.features import geometry_mask

# ============== plotting (matplotlib only, no seaborn) ==============
import matplotlib.pyplot as plt


# ========================= Config =========================

@dataclass
class HistoricalConfig:
    """
    Unified historical hydroclimate downloader + summariser (AWAP + AWRAL).

    Required
    --------
    chm_workspace : root project output directory
    catchment_path: path to catchment boundary (any GeoPandas-readable vector)
    start_year, end_year : inclusive year range

    Optional
    --------
    sites_gpkg    : path to sites GPKG (if None, defaults to <workspace>/<catchment>/Sites Datasets/<catchment> Sites Data.gpkg)
    make_site_csvs: whether to export per-site daily/annual CSVs + plot (default True)
    out_subdir    : subfolder under "Catchment Datasets/Hydroclimate" (fixed to "Historical data" by default)
    awap_root     : OPeNDAP root for AGCD/AWAP daily products
    awral_root    : OPeNDAP root for AWRAL products
    awap_vars     : short->pretty mapping for AWAP variables
    awap_reducers : short->reducer mapping (AGCD convention)
    awral_vars    : short->pretty mapping for AWRAL variables
    enable_awap   : toggle AWAP
    enable_awral  : toggle AWRAL
    """
    chm_workspace: str
    catchment_path: str
    start_year: int
    end_year: int

    sites_gpkg: Optional[str] = None
    make_site_csvs: bool = True

    out_subdir: str = "Historical data"

    awap_root: str = "https://thredds.nci.org.au/thredds/dodsC/zv2/agcd/v1-0-1"
    awral_root: str = (
        "https://thredds.nci.org.au/thredds/dodsC/"
        "iu04/australian-water-outlook/historical/v1/AWRALv7"
    )

    awap_vars: Dict[str, str] = None
    awap_reducers: Dict[str, str] = None
    awral_vars: Dict[str, str] = None

    enable_awap: bool = True
    enable_awral: bool = True

    def __post_init__(self):
        if self.awap_vars is None:
            self.awap_vars = {
                "precip": "Precipitation",
                "tmin": "Min Temperature",
                "tmax": "Max Temperature",
            }
        if self.awap_reducers is None:
            self.awap_reducers = {"precip": "total", "tmin": "mean", "tmax": "mean"}
        if self.awral_vars is None:
            self.awral_vars = {
                "qtot": "Runoff",
                "etot": "Actual ET",
                "s0":   "Upper Soil Moisture",
                "sd":   "Deeper Soil Moisture",
            }


# ========================= Helpers (filesystem/scaffold) =========================

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _scaffold(workspace: str, catchment_path: str, out_subdir: str) -> Tuple[str, str, str, str, str, str]:
    """Return common folder paths and ensure they exist."""
    catchment_name = os.path.splitext(os.path.basename(catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(workspace, catchment_name)
    catch_datasets   = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder     = os.path.join(catch_datasets, "Hydroclimate")
    hist_folder      = os.path.join(hydro_folder, out_subdir)  # <- single 'Historical data'
    sites_datasets   = os.path.join(catchment_folder, "Sites Datasets")
    catch_plots      = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_plots      = os.path.join(catchment_folder, "Sites Plots and Maps")
    _ensure_dirs([catchment_folder, catch_datasets, hydro_folder, hist_folder, sites_datasets, catch_plots, sites_plots])
    return catchment_name, catchment_folder, catch_datasets, hist_folder, sites_datasets, catch_plots, sites_plots


def _valid_nc(path: str, var_key: str) -> bool:
    """True if NetCDF exists, is non-empty, and contains var_key with time axis (uses ds.sizes to avoid warnings)."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        with xr.open_dataset(path) as ds_chk:
            if var_key not in ds_chk.variables:
                return False
            if "time" in ds_chk.sizes and ds_chk.sizes["time"] > 0:
                return True
            if "time" in ds_chk.coords and ds_chk.coords["time"].size > 0:
                return True
            return False
    except Exception:
        return False


# ========================= Grid utilities (generic lat/lon) =========================

def _lat_lon_names(obj: xr.Dataset | xr.DataArray) -> Tuple[str, str]:
    for lat_name in ["lat", "latitude", "y"]:
        if lat_name in obj.dims or lat_name in obj.coords:
            break
    else:
        raise KeyError("Latitude coordinate not found.")
    for lon_name in ["lon", "longitude", "x"]:
        if lon_name in obj.dims or lon_name in obj.coords:
            break
    else:
        raise KeyError("Longitude coordinate not found.")
    return lat_name, lon_name


def _subset_bbox_generic(ds: xr.Dataset, var: str, bbox: Tuple[float, float, float, float],
                         year_slice: Optional[Tuple[str, str]] = None) -> xr.Dataset:
    """
    Subset to lon/lat bbox; handles ascending/descending latitude.
    bbox: (minx, miny, maxx, maxy) in EPSG:4326
    Optionally slice time range with (start_iso, end_iso).
    """
    if var not in ds.variables:
        raise KeyError(f"Variable '{var}' missing (has: {list(ds.variables)})")
    minx, miny, maxx, maxy = bbox
    lat_name, lon_name = _lat_lon_names(ds)

    lat = ds[lat_name]
    lat_slice = slice(miny, maxy) if (lat[0] < lat[-1]) else slice(maxy, miny)
    lon_slice = slice(minx, maxx)

    sel = ds[var].sel({lon_name: lon_slice, lat_name: lat_slice})
    if year_slice is not None and "time" in sel.dims:
        sel = sel.sel(time=slice(*year_slice))
    return sel.to_dataset(name=var)


def _mask_mean_over_geoms(da: xr.DataArray, geoms: List) -> xr.DataArray:
    """
    Average over polygon(s) per timestep using a raster mask.
    Works for any lat/lon names; assumes EPSG:4326.
    """
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    transform = da.rio.transform()
    lat_name, lon_name = _lat_lon_names(da)

    mask = geometry_mask(
        geometries=geoms,
        transform=transform,
        invert=True,
        out_shape=(da.sizes[lat_name], da.sizes[lon_name]),
    )
    mask_da = xr.DataArray(
        mask,
        coords={lat_name: da.coords[lat_name], lon_name: da.coords[lon_name]},
        dims=(lat_name, lon_name),
    )
    return da.where(mask_da).mean(dim=(lat_name, lon_name), skipna=True)


def _sel_nearest(da: xr.DataArray, lon: float, lat: float) -> xr.DataArray:
    lat_name, lon_name = _lat_lon_names(da)
    return da.sel({lat_name: lat, lon_name: lon}, method="nearest")


# ========================= Data source specifics =========================

def _awap_url(root: str, var: str, reducer: str, year: int) -> str:
    return (
        f"{root}/{var}/{reducer}/r005/01day/"
        f"agcd_v1-0-1_{var}_{reducer}_r005_daily_{year}.nc"
    )


def _awral_url(root: str, var: str, year: int) -> str:
    return f"{root}/{var}_{year}.nc"


# ========================= Aggregation + Plot helpers =========================
def _aggregate_annual(df_daily: pd.DataFrame, var_pretty_list: List[str]) -> pd.DataFrame:
    """
    Aggregate daily dataframe to annual.
    - Fluxes (e.g., precipitation, runoff, ET): annual SUM with min_count=1 (all-NaN -> NaN)
    - States (e.g., soil moisture, temperatures): annual MEAN
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    if "Date" not in df.columns:
        if "time" in df.columns:
            df["Date"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise ValueError("Input dataframe needs a 'Date' or 'time' column.")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"]).copy()
    df["Year"] = df["Date"].dt.year

    agg = {}
    for pretty in var_pretty_list:
        low = pretty.lower()
        is_flux = (
            "precip" in low or
            "runoff" in low or
            "actual et" in low or "et" == low.strip()
        )
        if is_flux:
            agg[pretty] = (lambda s: s.sum(min_count=1))
        else:
            agg[pretty] = "mean"

    if not agg:
        return pd.DataFrame()

    annual = df.groupby("Year").agg(agg).reset_index()

    for c in [c for c in annual.columns if c != "Year"]:
        annual[c] = pd.to_numeric(annual[c], errors="coerce").round(2)
    return annual


def _plot_annual_subplots(
    annual_df: pd.DataFrame,
    var_order: List[str],
    title: str,
    outfile: str,
    units_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Subplot (1 col × N rows) for provided variables, saved to outfile.
    """
    if annual_df is None or annual_df.empty:
        return
    cols = [v for v in var_order if v in annual_df.columns]
    if not cols:
        return

    nrows = len(cols)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 2.4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    x = annual_df["Year"]
    for ax, var in zip(axes, cols):
        y = annual_df[var]
        ax.plot(x, y, color="black", marker="o", linewidth=1.5)
        unit = f" ({units_map.get(var, '')})" if units_map and var in units_map and units_map[var] else ""
        ax.set_ylabel(f"{var}{unit}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Year", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def _to_time_df(da: xr.DataArray, pretty_name: str) -> pd.DataFrame:
    """
    Make a clean (time, value) DataFrame from a 1D time DataArray.
    Drops non-index coords (eg. 'spatial_ref') to avoid merge collisions.
    """
    da = da.reset_coords(drop=True)
    df = da.to_dataframe(name=pretty_name).reset_index()
    keep = [c for c in df.columns if c in ("time", pretty_name)]
    return df[keep]


# ========================= Main API =========================

def hydroclimate_historical(cfg: HistoricalConfig) -> str:
    """
    Unified pipeline:
      (1) Download & cache clipped NetCDFs (AWAP + AWRAL) into one folder 'Historical data'
      (2) Build combined catchment-wide daily CSV (AWAP+AWRAL variables)
      (3) Build combined catchment-wide annual CSV + 1×7 subplot
      (4) (optional) Build combined site-level daily & annual CSVs + 1×7 subplot per site

    Returns
    -------
    hist_folder : str  Path to the 'Historical data' output folder.
    """
    print("Starting unified historical hydroclimate processing (AWAP + AWRAL)...")

    # ---- folders / paths ----
    (catchment_name, _catchment_folder, catch_datasets, hist_folder,
     sites_datasets, catch_plots, sites_plots) = _scaffold(cfg.chm_workspace, cfg.catchment_path, cfg.out_subdir)

    catch_layer = f"{catchment_name} Data"
    catch_gpkg = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg = cfg.sites_gpkg or os.path.join(sites_datasets, f"{catchment_name} Sites Data.gpkg")

    # ---- catchment geometry / bbox ----
    catch_gdf = gpd.read_file(cfg.catchment_path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = catch_gdf.total_bounds
    bbox = (minx, miny, maxx, maxy)
    catch_geoms = list(catch_gdf.geometry)
    catch_union = catch_gdf.unary_union
    catch_cent = catch_union.centroid

    combined_order_short = ["precip", "tmax", "tmin", "qtot", "etot", "s0", "sd"]
    pretty_map = {**cfg.awap_vars, **cfg.awral_vars}
    combined_pretty_order = [pretty_map[k] for k in combined_order_short if k in pretty_map]

    units = {
        "Precipitation": "mm/yr",
        "Min Temperature": "°C",
        "Max Temperature": "°C",
        "Runoff": "mm/yr",
        "Actual ET": "mm/yr",
        "Upper Soil Moisture": "",
        "Deeper Soil Moisture": "",
    }

    # ========================= (1) Download & cache (AWAP + AWRAL) =========================
    if cfg.enable_awap:
        for var, pretty in cfg.awap_vars.items():
            reducer = cfg.awap_reducers.get(var, "mean")
            print(f"\n[AWAP] {pretty} [{var}/{reducer}]...")
            for year in range(cfg.start_year, cfg.end_year + 1):
                out_nc = os.path.join(hist_folder, f"awap_{var}_{year}_clipped.nc")
                if _valid_nc(out_nc, var):
                    print(f"[cache] {os.path.basename(out_nc)} ok; skipping.")
                    continue
                url = _awap_url(cfg.awap_root, var, reducer, year)
                try:
                    ds = xr.open_dataset(url)
                    clipped = _subset_bbox_generic(ds, var, bbox)
                    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"}}
                    clipped.to_netcdf(out_nc, encoding=encoding)
                    print(f"[ok] wrote {os.path.basename(out_nc)}")
                except Exception as e:
                    print(f"[err] AWAP {var} {year}: {e}")
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass

    if cfg.enable_awral:
        for var, pretty in cfg.awral_vars.items():
            print(f"\n[AWRAL] {pretty} [{var}]...")
            for year in range(cfg.start_year, cfg.end_year + 1):
                out_nc = os.path.join(hist_folder, f"awral_{var}_{year}_clipped.nc")
                if _valid_nc(out_nc, var):
                    print(f"[cache] {os.path.basename(out_nc)} ok; skipping.")
                    continue
                url = _awral_url(cfg.awral_root, var, year)
                try:
                    ds = xr.open_dataset(url)
                    clipped = _subset_bbox_generic(ds, var, bbox, year_slice=(f"{year}-01-01", f"{year}-12-31"))
                    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"}}
                    clipped.to_netcdf(out_nc, encoding=encoding)
                    print(f"[ok] wrote {os.path.basename(out_nc)}")
                except Exception as e:
                    print(f"[err] AWRAL {var} {year}: {e}")
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass

    nc_by_var: Dict[str, List[str]] = {k: [] for k in combined_order_short if k in pretty_map}
    for fn in sorted(os.listdir(hist_folder)):
        if not fn.endswith(".nc"):
            continue
        for v in nc_by_var:
            if fn.startswith(f"awap_{v}_") or fn.startswith(f"awral_{v}_"):
                nc_by_var[v].append(os.path.join(hist_folder, fn))

    # ========================= (2) Catchment-wide daily (combined) =========================
    daily_df_by_var: Dict[str, pd.DataFrame] = {}
    for short, pretty in pretty_map.items():
        if short not in nc_by_var:
            continue
        series_parts = []
        for nc_path in nc_by_var[short]:
            try:
                with xr.open_dataset(nc_path) as ds:
                    ds = ds.rio.write_crs("EPSG:4326")
                    da = ds[short] if short in ds.variables else ds[list(ds.data_vars)[0]]
                    try:
                        masked_mean = _mask_mean_over_geoms(da, catch_geoms)
                        if np.count_nonzero(~np.isnan(masked_mean.isel(time=0))) == 0:
                            raise ValueError("zero-footprint")
                        daily_mean = masked_mean
                    except Exception:
                        daily_mean = _sel_nearest(da, lon=catch_cent.x, lat=catch_cent.y)

                    df = _to_time_df(daily_mean, pretty)
                    series_parts.append(df)
            except Exception as e:
                print(f"[warn] catchment read failed {os.path.basename(nc_path)}: {e}")

        daily_df_by_var[short] = (
            pd.concat(series_parts, ignore_index=True) if series_parts else pd.DataFrame()
        )

    catch_df: Optional[pd.DataFrame] = None
    for short in combined_order_short:
        if short not in daily_df_by_var:
            continue
        df = daily_df_by_var[short]
        if df is None or df.empty:
            continue
        df = df.sort_values("time")
        catch_df = df if catch_df is None else pd.merge(catch_df, df, on="time", how="outer")

    if catch_df is not None and not catch_df.empty:
        out_df = catch_df.rename(columns={"time": "Date"}).copy()
        out_df["Date"] = pd.to_datetime(out_df["Date"])
        out_df.sort_values("Date", inplace=True)

        pretty_cols = [pretty_map[k] for k in combined_order_short if pretty_map.get(k) in out_df.columns]
        csv_df = out_df[["Date"] + pretty_cols].copy()
        csv_df["Date"] = csv_df["Date"].dt.strftime("%d/%m/%Y")
        for c in pretty_cols:
            csv_df[c] = pd.to_numeric(csv_df[c], errors="coerce").round(2)

        out_csv = os.path.join(hist_folder, f"{catchment_name}_historical_hydroclimate_daily.csv")
        csv_df.to_csv(out_csv, index=False)
        out_csv_ = os.path.join(catch_plots, f"{catchment_name}_historical_hydroclimate_daily.csv")
        csv_df.to_csv(out_csv_, index=False)
        print(f"[OK] Catchment combined daily CSV: {out_csv}")
    else:
        print("[INFO] no combined catchment daily data assembled.")

    # ========================= (3) Catchment annual CSV + combined subplot =========================
    if catch_df is not None and not catch_df.empty:
        c_daily = catch_df.rename(columns={"time": "Date"})
        c_daily["Date"] = pd.to_datetime(c_daily["Date"])
        pretty_cols = [pretty_map[k] for k in combined_order_short if pretty_map.get(k) in c_daily.columns]

        annual_catch = _aggregate_annual(c_daily[["Date"] + pretty_cols], pretty_cols)
        if not annual_catch.empty:
            annual_csv = os.path.join(hist_folder, f"{catchment_name}_historical_hydroclimate_annual.csv")
            annual_catch.to_csv(annual_csv, index=False)
            print(f"[OK] Catchment combined annual CSV: {annual_csv}")

            var_order_pretty = [pretty_map[k] for k in combined_order_short if pretty_map.get(k) in annual_catch.columns]
            catch_png = os.path.join(catch_plots, "Catchment_Historical_annual.png")
            _plot_annual_subplots(
                annual_catch, var_order_pretty,
                title=f"{catchment_name} — Combined Annual Aggregates (AWAP + AWRAL)",
                outfile=catch_png,
                units_map=units
            )
            print(f"[OK] Catchment combined annual subplot PNG: {catch_png}")

    # ========================= (4) Sites: daily + annual + subplot (combined) =========================
    if cfg.make_site_csvs:
        try:
            sites_gdf = gpd.GeoDataFrame()

            if not os.path.exists(all_sites_gpkg):
                print(f"[WARN] sites GPKG not found: {all_sites_gpkg} — skipping sites.")
            else:
                try:
                    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)
                except Exception:
                    tried_layers = [f"{catchment_name} Sites Data", "Sites", "sites"]
                    for lyr in tried_layers:
                        try:
                            sites_gdf = gpd.read_file(all_sites_gpkg, layer=lyr).to_crs(epsg=4326)
                            print(f"[INFO] loaded sites layer '{lyr}'")
                            break
                        except Exception:
                            pass
                    if sites_gdf.empty:
                        print(f"[ERROR] could not load any sites layer from {all_sites_gpkg}; skipping sites.")

            if sites_gdf is not None and not sites_gdf.empty:
                for idx, site in sites_gdf.iterrows():
                    sid = site.get("Site_id") or site.get("site_id") or site.get("Site_ID") or idx
                    geom_list = [site.geometry]
                    site_folder = os.path.join(sites_plots, f"Site_{sid}")
                    _ensure_dirs([site_folder])

                    site_df: Optional[pd.DataFrame] = None

                    for short, pretty in pretty_map.items():
                        if short not in nc_by_var:
                            continue
                        parts = []
                        for nc_path in nc_by_var[short]:
                            try:
                                with xr.open_dataset(nc_path) as ds:
                                    ds = ds.rio.write_crs("EPSG:4326")
                                    da = ds[short] if short in ds.variables else ds[list(ds.data_vars)[0]]
                                    try:
                                        masked = _mask_mean_over_geoms(da, geom_list)
                                        if np.count_nonzero(~np.isnan(masked.isel(time=0))) == 0:
                                            raise ValueError("zero-footprint")
                                        daily_mean = masked
                                    except Exception:
                                        cen = site.geometry.centroid
                                        daily_mean = _sel_nearest(da, lon=cen.x, lat=cen.y)

                                    df_v = _to_time_df(daily_mean, pretty)
                                    parts.append(df_v)
                            except Exception as e:
                                print(f"[warn] site {sid}: {os.path.basename(nc_path)} failed: {e}")

                        if parts:
                            vdf = pd.concat(parts, ignore_index=True).sort_values("time")
                            site_df = vdf if site_df is None else pd.merge(site_df, vdf, on="time", how="outer")

                    if site_df is not None and not site_df.empty:
                        site_df.rename(columns={"time": "Date"}, inplace=True)
                        site_df["Date"] = pd.to_datetime(site_df["Date"])
                        site_df.sort_values("Date", inplace=True)

                        pretty_cols_site = [pretty_map[k] for k in combined_order_short if pretty_map.get(k) in site_df.columns]
                        out_daily = site_df.copy()
                        out_daily["Date"] = out_daily["Date"].dt.strftime("%d/%m/%Y")
                        out_daily = out_daily[["Date"] + pretty_cols_site].round(2)

                        out_csv_daily = os.path.join(site_folder, f"Site_{sid}_historical_hydroclimate_daily.csv")
                        out_daily.to_csv(out_csv_daily, index=False)

                        annual_site = _aggregate_annual(site_df[["Date"] + pretty_cols_site], pretty_cols_site)
                        if not annual_site.empty:
                            out_csv_annual = os.path.join(site_folder, f"Site_{sid}_historical_hydroclimate_annual.csv")
                            annual_site.to_csv(out_csv_annual, index=False)

                            var_order_pretty = [pretty_map[k] for k in combined_order_short if pretty_map.get(k) in annual_site.columns]
                            out_png = os.path.join(site_folder, f"Site_{sid}_Historical_annual.png")
                            _plot_annual_subplots(
                                annual_site, var_order_pretty,
                                title=f"Site {sid} — Combined Annual Aggregates (AWAP + AWRAL)",
                                outfile=out_png,
                                units_map=units
                            )
                            print(f"[OK] site {sid} combined annual CSV + subplot")
                    else:
                        print(f"[INFO] site {sid}: no daily data assembled (check .nc or geometry).")

                    # =========================
                    # Per-site cleanup
                    # =========================
                    plt.close("all")

                    try:
                        del geom_list
                    except Exception:
                        pass

                    try:
                        del site_df, out_daily, annual_site
                    except Exception:
                        pass

                    try:
                        del parts, df_v, vdf
                    except Exception:
                        pass

                    try:
                        del da, daily_mean, masked
                    except Exception:
                        pass

                    gc.collect()

                    print(f"[INFO] site {sid}: memory cleanup completed.")
        except Exception as e:
            print(f"[ERROR] site-level CSVs/plots failed: {e}")

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del catch_gdf, catch_geoms, catch_union, catch_cent
    except Exception:
        pass

    try:
        del sites_gdf, geom_list
    except Exception:
        pass

    try:
        del ds, da, clipped, masked_mean, daily_mean, masked
    except Exception:
        pass

    try:
        del catch_df, c_daily, annual_catch, csv_df, out_df
    except Exception:
        pass

    try:
        del daily_df_by_var, series_parts, parts
    except Exception:
        pass

    try:
        del df, df_v, vdf, site_df, out_daily, annual_site
    except Exception:
        pass

    try:
        del nc_by_var
    except Exception:
        pass

    gc.collect()

    print("[INFO] Memory cleanup completed.")
    print("Unified historical hydroclimate processing complete.")

    return