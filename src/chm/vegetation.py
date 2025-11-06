# src/chm_veg/veg_indices.py
# -*- coding: utf-8 -*-
"""
Vegetation indices (NDVI) and C-factor from DEA STAC; annual summaries,
riparian stats, per-site stats & plots.

Dependencies: geopandas, rasterio, rioxarray, xarray, numpy, pandas,
pystac_client, odc.stac, shapely, matplotlib, fiona (optional for layer replace)
"""

from __future__ import annotations

# --- Stdlib ---
import os
import re
import gc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# --- Scientific ---
import numpy as np
import pandas as pd
import xarray as xr

# --- Geo ---
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point
import fiona  # only used for fiona.remove(...)

# --- STAC / DEA ---
import pystac_client
import odc.stac

# --- Plotting ---
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # <-- add MaxNLocator here
from matplotlib.lines import Line2D
# ============================== Utilities ==============================

def _ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _years_from_interval(datetime_str: str) -> List[int]:
    """'YYYY-MM-DD/YYYY-MM-DD' -> list of years inclusive."""
    dt0, dt1 = datetime_str.split("/")
    y0, y1 = pd.Timestamp(dt0).year, pd.Timestamp(dt1).year
    return list(range(y0, y1 + 1))


def _both_exist(a: str, b: str) -> bool:
    return os.path.exists(a) and os.path.exists(b)


def _all_annual_outputs_exist(annual_ndvi_dir: str, annual_c_dir: str, years: List[int]) -> bool:
    for y in years:
        if not _both_exist(os.path.join(annual_ndvi_dir, f"NDVI_{y}.tif"),
                           os.path.join(annual_c_dir,    f"C_Factor_{y}.tif")):
            return False
    return True


def _zonal_mean(raster_path: str, geom, target_crs) -> float:
    """Mean over geometry; returns np.nan if file missing or empty."""
    if not os.path.exists(raster_path):
        return np.nan
    with rio.open(raster_path) as src:
        geom_proj = gpd.GeoSeries([geom], crs=target_crs).to_crs(src.crs).geometry.iloc[0]
        out, _ = rio_mask(src, [geom_proj], crop=True, filled=False)
        band = out[0]
        data = band.filled(np.nan).astype(float) if np.ma.isMaskedArray(band) else band.astype(float)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        m = np.nanmean(data)
        return float(m) if np.isfinite(m) else np.nan


def _masked_mean(src: rio.DatasetReader, geom) -> float:
    """Mean of masked region; np.nan on failure/empty."""
    try:
        out_img, _ = rio_mask(src, [geom], crop=True, filled=False)
        band = out_img[0]
        arr = band.filled(np.nan).astype(float) if np.ma.isMaskedArray(band) else band.astype(float)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        m = np.nanmean(arr)
        return float(m) if np.isfinite(m) else np.nan
    except Exception:
        return np.nan


def _plot_riparian_map_for_year(
    ST_gdf: gpd.GeoDataFrame,
    catch_gdf: gpd.GeoDataFrame,
    sites_path: str,
    year: int,
    out_png: str,
):
    """
    Plot riparian NDVI by stream segment for a single year, with fixed bins and legend.
    Expects a column on ST_gdf named 'Riparian NDVI - {year}'.
    """
    col = f"Riparian NDVI - {year}"
    if col not in ST_gdf.columns:
        print(f"[WARN] {_plot_riparian_map_for_year.__name__}: '{col}' not found on streams.")
        return

    # Fixed class breaks & colors (match the old style)
    breaks = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
    colors = ["#d7191c", "#fdae61", "#ffea00", "#a6d96a", "#1a9641", "#00441b"]
    labels = ["≤ 0.0", "0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "> 0.8"]

    # Plot in Web Mercator for consistency
    web = 3857
    streams_web = ST_gdf.to_crs(epsg=web)
    catch_web   = catch_gdf.to_crs(epsg=web)

    # Bin values
    vals = pd.to_numeric(streams_web[col], errors="coerce")
    cats = pd.cut(vals, breaks, right=True, labels=labels, include_lowest=True)
    streams_web = streams_web.assign(__class=cats)

    # Load sites (optional)
    try:
        sites = gpd.read_file(sites_path)
        sites_web = sites.to_crs(epsg=web) if not sites.empty else None
    except Exception:
        sites_web = None

    # Figure
    fig, ax = plt.subplots(figsize=(8, 10))
    xmin, ymin, xmax, ymax = catch_web.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Catchment outline
    catch_web.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=10, label="Catchment boundary")

    # Classed streams
    for lab, col_hex in zip(labels, colors):
        seg = streams_web[streams_web["__class"] == lab]
        if len(seg):
            seg.plot(ax=ax, color=col_hex, linewidth=3.0, zorder=20, label=lab)

    # Sites
    if sites_web is not None and not sites_web.empty:
        sites_web.plot(ax=ax, color="red", markersize=24, marker="o", zorder=30)
        for i, r in sites_web.iterrows():
            sid = r.get("id", i)
            ax.annotate(f"{sid}", (r.geometry.x, r.geometry.y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color="black")

    # Legend
    legend_lines = [Line2D([0],[0], color="black", lw=6, label="Catchment boundary")]
    legend_lines.append(Line2D([0],[0], marker="o", color="black", lw=0, markersize=8, label="Sites"))
    for lab, col_hex in zip(labels, colors):
        legend_lines.append(Line2D([0],[0], color=col_hex, lw=6, label=lab))

    ax.legend(handles=legend_lines, title="Riparian NDVI", loc="upper left", frameon=True)
    ax.set_title(f"Riparian NDVI by Stream Order — {year}",fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude",fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude",fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================== Config ==============================

@dataclass
class VegConfig:
    """Inputs & knobs for vegetation processing."""
    chm_workspace: str
    catchment_path: str
    sites_path: str
    datetime_range: str  # e.g., "2000-01-01/2024-12-31"
    # STAC/DEA parameters
    stac_url: str = "https://explorer.dea.ga.gov.au/stac"
    cloud_cover_lt: int = 20
    filter_query: Optional[dict] = None  # CQL2 (if supported) or None → use cloud cover filter
    # Pre/Post sensor split (DEA specifics)
    split_date: str = "2016-01-01"  # LS7 pre, S2A/B post
    # Bands per collection (override if DEA changes names)
    bands_ls7: Tuple[str, str] = ("nbart_red", "nbart_nir")
    bands_s2: Tuple[str, str] = ("nbart_red", "nbart_nir_1")  # renamed to nbart_nir later
    # Riparian analysis
    riparian_buffer_m: float = 30.0


# ============================== Main API ==============================

def veg_indices_and_c_factor(cfg: VegConfig):
    """
    End-to-end workflow:
      - Query DEA STAC for LS7 (pre-split) and S2 (post-split), load as xarray
      - Compute timestep NDVI & C; write rasters (cached)
      - Compute annual NDVI & C; write rasters (cached)
      - Append annual NDVI & C means to catchment GPKG layer
      - Compute riparian NDVI:
          • per stream segment (buffered), per year (columns on Streams layer)
          • per order × year summary (CSV + plot)
          • per-year maps by stream segment (classified color legend)
      - Per-site summaries & plots for all annual rasters under /Indices/**/Annual
      - Save per-site GPKG + CSV and combined all-sites files

    Returns
    -------
    all_sites_gpkg, dem_crs, sites_datasets, indices_output, annual_ndvi_dir, annual_c_dir
    """
    print("Starting processing vegetation and c factor...")

    # ---------- Folders ----------
    catchment_name = os.path.splitext(os.path.basename(cfg.catchment_path))[0].replace("_", " ")
    catchment_folder = os.path.join(cfg.chm_workspace, catchment_name)
    catch_datasets   = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots      = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets   = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots      = os.path.join(catchment_folder, "Sites Plots and Maps")
    veg_folder       = os.path.join(catch_datasets, "Vegetation")
    satellite_output = os.path.join(veg_folder, "Satellite data")
    indices_output   = os.path.join(veg_folder, "Indices")
    riparian_output  = os.path.join(veg_folder, "Riparian")
    ndvi_output      = os.path.join(indices_output, "NDVI")
    c_factor_output  = os.path.join(indices_output, "C Factor")

    for d in [catchment_folder, catch_datasets, catch_plots, sites_datasets, sites_plots,
              veg_folder, satellite_output, indices_output, ndvi_output, c_factor_output, riparian_output]:
        _ensure_dirs([d])

    catch_gpkg      = os.path.join(catch_datasets, f"{catchment_name} Data.gpkg")
    all_sites_gpkg  = os.path.join(sites_datasets,  f"{catchment_name} Sites.gpkg")

    # Annual subdirs
    annual_ndvi_dir = os.path.join(ndvi_output,     "Annual")
    annual_c_dir    = os.path.join(c_factor_output, "Annual")
    _ensure_dirs([annual_ndvi_dir, annual_c_dir])

    # ---------- Catchment & DEM ----------
    gdf = gpd.read_file(cfg.catchment_path)
    if gdf.empty:
        raise ValueError("Catchment file has no features.")
    crs = gdf.crs
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    bbox = gdf_wgs84.total_bounds

    topo_folder = os.path.join(catch_datasets, "Topography")
    dem_projected_file = os.path.join(topo_folder, "DEM.tif")
    if not os.path.exists(dem_projected_file):
        raise FileNotFoundError(f"DEM missing at {dem_projected_file}. Run dem_and_terrain first.")
    with rio.open(dem_projected_file) as dem_src:
        dem_crs       = dem_src.crs
        dem_transform = dem_src.transform
        dem_resolution = abs(dem_src.transform.a)
    stream_network = os.path.join(topo_folder, "Stream_Network.gpkg")

    # ---------- STAC: cache gate ----------
    try:
        requested_years = _years_from_interval(cfg.datetime_range)
    except Exception:
        # fallback: infer from files present
        requested_years = sorted({
            int(m.group(1)) for f in os.listdir(annual_ndvi_dir)
            if (m := re.match(r"NDVI_(\d{4})\.tif$", f))
        })

    ds: Optional[xr.Dataset] = None
    if requested_years and _all_annual_outputs_exist(annual_ndvi_dir, annual_c_dir, requested_years):
        print("[cache] All ANNUAL NDVI & C_Factor rasters already exist for requested years; skipping STAC load.")
    else:
        # ---------- STAC query & load ----------
        catalog = pystac_client.Client.open(cfg.stac_url)
        odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
        
        SPLIT_DATE = pd.Timestamp(cfg.split_date)
        dt0, dt1 = cfg.datetime_range.split("/")
        dt_start, dt_end = pd.Timestamp(dt0), pd.Timestamp(dt1)
        
        ranges = []
        if dt_start < SPLIT_DATE:
            ranges.append((max(dt_start, pd.Timestamp.min),
                           min(dt_end,   SPLIT_DATE - pd.Timedelta(seconds=1)),
                           ["ga_ls7e_ard_3"], list(cfg.bands_ls7)))
        if dt_end >= SPLIT_DATE:
            ranges.append((max(dt_start, SPLIT_DATE), dt_end,
                           ["ga_s2am_ard_3", "ga_s2bm_ard_3"], list(cfg.bands_s2)))
        
        loaded: List[xr.Dataset] = []
        for d0, d1, collections_sel, bands_sel in ranges:
            dt_str = f"{d0:%Y-%m-%d}/{d1:%Y-%m-%d}"
        
            # Prefer a CQL2 filter if available; otherwise fall back to legacy 'query='
            cql2_filter = (cfg.filter_query if cfg.filter_query is not None else {
                "op": "<",
                "args": [
                    {"property": "eo:cloud_cover"},
                    int(cfg.cloud_cover_lt),
                ],
            })
        
            # First attempt: use filter + filter_lang
            try:
                search = catalog.search(
                    bbox=bbox,
                    collections=collections_sel,
                    datetime=dt_str,
                    filter=cql2_filter,
                    filter_lang="cql2-json",
                )
                items = list(search.items())
            except Exception:
                # Fallback: many STAC APIs support 'query' instead of CQL2
                search = catalog.search(
                    bbox=bbox,
                    collections=collections_sel,
                    datetime=dt_str,
                    query={"eo:cloud_cover": {"lt": int(cfg.cloud_cover_lt)}},
                )
                items = list(search.items())
        
            if not items:
                print(f"[WARN] No items for {collections_sel} in {dt_str}")
                continue
        
            ds_part = odc.stac.load(
                items,
                bands=bands_sel,
                crs=dem_crs,
                resolution=dem_resolution,
                groupby="solar_day",
                bbox=bbox,
            )
        
            # Normalize NIR naming across sensors (S2 often 'nbart_nir_1')
            if "nbart_nir_1" in ds_part.data_vars and "nbart_nir" not in ds_part.data_vars:
                ds_part = ds_part.rename({"nbart_nir_1": "nbart_nir"})
        
            loaded.append(ds_part)
        
        if not loaded:
            raise RuntimeError("No DEA imagery found in the requested time range.")
        
        ds = xr.concat(loaded, dim="time").sortby("time") if len(loaded) > 1 else loaded[0]


        # ---------- Time-step NDVI & C (cached) ----------
        ts_ndvi_output   = os.path.join(ndvi_output,     "Time step")
        ts_c_factor_out  = os.path.join(c_factor_output, "Time step")
        _ensure_dirs([ts_ndvi_output, ts_c_factor_out])

        dem_ref = rxr.open_rasterio(dem_projected_file).squeeze()  # for match

        for t_idx, ts in enumerate(ds.time.values):
            stamp = pd.to_datetime(ts).strftime("%Y%m%d")
            ndvi_path = os.path.join(ts_ndvi_output, f"NDVI_{stamp}.tif")
            c_path    = os.path.join(ts_c_factor_out, f"C_Factor_{stamp}.tif")
            if _both_exist(ndvi_path, c_path):
                continue

            red = ds["nbart_red"].isel(time=t_idx)
            nir = ds["nbart_nir"].isel(time=t_idx)
            ndvi = (nir - red) / (nir + red)

            ndvi_clip = ndvi.rio.clip(gdf.geometry, gdf.crs, drop=True)
            ndvi_aln  = ndvi_clip.rio.reproject_match(dem_ref)
            ndvi_aln.rio.write_nodata(np.nan, inplace=True)
            ndvi_aln.rio.to_raster(ndvi_path)

            c_factor = np.clip(np.exp(-2 * ndvi_aln), 0, 1)
            cf_da = xr.DataArray(c_factor, coords=ndvi_aln.coords, dims=ndvi_aln.dims)
            cf_da.rio.write_crs(gdf.crs, inplace=True)
            cf_da.rio.write_nodata(np.nan, inplace=True)
            cf_da.rio.to_raster(c_path)

        # ---------- Annual NDVI (median) & C (cached) ----------
        ndvi_all = (ds["nbart_nir"] - ds["nbart_red"]) / (ds["nbart_nir"] + ds["nbart_red"])
        ndvi_annual = ndvi_all.groupby("time.year").median(dim="time")

        with rxr.open_rasterio(dem_projected_file, masked=True) as _ds_dem:
            dem_ref = _ds_dem.squeeze().copy()

        for year in ndvi_annual.year.values:
            ndvi_fp = os.path.join(annual_ndvi_dir, f"NDVI_{int(year)}.tif")
            c_fp    = os.path.join(annual_c_dir,    f"C_Factor_{int(year)}.tif")
            if _both_exist(ndvi_fp, c_fp):
                continue

            ndvi_year = ndvi_annual.sel(year=year).rio.clip(gdf.geometry, gdf.crs, drop=True)
            c_year = xr.apply_ufunc(lambda x: np.clip(np.exp(-2 * x), 0, 1), ndvi_year)

            ndvi_year = ndvi_year.rio.reproject_match(dem_ref)
            c_year    = c_year.rio.reproject_match(dem_ref)

            ndvi_year.rio.to_raster(ndvi_fp)
            c_year.rio.to_raster(c_fp)

        print("[OK] Saved timestep & annual NDVI/C rasters.")

    # ================= Catchment-level annual means (append to GPKG) =================
    catch_layer = f"{catchment_name} Data"
    y0, y1 = [pd.Timestamp(x).year for x in cfg.datetime_range.split("/")]
    years_all = list(range(y0, y1 + 1))

    try:
        base_gdf = gpd.read_file(catch_gpkg, layer=catch_layer)
    except Exception as e:
        print(f"[WARN] Could not read base layer from {catch_gpkg}: {e}")
        base_gdf = None

    if base_gdf is not None and not base_gdf.empty:
        catch_geom = base_gdf.geometry.iloc[0]
        if "Area_ha" in base_gdf.columns and pd.notna(base_gdf["Area_ha"].iloc[0]):
            area_ha_val = float(base_gdf["Area_ha"].iloc[0])
        else:
            area_ha_val = float(base_gdf.to_crs(3577).geometry.area.iloc[0]) / 10_000.0 if (
                base_gdf.crs is None or base_gdf.crs.is_geographic
            ) else float(base_gdf.geometry.area.iloc[0]) / 10_000.0

        rows = []
        for y in years_all:
            ndvi_path = os.path.join(annual_ndvi_dir, f"NDVI_{y}.tif")
            c_path    = os.path.join(annual_c_dir,    f"C_Factor_{y}.tif")
            ndvi_mean = _zonal_mean(ndvi_path, catch_geom, base_gdf.crs)
            c_mean    = _zonal_mean(c_path,    catch_geom, base_gdf.crs)
            rows.append({
                "Date": y,
                "NDVI (mean)": round(ndvi_mean, 4) if np.isfinite(ndvi_mean) else np.nan,
                "C factor (mean)": round(c_mean, 4) if np.isfinite(c_mean) else np.nan,
                "Area_ha": area_ha_val,
                "geometry": catch_geom
            })

        annual_df  = pd.DataFrame(rows)
        annual_df["Date"] = annual_df["Date"].astype("Int64")
        annual_gdf = gpd.GeoDataFrame(annual_df, geometry="geometry", crs=base_gdf.crs)

        try:
            try:
                fiona.remove(catch_gpkg, layer=catch_layer)
            except Exception:
                pass
            annual_gdf.to_file(catch_gpkg, layer=catch_layer, driver="GPKG")
            print(f"[OK] Wrote {len(annual_gdf)} catchment annual rows → {catch_gpkg} ({catch_layer}).")
        except Exception as e:
            print(f"[ERROR] Writing catchment annual layer failed: {e}")
    else:
        print("[WARN] Catchment layer missing/empty; skipped catchment annual write.")

    # ================= Riparian NDVI (per segment; per order × year) =================
    ST_gdf = gpd.read_file(stream_network)
    if ST_gdf.crs is None:
        ST_gdf = ST_gdf.set_crs(dem_crs)
    if ST_gdf.crs != dem_crs:
        ST_gdf = ST_gdf.to_crs(dem_crs)

    # Riparian polygons (union buffer around all segments)
    riparian_buf = ST_gdf.buffer(cfg.riparian_buffer_m).unary_union
    riparian_gdf = gpd.GeoDataFrame(geometry=[riparian_buf], crs=dem_crs)

    # Build years_list from annual NDVI files present
    years_list = sorted({
        int(m.group(1)) for f in os.listdir(annual_ndvi_dir)
        if (m := re.match(r"NDVI_(\d{4})\.tif$", f))
    })

    # Per-segment riparian NDVI columns + per-year maps
    for yy in years_list:
        colname = f"Riparian NDVI - {yy}"
        if colname not in ST_gdf.columns:
            ST_gdf[colname] = np.nan
        ndvi_path = os.path.join(annual_ndvi_dir, f"NDVI_{yy}.tif")
        if not os.path.exists(ndvi_path):
            print(f"[WARN] Missing annual NDVI for {yy}: {ndvi_path}")
            continue

        # Compute buffered segment means against NDVI raster
        with rio.open(ndvi_path) as src:
            st_proj = ST_gdf.to_crs(src.crs) if ST_gdf.crs != src.crs else ST_gdf
            vals = []
            for geom in st_proj.geometry:
                if geom is None or geom.is_empty:
                    vals.append(np.nan)
                    continue
                buf = geom.buffer(cfg.riparian_buffer_m)
                vals.append(_masked_mean(src, buf))

            # Write values back on the original GeoDataFrame (respecting index)
            ST_gdf[colname] = (
                pd.Series(vals, index=st_proj.index)
                  .reindex(ST_gdf.index)
                  .astype(float)
            )

        # Per-year riparian map (matches your old output)
        try:
            out_png_year = os.path.join(catch_plots, f"Riparian_NDVI_by_StreamSegment_{yy}.png")
            _plot_riparian_map_for_year(
                ST_gdf=ST_gdf,              # current streams (dem_crs)
                catch_gdf=gdf,              # catchment boundary
                sites_path=cfg.sites_path,  # site points for labels
                year=int(yy),
                out_png=out_png_year
            )
            print(f"[OK] Saved riparian map → {yy}")
        except Exception as e:
            print(f"[WARN] Riparian map for {yy} failed: {e}")

        # Also write union riparian raster (mask NDVI to global riparian buffer)
        try:
            with rio.open(ndvi_path) as src_union:
                rip_proj = riparian_gdf.to_crs(src_union.crs)
                out_img, out_tr = rio_mask(src_union, rip_proj.geometry, crop=True, filled=False)
                out_meta = src_union.meta.copy()
                out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_tr})
                rip_fp = os.path.join(riparian_output, f"Reparian_{yy}.tif")  # keep original name
                _ensure_dirs([os.path.dirname(rip_fp)])
                with rio.open(rip_fp, "w", **out_meta) as dst:
                    dst.write(out_img)
        except Exception as e:
            print(f"[WARN] Could not write union riparian raster for {yy}: {e}")

    # Per order × year summary + plot + CSV + append to catchment table
    try:
        riparian_year_cols = [c for c in ST_gdf.columns if isinstance(c, str) and c.startswith("Riparian NDVI - ")]
        if riparian_year_cols:
            years_sorted = sorted(int(c.split(" - ")[1]) for c in riparian_year_cols)
            if "order" not in ST_gdf.columns:
                raise ValueError("Streams layer lacks 'order' column (Strahler).")
            orders_int = pd.to_numeric(ST_gdf["order"], errors="coerce").astype("Int64")
            if orders_int.isna().all():
                raise ValueError("'order' column is all NaN after coercion.")
            ST_gdf = ST_gdf.assign(order_int=orders_int)

            long_rows = []
            for yy in years_sorted:
                col = f"Riparian NDVI - {yy}"
                grp = ST_gdf.groupby("order_int", dropna=True)[col].mean().reset_index()
                grp["Year"] = yy
                grp.rename(columns={col: "MeanRiparianNDVI"}, inplace=True)
                long_rows.append(grp)
            if long_rows:
                order_ts_df = pd.concat(long_rows, ignore_index=True)
                out_csv = os.path.join(riparian_output, "Riparian_NDVI_by_Order_and_Year.csv")
                order_ts_df.to_csv(out_csv, index=False)
                print(f"[OK] Wrote riparian NDVI order×year summary CSV → {out_csv}")

                pivot_df = (order_ts_df
                            .pivot_table(index="Year", columns="order_int", values="MeanRiparianNDVI", aggfunc="mean")
                            .sort_index())

                fig, ax = plt.subplots(figsize=(7, 4))
                for ord_val in sorted(pivot_df.columns.dropna()):
                    ax.plot(pivot_df.index.astype(int),
                            pivot_df[ord_val],
                            linewidth=1.2, marker="o", markersize=4,
                            label=f"Order {int(ord_val)}")
                yrs = np.array(sorted(pivot_df.index.astype(int)))
                ax.set_xlim(yrs.min(), yrs.max())
                tick_years = list(range((yrs.min() // 5) * 5, ((yrs.max() + 4) // 5) * 5 + 1, 5))
                tick_years = [y for y in tick_years if yrs.min() <= y <= yrs.max()]
                ax.set_xticks(tick_years)
                ax.set_xticklabels([str(y) for y in tick_years])
                ax.set_xticks(yrs, minor=True)
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Riparian NDVI — Mean by Stream Order (annual)", fontsize=11, fontweight="bold")
                ax.set_xlabel("Year", fontsize=10, fontweight="bold")
                ax.set_ylabel("Mean Riparian NDVI", fontsize=10, fontweight="bold")
                ax.tick_params(axis="both", labelsize=10)
                ax.grid(True, which="both", linestyle="--", alpha=0.6)
                ax.legend(title="Stream Order", loc="upper left", ncols=3, fontsize=9, title_fontsize=9, frameon=True)
                plt.tight_layout()
                out_png = os.path.join(catch_plots, "Riparian_NDVI_Mean_by_StreamOrder_Timeseries.png")
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[OK] Saved riparian NDVI time-series plot → {out_png}")

                # Append per-order columns per year into catchment GPKG
                def _ordinal_word(n: int) -> str:
                    mapping = {1:"First",2:"Second",3:"Third",4:"Fourth",5:"Fifth",
                               6:"Sixth",7:"Seventh",8:"Eighth",9:"Ninth",10:"Tenth"}
                    return mapping.get(int(n), f"{int(n)}th")

                wide = pivot_df.reset_index().rename(columns={"Year": "Date"})
                rename_map = {c: f"Riparian NDVI- {_ordinal_word(int(c))} Stream Order"
                              for c in wide.columns if c != "Date" and not pd.isna(c)}
                wide = wide.rename(columns=rename_map)
                wide["Date"] = pd.to_numeric(wide["Date"], errors="coerce").astype("Int64")

                try:
                    base_gdf = gpd.read_file(catch_gpkg, layer=catch_layer)
                    merged = base_gdf.merge(wide, on="Date", how="left")
                    try:
                        fiona.remove(catch_gpkg, layer=catch_layer)
                    except Exception:
                        pass
                    gpd.GeoDataFrame(merged, geometry="geometry", crs=base_gdf.crs).to_file(
                        catch_gpkg, layer=catch_layer, driver="GPKG"
                    )
                    print(f"[OK] Appended per-order riparian NDVI columns → {catch_gpkg} ({catch_layer}).")
                except Exception as e:
                    print(f"[ERROR] Appending per-order riparian NDVI failed: {e}")
        else:
            print("[WARN] No 'Riparian NDVI - YYYY' columns found on streams.")
    except Exception as e:
        print(f"[ERROR] Riparian NDVI order-timeseries failed: {e}")

    # Persist streams with new riparian columns
    try:
        ST_gdf.to_file(stream_network, layer="Streams", driver="GPKG", if_exists="replace")
        print(f"[OK] Wrote stream riparian NDVI columns → {stream_network} (Streams).")
    except Exception as e:
        print(f"[ERROR] Writing riparian NDVI columns failed: {e}")

    # ================= Per-site summaries & plots (Annual rasters only) =================
    sites_poly_gdf = gpd.read_file(all_sites_gpkg)  # polygons from previous module
    sites_point_gdf = gpd.read_file(cfg.sites_path)  # for plotting context (points)

    WH_rows = []
    for idx, row in sites_poly_gdf.iterrows():
        try:
            site_id = row.get("id", idx)
            site_geom = row.geometry
            attrs = row.drop(labels="geometry").to_dict()
            site_gdf = gpd.GeoDataFrame([attrs], geometry=[site_geom], crs=dem_crs)
            site_vals: Dict[str, float] = {}

            site_data_dir = os.path.join(sites_datasets, f"Site_{site_id}")
            site_plot_dir = os.path.join(sites_plots,    f"Site_{site_id}")
            _ensure_dirs([site_data_dir, site_plot_dir])

            # Iterate over /Indices/*/Annual/*.tif (NDVI, C, future indices…)
            for set1 in os.listdir(indices_output):
                set1_path = os.path.join(indices_output, set1)
                if not os.path.isdir(set1_path):
                    continue
                set2 = "Annual"
                set2_path = os.path.join(set1_path, set2)
                if not os.path.isdir(set2_path):
                    continue

                for file in os.listdir(set2_path):
                    if not file.lower().endswith(".tif"):
                        continue
                    raster_path = os.path.join(set2_path, file)
                    short_name = os.path.splitext(file)[0]

                    with rio.open(raster_path) as src:
                        # Context map (full raster + all site points)
                        full_data = src.read(1)
                        if src.nodata is not None:
                            full_data = np.where(full_data == src.nodata, np.nan, full_data)
                        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                        fig, ax = plt.subplots(figsize=(5, 5))
                        im = ax.imshow(full_data, cmap="viridis", extent=extent, origin="upper")
                        cbar = plt.colorbar(im, ax=ax, shrink=0.9); cbar.set_label(short_name)
                        pts = sites_point_gdf
                        if pts.crs != src.crs:
                            pts = pts.to_crs(src.crs)
                        pts.plot(ax=ax, markersize=15, color="red")
                        for s_idx, s_row in pts.iterrows():
                            ax.annotate(f"{s_row.get('id', s_idx)}",
                                        (s_row.geometry.x, s_row.geometry.y),
                                        xytext=(3, 3), textcoords="offset points",
                                        fontsize=7, color="black")

                        ax.set_title(short_name, fontsize=11, fontweight="bold")
                        ax.set_xlabel("Longitude",fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude",fontsize=10, fontweight="bold")
                        ax.tick_params(axis="both", labelsize=10)
                        # Limit number of ticks
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                
                        # Use scientific notation (e.g., 2.44×10⁶)
                        #fmt = ScalarFormatter(useMathText=True)
                        #fmt.set_powerlimits((-3, 3))
                        #ax.xaxis.set_major_formatter(fmt)
                        #ax.yaxis.set_major_formatter(fmt)
                
                        # Add grid for clarity (optional)
                        ax.grid(True, linestyle="--",alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(os.path.join(catch_plots, f"{short_name}.png"), dpi=300)
                        plt.close("all")

                        # Stats over this site's polygon
                        v = site_gdf.to_crs(src.crs) if site_gdf.crs != src.crs else site_gdf
                        geom = [v.geometry.iloc[0]]
                        out_image, out_transform = rio_mask(src, geom, crop=True, filled=False)
                        masked_band = out_image[0]
                        data = masked_band.filled(np.nan).astype(float)
                        site_vals[f"{short_name} (mean)"]   = round(float(np.nanmean(data)), 2) if data.size else np.nan
                        site_vals[f"{short_name} (median)"] = round(float(np.nanmedian(data)), 2) if data.size else np.nan

                        # Save clipped raster
                        clipped_meta = src.meta.copy()
                        clipped_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width":  out_image.shape[2],
                            "transform": out_transform,
                            "crs": src.crs,
                            "nodata": src.nodata
                        })
                        with rio.open(os.path.join(site_data_dir, f"{short_name}.tif"), "w", **clipped_meta) as dest:
                            dest.write(out_image)

                        # Plot clipped raster + site point (if X_site/Y_site exist)
                        try:
                            if out_transform:
                                left = out_transform[2]; top = out_transform[5]
                                pxw  = out_transform[0]; pxh = -out_transform[4]
                                right  = left + pxw * out_image.shape[2]
                                bottom = top  - pxh * out_image.shape[1]
                                ext = [left, right, bottom, top]
                            else:
                                ext = [0, out_image.shape[2], 0, out_image.shape[1]]
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(out_image[0], extent=ext, cmap="viridis", origin="upper")
                            cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
                            cbar.set_label(short_name, fontsize=9)
                            v.boundary.plot(ax=ax, color="black", linewidth=1.2)
                            if "X_site" in site_gdf.columns and "Y_site" in site_gdf.columns:
                                sx, sy = site_gdf["X_site"].iloc[0], site_gdf["Y_site"].iloc[0]
                                gpd.GeoSeries([Point(sx, sy)], crs=v.crs).plot(
                                    ax=ax, markersize=15, color="red", marker="o", label="Site location"
                                )
                                ax.annotate(f"{site_id}", (sx, sy),
                                            xytext=(5, 5), textcoords="offset points",
                                            fontsize=7, color="black")

                            ax.set_title(f"{short_name} - Site {site_id}",fontsize=11, fontweight="bold")
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
                            plt.savefig(os.path.join(site_plot_dir, f"{short_name}.png"), dpi=200)
                            plt.close("all")
                        except Exception as e:
                            print(f"[WARN] Plotting {short_name} for site {site_id} failed: {e}")

            # === Per-site mean of Reparian_{year}.tif (union riparian) ===
            if os.path.isdir(riparian_output):
                for rf in os.listdir(riparian_output):
                    if not (rf.lower().endswith(".tif") and rf.startswith("Reparian_")):
                        continue
                    year_str = os.path.splitext(rf)[0].split("_")[-1]
                    rip_fp = os.path.join(riparian_output, rf)
                    with rio.open(rip_fp) as src_rip:
                        v_rip = site_gdf.to_crs(src_rip.crs) if site_gdf.crs != src_rip.crs else site_gdf
                        geom = [v_rip.geometry.iloc[0]]
                        rip_img, _ = rio_mask(src_rip, geom, crop=True, filled=False)
                        rip_band = rip_img[0]
                        site_vals[f"Reparian_ndvi_{year_str}"] = (
                            np.nan if (rip_band.size == 0 or np.ma.count(rip_band) == 0)
                            else round(float(np.ma.mean(rip_band)), 4)
                        )

            # Per-site riparian NDVI timeseries plot (if >=2 years)
            rip_pairs = []
            for k, val in site_vals.items():
                if k.startswith("Reparian_ndvi_"):
                    try:
                        yr = int(k.split("_")[-1])
                        rip_pairs.append((yr, float(val)))
                    except Exception:
                        pass
            
            rip_pairs.sort(key=lambda t: t[0])
            years = [y for y, _v in rip_pairs]
            vals  = [v for _y, v in rip_pairs if np.isfinite(v)]

            if len(years) >= 2 and len(vals) >= 2:
                plt.figure(figsize=(7, 4))
                plt.plot(years, [vv for vv in vals], marker="o", color="black", linewidth=1.2)
                plt.xlabel("Year", fontsize=10, fontweight="bold")
                plt.ylabel("Riparian NDVI (mean)", fontsize=10, fontweight="bold")
                plt.title(f"Riparian Area mean NDVI - Site {site_id}", fontsize=11, fontweight="bold")          
                plt.grid(True, linestyle="--", alpha=0.6)
                first, last = years[0], years[-1]
                tick_years = list(range(first, last + 1, 4))
                plt.xlim(first, last); plt.xticks(tick_years, [str(y) for y in tick_years], fontsize=9)
                ax = plt.gca(); ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.tick_params(axis="both", labelsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(site_plot_dir, f"Site_{site_id}_Riparian_NDVI_Timeseries.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()

            # Persist per-site
            site_gdf = site_gdf.assign(**site_vals)
            WH_rows.append(site_gdf)
            site_gpkg = os.path.join(site_data_dir, f"Site_{site_id}.gpkg")
            site_csv  = os.path.join(site_data_dir, f"Site_{site_id}.csv")
            site_gdf.to_file(site_gpkg, driver="GPKG")
            site_gdf.drop(columns="geometry").to_csv(site_csv, index=False)
            print(f"[OK] Site {site_id} vegetation data & plots saved.")
        except Exception as e:
            print(f"[ERROR] Site {row.get('id', idx)} failed: {e}")
            continue

    # =========================================================================
    # DEACTIVATED: Combined "all sites" update/write to all_sites_gpkg & CSV
    # (as requested, we only comment these lines; no code removed or altered)
    # -------------------------------------------------------------------------
    # all_gdf = gpd.GeoDataFrame(pd.concat(WH_rows, ignore_index=True), crs=dem_crs)
    # all_sites_csv = os.path.join(sites_datasets, f"{catchment_name} Sites Data.csv")
    # all_gdf.to_file(all_sites_gpkg, driver="GPKG")
    # all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)
    # =========================================================================

    gc.collect()
    print("Done!")
    return all_sites_gpkg, dem_crs, sites_datasets, indices_output, annual_ndvi_dir, annual_c_dir
