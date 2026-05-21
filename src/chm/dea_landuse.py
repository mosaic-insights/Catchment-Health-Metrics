from __future__ import annotations

# ---------- stdlib ----------
import os
import re
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- third-party ----------
import requests
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# ========================= Config =========================

@dataclass
class DEALanduseConfig:
    chm_workspace: str
    catchment_path: str
    start_year: int
    end_year: int

    # Download/cache
    download_if_missing: bool = True
    dea_level: str = "level3"  # "level3" or "level4" (if available on server)
    dea_base_root: str = (
        "https://thredds.nci.org.au/thredds/fileServer/"
        "jw04/ga_ls_landcover_class_cyear_3/2-0-0/continental_mosaics"
    )
    request_timeout: int = 180
    chunk_bytes: int = 1024 * 1024

    # Behaviour
    make_maps: bool = True
    make_site_outputs: bool = True

    # Class mapping (code -> label/color). 255 = NoData (ignored)
    classes: Dict[int, str] = field(default_factory=lambda: {
        111: "Cultivated Terrestrial Vegetation",
        112: "Natural Terrestrial Vegetation",
        124: "Natural Aquatic Vegetation",
        215: "Artificial Surface",
        216: "Natural Bare Surface",
        220: "Water",
    })
    colors: Dict[int, str] = field(default_factory=lambda: {
        111: "#e6c35b",
        112: "#31a354",
        124: "#2b8cbe",
        215: "#6a3d9a",
        216: "#bdbdbd",
        220: "#1f78b4",
    })


# ========================= Helpers =========================

def _log(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _scaffold(cfg: DEALanduseConfig) -> Tuple[str, Path, Path, Path, Path, Path]:
    name = Path(cfg.catchment_path).stem.replace("_", " ")
    base = Path(cfg.chm_workspace) / name
    cds = base / "Catchment Datasets"
    land = cds / "Landuse"
    dea = land / "DEA Landcover"
    sds = base / "Sites Datasets"
    sps = base / "Sites Plots and Maps"
    cps = base / "Catchment Plots and Maps"
    _ensure_dirs([base, cds, land, dea, sds, sps, cps])
    return name, cds, dea, sds, sps, cps


def _stream_download(url: str, out_fp: Path, timeout: int, chunk: int) -> None:
    headers = {"User-Agent": "python-requests"}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        with out_fp.open("wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)


def _valid_tif(fp: Path) -> bool:
    try:
        if not fp.exists() or fp.stat().st_size == 0:
            return False
        with rio.open(fp) as src:
            _ = src.count
        return True
    except Exception:
        return False


def _clip_and_pct(
    raster_path: Path,
    gdf_geom: gpd.GeoDataFrame,
    class_codes: List[int],
) -> Dict[int, float]:
    """Clip raster to geometries and compute % by class code (exclude nodata = 255 by DEA convention)."""
    with rio.open(raster_path) as src:
        geom_in = gdf_geom.to_crs(src.crs)
        geoms = [g for g in geom_in.geometry if g and not g.is_empty]
        if not geoms:
            return {c: 0.0 for c in class_codes}

        img, _ = rio_mask(src, geoms, crop=True)
        arr = img[0]
        nd = src.nodata if src.nodata is not None else 255

    valid = (arr != nd)
    if not np.any(valid):
        return {c: 0.0 for c in class_codes}

    total = int(valid.sum())
    out: Dict[int, float] = {}
    for code in class_codes:
        cnt = int(np.count_nonzero(arr == code))
        out[code] = 100.0 * cnt / total if total > 0 else 0.0
    return out


def _reproject_to_catchment_crs(
    raster_path: Path,
    catch_gdf: gpd.GeoDataFrame,
) -> Path:
    """
    Reproject a clipped DEA raster to match the catchment CRS.
    Uses nearest-neighbour to preserve class codes.
    """
    out_path = raster_path.with_name(raster_path.stem + "_catchCRS.tif")
    if out_path.exists():
        return out_path

    dst_crs = catch_gdf.crs
    if dst_crs is None:
        raise ValueError("Catchment CRS is undefined.")

    with rio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rio.open(out_path, "w", **kwargs) as dst:
            reproject(
                source=rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

    return out_path


def _plot_catchment_map(
    catchment_name: str,
    raster_path: Path,
    gdf_catch: gpd.GeoDataFrame,
    code_to_label: Dict[int, str],
    code_to_color: Dict[int, str],
    year: int,
    out_png: Path,
) -> None:
    # Reproject raster to geographic coordinates for plotting only
    with rio.open(raster_path) as src:
        arr = src.read(1)
        nd = src.nodata if src.nodata is not None else 255
        src_meta = src.meta.copy()

    present_vals = [
        int(v) for v in np.unique(arr)
        if int(v) in code_to_label and int(v) != nd
    ]
    if not present_vals:
        return

    present_vals.sort()
    code_to_idx = {v: i for i, v in enumerate(present_vals)}
    idx = np.full(arr.shape, -1, dtype=np.int32)

    colors, labels = [], []
    for v in present_vals:
        idx[arr == v] = code_to_idx[v]
        colors.append(code_to_color.get(v, "#bbbbbb"))
        labels.append(f"{v} – {code_to_label[v]}")

    idx_ma = np.ma.masked_equal(idx, -1)
    cmap = ListedColormap(colors)

    # --- reproject for plotting in degrees ---
    src_crs = src_meta["crs"]
    src_transform = src_meta["transform"]
    src_width = src_meta["width"]
    src_height = src_meta["height"]

    left, bottom, right, top = rio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, "EPSG:4326", src_width, src_height, left, bottom, right, top
    )

    dst_array = np.full((dst_height, dst_width), -1, dtype=np.int32)

    reproject(
        source=idx.astype("int32"),
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=-1,
        dst_nodata=-1,
    )

    xmin, ymin, xmax, ymax = rio.transform.array_bounds(dst_height, dst_width, dst_transform)
    extent = (xmin, xmax, ymin, ymax)

    idx_ma_plot = np.ma.masked_equal(dst_array, -1)
    catch_plot = gdf_catch.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        idx_ma_plot,
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
    )

    catch_plot.boundary.plot(
        ax=ax, edgecolor="black", facecolor="none", linewidth=1.3
    )

    legend_patches = [
        Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        frameon=False,
    )

    ax.set_title(
        f"{catchment_name} – DEA Land Cover ({year})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Latitude (°)", fontsize=9, fontweight="bold")

    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(
    df_pct: pd.DataFrame,
    code_to_label: Dict[int, str],
    title: str,
    ylabel: str,
    out_png: Path,
) -> None:
    if df_pct.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for code in df_pct.columns:
        label = code_to_label.get(code, str(code))
        ax.plot(df_pct.index, df_pct[code], linewidth=1.3, label=f"{label}")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    years = df_pct.index.astype(int)
    ax.set_xticks(years)
    ax.set_xticklabels(years)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ========================= Main API =========================

def dea_landuse_change(cfg: DEALanduseConfig) -> Tuple[Path, Path]:
    """
    Download/cache DEA Land Cover annual mosaics, clip to catchment, compute % by class,
    produce catchment maps & time series; optionally do the same per site.

    Returns:
        (dea_dataset_folder, sites_dataset_folder)
    """
    _log("INFO", "Starting DEA land cover processing...")

    # ---- folders / names ----
    name, catch_ds, dea_ds, sites_ds, sites_plots, catch_plots = _scaffold(cfg)
    all_sites_gpkg = sites_ds / f"{name} Sites Data.gpkg"

    _log("OK", f"Workspace folders prepared for {name}")
    _log("INFO", f"DEA dataset folder: {dea_ds}")

    # ---- inputs ----
    catch_gdf = gpd.read_file(cfg.catchment_path)
    if catch_gdf.crs is None:
        raise ValueError("Catchment file has no CRS. Please define one before running.")
    _log("OK", f"Catchment loaded: {cfg.catchment_path}")

    class_codes = list(cfg.classes.keys())

    # ---- (1) download + clip per year ----
    if cfg.download_if_missing:
        _log("INFO", "Checking/downloading DEA Land Cover yearly mosaics...")
        for year in range(cfg.start_year, cfg.end_year + 1):
            fname = f"ga_ls_landcover_class_cyear_3_mosaic_{year}--P1Y_{cfg.dea_level}.tif"
            url = f"{cfg.dea_base_root}/{year}--P1Y/{fname}"
            raw_tif = dea_ds / fname
            clipped_tif = dea_ds / f"Landuse_{year}_clipped.tif"

            try:
                if not _valid_tif(raw_tif):
                    _log("INFO", f"Downloading DEA Land Cover for {year}")
                    _stream_download(url, raw_tif, timeout=cfg.request_timeout, chunk=cfg.chunk_bytes)
                    _log("OK", f"Downloaded raw DEA raster for {year}")
                else:
                    _log("INFO", f"Raw DEA raster already exists for {year}")

                if not _valid_tif(clipped_tif):
                    with rio.open(raw_tif) as src:
                        geom_in = catch_gdf.to_crs(src.crs)
                        geoms = [g for g in geom_in.geometry if g and not g.is_empty]
                        if not geoms:
                            _log("WARN", f"No valid geometries for clipping in {year}; skipping")
                            continue
                        img, tr = rio_mask(src, geoms, crop=True, nodata=src.nodata)
                        meta = src.meta.copy()
                        meta.update({"height": img.shape[1], "width": img.shape[2], "transform": tr})
                        with rio.open(clipped_tif, "w", **meta) as dst:
                            dst.write(img)
                    _log("OK", f"Saved clipped DEA raster for {year}")
                else:
                    _log("INFO", f"Clipped DEA raster already exists for {year}")

            except requests.HTTPError as e:
                _log("WARN", f"HTTP error for {year}: {e}")
            except requests.RequestException as e:
                _log("WARN", f"Network error for {year}: {e}")
            except Exception as e:
                _log("WARN", f"Error processing DEA raster for {year}: {e}")

    # ---- (2) catchment-level % by class, maps, time series ----
    _log("INFO", "Computing catchment class percentages and maps...")
    rows: List[Dict[int, float]] = []
    for year in range(cfg.start_year, cfg.end_year + 1):
        clipped_tif = dea_ds / f"Landuse_{year}_clipped.tif"
        if not _valid_tif(clipped_tif):
            _log("WARN", f"Missing clipped DEA raster for {year}; skipping")
            continue

        pct = _clip_and_pct(clipped_tif, catch_gdf, class_codes)
        rows.append({"Year": year, **pct})
        _log("OK", f"Catchment percentages computed for {year}")
        # Clean yearly raster arrays
        plt.close("all")

        try:
            del pct
        except Exception:
            pass

        gc.collect()

        if cfg.make_maps:
            try:
                _plot_catchment_map(
                    name,
                    clipped_tif,
                    catch_gdf,
                    code_to_label=cfg.classes,
                    code_to_color=cfg.colors,
                    year=year,
                    out_png=catch_plots / f"Catchment_DEA_LandCover_{year}.png",
                )
                _log("OK", f"Catchment map saved for {year}")
            except Exception as e:
                _log("WARN", f"Catchment map failed for {year}: {e}")

    catch_pct_df = pd.DataFrame(rows).set_index("Year").sort_index()
    if not catch_pct_df.empty:
        out_csv = dea_ds / "Catchment_LU_Percentages.csv"
        out_csv_ = catch_plots / "Catchment_LU_Percentages.csv"

        catch_pct_df_named = catch_pct_df.rename(columns=cfg.classes)
        catch_pct_df_named.to_csv(out_csv, float_format="%.3f")
        catch_pct_df_named.to_csv(out_csv_, float_format="%.3f")
        _log("OK", f"Catchment percentage CSV saved in datasets and plots folders")

        ts_pct = catch_plots / "Catchment_LU_Percentages_OverTime.png"
        _plot_timeseries(
            catch_pct_df[class_codes],
            cfg.classes,
            f"{name} – Class Percentages Over Time",
            "Percent (%)",
            ts_pct
        )
        _log("OK", "Catchment percentages over time plot saved")

        base = catch_pct_df.iloc[0]
        pp_change = catch_pct_df[class_codes].subtract(base[class_codes], axis="columns")

        pp_change_named = pp_change.rename(columns=cfg.classes)
        pp_change_csv = dea_ds / "Catchment_LU_PercentagePointChange.csv"
        pp_change_csv.parent.mkdir(parents=True, exist_ok=True)
        pp_change_named.to_csv(pp_change_csv, float_format="%.3f")
        _log("OK", "Catchment percentage-point change CSV saved")

        ts_pp = catch_plots / "Catchment_LU_PercentagePointChange.png"
        _plot_timeseries(
            pp_change,
            cfg.classes,
            f"{name} – Percentage-Point Change (vs {catch_pct_df.index[0]})",
            "Δ percentage points",
            ts_pp,
        )
        _log("OK", "Catchment percentage-point change plot saved")
    else:
        _log("INFO", "No catchment rows computed.")

    # ---- (3) per-site % by class (optional) ----
    if cfg.make_site_outputs:
        if all_sites_gpkg.exists():
            try:
                sites_gdf = gpd.read_file(all_sites_gpkg)
                if sites_gdf.crs and (sites_gdf.crs != catch_gdf.crs):
                    sites_gdf = sites_gdf.to_crs(catch_gdf.crs)
                _log("OK", f"Sites loaded: {all_sites_gpkg}")
            except Exception as e:
                _log("WARN", f"Could not read sites GPKG: {e}")
                sites_gdf = None

            if sites_gdf is not None and len(sites_gdf) > 0:
                _log("INFO", "Computing site class percentages and time series...")
                for _, row in sites_gdf.iterrows():
                    site_id = row.get("Site_id", None)
                    _log("INFO", f"Processing Site {site_id}...")

                    sgdf = gpd.GeoDataFrame([row.drop(labels=["geometry"])], geometry=[row.geometry], crs=sites_gdf.crs)

                    series: List[Dict[int, float]] = []
                    for year in range(cfg.start_year, cfg.end_year + 1):
                        clipped_tif = dea_ds / f"Landuse_{year}_clipped.tif"
                        if not _valid_tif(clipped_tif):
                            continue
                        try:
                            pct = _clip_and_pct(clipped_tif, sgdf, class_codes)
                            series.append({"Year": year, **pct})
                            _log("OK", f"Site {site_id}: percentages computed for {year}")
                        except Exception as e:
                            _log("WARN", f"Site {site_id}, year {year} failed: {e}")

                    site_df = pd.DataFrame(series).set_index("Year").sort_index()
                    site_dir = sites_ds / f"Site_{site_id}"
                    site_plot_dir = (Path(cfg.chm_workspace) / name / "Sites Plots and Maps" / f"Site_{site_id}")
                    _ensure_dirs([site_dir, site_plot_dir])

                    if site_df.empty:
                        _log("INFO", f"Site {site_id}: no valid rows computed")
                        continue

                    out_site_csv = site_plot_dir / f"Site_{site_id}_LU_Percentages.csv"
                    site_df_named = site_df.rename(columns=cfg.classes)
                    site_df_named.to_csv(out_site_csv, float_format="%.3f")
                    _log("OK", f"Site {site_id}: percentage CSV saved")

                    _plot_timeseries(
                        site_df[class_codes],
                        cfg.classes,
                        f"Site {site_id} – Class Percentages Over Time",
                        "Percent (%)",
                        site_plot_dir / f"Site_{site_id}_LU_Percentages_OverTime.png",
                    )
                    _log("OK", f"Site {site_id}: percentages over time plot saved")

                    base_site = site_df.iloc[0]
                    site_pp = site_df[class_codes].subtract(base_site[class_codes], axis="columns")

                    site_pp_named = site_pp.rename(columns=cfg.classes)
                    site_pp_named.to_csv(site_plot_dir / f"Site_{site_id}_LU_PercentagePointChange.csv", float_format="%.3f")
                    _log("OK", f"Site {site_id}: percentage-point change CSV saved")

                    _plot_timeseries(
                        site_pp,
                        cfg.classes,
                        f"Site {site_id} – Percentage-Point Change (vs {site_df.index[0]})",
                        "Δ percentage points",
                        site_plot_dir / f"Site_{site_id}_LU_PercentagePointChange.png",
                    )
                    _log("OK", f"Site {site_id}: percentage-point change plot saved")

                    # =========================
                    # Per-site cleanup
                    # =========================
                    plt.close("all")

                    try:
                        del sgdf
                    except Exception:
                        pass

                    try:
                        del site_df, site_df_named
                    except Exception:
                        pass

                    try:
                        del site_pp, site_pp_named
                    except Exception:
                        pass

                    try:
                        del series, pct
                    except Exception:
                        pass

                    try:
                        del base_site
                    except Exception:
                        pass

                    gc.collect()

                    _log("INFO", f"Site {site_id}: memory cleanup completed.")
                    _log("OK", f"Site {site_id} completed")
            else:
                _log("INFO", "No sites found; skipping site outputs.")
        else:
            _log("WARN", "Sites GPKG missing; site outputs will be skipped.")
    else:
        _log("INFO", "Site outputs disabled.")

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del catch_gdf
    except Exception:
        pass

    try:
        del sites_gdf, sgdf
    except Exception:
        pass

    try:
        del catch_pct_df, catch_pct_df_named
    except Exception:
        pass

    try:
        del pp_change, pp_change_named
    except Exception:
        pass

    try:
        del site_df, site_df_named
    except Exception:
        pass

    try:
        del site_pp, site_pp_named
    except Exception:
        pass

    try:
        del rows, series
    except Exception:
        pass

    try:
        del pct, base, base_site
    except Exception:
        pass

    gc.collect()

    _log("INFO", "Memory cleanup completed.")
    _log("OK", "DEA land cover processing completed.")

    return dea_ds, sites_ds