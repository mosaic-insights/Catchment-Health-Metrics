# src/chm/landuse_risk.py
from __future__ import annotations

# ===================== Imports =====================
import os, re, gc, requests
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize as rio_rasterize
import rioxarray as rxr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# ===================== VALUE -> CLASS =====================
landuse_lookup: Dict[int | Tuple[int, ...], str] = {
    **dict.fromkeys((110, 111, 112, 113, 114, 115, 116, 117), "Nature conservation"),
    **dict.fromkeys((120, 121, 122, 123, 124, 125), "Managed resource protection"),
    **dict.fromkeys((130, 131, 132, 133, 134), "Other minimal use"),
    210: "Grazing native vegetation",
    **dict.fromkeys((220, 221, 222), "Production native forests"),
    **dict.fromkeys((310, 311, 312, 313, 314), "Plantation forests"),
    **dict.fromkeys((320, 321, 322, 323, 324, 325), "Grazing modified pastures"),
    **dict.fromkeys((330, 331, 332, 333, 334, 335, 336, 337, 338), "Cropping"),
    **dict.fromkeys((340, 341, 342, 343, 344, 345, 346, 347, 348, 349), "Perennial horticulture"),
    **dict.fromkeys((350, 351, 352, 353), "Seasonal horticulture"),
    **dict.fromkeys((360, 361, 362, 363, 364, 365), "Land in transition"),
    **dict.fromkeys((410, 411, 412, 413, 414), "Irrigated plantation forests"),
    **dict.fromkeys((420, 421, 422, 423, 424), "Grazing irrigated modified pastures"),
    **dict.fromkeys((430, 431, 432, 433, 434, 435, 436, 437, 438, 439), "Irrigated cropping"),
    **dict.fromkeys((440, 441, 442, 443, 444, 445, 446, 447, 448, 449), "Irrigated perennial horticulture"),
    **dict.fromkeys((450, 451, 452, 453, 454), "Irrigated seasonal horticulture"),
    **dict.fromkeys((460, 461, 462, 463, 464, 465), "Irrigated land in transition"),
    **dict.fromkeys((510, 511, 512, 513, 514, 515), "Intensive horticulture"),
    **dict.fromkeys((520, 521, 522, 523, 524, 525, 526, 527, 528), "Intensive animal production"),
    **dict.fromkeys((530, 531, 532, 533, 534, 535, 536, 537, 538), "Manufacturing and industrial"),
    **dict.fromkeys((540, 541), "Urban residential"),
    **dict.fromkeys((542, 543, 544, 545), "Rural residential and farm infrastructure"),
    **dict.fromkeys((550, 551, 552, 553, 554, 555), "Services"),
    **dict.fromkeys((560, 561, 562, 563, 564, 565, 566, 567), "Utilities"),
    **dict.fromkeys((570, 571, 572, 573, 574, 575), "Transport and communication"),
    **dict.fromkeys((580, 581, 582, 583, 584), "Mining"),
    **dict.fromkeys((590, 591, 592, 593, 594, 595), "Waste treatment and disposal"),
    **dict.fromkeys((610, 611, 612, 613, 614), "Lake"),
    **dict.fromkeys((620, 621, 622, 623), "Reservoir/dam"),
    **dict.fromkeys((630, 631, 632, 633), "River"),
    **dict.fromkeys((640, 641, 642, 643), "Channel/aqueduct"),
    **dict.fromkeys((650, 651, 652, 653, 654), "Marsh/wetland"),
    **dict.fromkeys((660, 661, 662, 663), "Estuary/coastal waters"),
}

# ===================== CLASS -> 10 GROUPS =====================
target_landuse_groups_10: Dict[str, List[str]] = {
    "Conservation & minimal use": ["Nature conservation", "Managed resource protection", "Other minimal use"],
    "Native forests & native grazing": ["Grazing native vegetation", "Production native forests"],
    "Plantation forests": ["Plantation forests", "Irrigated plantation forests"],
    "Dryland agriculture": ["Grazing modified pastures", "Cropping", "Perennial horticulture", "Seasonal horticulture", "Land in transition"],
    "Irrigated agriculture": ["Grazing irrigated modified pastures", "Irrigated cropping", "Irrigated perennial horticulture", "Irrigated seasonal horticulture", "Irrigated land in transition"],
    "Intensive production": ["Intensive horticulture", "Intensive animal production"],
    "Urban, residential & services": ["Urban residential", "Rural residential and farm infrastructure", "Services"],
    "Industry, utilities, mining & waste": ["Manufacturing and industrial", "Utilities", "Mining", "Waste treatment and disposal"],
    "Transport & communication": ["Transport and communication"],
    "Water & wetlands": ["Lake", "Reservoir/dam", "River", "Channel/aqueduct", "Marsh/wetland", "Estuary/coastal waters"],
}

group_ids_10: Dict[str, int] = {
    "Conservation & minimal use": 100,
    "Native forests & native grazing": 101,
    "Plantation forests": 102,
    "Dryland agriculture": 103,
    "Irrigated agriculture": 104,
    "Intensive production": 105,
    "Urban, residential & services": 106,
    "Industry, utilities, mining & waste": 107,
    "Transport & communication": 108,
    "Water & wetlands": 109,
}

group_colors_10: Dict[str, str] = {
    "Water & wetlands": "#2b8cbe",
    "Native forests & native grazing": "#31a354",
    "Conservation & minimal use": "#b2df8a",
    "Plantation forests": "#006d2c",
    "Dryland agriculture": "#e6c35b",
    "Irrigated agriculture": "#66c2a5",
    "Intensive production": "#fc8d59",
    "Urban, residential & services": "#8c510a",
    "Industry, utilities, mining & waste": "#636363",
    "Transport & communication": "#6a3d9a",
}

# ===================== Config =====================
@dataclass
class LanduseRiskConfig:
    chm_workspace: str
    catchment_path: str
    landuse_vector_path: Optional[str] = None  # Branch A (vector with 'lu_class')

    # External rasters (TWI removed)
    sdr_folder_rel: str = "Surface and Groundwater Connectivity/SDR"

    # Remote ImageServer (original)
    image_service_url: str = (
        "https://di-daa.img.arcgis.com/arcgis/rest/services/"
        "Land_and_vegetation/Catchment_Scale_Land_Use_Agricultural_Industries/ImageServer/exportImage"
    )
    image_pixel_size: int = 50  # projected units (m) in EPSG:3857
    request_timeout: int = 180

    # Behavior flags
    write_site_plots: bool = True
    write_catchment_plot: bool = True

# ===================== Utilities =====================
def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def _safe_name(x: str | int) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(x))

def _scaffold(cfg: LanduseRiskConfig) -> Tuple[str, Path, Path, Path, Path, Path, Path]:
    catch_name = Path(cfg.catchment_path).stem.replace("_", " ")
    base = Path(cfg.chm_workspace) / catch_name
    cds = base / "Catchment Datasets"
    land = cds / "Landuse"
    sds = base / "Sites Datasets"
    cps = base / "Catchment Plots and Maps"
    sps = base / "Sites Plots and Maps"
    sdr = cds / cfg.sdr_folder_rel
    _ensure_dirs([base, cds, land, sds, cps, sps, sdr])
    return catch_name, base, cds, land, sds, cps, sps

def _pick_writable_path(p: Path) -> Path:
    if not p.exists():
        return p
    try:
        p.unlink()
        return p
    except PermissionError:
        alt = p.with_stem(p.stem + "_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        print(f"[warn] {p} is locked; writing to {alt} instead.")
        return alt

def _reverse_value_to_class(lookup: Dict[int | Tuple[int, ...], str]) -> Dict[int, str]:
    rev: Dict[int, str] = {}
    for k, v in lookup.items():
        if isinstance(k, tuple):
            for val in k:
                rev[int(val)] = v
        else:
            rev[int(k)] = v
    return rev

def _precompute_catchment_ranges(
    gdf: gpd.GeoDataFrame,
    sdr_folder: Path,
) -> Tuple[float, float, List[int]]:
    """Scan SDR rasters to get min/max and list of available years. (TWI removed)"""
    sdr_min, sdr_max = np.inf, -np.inf
    years: List[int] = []
    if sdr_folder.is_dir():
        for fn in sdr_folder.iterdir():
            name = fn.name
            if "SDR" in name and name.lower().endswith(".tif"):
                m = re.search(r"\d{4}", name)
                if m:
                    try:
                        years.append(int(m.group(0)))
                    except:
                        pass
                try:
                    with rio.open(fn) as src:
                        img, _ = rio_mask(src, gdf.geometry, crop=True)
                        arr = img[0].astype(float)
                        arr[arr == src.nodata] = np.nan
                        if np.isfinite(np.nanmin(arr)):
                            sdr_min = min(sdr_min, float(np.nanmin(arr)))
                            sdr_max = max(sdr_max, float(np.nanmax(arr)))
                except:
                    pass
    if not np.isfinite(sdr_min) or not np.isfinite(sdr_max):
        sdr_min, sdr_max = 0.0, 1.0
    return sdr_min, sdr_max, sorted(set(years))

def _plot_grouped_raster(
    raster_path: Path,
    catch_gdf: gpd.GeoDataFrame,
    group_ids: Dict[str, int],
    group_colors: Dict[str, str],
    out_png: Path,
    title: str,
) -> None:
    with rio.open(raster_path) as src:
        grp = src.read(1)
        bounds, crs = src.bounds, src.crs

    grp_ma = np.ma.masked_equal(grp, 0)
    ids_present = sorted(int(i) for i in np.unique(grp_ma.compressed())) if grp_ma.compressed().size else []
    if not ids_present:
        print("[info] grouped raster has no data to plot.")
        return

    id_to_group = {gid: g for g, gid in group_ids.items()}
    labels = [id_to_group[i] for i in ids_present]
    id_to_idx = {gid: i for i, gid in enumerate(ids_present)}
    idx = np.full(grp.shape, -1, dtype=np.int32)
    colors = []
    for gid in ids_present:
        idx[grp == gid] = id_to_idx[gid]
        colors.append(group_colors.get(id_to_group[gid], "#bbbbbb"))
    idx_ma = np.ma.masked_equal(idx, -1)
    cmap = ListedColormap(colors)
    gdf_plot = catch_gdf.to_crs(crs)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(
        idx_ma,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        cmap=cmap, interpolation="none", origin="upper",
    )
    gdf_plot.boundary.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1.3)
    legend_patches = [Patch(facecolor=colors[i], edgecolor="black", label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Land Use Groups", fontsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude", fontsize=10, fontweight="bold"); ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ===================== Main API =====================
def landuse_2023(cfg: LanduseRiskConfig) -> Tuple[Path, Path]:
    """
    Land-use risk profiling with TWI fully removed.
    SDR is used for exposure curves, AUCs, and per-class AUC time series at each site.
    """
    # ---- paths & folders
    name, base, cds, land, sds, cps, sps = _scaffold(cfg)
    all_sites_gpkg = sds / f"{name} Sites Data.gpkg"
    sdr_path = cds / cfg.sdr_folder_rel

    # ---- inputs
    catch_gdf = gpd.read_file(cfg.catchment_path)
    if catch_gdf.crs is None:
        raise ValueError("Catchment input has no CRS.")

    # ---- Precompute SDR ranges + years
    sdr_min, sdr_max, sdr_years = _precompute_catchment_ranges(catch_gdf, sdr_path)

    # ---------------------- BRANCH A: vector with 'lu_class' ----------------------
    if cfg.landuse_vector_path and Path(cfg.landuse_vector_path).exists():
        print(f("[Landuse] Using provided vector: {cfg.landuse_vector_path}"))
        lu_vec = gpd.read_file(cfg.landuse_vector_path)
        if lu_vec.crs is None: lu_vec = lu_vec.set_crs(catch_gdf.crs)
        if lu_vec.crs != catch_gdf.crs: lu_vec = lu_vec.to_crs(catch_gdf.crs)
        if "lu_class" not in lu_vec.columns:
            raise ValueError("Provided landuse vector must contain a 'lu_class' column.")

        lu_catch = gpd.clip(lu_vec, catch_gdf)
        (land / "Provided_Landuse_in_Catchment.gpkg").write_bytes(b"")
        lu_catch.to_file(land / "Provided_Landuse_in_Catchment.gpkg", driver="GPKG")

        # catchment plot
        if cfg.write_catchment_plot and not lu_catch.empty:
            classes = sorted([str(x) for x in lu_catch["lu_class"].dropna().unique()])
            palette = plt.cm.get_cmap("tab20", max(1, len(classes)))
            color_map = {cls: palette(i) for i, cls in enumerate(classes)}
            fig, ax = plt.subplots(figsize=(9, 6))
            for cls, sub in lu_catch.groupby("lu_class"):
                sub.plot(ax=ax, facecolor=color_map[str(cls)], edgecolor="black", linewidth=0.3)
            catch_gdf.boundary.plot(ax=ax, color="black", linewidth=1.3)
            patches = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in classes]
            if patches:
                ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Land Use Classes", fontsize=7)
            ax.set_title(f"{name} – Land Use", fontsize=11, fontweight="bold")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel("Longitude", fontsize=10, fontweight="bold"); ax.set_ylabel("Latitude", fontsize=10, fontweight="bold"); ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis="both", labelsize=10)
            plt.tight_layout()
            plt.savefig(cps / "Catchment_Landuse_Provided.png", dpi=300, bbox_inches="tight")
            plt.close()

        # per-site profiles (SDR AUC only) + AUC time-series plots
        if all_sites_gpkg.exists():
            sites_gdf = gpd.read_file(all_sites_gpkg)
            if sites_gdf.crs != catch_gdf.crs:
                sites_gdf = sites_gdf.to_crs(catch_gdf.crs)

            classes = sorted(lu_catch["lu_class"].astype(str).unique())
            # NOTE: use plain float dtype (not pandas nullable Float32) to avoid GPKG dtype error
            updates_df = pd.DataFrame(
                np.nan,
                index=sites_gdf.index,
                columns=[f"AUC_LU_{_safe_name(c)}_SDR_{y}" for c in classes for y in sdr_years],
                dtype=float,
            )

            for idx, row in sites_gdf.iterrows():
                try:
                    site_id = row.get("Site_id", idx)
                    sgdf = gpd.GeoDataFrame([row.drop("geometry")], geometry=[row.geometry], crs=sites_gdf.crs)
                    plot_dir = sps / f"Site_{site_id}"; plot_dir.mkdir(parents=True, exist_ok=True)

                    lu_site = gpd.overlay(lu_catch, sgdf, how="intersection", keep_geom_type=True)
                    if lu_site.empty: continue

                    # Collect AUC time series per class
                    auc_ts_map: Dict[str, Dict[int, float]] = {cls: {} for cls in classes}

                    # SDR curves + AUC (rasterize LU polygons to SDR grid)
                    if sdr_path.is_dir():
                        for fn in sdr_path.iterdir():
                            if "sdr" in fn.name.lower() and fn.name.lower().endswith(".tif"):
                                with rio.open(fn) as sdr_src:
                                    simg, strf = rio_mask(sdr_src, sgdf.geometry, crop=True)
                                    sarr = simg[0]; sarr = np.where(sarr == sdr_src.nodata, np.nan, sarr)
                                    valid = ~np.isnan(sarr); n_site = int(valid.sum())
                                ym = re.search(r"\d{4}", fn.name); ystr = ym.group(0) if ym else "Unknown"
                                try: yint = int(ystr)
                                except: yint = None

                                for cls in classes:
                                    geoms = lu_site[lu_site["lu_class"].astype(str) == cls].geometry
                                    if geoms.empty or n_site == 0: continue
                                    m = rio_rasterize([(g, 1) for g in geoms], out_shape=sarr.shape, transform=strf, fill=0, dtype="uint8")
                                    sel = sarr[(m == 1) & valid]
                                    if sel.size == 0: continue
                                    s = np.sort(sel); cum_frac = (np.arange(1, len(s) + 1) / n_site)
                                    if sdr_max <= sdr_min:
                                        auc_norm = 0.0
                                    else:
                                        mask_in = (s >= sdr_min) & (s <= sdr_max)
                                        auc_norm = float(np.trapz(cum_frac[mask_in], s[mask_in]) / (sdr_max - sdr_min)) if np.any(mask_in) else 0.0

                                    updates_df.loc[idx, f"AUC_LU_{_safe_name(cls)}_SDR_{ystr}"] = auc_norm
                                    if yint is not None:
                                        auc_ts_map[cls][yint] = auc_norm

                                    if cfg.write_site_plots:
                                        cum_pct = cum_frac * 100.0
                                        plt.figure(figsize=(6, 4))
                                        plt.plot(s, cum_pct, color="black", linewidth=1.3)
                                        plt.title(f"{cls} – Exposure (SDR {ystr}) – Site {site_id}", fontsize=11, fontweight="bold")
                                        plt.xlabel("SDR", fontsize=10, fontweight="bold"); plt.ylabel("Cumulative area (% of site)", fontsize=10, fontweight="bold")
                                        plt.xlim(sdr_min, sdr_max); plt.grid(True, linestyle="--", alpha=0.6)
                                        plt.tight_layout()
                                        plt.savefig(plot_dir / f"Site_{site_id}_LU_{_safe_name(cls)}_SDR_{ystr}.png") 
                                        plt.close()

                    # ---- AUC time-series plots (per class) ----
                    if cfg.write_site_plots:
                        for cls, ts in auc_ts_map.items():
                            if not ts:
                                continue
                            years = sorted(ts.keys())
                            series = [ts[y] for y in years]
                            plt.figure(figsize=(6, 4))
                            plt.plot(years, series, marker="o", color="black", linewidth=1.3)
                            plt.xlabel("Year", fontsize=10, fontweight="bold")
                            plt.ylabel("AUC (normalised)", fontsize=10, fontweight="bold")
                            plt.title(f"Land-use {cls} — AUC time series (SDR) — Site {site_id}", fontsize=11, fontweight="bold")
                            plt.gca().tick_params(axis="both", labelsize=10)
                            plt.grid(True, linestyle="--", alpha=0.6)
                            plt.tight_layout()
                            plt.savefig(plot_dir / f"Site_{site_id}_LU_{_safe_name(cls)}_AUC_TimeSeries.png", dpi=300)
                            plt.close()

                except Exception as e:
                    print(f"[site {row.get('Site_id','NA')}] {e}")
            """
            try:
                if not updates_df.empty:
                    sites_gdf = sites_gdf.join(updates_df).copy()
                sites_gdf.to_file(all_sites_gpkg, driver="GPKG")
                print(f"[OK] AUC columns written → {all_sites_gpkg}")
            except Exception as e:
                print(f"[warn] GPKG write failed: {e}")
            """
        print("[Landuse] Vector branch complete.")
        return land, sds

    # ---------------------- BRANCH B: download via ImageServer (original) ----------------------
    print("[Landuse] Downloading land-use raster via ImageServer (original method)...")
    gdf_3857 = catch_gdf.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    pixel_size = cfg.image_pixel_size
    width_px  = int((maxx - minx) / pixel_size)
    height_px = int((maxy - miny) / pixel_size)

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "f": "json",
    }
    r = requests.get(cfg.image_service_url, params=params, timeout=cfg.request_timeout); r.raise_for_status()
    image_url = r.json()["href"]
    img = requests.get(image_url, timeout=cfg.request_timeout); img.raise_for_status()

    lu_bbox = _pick_writable_path(land / "Catchment_Landuse_bbox.tif")
    with MemoryFile(img.content) as mem:
        with mem.open() as ds:
            data = ds.read(1).astype("int32", copy=False)
            profile = ds.profile.copy(); profile.update(dtype="int32", count=1, nodata=-9999)
            with rio.open(lu_bbox, "w", **profile) as dst:
                dst.write(data, 1)

    # clip to catchment (in 3857), then reproject directly to EPSG:3111
    with rxr.open_rasterio(lu_bbox, masked=True) as rds_raw:
        rds = rds_raw.rio.write_crs("EPSG:3857", inplace=False)
    rds = rds.rio.clip(gdf_3857.geometry.values.tolist(), gdf_3857.crs, drop=True, all_touched=True)
    rds = rds.rio.reproject("EPSG:3111", resampling=Resampling.nearest)

    lu_path = _pick_writable_path(land / "Catchment_Landuse.tif")
    rds.rio.to_raster(lu_path)

    # reclass → 10 groups (values remain the original LU codes; grouping is via lookup at analysis time)
    rev = _reverse_value_to_class(landuse_lookup)
    class_to_group = {cls: g for g, lst in target_landuse_groups_10.items() for cls in lst}

    # catchment grouped map (rendered by grouping codes to 10 classes)
    # Build a grouped raster for visualisation only
    with rio.open(lu_path) as src:
        arr = src.read(1); nd = src.nodata; prof = src.profile
    prof.update(dtype="uint16", count=1, nodata=0, compress="lzw")
    out = np.zeros_like(arr, dtype=np.uint16)
    valid = (arr != nd) if nd is not None else np.ones(arr.shape, dtype=bool)
    for v in np.unique(arr[valid]):
        cls = rev.get(int(v))
        if not cls: continue
        gname = class_to_group.get(cls)
        if not gname: continue
        out[arr == v] = group_ids_10[gname]

    lu_group = _pick_writable_path(land / "Catchment_Landuse_Grouped.tif")
    with rio.open(lu_group, "w", **prof) as dst:
        dst.write(out, 1)

    # catchment plot
    if cfg.write_catchment_plot:
        _plot_grouped_raster(
            lu_group, catch_gdf, group_ids_10, group_colors_10,
            cps / "Catchment_Landuse_Grouped.png", f"{name} – Land Use (Grouped)"
        )
        print(f"[OK] grouped map → {(cps / 'Catchment_Landuse_Grouped.png').as_posix()}")

    # per-site profiles (SDR-only) + AUC time-series plots
    if all_sites_gpkg.exists():
        sites_gdf = gpd.read_file(all_sites_gpkg)
        # Use plain float dtype (not pandas nullable Float32) to avoid GPKG dtype error
        auc_cols = [f"AUC_LU_{_safe_name(g)}_SDR_{y}" for g in target_landuse_groups_10.keys() for y in sdr_years]
        updates_df = pd.DataFrame(np.nan, index=sites_gdf.index, columns=auc_cols, dtype=float)

        # Pre-open LU raster once (rioxarray)
        lu_r_full = rxr.open_rasterio(lu_path, masked=True).squeeze()

        for idx, row in sites_gdf.iterrows():
            try:
                site_id = row.get("Site_id", idx)
                sgdf = gpd.GeoDataFrame([row.drop("geometry")], geometry=[row.geometry], crs=sites_gdf.crs)
                plot_dir = sps / f"Site_{site_id}"; plot_dir.mkdir(parents=True, exist_ok=True)

                # Collect AUC time series per LU group
                auc_ts_map: Dict[str, Dict[int, float]] = {gname: {} for gname in target_landuse_groups_10.keys()}

                if sdr_path.is_dir():
                    for fn in sdr_path.iterdir():
                        if "sdr" in fn.name.lower() and fn.name.lower().endswith(".tif"):
                            # Open SDR as rioxarray for grid reference
                            sdr_ref = rxr.open_rasterio(fn).squeeze()

                            # Reproject LU raster to SDR grid (ensures identical shape/transform)
                            lu_on_sdr = lu_r_full.rio.reproject_match(sdr_ref, resampling=Resampling.nearest)

                            # Clip BOTH to the site polygon on the SAME grid
                            sg_proj = sgdf.to_crs(sdr_ref.rio.crs)
                            sdr_clip = sdr_ref.rio.clip(sg_proj.geometry, sg_proj.crs, drop=True, all_touched=True)
                            lu_clip  = lu_on_sdr.rio.clip(sg_proj.geometry, sg_proj.crs, drop=True, all_touched=True)

                            sarr = sdr_clip.values
                            sarr = np.where(np.isnan(sarr), np.nan, sarr)  # keep NaNs
                            valid = ~np.isnan(sarr); n_site = int(valid.sum())

                            ym = re.search(r"\d{4}", Path(fn).name); ystr = ym.group(0) if ym else "Unknown"
                            try: yint = int(ystr)
                            except: yint = None

                            # Compute per-group AUCs using LU codes on SDR grid
                            lu_vals = lu_clip.values.astype("float64")
                            for gname, classes in target_landuse_groups_10.items():
                                codes = [code for code, lbl in rev.items() if lbl in classes]
                                if not codes or n_site == 0:
                                    continue
                                mask_lu = np.isin(lu_vals, codes)
                                sel = sarr[mask_lu & valid]
                                if sel.size == 0:
                                    continue
                                s = np.sort(sel.ravel())
                                cum_frac = (np.arange(1, len(s) + 1) / n_site)
                                if sdr_max <= sdr_min:
                                    auc_norm = 0.0
                                else:
                                    mask_in = (s >= sdr_min) & (s <= sdr_max)
                                    auc_norm = float(np.trapz(cum_frac[mask_in], s[mask_in]) / (sdr_max - sdr_min)) if np.any(mask_in) else 0.0

                                updates_df.loc[idx, f"AUC_LU_{_safe_name(gname)}_SDR_{ystr}"] = auc_norm
                                if yint is not None:
                                    auc_ts_map[gname][yint] = auc_norm

                                if cfg.write_site_plots:
                                    # Save cumulative curve for context
                                    cum_pct = cum_frac * 100.0
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.plot(s, cum_pct, color="black", linewidth=1.3)
                                    ax.set_title(f"{gname} – Exposure (SDR {ystr}) – Site {site_id}", fontsize=11, fontweight="bold")
                                    ax.set_xlabel("SDR", fontsize=10, fontweight="bold")
                                    ax.set_ylabel("Cumulative area (% of site)", fontsize=10, fontweight="bold")
                                    ax.set_xlim(sdr_min, sdr_max)
                                    ax.grid(True, linestyle="--", alpha=0.6)
                                    ax.tick_params(axis="both", labelsize=10)
                                    plt.tight_layout()
                                    plt.savefig(plot_dir / f"Site_{site_id}_LU_{_safe_name(gname)}_SDR_{ystr}.png")
                                    plt.close()

                # ---- AUC time-series plots (per LU group) ----
                if cfg.write_site_plots:
                    for gname, ts in auc_ts_map.items():
                        if not ts:
                            continue
                        years = sorted(ts.keys())
                        series = [ts[y] for y in years]
                        plt.figure(figsize=(6, 4))
                        plt.plot(years, series, marker="o", color="black", linewidth=1.3)
                        plt.xlabel("Year", fontsize=10, fontweight="bold")
                        plt.ylabel("AUC (normalised)", fontsize=10, fontweight="bold")
                        plt.title(f"Land-use {gname} — AUC (SDR) — Site {site_id}", fontsize=11, fontweight="bold")
                        plt.gca().tick_params(axis="both", labelsize=10)
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(plot_dir / f"Site_{site_id}_LU_{_safe_name(gname)}_AUC_TimeSeries.png", dpi=300)
                        plt.close()
                print(f"Site {site_id} is processed")
            except Exception as e:
                print(f"[site {row.get('Site_id','NA')}] {e}")
        """
        try:
            if not updates_df.empty:
                sites_gdf = sites_gdf.join(updates_df).copy()
            sites_gdf.to_file(all_sites_gpkg, driver="GPKG")
            print(f"[OK] AUC columns written → {all_sites_gpkg}")
        except Exception as e:
            print(f"[warn] GPKG write failed: {e}")
        """
    print("[Landuse] Raster branch complete.")
    return land, sds