from __future__ import annotations

# ---------- stdlib ----------
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable

# ---------- third-party ----------
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import gc


# ====================== configuration ======================
@dataclass
class MonitoringConfig:
    """
    Configure monitoring append + plotting.

    Pass your own `variables` list to control which variables are analysed & plotted.
    If you omit `variables` (None/[]), the code will try to auto-detect useful
    variables per your data (see `auto_detect_*` flags).

    Example:
        cfg = MonitoringConfig(
            chm_workspace="...",
            catchment_path="...",
            monitoring_folder="...",
            variables=["Total Phosphorus", "Total Nitrogen", "Turbidity", "pH"],  # <- fully customizable
            thresholds={"pH": 7.0},  # can be number or callable(mean)->float
        )
    """
    chm_workspace: str
    catchment_path: str
    monitoring_folder: str

    # file & column conventions
    site_file_pattern: str = "Site_{site_id}.xlsx"   # f-string with {site_id}
    date_column: str = "Date_Sampled"

    # >>> You can override this when creating cfg. If None/empty, we auto-detect.
    variables: Optional[List[str]] = None

    # Auto-detection behaviour when variables is None/[]:
    auto_detect_known_names: bool = True      # prefer common WQ names first
    auto_detect_all_numeric: bool = True      # then include other numeric columns
    auto_detect_exclude: List[str] = field(default_factory=lambda: [
        "east", "easting", "north", "northing", "lon", "longitude", "lat", "latitude",
        "x", "y", "geometry", "Site_id", "id", "station", "name"
    ])

    # thresholds: mapping variable -> callable(mean) or literal; fallback uses default_other_rule
    thresholds: Dict[str, float | Callable[[float], float]] = field(default_factory=lambda: {"pH": 7.0})
    default_other_rule: Callable[[float], float] = staticmethod(lambda m: 1.5 * m)  # WQT = 1.5 × mean

    # plotting
    figsize: Tuple[int, int] = (10, 12)
    dpi: int = 200
    line_kwargs: Dict = field(default_factory=lambda: dict(
        color="gray", linewidth=1, marker="o", markersize=4,
        markerfacecolor="black", markeredgecolor="black"
    ))

    # Fonts (axis labels & ticks)
    label_fontsize: int = 10
    label_fontweight: str = "bold"
    tick_fontsize: int = 10
    tick_fontweight: str = "bold"


# ====================== helpers ======================
def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _safe_mode(series: pd.Series) -> float:
    m = series.mode(dropna=True)
    return float(m.iloc[0]) if len(m) else np.nan


def _threshold_for(var: str, mean_val: float, cfg: MonitoringConfig) -> float:
    rule = cfg.thresholds.get(var, cfg.default_other_rule)
    return float(rule(mean_val)) if callable(rule) else float(rule)


def _read_sites_gpkg(base_sites: Path, catchment_name: str) -> Optional[gpd.GeoDataFrame]:
    gpkg = base_sites / f"{catchment_name} Sites Data.gpkg"
    if not gpkg.exists():
        return None
    gdf = gpd.read_file(gpkg)
    if "Site_id" not in gdf.columns:
        raise ValueError("Sites GPKG must have a 'Site_id' column.")
    return gdf


def _normalize_name(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def _auto_detect_variables(df: pd.DataFrame, cfg: MonitoringConfig) -> List[str]:
    """
    Build a variables list from a dataframe when cfg.variables is None/[].
    Strategy:
      1) find common WQ names (case-insensitive) if present
      2) add other numeric columns (exclude id/coord columns)
    """
    if df is None or df.empty:
        return []

    # Known names (case-insensitive) you can expand
    known_names = [
        "total phosphorus", "tp",
        "total nitrogen", "tn",
        "turbidity",
        "ph",
        "electrical conductivity", "ec",
        "dissolved oxygen", "do",
        "nh4", "nh3", "nox", "no3", "po4",
        "temperature", "water temperature",
    ] if cfg.auto_detect_known_names else []

    cols = list(df.columns)
    low_map = {_normalize_name(c): c for c in cols}

    picked: List[str] = []
    for k in known_names:
        if k in low_map:
            picked.append(low_map[k])

    if cfg.auto_detect_all_numeric:
        # add numeric columns not already picked & not excluded
        excluded = set(_normalize_name(c) for c in cfg.auto_detect_exclude)
        for c in cols:
            lc = _normalize_name(c)
            if lc in excluded or c in picked or c == cfg.date_column:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                picked.append(c)

    # Keep order as discovered; de-dup
    seen = set()
    out = []
    for p in picked:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _style_axis(ax, cfg: MonitoringConfig):
    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight(cfg.tick_fontweight)


def _plot_site_timeseries(
    site_id: str | int,
    df: pd.DataFrame,
    date_col: str,
    variables: List[str],
    out_png: Path,
    cfg: MonitoringConfig,
):
    if df.empty or date_col not in df.columns:
        return
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    if d.empty:
        return

    dates = d[date_col].values

    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=cfg.figsize, sharex=True)
    locator, formatter = AutoDateLocator(), AutoDateFormatter(AutoDateLocator())

    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(var, fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight)
        _style_axis(ax, cfg)

        if var not in d.columns:
            ax.text(0.5, 0.5, f"'{var}' not found", ha="center", va="center", transform=ax.transAxes)
            continue

        y = pd.to_numeric(d[var], errors="coerce").values
        ax.plot(dates, y, **cfg.line_kwargs)

        finite_mask = np.isfinite(y)
        if finite_mask.sum() >= 1:
            mean_val = float(np.nanmean(y))
            ax.axhline(mean_val, linestyle="--", linewidth=1.2, color="gray")
            ax.text(
                0.01, 0.90, f"Mean = {mean_val:.3g}",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight
            )

            wqt = _threshold_for(var, mean_val, cfg)
            ax.axhline(wqt, linestyle=":", linewidth=1.2, color="red")

            prop_exceed = float((y[finite_mask] > wqt).sum() / finite_mask.sum())
            ax.text(
                0.01, 0.78, f"WQT = {wqt:.3g}\n% > WQT = {prop_exceed*100:.1f}%",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight, color="black"
            )

    axes[-1].set_xlabel("Time", fontsize=cfg.label_fontsize, fontweight=cfg.label_fontweight)
    fig.autofmt_xdate()
    fig.suptitle(f"Site {site_id} – Monitoring Data", fontsize=cfg.label_fontsize + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)


def _site_summary(
    site_id: str | int,
    df: pd.DataFrame,
    variables: List[str],
    date_col: str,
    cfg: MonitoringConfig,
) -> pd.DataFrame:
    out: List[Dict] = []
    d = df.copy()

    # Parse dates if present (not strictly required for stats)
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col)

    for var in variables:
        if var in d.columns:
            s = pd.to_numeric(d[var], errors="coerce")
            n_valid = int(s.notna().sum())
            n_missing = int(s.isna().sum())
            gap_pct = float(n_missing / n_valid * 100) if n_valid > 0 else np.nan

            mean_v = float(np.nanmean(s)) if n_valid > 0 else np.nan
            med_v = float(np.nanmedian(s)) if n_valid > 0 else np.nan
            min_v = float(np.nanmin(s)) if n_valid > 0 else np.nan
            max_v = float(np.nanmax(s)) if n_valid > 0 else np.nan
            std_v = float(np.nanstd(s, ddof=1)) if n_valid >= 2 else np.nan
            var_v = float(np.nanvar(s, ddof=1)) if n_valid >= 2 else np.nan
            mode_v = _safe_mode(s)

            if np.isfinite(mean_v):
                wqt = _threshold_for(var, mean_v, cfg)
                prop_exceed = float((s > wqt).sum() / n_valid) if n_valid > 0 else np.nan
            else:
                wqt, prop_exceed = np.nan, np.nan
        else:
            mean_v = med_v = min_v = max_v = std_v = var_v = mode_v = gap_pct = wqt = prop_exceed = np.nan

        out.append({
            "Site ID": site_id,
            "Variable": var,
            "Mean": mean_v,
            "Median": med_v,
            "Min": min_v,
            "Max": max_v,
            "Std": std_v,
            "Variance": var_v,
            "Mode": mode_v,
            "Gap % (missing/observed)": gap_pct,
            "WQT": wqt,
            "Proportion > WQT": prop_exceed,
        })

    return pd.DataFrame(out)


# ====================== main API ======================
def monitoring_data(cfg: MonitoringConfig) -> Path:
    """
    Append per-site monitoring Excel to spatial site rows, write per-site GPKG/CSV,
    produce time-series plots with mean & WQT, and output per-site + combined summaries.
    Returns the base 'Sites Datasets' folder.

    Variables to analyse/plot are taken from:
      - cfg.variables, if provided; otherwise
      - auto-detected from each site's Excel file (may vary per site).
    """
    # ---- paths
    catchment_name = Path(cfg.catchment_path).stem.replace("_", " ")
    base = Path(cfg.chm_workspace) / catchment_name
    catch_plots = base / "Catchment Plots and Maps"
    sites_ds = base / "Sites Datasets"
    sites_plots = base / "Sites Plots and Maps"
    _ensure_dirs([base, catch_plots, sites_ds, sites_plots])

    # ---- load sites
    sites_gdf = _read_sites_gpkg(sites_ds, catchment_name)
    if sites_gdf is None:
        print(f"[info] sites GPKG missing for {catchment_name}; monitoring append skipped.")
        return sites_ds

    crs_sites = sites_gdf.crs
    combined_summaries: List[pd.DataFrame] = []

    for _, row in sites_gdf.iterrows():
        site_id = row["Site_id"]
        folder_name = f"Site_{site_id}"
        site_folder = sites_ds / folder_name
        site_folder.mkdir(parents=True, exist_ok=True)
        plot_folder = sites_plots / folder_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        # expected Excel path
        excel_path = Path(cfg.monitoring_folder) / cfg.site_file_pattern.format(site_id=site_id)
        if not excel_path.exists():
            print(f"[info] monitoring file missing for site {site_id}: {excel_path}")
            continue

        try:
            # 1) read data
            mdf = pd.read_excel(excel_path).copy()
            if cfg.date_column in mdf.columns:
                mdf[cfg.date_column] = pd.to_datetime(mdf[cfg.date_column], errors="coerce")
                mdf = mdf.dropna(subset=[cfg.date_column]).sort_values(cfg.date_column)

            # Resolve variables for THIS site:
            if cfg.variables and len(cfg.variables) > 0:
                vars_for_site = cfg.variables
            else:
                vars_for_site = _auto_detect_variables(mdf, cfg)
                if not vars_for_site:
                    print(f"[warn] no analyzable variables found for site {site_id}.")
                    continue

            # 2) build GeoDataFrame (single geometry replicated, attributes on first row only)
            site_attrs = row.drop(labels=["geometry"]).to_dict()
            rep = pd.DataFrame([site_attrs] * len(mdf))
            if len(mdf) > 1:
                rep.iloc[1:, :] = np.nan
            rep["geometry"] = [row.geometry] * len(mdf)
            combined = pd.concat([rep.reset_index(drop=True), mdf.reset_index(drop=True)], axis=1)
            gdf_out = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs_sites)

            # 3) write per-site GPKG + CSV
            gdf_out.to_file(site_folder / f"Site_{site_id}.gpkg", driver="GPKG")
            gdf_out.drop(columns="geometry").to_csv(site_folder / f"Site_{site_id}.csv", index=False)

            # 4) per-site summary
            summary_df = _site_summary(site_id, mdf, vars_for_site, cfg.date_column, cfg)
            summary_df.to_csv(site_folder / f"Site_{site_id}_Monitoring_Summary.csv", index=False)
            combined_summaries.append(summary_df)

            # 5) plot
            _plot_site_timeseries(
                site_id=site_id,
                df=mdf,
                date_col=cfg.date_column,
                variables=vars_for_site,
                out_png=plot_folder / f"Site_{site_id}_Monitoring_TimeSeries.png",
                cfg=cfg,
            )
            # =========================
            # Per-site cleanup
            # =========================
            plt.close("all")

            try:
                del mdf, rep, combined, gdf_out
            except Exception:
                pass

            try:
                del summary_df, vars_for_site, site_attrs
            except Exception:
                pass

            gc.collect()

            print(f"[INFO] site {site_id}: memory cleanup completed.")
            print(f"[OK] processed site {site_id}")

        except Exception as e:
            print(f"[error] site {site_id}: {e}")

    # combined summary across sites
    if combined_summaries:
        all_summary = pd.concat(combined_summaries, ignore_index=True)
        all_summary.to_csv(sites_ds / "All_Sites_Monitoring_Summary.csv", index=False)
        print(f"[OK] combined summary → {(sites_ds / 'All_Sites_Monitoring_Summary.csv').as_posix()}")

    # =========================
    # Memory cleanup
    # =========================
    plt.close("all")

    try:
        del sites_gdf
    except Exception:
        pass

    try:
        del combined_summaries, all_summary
    except Exception:
        pass

    try:
        del mdf, rep, combined, gdf_out
    except Exception:
        pass

    try:
        del summary_df, vars_for_site, site_attrs
    except Exception:
        pass

    gc.collect()

    print("[INFO] Memory cleanup completed.")
    print("[done] monitoring data appended.")

    return sites_ds