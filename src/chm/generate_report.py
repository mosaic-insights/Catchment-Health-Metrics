# report_builder.py
from __future__ import annotations

# ── stdlib
import os
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── third-party
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ===================== configuration =====================
@dataclass
class ReportConfig:
    chm_workspace: str
    catchment_path: str

    # NEW unified daily csv (preferred)
    historical_daily_filename: str = "{catchment}_historical_daily.csv"

    # Legacy fallbacks (kept for backward-compat)
    awap_filename: str = "{catchment}_AWAP_precip_temp.csv"
    awral_filename: str = "{catchment}_AWRAL_daily.csv"

    # subfolders
    catch_plots_dir: str = "Catchment Plots and Maps"
    catch_datasets_dir: str = "Catchment Datasets"
    sites_datasets_dir: str = "Sites Datasets"
    sites_plots_dir: str = "Sites Plots and Maps"

    # SPI/SRI products
    spi12_monthly_csv: str = "{catchment}_SPI12_monthly.csv"
    spi12_annual_csv: str = "{catchment}_SPI12_annual.csv"
    sri12_monthly_csv: str = "{catchment}_SRI12_monthly.csv"
    sri12_annual_csv: str = "{catchment}_SRI12_annual.csv"
    spi12_png: str = "{catchment}_SPI12.png"
    sri12_png: str = "{catchment}_SRI12.png"

    # sizes
    hydro_ts_size: Tuple[int, int] = (10, 15)  # taller to fit extra Mean Temperature panel
    bar_ts_size: Tuple[int, int] = (7, 4)
    default_dpi: int = 300

    # sites gpkg fallback
    all_sites_gpkg_name: str = "{catchment} Sites Data.gpkg"

    # land/erosion panel year window
    le_year_min: int = 2010
    le_year_max: int = 2024

    # dependency flags
    enable_spi_sri: bool = True

    # flexible name patterns
    hydro_ts_patterns: List[str] = field(default_factory=lambda: [
        "*Hydroclimate*Time*Series*Annual*.png",
        "*Hydroclimate*Annual*.png",
        "*Hydro*Annual*.png",
    ])
    monitoring_plot_patterns: List[str] = field(default_factory=lambda: [
        "Site_*_Monitoring*TimeSeries*.png",
        "*Monitoring*TimeSeries*.png",
        "*Monitoring*.png",
        "*TimeSeries*.png",
    ])
    basemap_patterns: List[str] = field(default_factory=lambda: [
        "*Catchment*Stream*Network*.png", "*Basemap*.png", "*Catchment*Map*.png"
    ])
    landuse_patterns: List[str] = field(default_factory=lambda: [
        "*Landuse*Grouped*.png", "*Landuse*.png", "*Land*Use*.png"
    ])

    # DEA Land Cover (two-panel image output name)
    dea_two_panel_name: str = "{catchment}_DEA_LandCover_FirstLast.png"

    # Canonical LU time-series images (insert after DEA two-panel)
    lu_pct_overtime_name: str = "Catchment_LU_Percentages_OverTime.png"
    lu_pct_pointchange_name: str = "Catchment_LU_PercentagePointChange.png"


# ===================== small utils =====================
def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _fmt_float(x: float) -> str:
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else f"{x:.3g}"
    return str(x)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def _catchment_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def _insert_figure(doc: Document, fig_path: Path, caption: str, width_in: float = 6.5) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(fig_path), width=Inches(width_in))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _safe_num(v) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def _glob_first(folder: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


# ===================== I/O: load unified & legacy =====================
def _load_historical(cfg: ReportConfig, base: Path, catchment_name: str) -> Tuple[pd.DataFrame, str]:
    datasets = base / cfg.catch_datasets_dir
    hydro = datasets / "Hydroclimate"
    hist_dir = hydro / "Historical data"

    # Preferred unified CSV
    candidates = [
        hist_dir / cfg.historical_daily_filename.format(catchment=catchment_name),
        datasets / cfg.historical_daily_filename.format(catchment=catchment_name),
    ]
    hist_path = _first_existing(candidates)

    if hist_path:
        try:
            df = pd.read_csv(hist_path)
            dcol = next((c for c in df.columns if c.lower().startswith("date")), "Date")
            df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[dcol]).sort_values(dcol)
            for c in ["Precipitation", "Min Temperature", "Max Temperature",
                      "Runoff", "Actual ET", "Upper Soil Moisture", "Deeper Soil Moisture"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df, dcol
        except Exception as e:
            print(f"[WARN] failed to read unified daily CSV at {hist_path}: {e}")

    # Legacy merge: AWAP + AWRAL
    awap_candidates = [
        datasets / cfg.awap_filename.format(catchment=catchment_name),
        hydro / "Historical AWAP" / cfg.awap_filename.format(catchment=catchment_name),
    ]
    awral_candidates = [
        datasets / cfg.awral_filename.format(catchment=catchment_name),
        hydro / "Historical AWRAL" / cfg.awral_filename.format(catchment=catchment_name),
    ]
    awap_path = _first_existing(awap_candidates)
    awral_path = _first_existing(awral_candidates)

    awap = pd.DataFrame(); awral = pd.DataFrame()
    d_awap = "Date"; d_awral = "Date"

    if awap_path:
        try:
            awap = pd.read_csv(awap_path)
            d_awap = next((c for c in awap.columns if c.lower().startswith("date")), "Date")
            awap[d_awap] = pd.to_datetime(awap[d_awap], dayfirst=True, errors="coerce")
            awap = awap.dropna(subset=[d_awap]).sort_values(d_awap)
            for c in ["Precipitation", "Min Temperature", "Max Temperature"]:
                if c in awap.columns:
                    awap[c] = pd.to_numeric(awap[c], errors="coerce")
        except Exception as e:
            print(f"[WARN] AWAP read failed: {e}")
            awap = pd.DataFrame()

    if awral_path:
        try:
            awral = pd.read_csv(awral_path)
            d_awral = next((c for c in awral.columns if c.lower().startswith("date")), "Date")
            awral[d_awral] = pd.to_datetime(awral[d_awral], dayfirst=True, errors="coerce")
            awral = awral.dropna(subset=[d_awral]).sort_values(d_awral)
            for c in ["Runoff", "Actual ET", "Upper Soil Moisture", "Deeper Soil Moisture"]:
                if c in awral.columns:
                    awral[c] = pd.to_numeric(awral[c], errors="coerce")
        except Exception as e:
            print(f"[WARN] AWRAL read failed: {e}")
            awral = pd.DataFrame()

    if awap.empty and awral.empty:
        return pd.DataFrame(), "Date"

    if not awap.empty and not awral.empty:
        dcol = "Date"
        awap2 = awap.rename(columns={d_awap: dcol})
        awral2 = awral.rename(columns={d_awral: dcol})
        merged = pd.merge(awap2, awral2, on=dcol, how="outer").sort_values(dcol)
        return merged, dcol
    elif not awap.empty:
        return awap, d_awap
    else:
        return awral, d_awral


# ===================== SPI/SRI =====================
def _plot_spi_bar(df_year_mean: pd.DataFrame, col: str, title: str, ylabel: str, outfile_png: Path, size=(7, 4), dpi=300):
    if df_year_mean is None or df_year_mean.empty:
        return
    fig, ax = plt.subplots(figsize=size)

    vals = df_year_mean[col].values
    years = df_year_mean["Year"].values
    colors = ["deepskyblue" if v >= 1 else "tomato" if v <= -1 else "lime" for v in vals]
    ax.bar(years, vals, color=colors)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    if len(years):
        ax.set_xlim(min(years) - 1, max(years) + 1)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(1, color="blue", linestyle="--", linewidth=1)
    ax.axhline(-1, color="red", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=dpi)
    plt.close(fig)


def _compute_spi_sri_from_daily(
    cfg: ReportConfig,
    base: Path,
    catchment_name: str,
    daily: pd.DataFrame,
    dcol: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    plots = base / cfg.catch_plots_dir
    datasets = base / cfg.catch_datasets_dir
    hydro = datasets / "Hydroclimate"
    hist_dir = hydro / "Historical data"
    _ensure_dirs([plots, datasets])

    spi_png = plots / cfg.spi12_png.format(catchment=catchment_name)
    sri_png = plots / cfg.sri12_png.format(catchment=catchment_name)

    if not cfg.enable_spi_sri:
        return (spi_png if spi_png.exists() else None, sri_png if sri_png.exists() else None)

    try:
        from standard_precip import spi  # optional dependency
        spi_calc = spi.SPI()

        if not daily.empty and ("Precipitation" in daily.columns):
            precip_m = (
                daily.set_index(dcol)["Precipitation"]
                .resample("ME").sum().rename("Precipitation").to_frame().reset_index()
                .rename(columns={dcol: "Datetime"})
            )
            spi12 = spi_calc.calculate(
                df=precip_m, date_col="Datetime", precip_cols=["Precipitation"],
                freq="M", scale=12, fit_type="lmom", dist_type="gam"
            )
            spi_col = spi12.columns[2]
            spi12 = spi12.rename(columns={spi_col: "SPI12"})
            spi12.to_csv(hydro / cfg.spi12_monthly_csv.format(catchment=catchment_name), index=False)
            spi12["Year"] = spi12["Datetime"].dt.year
            spi_year = spi12.groupby("Year", as_index=False)["SPI12"].mean()
            spi_year.to_csv(hydro / cfg.spi12_annual_csv.format(catchment=catchment_name), index=False)
            _plot_spi_bar(
                spi_year, "SPI12",
                f"{catchment_name} – SPI12 (Precipitation)", "SPI12",
                spi_png, size=cfg.bar_ts_size, dpi=cfg.default_dpi
            )
        else:
            print("[WARN] SPI12 skipped: 'Precipitation' not present in daily table.")

        if not daily.empty and ("Runoff" in daily.columns):
            runoff_m = (
                daily.set_index(dcol)["Runoff"]
                .resample("ME").sum().rename("Runoff").to_frame().reset_index()
                .rename(columns={dcol: "Datetime"})
            )
            sri12 = spi_calc.calculate(
                df=runoff_m, date_col="Datetime", precip_cols=["Runoff"],
                freq="M", scale=12, fit_type="lmom", dist_type="gam"
            )
            sri_col = sri12.columns[2]
            sri12 = sri12.rename(columns={sri_col: "SRI12"})
            sri12.to_csv(hydro / cfg.sri12_monthly_csv.format(catchment=catchment_name), index=False)
            sri12["Year"] = sri12["Datetime"].dt.year
            sri_year = sri12.groupby("Year", as_index=False)["SRI12"].mean()
            sri_year.to_csv(hydro / cfg.sri12_annual_csv.format(catchment=catchment_name), index=False)
            _plot_spi_bar(
                sri_year, "SRI12",
                f"{catchment_name} – SRI12 (Runoff)", "SRI12",
                sri_png, size=cfg.bar_ts_size, dpi=cfg.default_dpi
            )
        else:
            print("[WARN] SRI12 skipped: 'Runoff' not present in daily table.")
    except Exception as e:
        print(f"[WARN] SPI/SRI step skipped: {e}")

    return (spi_png if spi_png.exists() else None, sri_png if sri_png.exists() else None)


# ===================== annual aggregation (with Mean Temperature) =====================
def _annual_from_daily(daily: pd.DataFrame, dcol: str) -> pd.DataFrame:
    """
    Build annual aggregates from unified (or merged legacy) daily table.
    - Fluxes (Precip, Runoff, Actual ET): annual SUM with min_count=1  ➜ all-NaN year -> NaN (not 0)
    - State vars & temperatures: annual MEAN
    - Derived Mean Temperature = 0.5 * (Min + Max) (computed on daily, then annual mean)
    """
    if daily.empty or dcol not in daily.columns:
        return pd.DataFrame()

    df = daily.copy()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).set_index(dcol)

    # derive Mean Temperature at daily step if both min/max exist
    if ("Min Temperature" in df.columns) and ("Max Temperature" in df.columns):
        df["Mean Temperature"] = 0.5 * (
            pd.to_numeric(df["Min Temperature"], errors="coerce")
            + pd.to_numeric(df["Max Temperature"], errors="coerce")
        )

    sum_vars = [c for c in ["Precipitation", "Runoff", "Actual ET"] if c in df.columns]
    mean_vars = [c for c in ["Min Temperature", "Max Temperature", "Mean Temperature",
                             "Upper Soil Moisture", "Deeper Soil Moisture"] if c in df.columns]
    others = [c for c in df.columns if (c not in sum_vars and c not in mean_vars)]

    pieces = []
    if sum_vars:
        # KEY CHANGE: keep NaN if an annual window has no valid daily values
        pieces.append(df[sum_vars].resample("YE").sum(min_count=1))
    if mean_vars:
        pieces.append(df[mean_vars].resample("YE").mean())
    if others:
        pieces.append(df[others].resample("YE").mean())  # conservative default

    if not pieces:
        return pd.DataFrame()

    annual = pd.concat(pieces, axis=1)
    annual.index = annual.index.to_period("Y").to_timestamp("Y")
    return annual


# ===================== hydroclimate plot (now with Mean Temperature panel) =====================
def _hydro_ts_plot(cfg: ReportConfig, base: Path, catchment_name: str, annual: pd.DataFrame) -> Optional[Path]:
    plots_dir = base / cfg.catch_plots_dir

    reuse = _glob_first(plots_dir, cfg.hydro_ts_patterns)
    if reuse:
        return reuse

    if annual is None or annual.empty:
        return None

    panel_order = ["Precipitation", "Mean Temperature", "Runoff", "Actual ET", "Upper Soil Moisture", "Deeper Soil Moisture"]
    series = {k: annual[k].dropna() for k in panel_order if k in annual.columns and annual[k].dropna().size > 0}
    if not series:
        return None

    def _to_year_xy(s):
        years = s.index.to_period("Y").year if hasattr(s.index, "to_period") else s.index.year
        return years.astype(int), s.values

    fig, axes = plt.subplots(nrows=len(panel_order), ncols=1, figsize=cfg.hydro_ts_size, sharex=True)
    label_kwargs = dict(fontsize=10, fontweight="bold")
    line_kwargs = dict(color="gray", linewidth=1, marker="o", markersize=3,
                       markerfacecolor="black", markeredgecolor="black")

    xmin = None; xmax = None
    for name, s in series.items():
        x, _ = _to_year_xy(s)
        if x.size:
            xmin = int(x.min()) if xmin is None else min(xmin, int(x.min()))
            xmax = int(x.max()) if xmax is None else max(xmax, int(x.max()))

    for ax, name in zip(axes, panel_order):
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel(name, **label_kwargs)
        ax.tick_params(axis="both", labelsize=10)
        if name not in series:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
            continue
        s = series[name]
        x, y = _to_year_xy(s)
        ax.plot(x, y, **line_kwargs)
        if np.isfinite(y).sum() >= 1:
            mean_val = np.nanmean(y)
            ax.axhline(mean_val, linestyle="--", linewidth=1, color="gray")
            ax.text(0.01, 0.90, f"Mean = {mean_val:.3g}", transform=ax.transAxes,
                    ha="left", va="center", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Year", **label_kwargs)
    if xmin is not None and xmax is not None:
        axes[0].set_xlim([xmin - 0.5, xmax + 0.5])

    out_png = plots_dir / f"{catchment_name}_Hydroclimate_TimeSeries_Annual.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{catchment_name} – Hydroclimate Time Series (Annual)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_png, dpi=cfg.default_dpi)
    plt.close(fig)
    return out_png


# ===================== DEA landcover: build 2-row panel (first & last year) =====================
def _find_dea_first_last(catch_plots: Path) -> Optional[Tuple[Tuple[Path,int], Tuple[Path,int]]]:
    """
    Looks for files like 'Catchment_DEA_LandCover_YYYY.png' (any extension), returns
    ((first_path, first_year), (last_path, last_year)) if found.
    """
    hits = []
    for ext in ("png", "jpg", "jpeg"):
        hits.extend(catch_plots.glob(f"Catchment_DEA_LandCover_*.{ext}"))
    years = []
    for p in hits:
        m = re.search(r"Catchment_DEA_LandCover_(\d{4})", p.name)
        if m:
            years.append((p, int(m.group(1))))
    if not years:
        return None
    years_sorted = sorted(years, key=lambda t: t[1])
    first = years_sorted[0]
    last  = years_sorted[-1]
    return first, last


def _compose_dea_two_panel(cfg: ReportConfig, catchment_name: str, catch_plots: Path) -> Optional[Path]:
    pair = _find_dea_first_last(catch_plots)
    if not pair:
        return None
    (p_first, y_first), (p_last, y_last) = pair

    try:
        img1 = imread(str(p_first))
        img2 = imread(str(p_last))

        # One row, two columns – tighter and simpler
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10.0, 5.4),
            constrained_layout=False,
        )

        # --- Left / right panels ---
        panels = [(axes[0], img1, y_first), (axes[1], img2, y_last)]
        for ax, img, y in panels:
            ax.imshow(img)
            # one clear title per image
            #ax.set_title(f"DEA Land Cover {y}",fontsize=11,fontweight="bold",pad=2,)
            ax.axis("off")
            ax.margins(x=0, y=0)

        # If you want ONLY per-panel titles, comment this out:
        # fig.suptitle(f"{catchment_name} – DEA Land Cover",
        #              fontsize=12, fontweight="bold", y=0.97)

        # Tight margins + very small horizontal gap
        fig.subplots_adjust(
            left=0.02,
            right=0.99,
            bottom=0.03,
            top=0.92,   # leave room only for per-panel titles
            wspace=0.001 # controls space between panels
        )

        out_path = catch_plots / cfg.dea_two_panel_name.format(catchment=catchment_name)
        fig.savefig(out_path, dpi=cfg.default_dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return out_path

    except Exception as e:
        print(f"[WARN] Could not compose DEA two-panel: {e}")
        return None



def _add_csv_table(doc: Document, csv_path: Path, round_to: int = 3) -> None:
    """
    Insert a CSV as a Word table.
    - Keeps all columns.
    - Rounds float-like values to `round_to` decimals for readability.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            doc.add_paragraph(f"[{csv_path.name} is empty.]")
            return

        # Round numeric columns for clean display (without altering the saved CSV)
        disp = df.copy()
        for c in disp.columns:
            if pd.api.types.is_numeric_dtype(disp[c]):
                disp[c] = disp[c].round(round_to)

        # Build table
        tbl = doc.add_table(rows=1, cols=len(disp.columns))
        hdr = tbl.rows[0].cells
        for i, c in enumerate(disp.columns):
            hdr[i].text = str(c)

        for _, r in disp.iterrows():
            cells = tbl.add_row().cells
            for i, c in enumerate(disp.columns):
                val = r[c]
                if pd.isna(val):
                    cells[i].text = ""
                else:
                    cells[i].text = str(val)
    except Exception as e:
        doc.add_paragraph(f"[Could not add table from {csv_path.name}: {e}]")


# ===================== land/erosion helpers for sites =====================
def _extract_year_series(row: pd.Series, prefix: str, suffix: str, y0: int, y1: int) -> pd.DataFrame:
    out = []
    for y in range(y0, y1 + 1):
        col = f"{prefix}{y}{suffix}"
        if col in row.index:
            out.append((y, _safe_num(row[col])))
    return pd.DataFrame(out, columns=["Year", "Value"]).sort_values("Year")


def _site_le_panel(cfg: ReportConfig, row: pd.Series, site_id, out_dir: Path) -> Optional[Path]:
    ndvi = _extract_year_series(row, "NDVI_", " (mean)", cfg.le_year_min, cfg.le_year_max)
    cf   = _extract_year_series(row, "C_Factor_", " (mean)", cfg.le_year_min, cfg.le_year_max)
    rusl = _extract_year_series(row, "RUSLE total ", " (t/yr)", cfg.le_year_min, cfg.le_year_max)
    sdr  = _extract_year_series(row, "SDR-RUSLE total ", " (t/yr)", cfg.le_year_min, cfg.le_year_max)
    if all(df.empty for df in (ndvi, cf, rusl, sdr)):
        return None

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)
    panels = [
        ("NDVI (mean)", ndvi),
        ("C-Factor (mean)", cf),
        ("RUSLE total (t/yr)", rusl),
        ("SDR-RUSLE total (t/yr)", sdr),
    ]
    for ax, (title, df) in zip(axes, panels):
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        if df.empty:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
            continue
        # FIX: removed stray extra closing parenthesis
        ax.plot(
            df["Year"], df["Value"],
            color="gray", linewidth=1, marker="o",
            markersize=4, markerfacecolor="black", markeredgecolor="black"
        )
        if np.isfinite(df["Value"]).sum() >= 1:
            mean_val = np.nanmean(df["Value"])
            ax.axhline(mean_val, linestyle="--", linewidth=1, color="gray")
            ax.text(0.01, 0.90, f"Mean = {mean_val:.3g}", transform=ax.transAxes,
                    ha="left", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(cfg.le_year_min - 0.5, cfg.le_year_max + 0.5)

    axes[-1].set_xlabel("Year", fontsize=10, fontweight="bold")
    fig.suptitle(
        f"Site {site_id} – NDVI / C-Factor / RUSLE / SDR-RUSLE ({cfg.le_year_min}–{cfg.le_year_max})",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"Site_{site_id}_LandErosion_TimeSeries.png"
    fig.savefig(out_png, dpi=cfg.default_dpi)
    plt.close(fig)
    return out_png


def _site_le_summary_table(row: pd.Series, cfg: ReportConfig) -> pd.DataFrame:
    def summarize(df: pd.DataFrame) -> Dict:
        if df.empty:
            return dict(Start=np.nan, End=np.nan, Mean=np.nan, Median=np.nan, Min=np.nan, Max=np.nan, Std=np.nan)
        v = pd.to_numeric(df["Value"], errors="coerce")
        v = v[np.isfinite(v)]
        if v.empty:
            return dict(Start=df["Year"].min(), End=df["Year"].max(),
                        Mean=np.nan, Median=np.nan, Min=np.nan, Max=np.nan, Std=np.nan)
        return dict(Start=int(df["Year"].min()), End=int(df["Year"].max()),
                    Mean=float(np.nanmean(v)), Median=float(np.nanmedian(v)),
                    Min=float(np.nanmin(v)), Max=float(np.nanmax(v)),
                    Std=float(np.nanstd(v, ddof=1)) if v.size >= 2 else np.nan)

    ndvi = summarize(_extract_year_series(row, "NDVI_", " (mean)", cfg.le_year_min, cfg.le_year_max))
    cf   = summarize(_extract_year_series(row, "C_Factor_", " (mean)", cfg.le_year_min, cfg.le_year_max))
    rusl = summarize(_extract_year_series(row, "RUSLE total ", " (t/yr)", cfg.le_year_min, cfg.le_year_max))
    sdr  = summarize(_extract_year_series(row, "SDR-RUSLE total ", " (t/yr)", cfg.le_year_min, cfg.le_year_max))

    rows = [
        dict(Variable="NDVI (mean)", **ndvi),
        dict(Variable="C-Factor (mean)", **cf),
        dict(Variable="RUSLE total (t/yr)", **rusl),
        dict(Variable="SDR-RUSLE total (t/yr)", **sdr),
    ]
    return pd.DataFrame(rows)


# ===================== NEW: Riparian NDVI helper functions =====================
def _find_riparian_first_last(catch_plots: Path) -> Optional[Tuple[Tuple[Path, int], Tuple[Path, int]]]:
    """
    Find first and last 'Riparian_NDVI_by_StreamSegment_YYYY.png' images.
    """
    hits = []
    for ext in ("png", "jpg", "jpeg"):
        hits.extend(catch_plots.glob(f"Riparian_NDVI_by_StreamSegment_*.{ext}"))
    years = []
    for p in hits:
        m = re.search(r"Riparian_NDVI_by_StreamSegment_(\d{4})", p.name)
        if m:
            years.append((p, int(m.group(1))))
    if not years:
        return None
    years_sorted = sorted(years, key=lambda t: t[1])
    return years_sorted[0], years_sorted[-1]


def _compose_riparian_two_panel(cfg: ReportConfig, catchment_name: str, catch_plots: Path) -> Optional[Path]:
    pair = _find_riparian_first_last(catch_plots)
    if not pair:
        return None

    (p_first, y_first), (p_last, y_last) = pair
    try:
        img1 = imread(str(p_first))
        img2 = imread(str(p_last))

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10.0, 5.4),
            constrained_layout=False,
        )

        panels = [(axes[0], img1, y_first), (axes[1], img2, y_last)]
        for ax, img, y in panels:
            ax.imshow(img)
            #ax.set_title(f"Riparian NDVI by Stream Segment {y}",fontsize=11,fontweight="bold",pad=2,)
            ax.axis("off")
            ax.margins(x=0, y=0)

        # Again: comment this out if you *only* want per-panel titles
        # fig.suptitle(
        #     f"{catchment_name} – Riparian NDVI by Stream Segment",
        #     fontsize=12,
        #     fontweight="bold",
        #     y=0.97,
        # )

        fig.subplots_adjust(
            left=0.02,
            right=0.99,
            bottom=0.03,
            top=0.92,
            wspace=0.001,
        )

        out_path = catch_plots / f"{catchment_name}_Riparian_NDVI_StreamSegment_FirstLast.png"
        fig.savefig(out_path, dpi=cfg.default_dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[WARN] Could not compose Riparian NDVI two-panel: {e}")
        return None


# ===================== NEW: Site-level hydroclimate/SPI/SRI helpers =====================
def _load_site_daily(base: Path, cfg: ReportConfig, site_id: int) -> Tuple[pd.DataFrame, str]:
    """
    Load Site_{id}_historical_daily.csv.
    Priority: <Sites Plots and Maps>/Site_{id}/Site_{id}_historical_daily.csv
              <Sites Datasets>/Site_{id}/Site_{id}_historical_daily.csv
    Returns (df, date_col_name) or (empty_df, 'Date') if not found.
    """
    plots_csv = base / cfg.sites_plots_dir / f"Site_{site_id}" / f"Site_{site_id}_historical_daily.csv"
    data_csv  = base / cfg.sites_datasets_dir / f"Site_{site_id}" / f"Site_{site_id}_historical_daily.csv"
    path = _first_existing([plots_csv, data_csv])
    if not path:
        return pd.DataFrame(), "Date"

    try:
        df = pd.read_csv(path)
        dcol = next((c for c in df.columns if c.lower().startswith("date")), "Date")
        df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        # standardize numeric
        for c in ["Precipitation", "Min Temperature", "Max Temperature",
                  "Runoff", "Actual ET", "Upper Soil Moisture", "Deeper Soil Moisture"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, dcol
    except Exception as e:
        print(f"[WARN] failed reading site daily CSV for Site {site_id}: {e}")
        return pd.DataFrame(), "Date"


def _site_hydro_ts_plot(cfg: ReportConfig, base: Path, site_id: int, annual: pd.DataFrame) -> Optional[Path]:
    site_plot_dir = base / cfg.sites_plots_dir / f"Site_{site_id}"
    site_plot_dir.mkdir(parents=True, exist_ok=True)

    if annual is None or annual.empty:
        return None

    panel_order = ["Precipitation", "Mean Temperature", "Runoff", "Actual ET", "Upper Soil Moisture", "Deeper Soil Moisture"]
    series = {k: annual[k].dropna() for k in panel_order if k in annual.columns and annual[k].dropna().size > 0}
    if not series:
        return None

    def _to_year_xy(s):
        years = s.index.to_period("Y").year if hasattr(s.index, "to_period") else s.index.year
        return years.astype(int), s.values

    fig, axes = plt.subplots(nrows=len(panel_order), ncols=1, figsize=cfg.hydro_ts_size, sharex=True)
    label_kwargs = dict(fontsize=10, fontweight="bold")
    line_kwargs = dict(color="gray", linewidth=1, marker="o", markersize=3,
                       markerfacecolor="black", markeredgecolor="black")

    xmin = None; xmax = None
    for name, s in series.items():
        x, _ = _to_year_xy(s)
        if x.size:
            xmin = int(x.min()) if xmin is None else min(xmin, int(x.min()))
            xmax = int(x.max()) if xmax is None else max(xmax, int(x.max()))

    for ax, name in zip(axes, panel_order):
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel(name, **label_kwargs)
        ax.tick_params(axis="both", labelsize=10)
        if name not in series:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
            continue
        s = series[name]
        x, y = _to_year_xy(s)
        ax.plot(x, y, **line_kwargs)
        if np.isfinite(y).sum() >= 1:
            mean_val = np.nanmean(y)
            ax.axhline(mean_val, linestyle="--", linewidth=1, color="gray")
            ax.text(0.01, 0.90, f"Mean = {mean_val:.3g}", transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Year", **label_kwargs)
    if xmin is not None and xmax is not None:
        axes[0].set_xlim([xmin - 0.5, xmax + 0.5])

    out_png = site_plot_dir / f"Site_{site_id}_Hydroclimate_TimeSeries_Annual.png"
    fig.suptitle(f"Site {site_id} – Hydroclimate Time Series (Annual)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_png, dpi=cfg.default_dpi)
    plt.close(fig)
    return out_png


def _compute_spi_sri_for_site(
    cfg: ReportConfig,
    base: Path,
    site_id: int,
    daily: pd.DataFrame,
    dcol: str,
) -> Tuple[Optional[Path], Optional[Path], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute Site-level SPI12 (from Precipitation) and SRI12 (from Runoff).
    Outputs:
      - PNGs to <Sites Plots and Maps>/Site_{id}/
      - Also return annual mean SPI/SRI dataframes (for optional tabulation or checks).
    """
    site_plot_dir = base / cfg.sites_plots_dir / f"Site_{site_id}"
    site_plot_dir.mkdir(parents=True, exist_ok=True)

    spi_png = site_plot_dir / f"Site_{site_id}_SPI12.png"
    sri_png = site_plot_dir / f"Site_{site_id}_SRI12.png"

    spi_year = None
    sri_year = None

    if not cfg.enable_spi_sri:
        return (spi_png if spi_png.exists() else None,
                sri_png if sri_png.exists() else None, None, None)

    try:
        from standard_precip import spi  # optional dependency
        spi_calc = spi.SPI()

        if not daily.empty and ("Precipitation" in daily.columns):
            precip_m = (
                daily.set_index(dcol)["Precipitation"]
                .resample("ME").sum().rename("Precipitation").to_frame().reset_index()
                .rename(columns={dcol: "Datetime"}))
            spi12 = spi_calc.calculate(
                df=precip_m, date_col="Datetime", precip_cols=["Precipitation"],
                freq="M", scale=12, fit_type="lmom", dist_type="gam")
            spi_col = spi12.columns[2]
            spi12 = spi12.rename(columns={spi_col: "SPI12"})
            spi12["Year"] = spi12["Datetime"].dt.year
            spi_year = spi12.groupby("Year", as_index=False)["SPI12"].mean()

            _plot_spi_bar(
                spi_year, "SPI12",
                f"Site {site_id} – SPI12 (Precipitation)", "SPI12",
                spi_png, size=cfg.bar_ts_size, dpi=cfg.default_dpi)

        if not daily.empty and ("Runoff" in daily.columns):
            runoff_m = (
                daily.set_index(dcol)["Runoff"]
                .resample("ME").sum().rename("Runoff").to_frame().reset_index()
                .rename(columns={dcol: "Datetime"}))
            sri12 = spi_calc.calculate(
                df=runoff_m, date_col="Datetime", precip_cols=["Runoff"],
                freq="M", scale=12, fit_type="lmom", dist_type="gam")
            sri_col = sri12.columns[2]
            sri12 = sri12.rename(columns={sri_col: "SRI12"})
            sri12["Year"] = sri12["Datetime"].dt.year
            sri_year = sri12.groupby("Year", as_index=False)["SRI12"].mean()

            _plot_spi_bar(
                sri_year, "SRI12",
                f"Site {site_id} – SRI12 (Runoff)", "SRI12",
                sri_png, size=cfg.bar_ts_size, dpi=cfg.default_dpi)

    except Exception as e:
        print(f"[WARN] Site {site_id} SPI/SRI step skipped: {e}")

    return (spi_png if spi_png.exists() else None,
            sri_png if sri_png.exists() else None,
            spi_year, sri_year)


def _insert_site_plot_if_exists(doc: Document,
                                site_plot_dir: Path,
                                filename: str,
                                fig_counter: int,
                                caption: str,
                                width_in: float = 6.5
) -> int:
    """
    If <site_plot_dir>/<filename> exists, insert it with a caption and return updated fig_counter.
    Otherwise, leave a short note and return fig_counter unchanged.
    """
    p = site_plot_dir / filename
    if p.exists():
        fig_counter += 1
        _insert_figure(doc, p, f"Figure {fig_counter}. {caption}", width_in=width_in)
    else:
        doc.add_paragraph(f"[{filename} not found in: {site_plot_dir}]")
    return fig_counter


# Files we’ll try to include for each site, in this order.
# Use exact filenames as you generate them (including spaces & ampersands).
SITE_PLOTS_SEQUENCE = [
    # Riparian & LU time series
    ("Site_{sid}_Riparian_NDVI_Timeseries.png",
     "Site {sid} – Riparian NDVI time series"),
    ("Site_{sid}_LU_Percentages_OverTime.png",
     "Site {sid} – Land-use percentages over time"),
    ("Site_{sid}_LU_PercentagePointChange.png",
     "Site {sid} – Land-use percentage-point change"),

    # Area-under-curve (AUC) summary plots
    ("Site_{sid}_AUC_NDVI_SDR.png",
     "Site {sid} – AUC summary for NDVI & SDR"),
    ("Site_{sid}_AUC_Bushfire.png",
     "Site {sid} – AUC summary for Bushfire exposure"),
    ("Site_{sid}_RUSLE_SDR-RUSLE_Totals.png",
     "Site {sid} – Total RUSLE and SDR-RUSLE (t/yr)"),
    ("Site_{sid}_Road_AUC_TimeSeries.png",
     "Site {sid} – Roads: AUC time-series"),

    # Land-use class specific AUCs
    ("Site_{sid}_LU_Water_wetlands_AUC_TimeSeries.png","Site {sid} – AUC: Water & wetlands"),
    ("Site_{sid}_LU_Native_forests_native_grazing_AUC_TimeSeries.png","Site {sid} – AUC: Native forests & native grazing"),
    ("Site_{sid}_LU_Conservation_minimal_use_AUC_TimeSeries.png","Site {sid} – AUC: Conservation & minimal use"),
    ("Site_{sid}_LU_Plantation_forests_AUC_TimeSeries.png","Site {sid} – AUC: Plantation forests"),
    ("Site_{sid}_LU_Dryland_agriculture_AUC_TimeSeries.png","Site {sid} – AUC: Dryland agriculture"),
    ("Site_{sid}_LU_Irrigated_agriculture_AUC_TimeSeries.png","Site {sid} – AUC: Irrigated agriculture"),
    ("Site_{sid}_LU_Intensive_production_AUC_TimeSeries.png","Site {sid} – AUC: Intensive production"),
    ("Site_{sid}_LU_Urban_residential_services_AUC_TimeSeries.png", "Site {sid} – AUC: Urban, residential & services"),
    ("Site_{sid}_LU_Industry_utilities_mining_waste_AUC_TimeSeries.png","Site {sid} – AUC: Industry, utilities, mining & waste"),
    ("Site_{sid}_LU_Transport_communication_AUC_TimeSeries.png","Site {sid} – AUC: Transport & communication"),
]

def _site_location_map(
    cfg: ReportConfig,
    base: Path,
    catch_gdf: Optional[gpd.GeoDataFrame],
    site_geom,                     # <- a Shapely geometry (Polygon/MultiPolygon/Point/LineString)
    sid: int,
    sites_crs=None                 # <- CRS of the sites layer (e.g., sites_gdf.crs)
) -> Optional[Path]:
    """
    Create a simple location map for a site:
    - catchment boundary in black outline (if available)
    - site polygon/point/line overlaid in red
    - saved to <Sites Plots and Maps>/Site_{sid}/Site_{sid}_Location.png

    NOTE:
        Shapely geometries do NOT have .crs. We must pass the CRS separately (sites_crs).
    """
    try:
        # ---- 0) Validate geometry
        if site_geom is None or (hasattr(site_geom, "is_empty") and site_geom.is_empty):
            return None

        # ---- 1) Build a GeoDataFrame for the single site geometry with the provided CRS
        site_gdf = gpd.GeoDataFrame({"Site_id": [sid]}, geometry=[site_geom], crs=sites_crs)

        # ---- 2) Align CRS to the catchment's CRS (if both known)
        if catch_gdf is not None and catch_gdf.crs is not None:
            if site_gdf.crs is None:
                # If site CRS is unknown but catchment CRS exists, assume the site is already in that CRS
                site_gdf = site_gdf.set_crs(catch_gdf.crs, allow_override=True)
            elif str(site_gdf.crs) != str(catch_gdf.crs):
                site_gdf = site_gdf.to_crs(catch_gdf.crs)

        # ---- 3) Output path
        out_dir = base / cfg.sites_plots_dir / f"Site_{sid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"Site_{sid}_Location.png"

        # ---- 4) Figure
        fig, ax = plt.subplots(figsize=(6.8, 6.0))  # compact, Word-friendly

        # 4a) Plot catchment outline if available
        if catch_gdf is not None and not catch_gdf.empty:
            try:
                catch_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=1)
            except Exception:
                catch_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0, zorder=1)

        # 4b) Plot site geometry (polygon/point/line) in crimson
        gtype = site_gdf.geom_type.iloc[0] if not site_gdf.empty else ""
        if gtype in {"Polygon", "MultiPolygon"}:
            site_gdf.boundary.plot(ax=ax, color="crimson", linewidth=1.8, zorder=3)
            site_gdf.plot(ax=ax, facecolor="none", edgecolor="crimson", linewidth=1.8, zorder=3)
        elif gtype in {"LineString", "MultiLineString"}:
            site_gdf.plot(ax=ax, color="crimson", linewidth=2.0, zorder=3)
        else:  # Points, etc.
            site_gdf.plot(ax=ax, color="crimson", markersize=35, zorder=3)

        # ---- 5) Style
        ax.set_title(f"Site {sid} – Location within Catchment", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Longitude", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        # ---- 6) Extent: prefer full catchment for context; fallback to site buffer
        if catch_gdf is not None and not catch_gdf.empty:
            minx, miny, maxx, maxy = catch_gdf.total_bounds
        else:
            minx, miny, maxx, maxy = site_gdf.total_bounds
            # Add a small buffer for nicer framing if only the site is available
            dx = (maxx - minx) or 1.0
            dy = (maxy - miny) or 1.0
            minx, maxx = minx - 0.1 * dx, maxx + 0.1 * dx
            miny, maxy = miny - 0.1 * dy, maxy + 0.1 * dy
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        fig.tight_layout()
        fig.savefig(out_png, dpi=cfg.default_dpi)
        plt.close(fig)
        return out_png

    except Exception as e:
        print(f"[WARN] Site {sid} location map failed: {e}")
        return None

# ===================== document assembly =====================
def build_report(cfg: ReportConfig) -> Path:
    catch_path = Path(cfg.catchment_path)
    catchment_name = _catchment_name(catch_path)
    base = Path(cfg.chm_workspace) / catchment_name
    # Load catchment outline once
    catch_gdf = None
    try:
        catch_gdf = gpd.read_file(cfg.catchment_path)
    except Exception as e:
        print(f"[WARN] Could not read catchment_path: {e}")


    # scaffold
    catch_plots = base / cfg.catch_plots_dir
    catch_datasets = base / cfg.catch_datasets_dir
    sites_datasets = base / cfg.sites_datasets_dir
    sites_plots = base / cfg.sites_plots_dir
    _ensure_dirs([base, catch_plots, catch_datasets, sites_datasets, sites_plots])

    # 1) Load daily
    daily, dcol = _load_historical(cfg, base, catchment_name)

    # 2) SPI/SRI
    spi_png, sri_png = _compute_spi_sri_from_daily(cfg, base, catchment_name, daily, dcol)

    # 3) Annual aggregation (with Mean Temperature)
    annual = _annual_from_daily(daily, dcol)

    # 4) Summary table
    summary_rows = []
    for var in ["Precipitation", "Mean Temperature", "Runoff", "Actual ET",
                "Upper Soil Moisture", "Deeper Soil Moisture",
                "Min Temperature", "Max Temperature"]:
        if isinstance(annual, pd.DataFrame) and (var in annual.columns):
            s = pd.to_numeric(annual[var], errors="coerce")
            n = int(s.notna().sum())
            summary_rows.append({
                "Variable": var,
                "Mean":   float(np.nanmean(s)) if n else np.nan,
                "Median": float(np.nanmedian(s)) if n else np.nan,
                "Min":    float(np.nanmin(s)) if n else np.nan,
                "Max":    float(np.nanmax(s)) if n else np.nan,
                "Std":    float(np.nanstd(s, ddof=1)) if n >= 2 else np.nan,
            })
    summary_df = pd.DataFrame(summary_rows)

    # 5) DOCX
    doc = Document()
    title = doc.add_heading(level=0)
    run = title.add_run(f"This is an automated report for {catchment_name} catchment health condition")
    run.font.size = Pt(16); run.bold = True
    doc.add_heading(catchment_name, level=1)

    fig_counter = 0

    # 5a) Basemap
    basemap = _glob_first(catch_plots, cfg.basemap_patterns)
    if basemap and basemap.exists():
        fig_counter += 1
        _insert_figure(doc, basemap, f"Figure {fig_counter}. {catchment_name} and the location of sites.", width_in=4.5)
    else:
        doc.add_paragraph(f"[Catchment basemap not found in: {catch_plots}]")

    # 5b) Land use (existing grouped)
    lu_png = _glob_first(catch_plots, cfg.landuse_patterns)
    if lu_png and lu_png.exists():
        fig_counter += 1
        _insert_figure(doc, lu_png, f"Figure {fig_counter}. {catchment_name} land use.", width_in=6.5)
    else:
        doc.add_paragraph(f"[Grouped land-use image not found in: {catch_plots}]")

    # 5c) DEA Land Cover 2-row panel (first & last year) — insert right after Figure 2
    dea_two = _compose_dea_two_panel(cfg, catchment_name, catch_plots)
    if dea_two and dea_two.exists():
        fig_counter += 1
        _insert_figure(doc, dea_two,
                       f"Figure {fig_counter}. {catchment_name} DEA Land Cover.",
                       width_in=6.5)
    else:
        doc.add_paragraph("[DEA Land Cover first/last panel not generated (files not found).]")

    # 5d) Catchment_LU_Percentages_OverTime.png
    lu_pct_over = catch_plots / cfg.lu_pct_overtime_name
    if lu_pct_over.exists():
        fig_counter += 1
        _insert_figure(doc, lu_pct_over,
                       f"Figure {fig_counter}. {catchment_name} – Land-use percentages over time.", width_in=6.5)
    else:
        doc.add_paragraph(f"[{cfg.lu_pct_overtime_name} not found in: {catch_plots}]")

    # 5e) Catchment_LU_PercentagePointChange.png
    lu_pct_change = catch_plots / cfg.lu_pct_pointchange_name
    if lu_pct_change.exists():
        fig_counter += 1
        _insert_figure(doc, lu_pct_change,
                       f"Figure {fig_counter}. {catchment_name} – Land-use percentage-point change.", width_in=6.5)
    else:
        doc.add_paragraph(f"[{cfg.lu_pct_pointchange_name} not found in: {catch_plots}]")

    # 5f-new) Land-use percentages table (from CSV), placed right after Figure 5.
    landuse_output = catch_datasets / "Landuse" / "DEA Landcover"
    lu_pct_csv = landuse_output / "Catchment_LU_Percentages.csv"
    doc.add_paragraph("")  # small spacer
    doc.add_heading(f"{catchment_name} – Land-use percentages (table)", level=2)
    if lu_pct_csv.exists():
        _add_csv_table(doc, lu_pct_csv, round_to=3)
    else:
        doc.add_paragraph(f"[Catchment_LU_Percentages.csv not found at: {lu_pct_csv}]")

    # === NEW SECTION: Riparian NDVI by Stream Segment + Mean NDVI Timeseries ===
    riparian_two_panel = _compose_riparian_two_panel(cfg, catchment_name, catch_plots)
    if riparian_two_panel and riparian_two_panel.exists():
        fig_counter += 1
        _insert_figure(doc, riparian_two_panel,f"Figure {fig_counter}. {catchment_name} – Riparian NDVI by Stream Segment.",width_in=6.5)
    else:
        doc.add_paragraph("[Riparian NDVI by Stream Segment images not found.]")

    riparian_timeseries = catch_plots / "Riparian_NDVI_Mean_by_StreamOrder_Timeseries.png"
    if riparian_timeseries.exists():
        fig_counter += 1
        _insert_figure( doc, riparian_timeseries, f"Figure {fig_counter}. {catchment_name} – Riparian NDVI mean by stream order over time.",width_in=6.5)
    else:
        doc.add_paragraph("[Riparian NDVI mean by stream order timeseries missing.]")

    # 5f) Hydro TS (with Mean Temperature)
    hydro_png = _hydro_ts_plot(cfg, base, catchment_name, annual)
    if hydro_png and hydro_png.exists():
        fig_counter += 1
        _insert_figure(doc, hydro_png,f"Figure {fig_counter}. Annual hydroclimate time series for {catchment_name}.", width_in=6.5)
    else:
        doc.add_paragraph("[Hydroclimate time series not generated.]")

    # 5g) Summary table
    doc.add_heading(f"Statistical summary for annual hydroclimate variables across {catchment_name}", level=2)
    cols = ["Variable", "Mean", "Median", "Min", "Max", "Std"]
    if not summary_df.empty:
        tbl = doc.add_table(rows=1, cols=len(cols))
        hdr = tbl.rows[0].cells
        for i, c in enumerate(cols): hdr[i].text = c
        for _, r in summary_df.iterrows():
            row_cells = tbl.add_row().cells
            for i, c in enumerate(cols):
                row_cells[i].text = _fmt_float(r.get(c, np.nan))
    else:
        doc.add_paragraph("[No annual summary could be computed (missing inputs).]")

    # 5h) SPI/SRI figs
    if spi_png and spi_png.exists():
        fig_counter += 1
        _insert_figure(doc, spi_png, f"Figure {fig_counter}. {catchment_name} SPI-12 (annual mean of 12-month SPI).")
    else:
        doc.add_paragraph("[SPI12 figure not generated.]")

    if sri_png and sri_png.exists():
        fig_counter += 1
        _insert_figure(doc, sri_png, f"Figure {fig_counter}. {catchment_name} SRI-12 (annual mean of 12-month SRI).")
    else:
        doc.add_paragraph("[SRI12 figure not generated.]")

    # 5i) RUSLE and SDR-RUSLE Totals plot (after Figure 10)
    rusle_sdr_totals = catch_plots / f"{catchment_name}_RUSLE_SDR-RUSLE_Totals.png"
    if rusle_sdr_totals.exists():
        fig_counter += 1
        _insert_figure(doc,rusle_sdr_totals,f"Figure {fig_counter}. {catchment_name} – Total RUSLE and SDR-RUSLE (t/yr).",width_in=6.5,)
    else:
        doc.add_paragraph(f"[{rusle_sdr_totals.name} not found in: {catch_plots}]")

    # 6) Sites section (enhanced with Site-level hydroclimate, summary, SPI, SRI)
    doc.add_heading("Sites", level=1)

    site_ids: List[str] = []
    gpkg_candidates = [sites_datasets / cfg.all_sites_gpkg_name.format(catchment=catchment_name),
                       sites_datasets / f"{catchment_name} Sites Data.gpkg",
                       sites_datasets / "All Sites Data.gpkg",]
    gpkg_path = _first_existing(gpkg_candidates)
    sites_gdf = None
    if gpkg_path and gpkg_path.exists():
        try:
            sites_gdf = gpd.read_file(gpkg_path)
            if "Site_id" in sites_gdf.columns:
                # keep raw IDs as strings (works for 1, 2, 3 and DNE6, E601, ...)
                site_ids = sorted({str(v) for v in sites_gdf["Site_id"].dropna().unique()})
        except Exception as e:
            print(f"[WARN] could not read sites GPKG: {e}")

    if not site_ids and sites_plots.exists():
        for p in (p for p in sites_plots.iterdir() if p.is_dir()):
            if p.name.startswith("Site_"):
                site_ids.append(p.name.replace("Site_", ""))
    
    site_ids = sorted(set(site_ids))

    for sid in site_ids:
        doc.add_heading(f"Site {sid}", level=2)
        # --- First figure for each site: location map (site polygon over catchment) ---
        if catch_gdf is not None and sites_gdf is not None and not sites_gdf.empty and ("Site_id" in sites_gdf.columns):
            row_match = sites_gdf[sites_gdf["Site_id"].astype(str) == str(sid)]
            if not row_match.empty:
                site_row = row_match.iloc[0]
                # Pass the geometry AND the CRS of the sites layer (sites_gdf.crs). Geometry itself has no CRS.
                loc_png = _site_location_map(
                    cfg, base, catch_gdf,
                    site_row.geometry,  # Shapely geometry
                    sid,
                    getattr(sites_gdf, "crs", None)
                )
                if loc_png and loc_png.exists():
                    fig_counter += 1
                    _insert_figure(doc, loc_png, f"Figure {fig_counter}. Site {sid} location within {catchment_name}.", width_in=6.0)
                else:
                    doc.add_paragraph(f"[Site {sid}: location map not generated.]")
        else:
            doc.add_paragraph("[Catchment or sites layer missing – cannot draw site location maps.]")


        site_plot_dir = sites_plots / f"Site_{sid}"
        # monitoring plot
        site_plot_dir = sites_plots / f"Site_{sid}"
        plot_path = _glob_first(site_plot_dir, cfg.monitoring_plot_patterns) if site_plot_dir.exists() else None
        if plot_path and plot_path.exists():
            fig_counter += 1
            _insert_figure(doc, plot_path, f"Figure {fig_counter}. Monitoring data for Site {sid}.", width_in=6.5)
        else:
            doc.add_paragraph(f"[Monitoring plot not found for Site {sid}. Looked in: {site_plot_dir}]")

        # monitoring summary table
        site_data_dir = sites_datasets / f"Site_{sid}"
        site_summary_csv = None
        if site_data_dir.exists():
            matches = list(site_data_dir.glob("*Monitoring*Summary*.csv"))
            if matches:
                site_summary_csv = matches[0]
        if site_summary_csv and site_summary_csv.exists():
            try:
                df_sum = pd.read_csv(site_summary_csv)
                if not df_sum.empty:
                    doc.add_paragraph("")
                    doc.add_heading(f"Monitoring summary for Site {sid}", level=3)
                    cols_site = list(df_sum.columns)
                    tbl = doc.add_table(rows=1, cols=len(cols_site))
                    hdr = tbl.rows[0].cells
                    for i, c in enumerate(cols_site):
                        hdr[i].text = str(c)
                    for _, r in df_sum.iterrows():
                        cells = tbl.add_row().cells
                        for i, c in enumerate(cols_site):
                            val = r.get(c, np.nan)
                            if isinstance(val, (float, np.floating)):
                                cells[i].text = "" if not np.isfinite(val) else f"{val:.3g}"
                            else:
                                cells[i].text = "" if pd.isna(val) else str(val)
            except Exception as e:
                doc.add_paragraph(f"[Could not add monitoring summary table for Site {sid}: {e}]")

        # land/erosion panel & summary
        if sites_gdf is not None and not sites_gdf.empty and ("Site_id" in sites_gdf.columns):
            row_match = sites_gdf[pd.to_numeric(sites_gdf["Site_id"], errors="coerce") == sid]
            if not row_match.empty:
                row = row_match.iloc[0]
                le_png = _site_le_panel(cfg, row, sid, site_plot_dir)
                if le_png and le_png.exists():
                    fig_counter += 1
                    _insert_figure(
                        doc, le_png,
                        f"Figure {fig_counter}. NDVI, C-Factor, RUSLE, and SDR-RUSLE ({cfg.le_year_min}–{cfg.le_year_max}) for Site {sid}.",
                        width_in=6.5,
                    )
                """
                try:
                    le_df = _site_le_summary_table(row, cfg)
                    if not le_df.empty:
                        doc.add_paragraph("")
                        doc.add_heading(f"NDVI / C-Factor / RUSLE / SDR-RUSLE summary for Site {sid}", level=3)
                        cols_le = ["Variable", "Start", "End", "Mean", "Median", "Min", "Max", "Std"]
                        tbl = doc.add_table(rows=1, cols=len(cols_le))
                        hdr = tbl.rows[0].cells
                        for i, c in enumerate(cols_le): hdr[i].text = c
                        for _, r2 in le_df.iterrows():
                            cells = tbl.add_row().cells
                            for i, c in enumerate(cols_le):
                                v = r2.get(c, np.nan)
                                if isinstance(v, (float, np.floating)):
                                    cells[i].text = "" if not np.isfinite(v) else f"{v:.3g}"
                                else:
                                    cells[i].text = "" if pd.isna(v) else str(v)
                except Exception as e:
                    doc.add_paragraph(f"[Could not add land/erosion summary for Site {sid}: {e}]")
                 """
        # === NEW: Additional per-site figures (Riparian NDVI, LU, AUCs) ===
        doc.add_heading(f"Additional figures for Site {sid}", level=3)
        # We'll remember when we've just inserted the LU-over-time plot
        just_inserted_lu_overtime = False  # retained for future logic
        site_lu_pct_csv_name = f"Site_{sid}_LU_Percentages.csv"
        site_data_dir = sites_plots / f"Site_{sid}"

        for fname_tpl, cap_tpl in SITE_PLOTS_SEQUENCE:
            filename = fname_tpl.format(sid=sid)
            caption  = cap_tpl.format(sid=sid)

            # Insert the figure if it exists
            prev_fig_counter = fig_counter
            fig_counter = _insert_site_plot_if_exists(
                doc, site_plot_dir, filename, fig_counter, caption, width_in=6.5
            )

            # If we just inserted the LU percentages over time figure, insert the matching CSV table right after it
            if (
                prev_fig_counter != fig_counter
                and fname_tpl == "Site_{sid}_LU_Percentages_OverTime.png"
            ):
                lu_pct_csv_path = site_data_dir / site_lu_pct_csv_name
                doc.add_paragraph("")  # small spacer
                doc.add_heading(f"Site {sid} – Land-use percentages table", level=4)
                if lu_pct_csv_path.exists():
                    _add_csv_table(doc, lu_pct_csv_path, round_to=3)
                else:
                    doc.add_paragraph(f"[{site_lu_pct_csv_name} not found at: {lu_pct_csv_path}]")

        # === NEW: Site-level hydroclimate (from Site_{sid}_historical_daily.csv) ===
        site_daily, dcol_site = _load_site_daily(base, cfg, sid)
        if not site_daily.empty:
            site_annual = _annual_from_daily(site_daily, dcol_site)

            # Site time-series figure (same style as catchment)
            site_hydro_png = _site_hydro_ts_plot(cfg, base, sid, site_annual)
            if site_hydro_png and site_hydro_png.exists():
                fig_counter += 1
                _insert_figure(
                    doc, site_hydro_png,
                    f"Figure {fig_counter}. Annual hydroclimate time series for Site {sid}.",
                    width_in=6.5
                )
            else:
                doc.add_paragraph(f"[Site {sid}: hydroclimate time series not generated.]")

            # Site summary table (same columns set as catchment)
            doc.add_heading(f"Statistical summary for annual hydroclimate variables – Site {sid}", level=3)
            cols = ["Variable", "Mean", "Median", "Min", "Max", "Std"]
            rows = []
            for var in ["Precipitation", "Mean Temperature", "Runoff", "Actual ET",
                        "Upper Soil Moisture", "Deeper Soil Moisture",
                        "Min Temperature", "Max Temperature"]:
                if isinstance(site_annual, pd.DataFrame) and (var in site_annual.columns):
                    s = pd.to_numeric(site_annual[var], errors="coerce")
                    n = int(s.notna().sum())
                    rows.append({
                        "Variable": var,
                        "Mean":   float(np.nanmean(s)) if n else np.nan,
                        "Median": float(np.nanmedian(s)) if n else np.nan,
                        "Min":    float(np.nanmin(s)) if n else np.nan,
                        "Max":    float(np.nanmax(s)) if n else np.nan,
                        "Std":    float(np.nanstd(s, ddof=1)) if n >= 2 else np.nan,
                    })
            if rows:
                df_site_summary = pd.DataFrame(rows)
                tbl = doc.add_table(rows=1, cols=len(cols))
                hdr = tbl.rows[0].cells
                for i, c in enumerate(cols): hdr[i].text = c
                for _, rsum in df_site_summary.iterrows():
                    rc = tbl.add_row().cells
                    for i, c in enumerate(cols):
                        rc[i].text = _fmt_float(rsum.get(c, np.nan))
            else:
                doc.add_paragraph(f"[Site {sid}: no annual summary computed (missing inputs).]")

            # Site SPI/SRI (SPI from Precip, SRI from Runoff)
            spi_site_png, sri_site_png, _spi_year, _sri_year = _compute_spi_sri_for_site(cfg, base, sid, site_daily, dcol_site)
            if spi_site_png and spi_site_png.exists():
                fig_counter += 1
                _insert_figure(doc, spi_site_png, f"Figure {fig_counter}. Site {sid} SPI-12 (annual mean of 12-month SPI).")
            else:
                doc.add_paragraph(f"[Site {sid}: SPI12 figure not generated.]")

            if sri_site_png and sri_site_png.exists():
                fig_counter += 1
                _insert_figure(doc, sri_site_png, f"Figure {fig_counter}. Site {sid} SRI-12 (annual mean of 12-month SRI).")
            else:
                doc.add_paragraph(f"[Site {sid}: SRI12 figure not generated.]")
        else:
            doc.add_paragraph(f"[Site {sid}: daily hydroclimate CSV not found.]")

    # 7) Save DOCX
    out_docx = base / f"{catchment_name}_Automated_Report.docx"
    try:
        doc.save(out_docx)
        print(f"[OK] Report written: {out_docx}")
    except PermissionError:
        print(f"[ERR] Could not write '{out_docx}'. Please close it in Word and run again.")
    return out_docx