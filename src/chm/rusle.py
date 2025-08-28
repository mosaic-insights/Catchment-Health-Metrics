# === Standard Libraries ===
import os
import glob
import re

# === Scientific & Array Processing ===
import numpy as np
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.mask import geometry_mask
from shapely.geometry import mapping


### BLOCK 2: Soil factors (C/K/AI) ###
def rusle_and_sdr_rusle(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path, K_Factor_Path, P_Factor_Path):
    print("Starting processesing RUSLE ...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    rusle_folder = os.path.join(catch_datasets, 'RUSLE and SDR_RUSLE')
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    
    # List of folders to create
    folders_to_create = [catchment_folder, catch_datasets, sites_datasets, rusle_folder]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)


    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    dem_projected_file = os.path.join(catch_datasets, "Topography","DEM.tif")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation","Indices","C Factor","Annual")
    annual_ndvi_dir = os.path.join(catch_datasets, "Vegetation","Indices","NDVI","Annual")
    sdr_output_dir = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")
    TWI_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")
    LS_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Slope length-gradient factor.tif") 
    awap_folder_hist = os.path.join(catch_datasets, "Hydroclimate", "Historical AWAP")
    
    def process_rasters(catch_shp_path, K_Factor_Path, P_Factor_Path, output_path, buffer_distance_km=10):
        def buffered_bounds(catch_shp_path, raster_path, buffer_distance_km):
            gdf = gpd.read_file(catch_shp_path)
            with rio.open(raster_path) as src:
                raster_crs = src.crs
            gdf_projected = gdf.to_crs(raster_crs)
            bbox = gdf_projected.total_bounds
            buffer_degrees = buffer_distance_km / 111
            bbox_with_buffer = [
                bbox[0] - buffer_degrees,
                bbox[1] - buffer_degrees,
                bbox[2] + buffer_degrees,
                bbox[3] + buffer_degrees
            ]
            return bbox_with_buffer
    
        def clip_raster_by_bbox(raster_path, output_path, bbox_with_buffer):
            with rio.open(raster_path) as src:
                window = from_bounds(bbox_with_buffer[0], bbox_with_buffer[1], bbox_with_buffer[2], bbox_with_buffer[3], transform=src.transform)
                if window.height > 0 and window.width > 0:
                    clipped_raster = src.read(1, window=window, masked=True)
                    meta = src.meta.copy()
                    meta.update({
                        "driver": "GTiff",
                        "height": window.height,
                        "width": window.width,
                        "transform": src.window_transform(window)
                    })
                    with rio.open(output_path, 'w', **meta) as dest:
                        dest.write(clipped_raster, 1)
    
        def reproject_and_resample_to_dem(src_raster_path, output_raster_path):
            with rio.open(dem_projected_file) as dem_src:
                dem_crs = dem_src.crs
                dem_transform = dem_src.transform
                dem_width = dem_src.width
                dem_height = dem_src.height
                dem_nodata = dem_src.nodata
        
                # Prepare an array filled with DEM nodata
                reprojected_data = np.full((dem_height, dem_width), dem_nodata, dtype=np.float32)
        
                with rio.open(src_raster_path) as src:
                    # Perform reprojection directly into the array
                    reproject(
                        source=rio.band(src, 1),
                        destination=reprojected_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dem_transform,
                        dst_crs=dem_crs,
                        resampling=Resampling.nearest
                    )
        
                # Mask using DEM nodata mask
                dem_mask = dem_src.read(1, masked=True).mask  # True where DEM has nodata
                reprojected_data = np.where(dem_mask, dem_nodata, reprojected_data)
        
                # Save the reprojected + masked raster
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dem_crs,
                    'transform': dem_transform,
                    'width': dem_width,
                    'height': dem_height,
                    'nodata': dem_nodata,
                    'dtype': 'float32'
                })
                with rio.open(output_raster_path, 'w', **kwargs) as dst:
                    dst.write(reprojected_data, 1)
                    
        output_clipped_k_factor = os.path.join(rusle_folder, "k_factor_clipped.tif")
        output_clipped_p_factor = os.path.join(rusle_folder, "p_factor_clipped.tif")
    
        raster_paths = [K_Factor_Path, P_Factor_Path]
        output_paths = [output_clipped_k_factor, output_clipped_p_factor]
    
        for raster_path, output_path in zip(raster_paths, output_paths):
            bbox_with_buffer = buffered_bounds(catch_shp_path, raster_path, buffer_distance_km)
            clip_raster_by_bbox(raster_path, output_path, bbox_with_buffer)
    
        output_reprojected_k_factor = os.path.join(rusle_folder, "k_factor.tif")
        output_reprojected_r_factor = os.path.join(rusle_folder, "r_factor.tif")
        output_reprojected_p_factor = os.path.join(rusle_folder, "p_factor.tif")
    
        reproject_and_resample_to_dem(output_clipped_k_factor, output_reprojected_k_factor)
        reproject_and_resample_to_dem(output_clipped_p_factor, output_reprojected_p_factor)
        # === CLEAN TEMP FILES ===
        os.remove(output_clipped_k_factor)
        os.remove(output_clipped_p_factor)
        # Return reprojected raster paths
        return output_reprojected_k_factor, output_reprojected_p_factor
    
    process_rasters(Catchment_Shapefile_Path, K_Factor_Path, P_Factor_Path, catchment_folder)
    
    # Run and capture raster outputs
    output_reprojected_k_factor, output_reprojected_p_factor = process_rasters(
        Catchment_Shapefile_Path, K_Factor_Path, P_Factor_Path, catchment_folder
    )
    
        # 2: Calculate base RUSLE layer
    with rio.open(LS_path) as ls:
        ls_array = ls.read(1)
        profile = ls.meta.copy()
        ls_meta, ls_transform, ls_crs = ls.meta.copy(), ls.transform, ls.crs
        ls_width, ls_height = ls.width, ls.height
    cell_area_m2 = abs(ls_transform.a) * abs(ls_transform.e)
    cell_area_ha = cell_area_m2 / 10000.0
    
    with rio.open(output_reprojected_k_factor) as k_factor, \
         rio.open(output_reprojected_p_factor) as p_factor:
    
        k_array = k_factor.read(1)
        p_array = p_factor.read(1)
    
    # --- Build C/SDR availability maps once ---
    c_rasters = sorted(glob.glob(os.path.join(annual_c_dir, "C_Factor_*.tif")))
    sdr_rasters = sorted(glob.glob(os.path.join(sdr_output_dir, "SDR_*.tif")))
    
    def extract_year(name_or_path):
        #m = re.search(r"(19|20)\d{4}", "BAD")  # guard (won't match)
        name = os.path.basename(str(name_or_path))
        m = re.search(r"(19|20)\d{2}", name)  # find 4-digit year
        return int(m.group(0)) if m else None
    
    c_by_year   = {extract_year(p): p for p in c_rasters   if extract_year(p) is not None}
    sdr_by_year = {extract_year(p): p for p in sdr_rasters if extract_year(p) is not None}
    
    c_years   = sorted(c_by_year.keys())
    sdr_years = sorted(sdr_by_year.keys())
    
    def nearest_year(target, available):
        """
        Pick the available year closest to 'target'.
        In a tie, prefer the later year.
        """
        if not available:
            return None
        return min(available, key=lambda y: (abs(y - target), -y))
    
    # --- Gather precip files once and iterate by precip year ---
    precip_files = [f for f in os.listdir(awap_folder_hist) if f.endswith(".nc") and "precip" in f.lower()]
    precip_paths_sorted = sorted((os.path.join(awap_folder_hist, f) for f in precip_files),
                                 key=lambda p: extract_year(p) if extract_year(p) is not None else 0)
    
    # Optionally: require C/SDR not older/newer than this many years; set to None to disable
    max_allowed_gap_years = None  # e.g., 3
    dem = rxr.open_rasterio(dem_projected_file, masked=True).squeeze(drop=True)  # (y,x)
    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(dem.rio.crs)
    # Grid metadata used for rasterizing the site polygons
    tform  = dem.rio.transform()
    height = dem.rio.height
    width  = dem.rio.width
    for nc_path in precip_paths_sorted:
        # 1) Determine precip year (prefer filename; fallback to time axis)
        p_year = extract_year(nc_path)
        if p_year is None:
            with xr.open_dataset(nc_path) as _tmp:
                p_year = int(pd.DatetimeIndex(_tmp["time"].values).year[0])
        print(f"\n--- Processing precipitation year {p_year} from {os.path.basename(nc_path)} ---")
    
        # 2) Find exact/nearest C & SDR years
        c_year_pick   = p_year if p_year in c_years   else nearest_year(p_year, c_years)
        sdr_year_pick = p_year if p_year in sdr_years else nearest_year(p_year, sdr_years)
    
        # Enforce a max gap if desired
        if max_allowed_gap_years is not None:
            if c_year_pick is None or abs(c_year_pick - p_year) > max_allowed_gap_years:
                print(f"Skipping {p_year}: no C-factor within ±{max_allowed_gap_years} years.")
                continue
            if sdr_year_pick is None or abs(sdr_year_pick - p_year) > max_allowed_gap_years:
                print(f"Skipping {p_year}: no SDR within ±{max_allowed_gap_years} years.")
                continue
    
        if c_year_pick is None or sdr_year_pick is None:
            print(f"Skipping {p_year}: no C/SDR years available at all.")
            continue
    
        c_path  = c_by_year[c_year_pick]
        sdr_path = sdr_by_year[sdr_year_pick]
        if c_year_pick != p_year or sdr_year_pick != p_year:  # this is for when we dont have c factor or sdr for each corrsponding year of the rainfall
            print(f"Using nearest layers -> C:{c_year_pick}  SDR:{sdr_year_pick}  (precip:{p_year})")
    
        # 3) Read C and SDR rasters (assumed already aligned to DEM grid by your earlier steps)
        with rio.open(c_path) as csrc:
            c_array = csrc.read(1).astype("float64")
            c_nd = csrc.nodata
            if c_nd is not None:
                c_array = np.where(c_array == c_nd, np.nan, c_array)
    
        with rio.open(sdr_path) as ssrc:
            sdr_array = ssrc.read(1).astype("float64")
            sdr_nd = ssrc.nodata
            if sdr_nd is not None:
                sdr_array = np.where(sdr_array == sdr_nd, np.nan, sdr_array)
    
        # 4) Base factors for this (precip) year using chosen C/SDR
        base = (c_array * k_array * ls_array * p_array).astype("float64")         # (y,x)
        base_da = xr.DataArray(base, coords={"y": dem["y"], "x": dem["x"]}, dims=("y","x"), name="rusle_base")
        sdr_da  = xr.DataArray(sdr_array, coords={"y": dem["y"], "x": dem["x"]}, dims=("y","x"), name="sdr")
    
        # 5) Open precip, reproject to DEM, compute erosivity, then RUSLE stacks
        ds = xr.open_dataset(nc_path)
        ds = ds.rio.write_crs("EPSG:4326")  # Attach CRS
    
        cand_vars = [v for v in ["precip","pr","rain","precipitation"] if v in ds.data_vars]
        varname = cand_vars[0] if cand_vars else list(ds.data_vars)[0]
        da = ds[varname]
    
        # Ensure lon/lat naming and set spatial dims/CRS
        x_name = "lon" if "lon" in da.dims else ("longitude" if "longitude" in da.dims else None)
        y_name = "lat" if "lat" in da.dims else ("latitude"  if "latitude"  in da.dims else None)
        if x_name is None or y_name is None:
            raise ValueError("Could not find 1D lon/lat dims in precip NetCDF.")
        if x_name != "lon" or y_name != "lat":
            da = da.rename({x_name:"lon", y_name:"lat"})
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        da = da.rio.write_crs("EPSG:4326", inplace=False).rio.write_nodata(np.nan)
        if "time" in da.dims:
            da = da.chunk({"time": 30})
        
        precip_matched = da.rio.reproject_match(dem, resampling=Resampling.nearest).transpose("time","y","x")
        precip_matched.name = "precip"
        precip_matched.attrs.pop("grid_mapping", None); precip_matched.encoding.pop("grid_mapping", None)
    
        # === Erosivity proxy (as before) ===
        intensity = precip_matched / 24.0
        e_r = 0.29 * (1.0 - 0.72 * np.exp(-Empirical_coefficient * intensity))
        E = e_r * precip_matched
        Erosivity = E * intensity  # (time,y,x)
    
        # === Daily stacks ===
        rusle_stack     = Erosivity * base_da          # (time,y,x)
        sdr_rusle_stack = rusle_stack * sdr_da         # (time,y,x)
        
        for idx, site in sites_gdf.iterrows():
            site_id   = site.get("id", idx)
            site_name = str(site.get("Site_name", f"Site_{site_id}")).replace(os.sep, "_")
            geom      = site.geometry
        
            # 1) Raster mask for this site on the DEM grid (True = inside polygon)
            mask_np = geometry_mask(
                geometries=[mapping(geom)],
                out_shape=(height, width),
                transform=tform,
                invert=True
            )
        
            # Wrap as xarray so we can use .where() cleanly with (time,y,x) fields
            mask_da = xr.DataArray(
                mask_np,
                coords={"y": precip_matched["y"], "x": precip_matched["x"]},
                dims=("y", "x"),
                name="site_mask"
            )
        
            # 2) Site-mean time series for each variable (area mean over polygon)
            #    If the polygon is smaller than a pixel, these may be all-NaN; we handle a fallback below.
            rain_ts       = precip_matched.where(mask_da).mean(("y", "x"), skipna=True)  # mm/day
            erosivity_ts  = Erosivity.where(mask_da).mean(("y", "x"), skipna=True)
            rusle_ts      = rusle_stack.where(mask_da).mean(("y", "x"), skipna=True)
            sdr_rusle_ts  = sdr_rusle_stack.where(mask_da).mean(("y", "x"), skipna=True)
        
            # 3) Fallback for tiny sites: nearest-cell time series at polygon centroid
            if np.isnan(rain_ts.isel(time=0)).item():
                centroid_xy = gpd.GeoSeries([geom], crs=sites_gdf.crs).to_crs(dem.rio.crs).iloc[0].centroid
                x0, y0 = float(centroid_xy.x), float(centroid_xy.y)
                rain_ts      = precip_matched.sel(x=x0, y=y0, method="nearest")
                erosivity_ts = Erosivity.sel(x=x0, y=y0, method="nearest")
                rusle_ts     = rusle_stack.sel(x=x0, y=y0, method="nearest")
                sdr_rusle_ts = sdr_rusle_stack.sel(x=x0, y=y0, method="nearest")
        
            # 4) Build the per-site daily DataFrame
            dates = pd.to_datetime(rain_ts["time"].values)
            df_site = pd.DataFrame({
                "Date": dates.strftime("%Y-%m-%d"),
                "Rainfall":       rain_ts.values.astype("float64"),
                "Rainfall erosivity": erosivity_ts.values.astype("float64"),
                "RUSLE":          np.nan_to_num(rusle_ts.values.astype("float64"), nan=0.0),
                "SDR-RUSLE":      np.nan_to_num(sdr_rusle_ts.values.astype("float64"), nan=0.0),
            })
        
            # 5) Save/append one CSV per site (dedupe on Date to accumulate across years)
            site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
            os.makedirs(site_folder, exist_ok=True)
            out_csv = os.path.join(site_folder, f"Daily RUSLE and SDR-RUSLE {site_id}.csv")
        
            if os.path.exists(out_csv):
                df_prev = pd.read_csv(out_csv, parse_dates=["Date"])
                df_prev["Date"] = df_prev["Date"].dt.strftime("%Y-%m-%d")
                df_all = (pd.concat([df_prev, df_site], ignore_index=True)
                            .drop_duplicates(subset=["Date"])
                            .sort_values("Date"))
            else:
                df_all = df_site.sort_values("Date")
        
            df_all.to_csv(out_csv, index=False)
            #print(f"Saved site CSV: {out_csv}")
        ds.close()
    return all_sites_gpkg, sites_datasets