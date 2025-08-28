# === Standard Libraries ===
import os

# === Scientific & Array Processing ===
import numpy as np
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import xarray as xr
import rioxarray  # to enable .rio accessor on xarray
from rasterio.features import geometry_mask

def download_awap_historical_data(CHM_Work_Space, Catchment_Shapefile_Path, Start_Year, End_Year):
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder = os.path.join(catch_datasets, "Hydroclimate")
    awap_folder_hist = os.path.join(hydro_folder, "Historical AWAP")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    for folder in [catchment_folder, catch_datasets, hydro_folder, awap_folder_hist, sites_datasets]:
        os.makedirs(folder, exist_ok=True)

    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    # === Variables and download strategy ===
    var_stat = {"precip": "total", "tmin": "mean", "tmax": "mean"}
    var_name = { "precip": "Precipitation", "tmax": "Max Temperature", "tmin": "Min Temperature",}
    # === Read geometries and extract bounds ===
    gdf = gpd.read_file(Catchment_Shapefile_Path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)
    site_dfs = {row['id']: pd.DataFrame() for _, row in sites_gdf.iterrows()}
    # === Loop over variables and years ===
    for var, longname in var_name.items():
        print(f"\nProcessing {longname} ({var})...")
        stat = var_stat[var]
        for year in range(Start_Year, End_Year + 1):
            try:
                # === Build OPeNDAP URL ===
                url = f"https://thredds.nci.org.au/thredds/dodsC/zv2/agcd/v1-0-1/{var}/{stat}/r005/01day/agcd_v1-0-1_{var}_{stat}_r005_daily_{year}.nc"

                # === Open and spatially subset ===
                ds = xr.open_dataset(url)
                clipped = ds[var].sel(lon=slice(minx, maxx),lat=slice(miny, maxy))  # Reversed because AGCD has descending lats
                clipped_ds = clipped.to_dataset(name=var)
                out_path = os.path.join(awap_folder_hist, f"{var}_{year}_clipped.nc")
                clipped_ds.to_netcdf(out_path)
            except Exception as e:
                print(f"Failed to download {var} {year}: {e}")
    return awap_folder_hist
# =========================================================================================================================
def extract_awap_to_site_csv(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Start extracting AWA data for sites...")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder = os.path.join(catch_datasets, "Hydroclimate")
    awap_folder_hist = os.path.join(hydro_folder, "Historical AWAP")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    for folder in [catchment_folder, catch_datasets, hydro_folder, awap_folder_hist, sites_datasets]:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    
    var_map = {"precip": "Precipitation", "tmax": "Max Temperature", "tmin": "Min Temperature"}
    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)  # Ensure lat/lon CRS
    nc_files_by_var = {var: [] for var in var_map}
    for f in os.listdir(awap_folder_hist):
        if f.endswith(".nc"):
            for var in var_map:
                if f.startswith(var):
                    nc_files_by_var[var].append(os.path.join(awap_folder_hist, f))
    # Sort each variable’s files by year
    for var in nc_files_by_var:
        nc_files_by_var[var].sort()
    # === Loop over sites ===
    for idx, site in sites_gdf.iterrows():
        site_id = site.get("id", idx)
        site_geom = [site.geometry]  # GeoJSON-like list
        site_df = None  # Will hold merged dataframe for this site
        site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
        os.makedirs(site_folder, exist_ok=True)
        for var, col_name in var_map.items():
            var_dfs = []
            for nc_path in nc_files_by_var[var]:
                ds = xr.open_dataset(nc_path)
                ds = ds.rio.write_crs("EPSG:4326")  # Attach CRS
                # Polygon mask
                mask = geometry_mask(
                    geometries=site_geom,
                    transform=ds.rio.transform(),
                    invert=True,
                    out_shape=(ds.rio.height, ds.rio.width)
                )
                mask_da = xr.DataArray(mask, coords={"lat": ds.lat, "lon": ds.lon}, dims=("lat", "lon"))
                site_vals = ds[var].where(mask_da)

                # Fallback: nearest cell if no pixels in mask
                if np.count_nonzero(~np.isnan(site_vals.isel(time=0))) == 0:
                    lon_pt, lat_pt = site.geometry.centroid.x, site.geometry.centroid.y
                    daily_mean = ds[var].sel(lat=lat_pt, lon=lon_pt, method="nearest")
                else:
                    daily_mean = site_vals.mean(dim=("lat", "lon"), skipna=True)

                df_var = daily_mean.to_dataframe(name=col_name).reset_index()
                var_dfs.append(df_var)
                ds.close()
            # Combine all years for this variable
            var_all_years = pd.concat(var_dfs, ignore_index=True)
            if site_df is None:
                site_df = var_all_years
            else:
                site_df = pd.merge(site_df, var_all_years, on="time", how="outer")
        # Final formatting for site CSV
        if site_df is not None and not site_df.empty:
            site_df.rename(columns={"time": "Date"}, inplace=True)
            site_df.sort_values("Date", inplace=True)
            site_df = site_df[["Date", "Precipitation", "Max Temperature", "Min Temperature"]]
            site_df = site_df.round(1)
            # Save to CSV in same folder as GPKG
            out_csv = os.path.join(site_folder, f"Site_{site_id}_precip_temp.csv")
            site_df.to_csv(out_csv, index=False)


def awap_historical_data(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path):   
    # Step 1 : downlaod hydrological data
    awap_folder_hist = download_awap_historical_data(
        CHM_Work_Space, 
        Catchment_Shapefile_Path, 
        Start_Year, End_Year)
    
    # Step 2 : extract for sitesa
    extract_awap_to_site_csv(
        CHM_Work_Space, 
        Catchment_Shapefile_Path)