# === Standard Libraries ===
import os

# === Scientific & Array Processing ===
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping

# === Progress ===
from tqdm import tqdm

def awra_historical_data(CHM_Work_Space, Catchment_Shapefile_Path, Start_Year, End_Year):
    """
    Downloads AWRA-L variables, clips to catchment, and outputs:
    - Clipped NetCDF files
    - Daily CSVs of catchment-averaged values

    Parameters
    ----------
    CHM_Work_Space : str
        Base workspace directory
    Shapefile_Dir : str
        Full path to the catchment shapefile
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)
    """
    print("Start downloading awral hydrological data...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder = os.path.join(catch_datasets, "Hydroclimate")
    awral_folder_hist = os.path.join(hydro_folder, "Historical AWRAL")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    
    folders_to_create = [catchment_folder, catch_datasets, hydro_folder,awral_folder_hist, sites_datasets]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    var_name = {
        "rain_day": "Rainfall",
        "qtot": "Runoff",
        "etot": "Actual ET",
        "s0": "Upper Soil Moisture",
        "sd": "Deeper Soil Moisture"
    }
    # === Read and reproject catchment shapefile ===
    gdf = gpd.read_file(Catchment_Shapefile_Path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)
    
    # === Dictionary to accumulate data for each site ===
    site_dfs = {row['id']: pd.DataFrame() for _, row in sites_gdf.iterrows()}
    # === Loop through variables ===
    for var, name in var_name.items():
        print(f"Processing {var} → {name}")
        for year in tqdm(range(Start_Year, End_Year + 1), desc=f"{var}"):
            try:
                url = f"https://thredds.nci.org.au/thredds/dodsC/iu04/australian-water-outlook/historical/v1/AWRALv7/{var}_{year}.nc"
                ds = xr.open_dataset(url)

                ds_subset = ds.sel(
                    longitude=slice(minx, maxx),
                    latitude=slice(maxy, miny),
                    time=slice(f"{year}-01-01", f"{year}-12-31")
                )
                ds_subset.rio.write_crs("EPSG:4326", inplace=True)
                # Save clipped NetCDF
                clipped_nc_path = os.path.join(awral_folder_hist, f"{var}_{year}.nc")
                ds_subset.to_netcdf(clipped_nc_path)

                # === Clip to each site and extract mean values ===
                for idx, row in sites_gdf.iterrows():
                    site_id = row['id']
                    geom = [mapping(row.geometry)]
                    site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
                    os.makedirs(site_folder, exist_ok=True)

                    try:
                        ds_site = ds_subset.rio.clip(geom, all_touched=True, drop=True)
                        #site_nc_path = os.path.join(site_folder, f"{var}_{year}.nc")
                        #ds_site.to_netcdf(site_nc_path)
                        # Extract daily mean for the site
                        da = ds_site[var]
                        daily_mean = da.mean(dim=["latitude", "longitude"], skipna=True)
                        df = daily_mean.to_dataframe(name=name).reset_index()[["time", name]]
                        df.rename(columns={"time": "Date"}, inplace=True)

                        if site_dfs[site_id].empty:
                            site_dfs[site_id] = df
                        else:
                            merged_df = pd.merge(site_dfs[site_id], df, on="Date", how="outer")
                            # Clean up duplicate columns (from pandas merge auto-suffixing)
                            for col in merged_df.columns:
                                if col.endswith("_x") and col.replace("_x", "_y") in merged_df.columns:
                                    base = col.replace("_x", "")
                                    x_col = col
                                    y_col = f"{base}_y"
                                    if merged_df[y_col].isna().all():
                                        merged_df.rename(columns={x_col: base}, inplace=True)
                                        merged_df.drop(columns=[y_col], inplace=True)
                                    elif merged_df[x_col].isna().all():
                                        merged_df.rename(columns={y_col: base}, inplace=True)
                                        merged_df.drop(columns=[x_col], inplace=True)
                                    else:
                                        # If both have values, prefer non-NaN from x_col, then y_col
                                        merged_df[base] = merged_df[x_col].combine_first(merged_df[y_col])
                                        merged_df.drop(columns=[x_col, y_col], inplace=True)                            
                            site_dfs[site_id] = merged_df
                    except Exception as site_error:
                        print(f"  Skipping site {site_id} for {var} in {year}: {site_error}")
            except Exception as year_error:
                print(f"Failed for {var} in {year}: {year_error}")
    # === Save one CSV per site with all years/variables merged ===
    for site_id, df in site_dfs.items():
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        csv_path = os.path.join(sites_datasets, f"Site_{site_id}", f"Site {site_id} Daily Hydrological Data.csv")
        df.to_csv(csv_path, index=False)

    print("Finished downloading and processing AWRA-L site-based data.")
    return all_sites_gpkg, sites_datasets