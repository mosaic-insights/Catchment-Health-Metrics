# === Standard Libraries ===
import os

# === Scientific & Array Processing ===
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping

def awra_projections_data(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Start downloading projected hydrological data...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    hydro_folder = os.path.join(catch_datasets, "Hydroclimate")
    awral_folder_proj = os.path.join(hydro_folder, "Future AWRAL")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")

    for folder in [catchment_folder, catch_datasets, hydro_folder, awral_folder_proj, sites_datasets]:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    
    # === Projections and variables to download ===
    projections = {
        "historical": "Historical",
        "rcp45": "RCP4.5",
        "rcp85": "RCP8.5"
    }
    var_names = {
        "qtot": "Runoff",
        "etot": "Actual ET",
        "s0": "Upper Soil Moisture",
        "sd": "Deeper Soil Moisture"
    }

    # === Load catchment and sites
    gdf = gpd.read_file(Catchment_Shapefile_Path).to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    sites_gdf = gpd.read_file(all_sites_gpkg).to_crs(epsg=4326)
    site_dfs = {row['id']: pd.DataFrame() for _, row in sites_gdf.iterrows()}

    for proj, projname in projections.items():
        tim = "19600101-20051231" if proj == "historical" else "20060101-20991231"

        for var, short_name in var_names.items():
            print(f"Processing {proj.upper()} → {var}")

            try:
                # === Dataset URL
                url = f"https://thredds.nci.org.au/thredds/dodsC/iu04/australian-water-outlook/hydrologic-projections/hydrologic-output-variables/output/AUS-5/BoM/AWRALv6-1-CNRM-CERFACS-CNRM-CM5/{proj}/r1i1p1/CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP/latest/day/{var}/AWRALv6-1-CNRM-CERFACS-CNRM-CM5_CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP_{proj}_r1i1p1_{var}_AUS-5_day_v1_{tim}.nc"

                ds = xr.open_dataset(url)
                lat_slice = slice(maxy, miny) if ds['lat'][0] > ds['lat'][-1] else slice(miny, maxy)
                ds_clipped = ds.sel(lon=slice(minx, maxx), lat=lat_slice)

                #ds_clipped = ds_clipped.isel(time=slice(0, 10))
                ds_clipped = ds_clipped.sel(time=slice("2006-01-01", "2006-01-10"))
                
                ds_clipped = ds_clipped.rename({'lat': 'y', 'lon': 'x'})
                ds_clipped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
                ds_clipped.rio.write_crs("EPSG:4326", inplace=True)

                # === Save clipped NetCDF
                nc_path = os.path.join(awral_folder_proj, f"{proj}_{var}.nc")
                if os.path.exists(nc_path):
                    try:
                        os.remove(nc_path)
                    except PermissionError:
                        print(f"File is in use: {nc_path}. Skipping save.")
                        continue
                ds_clipped.to_netcdf(nc_path)

                # === Per-site extraction
                #for idx, row in sites_gdf[sites_gdf['id'] == 1].iterrows():
                for idx, row in sites_gdf.iterrows():
                    site_id = row['id']
                    site_geom = [mapping(row.geometry)]
                    site_folder = os.path.join(sites_datasets, f"Site_{site_id}")
                    os.makedirs(site_folder, exist_ok=True)

                    try:
                        da = ds_clipped[var]
                        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
                        da.rio.write_crs("EPSG:4326", inplace=True)

                        da_clipped = da.rio.clip(site_geom, all_touched=True, drop=True)
                        daily_mean = da_clipped.mean(dim=["y", "x"], skipna=True)

                        df = daily_mean.to_dataframe(name=f"{short_name} - {projname}").reset_index()
                        df["Date"] = pd.to_datetime(df["time"]).dt.date
                        df.drop(columns=["time"], inplace=True)

                        # Drop all unwanted columns before merge
                        drop_cols = [col for col in df.columns if col.startswith("depth") or col == "spatial_ref" or col.endswith("_y")]
                        df.drop(columns=drop_cols, inplace=True, errors='ignore')


                        # Place Date column at front
                        cols = ['Date'] + [col for col in df.columns if col != 'Date']
                        df = df[cols]

                        # Merge to existing site_df
                        if site_dfs[site_id].empty:
                            site_dfs[site_id] = df
                        else:
                            site_dfs[site_id] = pd.merge(site_dfs[site_id], df, on="Date", how="outer")

                    except Exception as e:
                        print(f"Skipping site {site_id} for {var} in {proj}: {e}")

            except Exception as e:
                print(f"Error downloading {proj} {var}: {e}")

    # === Save final CSVs per site
    for site_id, df in site_dfs.items():
        if not df.empty and "Date" in df.columns:
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            out_csv = os.path.join(sites_datasets, f"Site_{site_id}", f"Site {site_id} Projected Hydrological Data.csv")
            df.to_csv(out_csv, index=False)
        else:
            print(f"Site {site_id} has no valid data — skipped saving.")

    print("Finished downloading and processing projected AWRA-L data.")
    return all_sites_gpkg, sites_datasets