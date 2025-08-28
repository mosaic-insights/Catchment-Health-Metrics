import os
import numpy as np
import pandas as pd
import geopandas as gpd

## appending monitoring data
def appending_monitoring_data(CHM_Work_Space, Catchment_Shapefile_Path, Monitoring_Data):
    print("Starting appending monitoring data ...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")

    folders_to_create = [catchment_folder, sites_datasets]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    sites = gpd.read_file(all_sites_gpkg)
    for idx, row in sites.iterrows():
        site_id = row["id"]
        folder_name = f"Site_{site_id}"
        site_folder = os.path.join(sites_datasets, folder_name)
        if not os.path.isdir(site_folder):
            print(f"Folder not found for site {site_id}: {site_folder}")
            continue
        excel_file = os.path.join(Monitoring_Data, f"Site_{site_id}.xlsx")
        if not os.path.exists(excel_file):
            print(f"Monitoring Excel file not found for site {site_id}")
            continue
        try:
            monitoring_df = pd.read_excel(excel_file)
            site_attrs = row.drop("geometry").to_dict()
            site_geom = row.geometry
            site_crs = sites.crs

            site_info_df = pd.DataFrame([site_attrs] * len(monitoring_df))
            site_info_df.iloc[1:, :] = np.nan
            site_info_df["geometry"] = [site_geom] * len(monitoring_df)

            combined_df = pd.concat([site_info_df, monitoring_df.reset_index(drop=True)], axis=1)
            combined_gdf = gpd.GeoDataFrame(combined_df, geometry="geometry", crs=site_crs)

            out_gpkg = os.path.join(site_folder, f"Site_{site_id}.gpkg")
            combined_gdf.to_file(out_gpkg, driver="GPKG")

            out_csv = os.path.join(site_folder, f"Site_{site_id}.csv")
            combined_gdf.drop(columns="geometry").to_csv(out_csv, index=False)

            print(f"Saved monitoring data for site {site_id}")
        except Exception as e:
            print(f"Error processing site {site_id}: {e}")