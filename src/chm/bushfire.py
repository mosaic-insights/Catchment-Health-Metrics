# === Standard Libraries ===
import os
import re

# === Scientific & Array Processing ===
import numpy as np
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.features import rasterize

# === HTTP Requests ===
import requests

# === Visualization ===
import matplotlib.pyplot as plt


def historical_bushfire_risk_profile(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Starting downlaoding and processing bushfire data...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    bushfire_output = os.path.join(catch_datasets, 'Historical Bushfire')
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    
    folders_to_create = [catchment_folder, catch_datasets, catch_plots, bushfire_output, sites_datasets, sites_plots]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    SDR_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")
    TWI_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")

    gdf = gpd.read_file(Catchment_Shapefile_Path)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    bbox = f"{minx},{miny},{maxx},{maxy}"
    # ArcGIS REST API query
    url = Bushfire_url
    params = {
        "f": "geojson",
        "where": "1=1",
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true"
    }    
    # Request data
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    bushfire_all = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    
    # Now clip to your exact catchment boundary
    bushfire_in_catchment = gpd.clip(bushfire_all, gdf_wgs84)
    bushfire_in_catchment = bushfire_in_catchment.to_crs(gdf.crs)
    bushfire_in_catchment["ignition_date"] = pd.to_datetime(bushfire_in_catchment["ignition_date"], unit='ms') # convert the milisecond time format to date format
    # Save result
    bushfire_path = os.path.join(bushfire_output, "Historical_Bushfires_Boundary.gpkg")
    bushfire_in_catchment.to_file(bushfire_path, driver="GPKG")
    
    # Drop geometry and save attribute table as CSV
    bushfire_table = bushfire_in_catchment.drop(columns='geometry')
    csv_path = os.path.join(bushfire_output, "Historical_Bushfires_Attributes.csv")
    bushfire_table.to_csv(csv_path, index=False)

    sites_gdf = gpd.read_file(all_sites_gpkg)
    for idx, row in sites_gdf.iterrows():
        try:
            site_id = row['id']
            site_geom = row.geometry
            attrs = row.drop(labels="geometry").to_dict()
            site_gdf = gpd.GeoDataFrame([attrs], geometry=[site_geom], crs=sites_gdf.crs)
    
            site_data = os.path.join(sites_datasets, f"Site_{site_id}")
            os.makedirs(site_data, exist_ok=True)
            site_plot = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_plot, exist_ok=True)
            # Clip bushfire data to site polygon
            site_fire = gpd.clip(bushfire_in_catchment, site_gdf)
            if not site_fire.empty:
                site_fire_gpkg = os.path.join(site_data, f"Site_{site_id}_Bushfires.gpkg")
                site_fire_csv = os.path.join(site_data, f"Site_{site_id}_Bushfires.csv")

                site_fire.to_file(site_fire_gpkg, driver="GPKG")
                site_fire.drop(columns="geometry").to_csv(site_fire_csv, index=False)
                
            # === Process SDR rasters ===
            for fname in os.listdir(SDR_path):
                if "SDR" in fname and fname.endswith(".tif"):
                    sdr_file = os.path.join(SDR_path, fname)
                    with rio.open(sdr_file) as src:
                        max_sdr = np.nanmax(src.read(1))
                        out_image, out_transform = mask(src, site_gdf.geometry, crop=True)
                        out_meta = src.meta.copy()
                        sdr_data = out_image[0]
                        sdr_data = np.where(sdr_data == src.nodata, np.nan, sdr_data)

                        if not site_fire.empty:
                            bushfire_raster = rasterize(
                                [(geom, 1) for geom in site_fire.geometry],
                                out_shape=sdr_data.shape,
                                transform=out_transform,
                                fill=0,
                                dtype='uint8'
                            )
                            # === Cumulative Risk Profile Plot ===
                            valid_mask = ~np.isnan(sdr_data)
                            bushfire_mask = bushfire_raster == 1
                            sdr_burned = sdr_data[bushfire_mask & valid_mask]     
                            # === Skip if no fire found in this SDR ===
                            if sdr_burned.size == 0:
                                continue
                            sorted_sdr = np.sort(sdr_burned)# === Sort SDR values for burned pixels ===
                            cumulative_counts = np.arange(1, len(sorted_sdr) + 1)  # === Cumulative count ===
                            cumulative_percent = cumulative_counts / cumulative_counts[-1] * 100# === Convert to percentage ===
                            
                            # === Plot cumulative risk profile ===
                            plt.figure(figsize=(6, 4))
                            plt.plot(sorted_sdr, cumulative_percent, color='red', linewidth=2)
                            plt.xlabel("SDR Value", fontsize=10, fontweight='bold')
                            plt.ylabel("Cumulative Bushfire Pixels (%)", fontsize=10, fontweight='bold')
                            # Extract year from filename like "SDR_2024.tif"
                            match = re.search(r'\d{4}', fname)
                            year_str = match.group(0) if match else "Unknown"
                            plt.title(f"Cumulative Bushfire Risk Profile Site {year_str} - (Site {site_id})", fontsize=11, fontweight='bold')
                            #plt.xlim(0, 0.05)
                            plt.grid(True, linestyle="--", alpha=0.6)
                            #plt.legend(fontsize=8, loc="upper right")
                            plt.tight_layout()                           
                            plot_file = os.path.join(site_plot, f"Site_{site_id}_Cumulative_Bushfire_Risk_{fname.replace('.tif','')}.png")
                            plt.savefig(plot_file)
                            plt.close()
            # === Process TWI raster (single file) ===
            with rio.open(TWI_path) as twi_src:
                max_twi = np.nanmax(twi_src.read(1))
                twi_image, twi_transform = mask(twi_src, site_gdf.geometry, crop=True)
                twi_data = twi_image[0]
                twi_data = np.where(twi_data == twi_src.nodata, np.nan, twi_data)

                if not site_fire.empty:
                    twi_fire_raster = rasterize(
                        [(geom, 1) for geom in site_fire.geometry],
                        out_shape=twi_data.shape,
                        transform=twi_transform,
                        fill=0,
                        dtype='uint8'
                    )

                    valid_mask = ~np.isnan(twi_data)
                    bushfire_mask = twi_fire_raster == 1
                    twi_burned = twi_data[bushfire_mask & valid_mask]

                    if twi_burned.size > 0:
                        sorted_twi = np.sort(twi_burned)
                        cumulative_counts = np.arange(1, len(sorted_twi) + 1)
                        cumulative_percent = cumulative_counts / cumulative_counts[-1] * 100

                        plt.figure(figsize=(6, 4))
                        plt.plot(sorted_twi, cumulative_percent, color='red', linewidth=2)
                        plt.xlabel("Topographic wetness Value", fontsize=10, fontweight='bold')
                        plt.ylabel("Cumulative Bushfire Pixels (%)", fontsize=10, fontweight='bold')
                        plt.title(f"Cumulative Bushfire Risk Profile - (Site {site_id})", fontsize=11, fontweight='bold')
                        #plt.xlim(0, max_twi)
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()

                        plot_file = os.path.join(site_plot, f"Site_{site_id}_Cumulative_Bushfire_Risk_TWI.png")
                        plt.savefig(plot_file)
                        plt.close()
        except Exception as e:
            print(f"Error processing site {site_id}: {e}")

    print('Done!')