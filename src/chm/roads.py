# === Standard Libraries ===
import os
import re

# === HTTP ===
import requests

# === Numerical ===
import numpy as np

# === Geospatial ===
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.features import rasterize

# === Plotting ===
import matplotlib.pyplot as plt

def national_roads_risk_profile(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Start downloading and processing national road data...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    roads_output = os.path.join(catch_datasets, "National Roads")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    
    folders_to_create = [catchment_folder, catch_datasets, roads_output, sites_datasets, sites_plots]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    SDR_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")
    TWI_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")   
    # Read and reproject catchment to EPSG:4326 for API query
    gdf = gpd.read_file(Catchment_Shapefile_Path)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    bbox = f"{minx},{miny},{maxx},{maxy}"

    # ArcGIS REST API query
    url = Road_url
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
    roads_all = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    # Clip roads to the actual catchment polygon
    roads_in_catchment = gpd.clip(roads_all, gdf_wgs84)
    # Reproject to original catchment CRS
    roads_in_catchment = roads_in_catchment.to_crs(gdf.crs)
    # Save GeoPackage
    roads_path = os.path.join(roads_output, "National_Roads_in_Catchment.gpkg")
    roads_in_catchment.to_file(roads_path, driver="GPKG")
    # Save attributes as CSV
    roads_table = roads_in_catchment.drop(columns='geometry')
    csv_path = os.path.join(roads_output, "National_Roads_Attributes.csv")
    roads_table.to_csv(csv_path, index=False)

    # === Per site road risk profile ===
    sites_gdf = gpd.read_file(all_sites_gpkg)
    for idx, row in sites_gdf.iterrows():
        try:
            site_id = row['id']
            site_geom = row.geometry
            site_gdf = gpd.GeoDataFrame([row.drop('geometry')], geometry=[site_geom], crs=sites_gdf.crs)

            site_data = os.path.join(sites_datasets, f"Site_{site_id}")
            site_plot = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_data, exist_ok=True)
            os.makedirs(site_plot, exist_ok=True)

            site_roads = gpd.clip(roads_in_catchment, site_gdf)

            # === Process SDR rasters ===
            for fname in os.listdir(SDR_path):
                if "SDR" in fname and fname.endswith(".tif"):
                    sdr_file = os.path.join(SDR_path, fname)
                    with rio.open(sdr_file) as src:
                        out_image, out_transform = mask(src, site_gdf.geometry, crop=True)
                        sdr_data = out_image[0]
                        sdr_data = np.where(sdr_data == src.nodata, np.nan, sdr_data)

                        if not site_roads.empty:
                            roads_raster = rasterize(
                                [(geom, 1) for geom in site_roads.geometry],
                                out_shape=sdr_data.shape,
                                transform=out_transform,
                                fill=0,
                                dtype='uint8'
                            )

                            valid_mask = ~np.isnan(sdr_data)
                            road_mask = roads_raster == 1
                            sdr_on_road = sdr_data[road_mask & valid_mask]

                            if sdr_on_road.size == 0:
                                continue

                            sorted_vals = np.sort(sdr_on_road)
                            cumulative_counts = np.arange(1, len(sorted_vals) + 1)
                            cumulative_percent = cumulative_counts / cumulative_counts[-1] * 100

                            match = re.search(r'\d{4}', fname)
                            year_str = match.group(0) if match else "Unknown"

                            plt.figure(figsize=(6, 4))
                            plt.plot(sorted_vals, cumulative_percent, color='red', linewidth=2)
                            plt.xlabel("SDR Value", fontsize=10, fontweight='bold')
                            plt.ylabel("Cumulative Road Pixels (%)", fontsize=10, fontweight='bold')
                            plt.title(f"Cumulative Road Risk Profile {year_str} (Site {site_id})", fontsize=11, fontweight='bold')
                            plt.grid(True, linestyle="--", alpha=0.6)
                            plt.tight_layout()
                            plt.savefig(os.path.join(site_plot, f"Site_{site_id}_Cumulative_Roads_Risk_SDR_{year_str}.png"))
                            plt.close()

            # === TWI Risk ===
            with rio.open(TWI_path) as twi_src:
                twi_image, twi_transform = mask(twi_src, site_gdf.geometry, crop=True)
                twi_data = twi_image[0]
                twi_data = np.where(twi_data == twi_src.nodata, np.nan, twi_data)

                if not site_roads.empty:
                    roads_raster = rasterize(
                        [(geom, 1) for geom in site_roads.geometry],
                        out_shape=twi_data.shape,
                        transform=twi_transform,
                        fill=0,
                        dtype='uint8'
                    )

                    valid_mask = ~np.isnan(twi_data)
                    road_mask = roads_raster == 1
                    twi_on_road = twi_data[road_mask & valid_mask]

                    if twi_on_road.size > 0:
                        sorted_vals = np.sort(twi_on_road)
                        cumulative_counts = np.arange(1, len(sorted_vals) + 1)
                        cumulative_percent = cumulative_counts / cumulative_counts[-1] * 100

                        plt.figure(figsize=(6, 4))
                        plt.plot(sorted_vals, cumulative_percent, color='red', linewidth=2)
                        plt.xlabel("Topographic Wetness", fontsize=10, fontweight='bold')
                        plt.ylabel("Cumulative Road Pixels (%)", fontsize=10, fontweight='bold')
                        plt.title(f"Cumulative Road Risk Profile (Site {site_id})", fontsize=11, fontweight='bold')
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()

                        plt.savefig(os.path.join(site_plot, f"Site_{site_id}_Cumulative_Roads_Risk_TWI.png"))
                        plt.close()

        except Exception as e:
            print(f"Error processing site {site_id}: {e}")

    print("Done!")