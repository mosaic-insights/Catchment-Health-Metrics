# === Standard Libraries ===
import os
import gc
import re

# === HTTP ===
import requests

# === Numerical ===
import numpy as np

# === Geospatial ===
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import rioxarray as rxr
import xarray as xr

# === Plotting ===
import matplotlib.pyplot as plt


# Land use lookup dictionary (VALUE -> CLASS)
landuse_lookup = {
    **dict.fromkeys([110, 111, 112, 113, 114, 115, 116, 117], 'Nature conservation'),
    **dict.fromkeys([120, 121, 122, 123, 124, 125], 'Managed resource protection'),
    **dict.fromkeys([130, 131, 132, 133, 134], 'Other minimal use'),
    210: 'Grazing native vegetation',
    **dict.fromkeys([220, 221, 222], 'Production native forests'),
    **dict.fromkeys([310, 311, 312, 313, 314], 'Plantation forests'),
    **dict.fromkeys([320, 321, 322, 323, 324, 325], 'Grazing modified pastures'),
    **dict.fromkeys([330, 331, 332, 333, 334, 335, 336, 337, 338], 'Cropping'),
    **dict.fromkeys([340, 341, 342, 343, 344, 345, 346, 347, 348, 349], 'Perennial horticulture'),
    **dict.fromkeys([350, 351, 352, 353], 'Seasonal horticulture'),
    **dict.fromkeys([360, 361, 362, 363, 364, 365], 'Land in transition'),
    **dict.fromkeys([410, 411, 412, 413, 414], 'Irrigated plantation forests'),
    **dict.fromkeys([420, 421, 422, 423, 424], 'Grazing irrigated modified pastures'),
    **dict.fromkeys([430, 431, 432, 433, 434, 435, 436, 437, 438, 439], 'Irrigated cropping'),
    **dict.fromkeys([440, 441, 442, 443, 444, 445, 446, 447, 448, 449], 'Irrigated perennial horticulture'),
    **dict.fromkeys([450, 451, 452, 453, 454], 'Irrigated seasonal horticulture'),
    **dict.fromkeys([460, 461, 462, 463, 464, 465], 'Irrigated land in transition'),
    **dict.fromkeys([510, 511, 512, 513, 514, 515], 'Intensive horticulture'),
    **dict.fromkeys([520, 521, 522, 523, 524, 525, 526, 527, 528], 'Intensive animal production'),
    **dict.fromkeys([530, 531, 532, 533, 534, 535, 536, 537, 538], 'Manufacturing and industrial'),
    **dict.fromkeys([540, 541], 'Urban residential'),
    **dict.fromkeys([542, 543, 544, 545], 'Rural residential and farm infrastructure'),
    **dict.fromkeys([550, 551, 552, 553, 554, 555], 'Services'),
    **dict.fromkeys([560, 561, 562, 563, 564, 565, 566, 567], 'Utilities'),
    **dict.fromkeys([570, 571, 572, 573, 574, 575], 'Transport and communication'),
    **dict.fromkeys([580, 581, 582, 583, 584], 'Mining'),
    **dict.fromkeys([590, 591, 592, 593, 594, 595], 'Waste treatment and disposal'),
    **dict.fromkeys([610, 611, 612, 613, 614], 'Lake'),
    **dict.fromkeys([620, 621, 622, 623], 'Reservoir/dam'),
    **dict.fromkeys([630, 631, 632, 633], 'River'),
    **dict.fromkeys([640, 641, 642, 643], 'Channel/aqueduct'),
    **dict.fromkeys([650, 651, 652, 653, 654], 'Marsh/wetland'),
    **dict.fromkeys([660, 661, 662, 663], 'Estuary/coastal waters'),
}
target_landuse_groups = {
    'Plantation forests': ['Plantation forests'],
    'Mining': ['Mining'],
    'Waste treatment and disposal': ['Waste treatment and disposal'],
    'Industry': ['Manufacturing and industrial'],
    'Transport': ['Transport and communication'],
    'Animal Production': ['Intensive animal production'],
    'Urban and Rural residential': ['Urban residential', 'Rural residential and farm infrastructure'],
    'Irrigation, Cropping, and Horticulture': [
        'Cropping', 'Perennial horticulture', 'Land in transition', 'Irrigated plantation forests',
        'Grazing irrigated modified pastures', 'Irrigated cropping', 'Irrigated perennial horticulture',
        'Irrigated seasonal horticulture', 'Irrigated land in transition', 'Intensive horticulture'
    ]
}
def landuse_risk_profile(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Start downloading and processing land use data...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    landuse_output = os.path.join(catch_datasets, "Landuse")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    
    folders_to_create = [catchment_folder, catch_datasets, landuse_output, sites_datasets, sites_plots]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    SDR_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")
    TWI_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")
    # Load catchment and convert to Web Mercator (EPSG:3857)
    gdf = gpd.read_file(Catchment_Shapefile_Path)
    gdf_crs = gdf.crs
    gdf_3857 = gdf.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf_3857.total_bounds

    # ExportImage request to ArcGIS ImageServer
    url = Landuse_url
    
    # Calculate appropriate image size based on 50m resolution
    width_m = maxx - minx
    height_m = maxy - miny
    pixel_size = 50
    width_px = int(width_m / pixel_size)
    height_px = int(height_m / pixel_size)
    
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{width_px},{height_px}",  # resolution-aware sizing
        "format": "tiff",
        "f": "json"
    }
    # Step 1: Get image download URL
    response = requests.get(url, params=params)
    response.raise_for_status()
    image_url = response.json()["href"]
    # Step 2: Download raster
    img_response = requests.get(image_url)
    img_response.raise_for_status()
    # Step 3: Save raster locally from in-memory download
    landuse_bbox_path = os.path.join(landuse_output, "Catchment_Landuse_bbox.tif")
    # Ensure no locks exist
    gc.collect()  # Trigger garbage collection to release unclosed references
    if os.path.exists(landuse_bbox_path):
        try:
            os.remove(landuse_bbox_path)
        except PermissionError:
            print(f"File {landuse_bbox_path} is locked. Skipping overwrite for safety.")
            return  # Exit early to prevent crash
    # Write image from MemoryFile
    with MemoryFile(img_response.content) as memfile:
        with memfile.open() as dataset:
            data = dataset.read().astype("float32")
            profile = dataset.profile.copy()
            profile.update({'dtype': 'float32', 'nodata': np.nan})
            with rio.open(landuse_bbox_path, 'w', **profile) as dst:
                dst.write(data)

    with rxr.open_rasterio(landuse_bbox_path, masked=True) as rds_raw:
        rds = rds_raw.rio.write_crs("EPSG:3857", inplace=False)  # Return a new object
    # Now you can proceed to clip and reproject
    rds_clipped = rds.rio.clip(gdf_3857.geometry.values.tolist(), gdf_3857.crs, drop=True, all_touched=True)
    # === Reproject and resample clipped land use to match TWI ===
    with rio.open(TWI_path) as twi_ref:
        twi_profile = twi_ref.profile
        twi_shape = (twi_ref.height, twi_ref.width)
        twi_transform = twi_ref.transform
        twi_crs = twi_ref.crs

    # Align land use raster to TWI grid
    rds_clipped = rds_clipped.rio.reproject_match(xr.open_rasterio(TWI_path).squeeze())
    rds_clipped = rds_clipped.rio.reproject("EPSG:3111")# Reproject land use clipped raster to match EPSG:3111
    LU_path = os.path.join(landuse_output, "Catchment_Landuse.tif")# Save clipped result with original dtype and resolution
    # If it exists and is locked, delete safely before overwrite
    if os.path.exists(LU_path):
        try:
            os.remove(LU_path)
        except PermissionError:
            print(f"File {LU_path} is locked. Trying to close and overwrite...")
    rds_clipped.rio.to_raster(LU_path)# Then write the raster
    
    # === Per site road risk profile ===
    sites_gdf = gpd.read_file(all_sites_gpkg)
    #print(sites_gdf.crs)
    for idx, row in sites_gdf.iterrows():
        try:
            site_id = row['id']
            site_geom = row.geometry
            site_gdf = gpd.GeoDataFrame([row.drop('geometry')], geometry=[site_geom], crs=sites_gdf.crs)
            site_data = os.path.join(sites_datasets, f"Site_{site_id}")
            site_plot = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_data, exist_ok=True)
            os.makedirs(site_plot, exist_ok=True)

            landuse_raster = rxr.open_rasterio(LU_path, masked=True).squeeze()  # Already EPSG:3111
            #print("Landuse raster CRS:", landuse_raster.rio.crs)
            site_gdf_proj = site_gdf  # Already EPSG:3111

            site_gdf_proj = site_gdf.to_crs(landuse_raster.rio.crs)
            landuse_clipped = landuse_raster.rio.clip(site_gdf_proj.geometry, site_gdf_proj.crs, drop=True, all_touched=True)
            #landuse_array = landuse_clipped.values.astype(int)
            landuse_array = np.where(np.isnan(landuse_clipped.values), -9999, landuse_clipped.values).astype(int)
            # === Reverse lookup to get VALUE -> CLASS mapping
            reverse_lookup = {}
            for code, cls in landuse_lookup.items():
                if isinstance(code, int):
                    reverse_lookup[code] = cls
                else:
                    reverse_lookup.update(dict.fromkeys(code, cls))  
            # === Mask nodata
            landuse_array = np.where(np.isnan(landuse_array), -9999, landuse_array)
            # === Load TWI
            with rio.open(TWI_path) as twi_src:
                twi_img, twi_transform = mask(twi_src, site_gdf.geometry, crop=True)
                twi_data = twi_img[0]
                twi_data = np.where(twi_data == twi_src.nodata, np.nan, twi_data)
            
            # === For each group of land use classes
            for group_name, classes in target_landuse_groups.items():
                codes = [code for code, label in reverse_lookup.items() if label in classes]
                mask_lu = np.isin(landuse_array, codes)
                # === Risk profile vs TWI
                twi_valid = ~np.isnan(twi_data)
                twi_selected = twi_data[mask_lu & twi_valid]
                if twi_selected.size > 0:
                    sorted_twi = np.sort(twi_selected)
                    cum_count = np.arange(1, len(sorted_twi)+1)
                    cum_percent = cum_count / cum_count[-1] * 100
            
                    plt.figure(figsize=(6, 4))
                    plt.plot(sorted_twi, cum_percent, color='red', linewidth=2)
                    plt.title(f'{group_name} – Cumulative Risk Profile (Site {site_id})', fontsize=11, fontweight='bold')
                    plt.xlabel("Topographic Wetness", fontsize=10, fontweight='bold')
                    plt.ylabel("Cumulative % Pixels", fontsize=10, fontweight='bold')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(site_plot, f"Site_{site_id}_LU_{group_name.replace(' ', '_')}_TWI.png"))
                    plt.close()
            
            # === Repeat similarly for SDR rasters
            for fname in os.listdir(SDR_path):
                if "SDR" in fname and fname.endswith(".tif"):
                    with rio.open(os.path.join(SDR_path, fname)) as sdr_src:
                        sdr_img, sdr_transform = mask(sdr_src, site_gdf.geometry, crop=True)
                        sdr_data = sdr_img[0]
                        sdr_data = np.where(sdr_data == sdr_src.nodata, np.nan, sdr_data)
                        for group_name, classes in target_landuse_groups.items():
                            codes = [code for code, label in reverse_lookup.items() if label in classes]
                            mask_lu = np.isin(landuse_array, codes)   
                            sdr_valid = ~np.isnan(sdr_data)
                            sdr_selected = sdr_data[mask_lu & sdr_valid]
                            if sdr_selected.size > 0:
                                sorted_sdr = np.sort(sdr_selected)
                                cum_count = np.arange(1, len(sorted_sdr)+1)
                                cum_percent = cum_count / cum_count[-1] * 100
                                match = re.search(r'\d{4}', fname)
                                year_str = match.group(0) if match else "Unknown"
                                plt.figure(figsize=(6, 4))
                                plt.plot(sorted_sdr, cum_percent, color='red', linewidth=2)
                                plt.title(f'{group_name} – Cumulative Risk Profile {year_str} (Site {site_id})', fontsize=11, fontweight='bold')
                                plt.xlabel("SDR", fontsize=10, fontweight='bold')
                                plt.ylabel("Cumulative % Pixels", fontsize=10, fontweight='bold')
                                plt.grid(True, linestyle='--', alpha=0.6)
                                plt.tight_layout()
                                plt.savefig(os.path.join(site_plot, f"Site_{site_id}_LU_{group_name.replace(' ', '_')}_SDR_{year_str}.png"))
                                plt.close()

        except Exception as e:
            print(f"Error processing site {site_id}: {e}")
    print("Done!")