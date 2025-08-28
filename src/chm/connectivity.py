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
from shapely.geometry import Point
from pysheds.grid import Grid

# === Visualization ===
import matplotlib.pyplot as plt


def process_surface_and_groundwater_connectivity(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path):
    print("Starting processesing surface and groundwater connectivity...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    con_folder = os.path.join(catch_datasets, 'Surface and Groundwater Connectivity')
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    # List of folders to create
    folders_to_create = [catchment_folder, catch_datasets, sites_datasets, con_folder,]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation","Indices","C Factor","Annual")

    # Open the DEM raster and extract the elevation data
    dem_projected_file = os.path.join(catch_datasets, "Topography","DEM.tif")
    with rio.open(dem_projected_file) as src: # Read reprojected DEM
        dem_data = src.read(1)  # Read the first band (DEM data)
        transform = src.transform  # Affine transformation for pixel to geographic coordinates
        nodata = src.nodata  # NoData value used in the DEM
        dem_meta = src.meta.copy()
        dem_profile = src.profile
        dem_crs = src.crs
    # Get pixel resolution (grid cell size)
    xres = transform[0]  # Width of a pixel (east-west direction)
    yres = abs(transform[4])  # Height of a pixel (north-south direction)
    pixel_area = xres * yres  # Area of a single pixel
    dem_data = np.where(dem_data == nodata, np.nan, dem_data) # Replace NoData values with NaN for numerical operations
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)  # Elevation gradient in x and y directions
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_percent = np.sqrt(dz_dx**2 + dz_dy**2) * 100 # Slope as a percentage
    slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))  # Slope in radians
    slope_degrees = np.degrees(slope_radians)  # Slope in degrees
    aspect_radians = np.arctan2(dz_dy, -dz_dx)  # Aspect in radian  # Calculate the aspect (direction of steepest descent) using arctan2
    # Ensure aspect is in the range [0, 2π]
    aspect_radians = np.where(aspect_radians < 0, 2 * np.pi + aspect_radians, aspect_radians)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Calculate flow direction and accumulation                    
    grid = Grid.from_raster(dem_projected_file)                                                        
    dem_grid = grid.read_raster(dem_projected_file)                                                                                                                                           
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.fill_depressions(dem_filled)
    inflated_dem = grid.resolve_flats(dem_filled)  # Ensure that flat areas drain correctly
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # Directional map for flow direction
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # Calculate flow accumulation (number of upstream cells contributing to each cell)
    acc = grid.accumulation(fdir, dirmap=dirmap)
    acc_data = np.array(acc, dtype=np.float32)  # Convert to numpy array
    area = acc_data * pixel_area  # This is area in meter square (Multiply the flow accumulation by the resolution to get the area)
    streams = (area > 1.3e4)
    stream_cells = np.where(streams)
    streams = np.where(np.isnan(dem_data), np.nan, streams.astype(np.float32))
    streams_path = os.path.join(con_folder, "Streams.tif")
    with rio.open(streams_path, 'w', **dem_meta) as dest:
        dest.write(np.where(np.isnan(streams), np.nan, streams).astype(np.float32), 1)
    Sth = np.where(slope_ratio < 0.005, 0.005, np.where(slope_ratio <= 1, slope_ratio, 1))
    nan_mask = np.isnan(slope_ratio)
    Sth[nan_mask] = np.nan
    Sth_path = os.path.join(con_folder, "Average Thresholded Slopes.tif")
    with rio.open(Sth_path, 'w', **dem_meta) as dest:
        dest.write(Sth.astype('float32'), 1)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Calculate TWI (Topographic Wetness Index)
    TWI = np.log(acc_data / np.tan(slope_radians))
    TWI = np.where(np.isfinite(TWI), TWI, np.nan) # Replace invalid values with NaN
    TWI_path = os.path.join(con_folder, "Topographic Wetness Index.tif")
    with rio.open(TWI_path, 'w', **dem_meta) as dest:
        dest.write(TWI.astype('float32'), 1)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Calculate LSi factor (slope length-gradient factor)
    specific_area = np.sqrt(acc_data * pixel_area)  # Estimate specific catchment area (Ai_in) in meter
    # Cap specific_area values to a maximum of 141
    specific_area = np.where(specific_area > 122, 122, specific_area) # set a max for slope length (sqrt(area in m2)) to avoid overestimation of the LS factor in heterogeneous landscapes
    # Calculate the aspect length (xi) as the sum of the absolute sine and cosine of the aspect angle
    xi = np.abs(np.sin(aspect_radians)) + np.abs(np.cos(aspect_radians))
    # Compute slope factor (Si) using slope in degrees
    Si = np.zeros_like(slope_percent)
    Si[slope_percent < 9] = 10.8 * np.sin(np.radians(slope_degrees[slope_percent < 9])) + 0.03
    Si[slope_percent >= 9] = 16.8 * np.sin(np.radians(slope_degrees[slope_percent >= 9])) - 0.50
    # Modify slope percentage
    m = np.zeros_like(slope_percent)  # Initialize m with the same shape as slope_percent
    m[slope_percent <= 1] = 0.2
    m[(slope_percent > 1) & (slope_percent <= 3.5)] = 0.3
    m[(slope_percent > 3.5) & (slope_percent <= 5)] = 0.4
    m[(slope_percent > 5) & (slope_percent <= 9)] = 0.5
    # For slopes greater than 9%, use a more detailed calculation
    slope_mask = slope_percent > 9  # instead of mask = slope_percent > 9
    if np.any(slope_mask):
        slope_radians_high = np.arctan(slope_percent[slope_mask] / 100)
        beta = (np.sin(slope_radians_high) / 0.0896) / ((3 * np.sin(slope_radians_high)**0.8) + 0.56)
        m[slope_mask] = beta / (1 + beta)
    # Calculate LSi factor (slope length-gradient factor)
    D = xres  # Grid cell dimension (same as pixel size in DEM)
    LSi = Si * (((specific_area + D**2)**(m + 1)) - (specific_area**(m + 1))) / ((D**(m + 2)) * (xi**m) * (22.13**m))
    # Replace NaN values in LSi with the NoData value before saving
    LSi = np.where(np.isnan(dem_data), np.nan, LSi)
    # Save layers
    LS_path = os.path.join(con_folder, "Slope length-gradient factor.tif")
    with rio.open(LS_path, 'w', **dem_meta) as dest:
        dest.write(LSi.astype('float32'), 1)
    terrain_layers = {"Flow Accumulation.tif": acc_data}
    for key in terrain_layers:# Ensure all terrain layers have NaN outside valid DEM area
        terrain_layers[key] = np.where(np.isnan(dem_data), np.nan, terrain_layers[key])
    with rio.open(dem_projected_file) as src:
        meta = src.meta.copy()
        meta.update({'dtype': rio.float32,'nodata': np.nan})
    # Save each terrain layer as a raster file
    for name, data in terrain_layers.items():
        out_path = os.path.join(con_folder, name)
        with rio.open(out_path, "w", **meta) as out_dst:
            out_dst.write(data.astype(np.float32), 1)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Calculate SDR
    sdr_output_dir = os.path.join(con_folder, "SDR")
    os.makedirs(sdr_output_dir, exist_ok=True)
    Sth_raster = grid.read_raster(Sth_path)
    acc_Sth = grid.accumulation(fdir=fdir, weights=Sth_raster)
    acc_no0 = np.where(acc == 0, np.nan, acc)
    Av_Sth = acc_Sth / acc_no0  # avarage Sth
    # C factors
    for file in os.listdir(annual_c_dir):
        if file.endswith(".tif") and "C_Factor" in file:
            year = file.split("_")[-1].replace(".tif", "")
            c_path = os.path.join(annual_c_dir, file)
            with rio.open(c_path) as src:
                c_factor = src.read(1)
        Cth = np.where(c_factor < 0.001, 0.001, c_factor)
        Cth_path = os.path.join(sdr_output_dir, f"Average Thresholded C factor_{year}.tif")
        dem_meta.update(dtype='float32')
        with rio.open(Cth_path, 'w', **dem_meta) as dest:
            dest.write(Cth.astype('float32'), 1)
        Cth_raster = grid.read_raster(Cth_path)
        acc_Cth = grid.accumulation(fdir=fdir, weights=Cth_raster)
        Av_Cth = acc_Cth / acc_no0
    
        distance_to_stream = np.full_like(dem_data, 0)
        Ddn = np.full_like(dem_data, 0.0, dtype=np.float32)
        st_indices = list(zip(stream_cells[0], stream_cells[1]))
        dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
        dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        diag_cell_size = (xres**2 + yres**2) ** 0.5
        grid_lengths = np.array([yres, diag_cell_size, xres, diag_cell_size,
                                 yres, diag_cell_size, xres, diag_cell_size])
        visited = np.zeros_like(dem_data, dtype=bool)
        visited[stream_cells] = True
    
        while st_indices:
            row, col = st_indices.pop(0)
            current_distance = distance_to_stream[row, col]
            for i in range(8):
                new_row = row + dy[i]
                new_col = col + dx[i]
                if 0 <= new_row < dem_data.shape[0] and 0 <= new_col < dem_data.shape[1]:
                    if fdir[new_row, new_col] == dirmap[(i + 4) % 8]:
                        if not visited[new_row, new_col]:
                            visited[new_row, new_col] = True
                            downslope_component = grid_lengths[i] / (Cth[new_row, new_col] * Sth[new_row, new_col]) if Cth[new_row, new_col] > 0 and Sth[new_row, new_col] > 0 else 0
                            Ddn[new_row, new_col] = Ddn[row, col] + downslope_component
                            distance_to_stream[new_row, new_col] = current_distance + grid_lengths[i]
                            st_indices.append((new_row, new_col))
    
        distance_to_stream[(dem_data == nodata)] = np.nan #Distance to Stream
        Ddn[(dem_data == nodata)] = np.nan #Downslope Path

        Dup = Av_Cth * Av_Sth * np.sqrt(area) # Upslope Area
        Dup[(dem_data == nodata)] = np.nan
    
        Ddn_safe = np.where(Ddn == 0, np.nan, Ddn) #Connectivity Index
        IC = np.log10(Dup / Ddn_safe)
        IC[(dem_data == nodata)] = np.nan
    
        SDR = SDR_max / (1 + np.exp((IC0 - IC) / k)) # SDR
        # Save layers
        terrain2_layers = {
            f"Distance to Stream_{year}.tif": distance_to_stream,
            f"Downslope Path_{year}.tif": Ddn,
            f"Upslope Area_{year}.tif": Dup,
            f"Connectivity Index_{year}.tif": IC}
        # Ensure all terrain layers have NaN outside valid DEM area
        for key in terrain2_layers:
            terrain2_layers[key] = np.where(np.isnan(dem_data), np.nan, terrain2_layers[key])
        # Update metadata for output rasters
        with rio.open(dem_projected_file) as src:
            meta = src.meta.copy()
            meta.update({'dtype': rio.float32,'nodata': np.nan})
        # Save each terrain layer as a raster file
        for name, data in terrain2_layers.items():
            out_path = os.path.join(sdr_output_dir, name)
            with rio.open(out_path, "w", **meta) as out_dst:
                out_dst.write(data.astype(np.float32), 1)

        SDR_path = os.path.join(sdr_output_dir, f"SDR_{year}.tif")
        with rio.open(SDR_path, 'w', **dem_meta) as dest:
            dest.write(SDR.astype('float32'), 1)

        # =======================================================================================================================
        sites_gdf = gpd.read_file(all_sites_gpkg)
        sites_point = gpd.read_file(Sites_Shapefile_Path)
        WH_df = []
        
        # === Make sure these are defined before the loop
        raster_files = [TWI_path, SDR_path]  # You must define TWI_path and SDR_path before this block
        
        for idx, row in sites_gdf.iterrows():
            try:
                site_id = row['id']
                site_geom = row.geometry
                attrs = row.drop(labels="geometry").to_dict()
                site_gdf = gpd.GeoDataFrame([attrs], geometry=[site_geom], crs=dem_crs)
        
                site_data = os.path.join(sites_datasets, f"Site_{site_id}")
                os.makedirs(site_data, exist_ok=True)
                site_plot = os.path.join(sites_plots, f"Site_{site_id}")
                os.makedirs(site_plot, exist_ok=True)
        
                for raster_path in raster_files:
                    short_name = os.path.splitext(os.path.basename(raster_path))[0]
                    site_gdf[f"{short_name} (mean)"] = np.nan
                    site_gdf[f"{short_name} (median)"] = np.nan
                    site_gdf[f"{short_name} (at site)"] = np.nan
        
                    try:
                        with rio.open(raster_path) as src:
                            band_data = src.read(1)
                            band_data = np.where(band_data == src.nodata, np.nan, band_data)
                            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
                            # === Reproject site_gdf to raster CRS before any geoprocessing
                            if site_gdf.crs != src.crs:
                                site_gdf = site_gdf.to_crs(src.crs)
        
                            # === Plot full raster with all site points
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(band_data, cmap='viridis', extent=extent, origin='upper')
                            plt.colorbar(im, ax=ax, shrink=0.9).set_label(short_name)
        
                            if sites_point.crs != src.crs:
                                sites_point = sites_point.to_crs(src.crs)
                            sites_point.plot(ax=ax, color='red', markersize=15)
                            for s_idx, s_row in sites_point.iterrows():
                                ax.annotate(text=f"Site {s_row['id']}", xy=(s_row.geometry.x, s_row.geometry.y),
                                            xytext=(3, 3), textcoords='offset points', fontsize=7, color='red')
                            ax.set_title(f"{short_name}", fontsize=10)
                            ax.set_xlabel("Longitude")
                            ax.set_ylabel("Latitude")
                            plt.tight_layout()
                            plt.savefig(os.path.join(catch_plots, f"{short_name}.png"), dpi=300)
                            plt.close()
        
                            # === Clip and summarize raster for site
                            geom = [site_gdf.geometry.iloc[0]]
                            out_image, out_transform = rio.mask.mask(src, geom, crop=True, filled=False)
                            masked_band = out_image[0].filled(np.nan).astype(float)
        
                            site_gdf.at[0, f"{short_name} (mean)"] = round(np.nanmean(masked_band), 2)
                            site_gdf.at[0, f"{short_name} (median)"] = round(np.nanmedian(masked_band), 2)
        
                            # === Get raster value at site point
                            site_x, site_y = site_gdf['X_site'].iloc[0], site_gdf['Y_site'].iloc[0]
                            rowcol = src.index(site_x, site_y)
                            site_val = band_data[rowcol[0], rowcol[1]]
                            if site_val == src.nodata:
                                site_val = np.nan
                            site_gdf.at[0, f"{short_name} (at site)"] = round(site_val, 2) if not np.isnan(site_val) else np.nan
        
                            # === Save clipped raster
                            clipped_meta = src.meta.copy()
                            clipped_meta.update({
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform,
                                "crs": src.crs,
                                "nodata": src.nodata
                            })
                            clipped_path = os.path.join(site_data, f"{short_name}.tif")
                            with rio.open(clipped_path, "w", **clipped_meta) as dest:
                                dest.write(out_image)
        
                            # === Plot clipped raster
                            fig, ax = plt.subplots(figsize=(5, 5))
                            left, top = out_transform[2], out_transform[5]
                            width, height = out_image.shape[2], out_image.shape[1]
                            px_w, px_h = out_transform[0], -out_transform[4]
                            extent = [left, left + px_w * width, top - px_h * height, top]
        
                            im = ax.imshow(out_image[0], extent=extent, cmap="viridis", origin='upper')
                            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8).set_label(short_name)
                            site_gdf.boundary.plot(ax=ax, color='black')
                            gpd.GeoSeries([Point(site_x, site_y)], crs=site_gdf.crs).plot(ax=ax, color='red', markersize=15)
                            ax.annotate(f"Site {site_id}", xy=(site_x, site_y), xytext=(5, 5), textcoords='offset points',
                                        fontsize=7, color='red')
                            ax.set_title(f"{short_name} - Site {site_id}", fontsize=10)
                            ax.set_xlabel("Longitude")
                            ax.set_ylabel("Latitude")
                            plt.tight_layout()
                            plt.savefig(os.path.join(site_plot, f"{short_name}.png"), dpi=200)
                            plt.close()
        
                    except Exception as e:
                        print(f"Error processing raster {short_name} for site {site_id}: {e}")
                        continue
                WH_df.append(site_gdf)
                file_gpkg = os.path.join(site_data, f"Site_{site_id}.gpkg")
                file_csv = os.path.join(site_data, f"Site_{site_id}.csv")
                site_gdf.to_file(file_gpkg, driver="GPKG")
                site_gdf.drop(columns="geometry").to_csv(file_csv, index=False)
                print(f"Site {site_id} vegetation data and plots saved")
            except Exception as e:
                print(f"Error processing site {row.get('id', idx)}: {e}")
                continue
        # Save all sites data
        all_gdf = gpd.GeoDataFrame(pd.concat(WH_df, ignore_index=True), crs=dem_crs)
        all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
        all_sites_csv = os.path.join(sites_datasets, "All Sites Data.csv")
        all_gdf.to_file(all_sites_gpkg, driver="GPKG")
        all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)
    return all_sites_gpkg, LS_path, TWI_path, sdr_output_dir, sites_datasets
# ----------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
# NDVI risk profile
def ndvi_cumulative_risk_profiles(CHM_Work_Space, Catchment_Shapefile_Path):
    print("Starting NDVI cumulative risk profiles by SDR and TWI...")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    # List of folders to create
    folders_to_create = [catchment_folder, catch_datasets, sites_datasets, sites_plots]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    dem_projected_file = os.path.join(catch_datasets, "Topography","DEM.tif")
    annual_c_dir = os.path.join(catch_datasets, "Vegetation","Indices","C Factor","Annual")
    annual_ndvi_dir = os.path.join(catch_datasets, "Vegetation","Indices","NDVI","Annual")
    sdr_output_dir = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "SDR")
    TWI_path = os.path.join(catch_datasets, "Surface and Groundwater Connectivity", "Topographic Wetness Index.tif")

    sites_gdf = gpd.read_file(all_sites_gpkg)
    # Collect all NDVI and SDR filenames
    ndvi_files = [f for f in os.listdir(annual_ndvi_dir) if f.endswith(".tif") and "NDVI" in f]
    sdr_files = [f for f in os.listdir(sdr_output_dir) if f.endswith(".tif") and "SDR" in f]
    # Extract years from filenames
    ndvi_dict = {re.search(r'\d{4}', f).group(): f for f in ndvi_files if re.search(r'\d{4}', f)}
    sdr_dict = {re.search(r'\d{4}', f).group(): f for f in sdr_files if re.search(r'\d{4}', f)}
    # Match NDVI and SDR years
    common_years = sorted(set(ndvi_dict.keys()) & set(sdr_dict.keys()))
    for idx, row in sites_gdf.iterrows():
        try:
            site_id = row['id']
            site_geom = row.geometry
            site_gdf = gpd.GeoDataFrame([row.drop(labels="geometry")], geometry=[site_geom], crs=sites_gdf.crs)
            site_plot = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_plot, exist_ok=True)
            # TWI once for all years
            with rio.open(TWI_path) as twi_src:
                twi_img, twi_transform = mask(twi_src, site_gdf.geometry, crop=True)
                twi_data = np.where(twi_img[0] == twi_src.nodata, np.nan, twi_img[0])
            for year in common_years:
                ndvi_file = os.path.join(annual_ndvi_dir, ndvi_dict[year])
                sdr_file = os.path.join(sdr_output_dir, sdr_dict[year])
                # Open and mask NDVI
                with rio.open(ndvi_file) as ndvi_src:
                    ndvi_img, _ = mask(ndvi_src, site_gdf.geometry, crop=True)
                    ndvi_data = np.where(ndvi_img[0] == ndvi_src.nodata, np.nan, ndvi_img[0])
                # Open and mask SDR
                with rio.open(sdr_file) as sdr_src:
                    sdr_img, _ = mask(sdr_src, site_gdf.geometry, crop=True)
                    sdr_data = np.where(sdr_img[0] == sdr_src.nodata, np.nan, sdr_img[0])
                # Ensure same shape and valid mask
                if sdr_data.shape == ndvi_data.shape:
                    valid_mask = (~np.isnan(ndvi_data)) & (~np.isnan(sdr_data))
                    ndvi_vals = ndvi_data[valid_mask]
                    sdr_vals = sdr_data[valid_mask]
                    # Sort SDR and align NDVI
                    sorted_idx = np.argsort(sdr_vals)
                    sorted_sdr = sdr_vals[sorted_idx]
                    sorted_ndvi = ndvi_vals[sorted_idx]
                    cumulative_pct = np.arange(1, len(sorted_ndvi) + 1) / len(sorted_ndvi) * 100
                    plt.figure(figsize=(6, 4))
                    plt.plot(sorted_sdr, cumulative_pct, color='red', linewidth=2)
                    plt.xlabel("SDR Value", fontsize=10, fontweight='bold')
                    plt.ylabel("Cumulative NDVI Pixels (%)", fontsize=10, fontweight='bold')
                    plt.title(f"NDVI Risk Profile by SDR - {year} (Site {site_id})", fontsize=11, fontweight='bold')
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plot_file = os.path.join(site_plot, f"Site_{site_id}_NDVI_Cumulative_SDR_{year}.png")
                    plt.savefig(plot_file)
                    plt.close()

                # TWI-based cumulative NDVI
                if twi_data.shape == ndvi_data.shape:
                    valid_mask = (~np.isnan(ndvi_data)) & (~np.isnan(twi_data))
                    ndvi_vals = ndvi_data[valid_mask]
                    twi_vals = twi_data[valid_mask]
                    
                    sorted_idx = np.argsort(twi_vals)
                    sorted_twi = twi_vals[sorted_idx]
                    sorted_ndvi = ndvi_vals[sorted_idx]
                    cumulative_pct = np.arange(1, len(sorted_ndvi) + 1) / len(sorted_ndvi) * 100

                    plt.figure(figsize=(6, 4))
                    plt.plot(sorted_twi, cumulative_pct, color='red', linewidth=2)
                    plt.xlabel("Topographic Wetness Index", fontsize=10, fontweight='bold')
                    plt.ylabel("Cumulative NDVI Pixels (%)", fontsize=10, fontweight='bold')
                    plt.title(f"NDVI Risk Profile by TWI - {year} (Site {site_id})", fontsize=11, fontweight='bold')
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plot_file = os.path.join(site_plot, f"Site_{site_id}_NDVI_Cumulative_TWI_{year}.png")
                    plt.savefig(plot_file)
                    plt.close()
        except Exception as e:
            print(f"Error processing site {site_id}: {e}")
    print("NDVI cumulative profiles by SDR and TWI complete.")


def surface_ground_water_connectivity(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path):
    # Step 1 : calculate surface and ground water connectivity
    all_sites_gpkg, LS_path, TWI_path, sdr_output_dir, sites_datasets = process_surface_and_groundwater_connectivity(
        CHM_Work_Space, 
        Catchment_Shapefile_Path,               
        Sites_Shapefile_Path,        
    )
    # Step 2 : create risk profiles
    ndvi_cumulative_risk_profiles(
        CHM_Work_Space,
        Catchment_Shapefile_Path)