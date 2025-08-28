# === Standard Libraries ===
import os
import copy
import gc

# === Scientific & Array Processing ===
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter

# === Geospatial Processing ===
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.plot import plotting_extent
from rasterio.transform import from_origin
from shapely.geometry import Point, shape
from affine import Affine

# === Visualization ===
import matplotlib.pyplot as plt

# === Hydrological Modeling ===
from pysheds.grid import Grid

# === HTTP Requests ===
import requests


def dem_and_terrain(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path, ):
    print("Starting processes dem and topography ...")
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    topo_folder = os.path.join(catch_datasets, 'Topography')
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")

    folders_to_create = [catchment_folder, catch_datasets, catch_plots, topo_folder, sites_datasets, sites_plots]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    # ===========================
    gdf = gpd.read_file(Catchment_Shapefile_Path).dissolve()  # Load the shapefile
    catchment_crs = gdf.crs   # Get the CRS of the shapefile
    # Reproject the shapefile to WGS84 (EPSG:4326) and get the bounding box
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    
    # Calculate width and height at 1 arc-second resolution (~0.000277778 deg)
    resolution_deg = 1 / 3600
    width = int((maxx - minx) / resolution_deg)
    height = int((maxy - miny) / resolution_deg)
    bbox = f"{minx},{miny},{maxx},{maxy}"
    # WCS request parameters
    wcs_url = DEM_url
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": "1",  # Coverage ID for the DEM layer (might vary depending on service)
        "crs": "EPSG:4326",
        "bbox": bbox,
        "width": width,
        "height": height,
        "format": "GeoTIFF"}
    print("Requesting DEM from WCS...")
    response = requests.get(wcs_url, params=params, stream=True)
    # Save the output
    dem_temp = os.path.join(topo_folder, "DEM_temp.tif")
    if response.status_code == 200 and 'image/tiff' in response.headers.get('Content-Type', ''):
        with open(dem_temp, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"DEM saved to {topo_folder}")
    else:
        print("Failed to download DEM. Status code:", response.status_code)
        print("Content type:", response.headers.get('Content-Type'))
        print("Response content (first 500 chars):", response.content[:500])
    # ===========================
    # CLIP THE DEM USING SHAPEFILE & SET VALUES OUTSIDE BOUNDARY TO NAN
    with rxr.open_rasterio(dem_temp, masked=True) as dem_wgs84:
        if dem_wgs84.rio.crs != gdf_wgs84.crs:  # Ensure CRS matches before clipping
            gdf_wgs84 = gdf.to_crs(dem_wgs84.rio.crs)
        # Clip the DEM using the shapefile boundary and set nodata values to NaN
        clipped_dem = dem_wgs84.rio.clip(gdf_wgs84.geometry, gdf_wgs84.crs, drop=True, all_touched=True)  # Include pixels that touch the polygon boundary
        clipped_dem = clipped_dem.where(clipped_dem != clipped_dem.rio.nodata, np.nan)  # Convert nodata to NaN
    # ===========================
    dem_clipped_wgs84_file = os.path.join(topo_folder, "DEM_WGS84.tif")    # SAVE THE CLIPPED DEM IN WGS84
    clipped_dem.rio.to_raster(dem_clipped_wgs84_file)
    # ===========================
    dem_projected_file = os.path.join(topo_folder, "DEM.tif") # REPROJECT CLIPPED DEM TO MATCH SHAPEFILE CRS
    with rio.open(dem_clipped_wgs84_file) as src: # Read the clipped DEM and reproject it
        transform, width, height = calculate_default_transform(
            src.crs, catchment_crs, src.width, src.height, *src.bounds, resolution=(30, 30))  # Target 30m resolution
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': catchment_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': np.nan})
        with rio.open(dem_projected_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=catchment_crs,
                    resampling=Resampling.nearest  # Use bilinear for smoother resampling
                )
    # Open the DEM raster and extract the elevation data
    with rio.open(dem_projected_file) as src:
        dem_data = src.read(1)  # Read the first band (DEM data)
        transform = src.transform  # Affine transformation for pixel to geographic coordinates
        nodata = src.nodata  # NoData value used in the DEM
        meta = src.meta.copy()
    # Get pixel resolution (grid cell size)
    xres = transform[0]  # Width of a pixel (east-west direction)
    yres = abs(transform[4])  # Height of a pixel (north-south direction)
    pixel_area = xres * yres  # Area of a single pixel
    dem_data = np.where(dem_data == nodata, np.nan, dem_data)# Replace NoData values with NaN for numerical operations
    # Calculate slope using the gradient of the elevation data
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)  # Elevation gradient in x and y directions
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)  # Slope as a ratio (rise/run)
    slope_radians = np.arctan(slope_ratio)  # Slope in radians
    slope_degrees = np.degrees(slope_radians)  # Slope in degrees
    slope_percent = slope_ratio * 100  # Slope as a percentage
    # Calculate the aspect (direction of steepest descent) using arctan2
    aspect_radians = np.arctan2(dz_dy, -dz_dx)  # Aspect in radians
    # Ensure aspect is in the range [0, 2π]
    aspect_radians = np.where(aspect_radians < 0, 2 * np.pi + aspect_radians, aspect_radians) # 0 radians (or 0°) indicates a north-facing slope
    # Initialize a Pysheds Grid object and read the DEM data into it                          # π/2 radians (or 90°) indicates an east-facing slope.
    aspect_degrees = np.degrees(aspect_radians)                                               # π radians (or 180°) indicates a south-facing slope.
                                                                                              # 3π/2 radians (or 270°) indicates a west-facing slope.
     # === TPI: Focal elevation - mean of 8 neighbors
    def tpi_func(window):
        center = window[len(window) // 2]
        neighbors = np.delete(window, len(window) // 2)
        return center - np.nanmean(neighbors)
    # === TRI: Mean absolute elevation difference with neighbors
    def tri_func(window):
        center = window[len(window) // 2]
        neighbors = np.delete(window, len(window) // 2)
        return np.nanmean(np.abs(neighbors - center))
    # Apply 3x3 window (footprint) to DEM
    footprint = np.ones((3, 3))
    tpi = generic_filter(dem_data, tpi_func, footprint=footprint, mode='constant', cval=np.nan)
    tri = generic_filter(dem_data, tri_func, footprint=footprint, mode='constant', cval=np.nan)
    # === Add all layers to output
    terrain_layers = {
        "Slope in degree.tif": slope_degrees,
        "Slope in radian.tif": slope_radians,
        "Slope in percent.tif": slope_percent,
        "Aspect in radian.tif": aspect_radians,
        "Aspect in degree.tif": aspect_degrees,
        "Topographic Position Index.tif": tpi,
        "Terrain Ruggedness Index.tif": tri}
    # === Read and add DEM to terrain_layers
    with rio.open(dem_projected_file) as dem_src:
        dem_data = dem_src.read(1)
        dem_extent = plotting_extent(dem_src)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_data = np.where(dem_data == dem_src.nodata, np.nan, dem_data)
    terrain_layers["DEM.tif"] = (dem_data, dem_extent, dem_meta)  # Add DEM to terrain layers (plotting last)
    sites_gdf = gpd.read_file(Sites_Shapefile_Path).to_crs(dem_crs)# === Read and reproject sites to match DEM CRS
    # === Plot each terrain layer, including DEM
    for name, value in terrain_layers.items():
        if isinstance(value, tuple):
            data, extent, meta_layer = value
        else:
            data = value
            extent = plotting_extent(rio.open(dem_projected_file))  # fallback for other layers
            meta_layer = meta
        out_path = os.path.join(topo_folder, name)
        with rio.open(out_path, "w", **meta_layer) as out_dst:
            out_dst.write(data.astype(np.float32), 1)
        # === Plotting
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(data, cmap='viridis', extent=extent, origin='upper')
    
        gdf.boundary.plot(ax=ax, color='black', linewidth=1.2) # Catchment boundary
        sites_gdf.plot(ax=ax, markersize=15, color='red') # Site points
        # Annotate each site
        for idx, row in sites_gdf.iterrows():
            ax.annotate(
                text=str(row['id']),  # Replace if different
                xy=(row.geometry.x, row.geometry.y), xytext=(3, 3), textcoords='offset points', fontsize=7, color='red')
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=1)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(os.path.splitext(name)[0], fontsize=9)
        # Axis limits and ticks
        xmin, xmax, ymin, ymax = extent
        ax.set_xticks(np.linspace(xmin, xmax, 4))
        #ax.set_yticks(np.linspace(ymin, ymax, 4))
        ax.set_title(os.path.splitext(name)[0], fontsize=10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        # Save plot
        plot_filename = os.path.splitext(name)[0] + ".png"
        plot_path = os.path.join(catch_plots, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close("all")
    os.remove(dem_temp)

        # === Read DEM metadata ===
    with rio.open(dem_projected_file) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
    sites_gdf = gpd.read_file(Sites_Shapefile_Path).to_crs(dem_crs) # === Load site points
    grid = Grid.from_raster(dem_projected_file) # === Load and preprocess DEM
    dem = grid.read_raster(dem_projected_file)
    inflated_dem = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32) # === Flow direction and accumulation
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)
    
    WH_df = []
    for idx, row in sites_gdf.iterrows():
        try:
            site_id = row['id']
            site_geom = row.geometry
            x, y = site_geom.x, site_geom.y
            # === Choose polygon source: buffer OR hydrologic catchment ===
            if Buffer_dist is not None:
                # ---- Buffer path: build a simple metric buffer around the site point ----
                # Assumes dem_crs units are meters. If dem_crs is geographic, reproject site_geom before buffering.
                combined_geom = site_geom.buffer(Buffer_dist)
            else:
                # ---- Catchment path: original delineation (unchanged) ----
                grid_local = copy.deepcopy(grid)
                x_snap, y_snap = grid_local.snap_to_mask(acc > 25, (x, y))  # Snap pour point to highest accumulation
                catch = grid_local.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate")
                grid_local.clip_to(catch)
                clipped_catch = np.array(grid_local.view(catch), dtype=np.int16)
                # === Extract extent and transform
                min_x, min_y, max_x, max_y = grid_local.extent
                pixel_size_x = dem_transform.a
                pixel_size_y = -dem_transform.e
                transform_clipped = from_origin(min_x, max_y, pixel_size_x, pixel_size_y)
                shapes_gen = shapes(clipped_catch, transform=transform_clipped)  # === Vectorize
                geometries = [shape(geom) for geom, val in shapes_gen if val == 1]
                if not geometries:
                    print(f"No catchment found for site {site_id}")
                    continue
                combined_geom = gpd.GeoSeries(geometries).unary_union

            # Drop geometry and convert to dictionary, ensuring clean alignment
            attrs = row.drop(labels="geometry").to_dict()
            site_gdf = gpd.GeoDataFrame([attrs], geometry=[combined_geom], crs=dem_crs)
            site_gdf['X_site'] = x
            site_gdf['Y_site'] = y
            site_gdf['Area_m2'] = round(site_gdf.geometry.area.values[0], 0)
            site_gdf['Area_ha'] = round(site_gdf['Area_m2'] / 10000, 1)
            # === create folder for this site
            site_data = os.path.join(sites_datasets, f"Site_{site_id}")
            os.makedirs(site_data, exist_ok=True)
            site_plot = os.path.join(sites_plots, f"Site_{site_id}")
            os.makedirs(site_plot, exist_ok=True)
            # === summarise raster layers for the site
            raster_files = ["Aspect in degree.tif", "DEM.tif", "Slope in degree.tif", "Terrain Ruggedness Index.tif", "Topographic Position Index.tif"]
            for raster_name in raster_files:
                raster_path = os.path.join(topo_folder, raster_name)
                if not os.path.exists(raster_path):
                    print(f"{raster_name} not found — skipping.")
                    continue
                short_name = os.path.splitext(raster_name)[0]   #.replace(" ", "_")
                site_gdf[f"{short_name} (mean)"] = np.nan
                site_gdf[f"{short_name} (median)"] = np.nan
                site_gdf[f"{short_name} (at site)"] = np.nan

                with rio.open(raster_path) as src:
                    if site_gdf.crs != src.crs:
                        site_gdf = site_gdf.to_crs(src.crs)
                    try:
                        geom = [site_gdf.geometry.iloc[0]]
                        out_image, out_transform = rio.mask.mask(src, geom, crop=True)
                        data = out_image[0]
                        data = np.where(data == src.nodata, np.nan, data)
                        site_gdf.at[0, f"{short_name} (mean)"] = round(np.nanmean(data), 2)
                        site_gdf.at[0, f"{short_name} (median)"] = round(np.nanmedian(data), 2)
                        # sample at site
                        site_x, site_y = site_gdf['X_site'].iloc[0], site_gdf['Y_site'].iloc[0]
                        rowcol = src.index(site_x, site_y)
                        band = src.read(1)
                        site_value = band[rowcol[0], rowcol[1]]
                        if site_value == src.nodata:
                            site_value = np.nan
                        site_gdf.at[0, f"{short_name} (at site)"] = round(site_value, 2) if not np.isnan(site_value) else np.nan
                        # === Save clipped raster ===
                        clipped_meta = src.meta.copy()
                        clipped_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "crs": src.crs,
                            "nodata": src.nodata
                        })
                        clipped_raster_path = os.path.join(site_data, f"{short_name}.tif")
                        with rio.open(clipped_raster_path, "w", **clipped_meta) as dest:
                            dest.write(out_image)
                        try:
                            if out_transform:
                                left = out_transform[2]
                                top = out_transform[5]
                                pixel_width = out_transform[0]
                                pixel_height = -out_transform[4]
                                right = left + pixel_width * out_image.shape[2]
                                bottom = top - pixel_height * out_image.shape[1]
                                extent = [left, right, bottom, top]
                            else:
                                extent = [0, out_image.shape[2], 0, out_image.shape[1]]  # fallback                        
                            # === Fixed figure size
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(out_image[0], extent=extent, cmap="viridis", origin='upper')# === Plot raster
                            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)# === Colorbar
                            cbar.ax.tick_params(labelsize=9)
                            cbar.set_label(f'{short_name}', fontsize=9)
                            # === Overlay catchment boundary # === Plot site point (cyan circle)
                            site_gdf.boundary.plot(ax=ax, color='black', linewidth=1.2)
                            gpd.GeoSeries([Point(site_x, site_y)], crs=site_gdf.crs).plot(ax=ax, markersize=15, color='red', marker='o', label='Site location')
                            # === Annotate site
                            ax.annotate(f"Site {site_id}", xy=(site_x, site_y), xytext=(5, 5), textcoords='offset points', fontsize=7, color='red')                          
                            ax.set_title(f"{short_name} - Site {site_id}", fontsize=10) # === Labels and layout
                            ax.set_xlabel("Longitude", fontsize=9)
                            ax.set_ylabel("Latitude", fontsize=9)
                            ax.tick_params(axis='both', labelsize=9)
                            plt.tight_layout()
                            plot_path = os.path.join(site_plot, f"{short_name}.png")
                            plt.savefig(plot_path, dpi=200)
                            plt.close("all")
                        except Exception as e:
                            print(f"Error plotting {short_name} for site {site_id}: {e}")
                    except Exception as e:
                        print(f"Error summarising raster {raster_name} for site {site_id}: {e}")
                        continue
            # Append to master list and save
            WH_df.append(site_gdf)
            file_gpkg = os.path.join(site_data, f"Site_{site_id}.gpkg")
            file_csv = os.path.join(site_data, f"Site_{site_id}.csv")
            site_gdf.to_file(file_gpkg, driver="GPKG")
            site_gdf.drop(columns="geometry").to_csv(file_csv, index=False)
            print(f"Site {site_id} topogrphic data and plots saved.")
        except Exception as e:
            print(f"Error processing site {row.get('id', idx)}: {e}")
            continue
    # Save all sites data
    all_gdf = gpd.GeoDataFrame(pd.concat(WH_df, ignore_index=True), crs=dem_crs)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    all_sites_csv = os.path.join(sites_datasets, "All Sites Data.csv")
    all_gdf.to_file(all_sites_gpkg, driver="GPKG")
    all_gdf.drop(columns="geometry").to_csv(all_sites_csv, index=False)
    gc.collect()

    print(f"All sites summary saved in {sites_datasets}")
    return dem_projected_file, all_sites_gpkg, sites_datasets, sites_plots