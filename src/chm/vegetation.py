# === Standard Libraries ===
import os
import gc

# === Scientific & Array Processing ===
import numpy as np
import pandas as pd

# === Geospatial Processing ===
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import Point

# === Visualization ===
import matplotlib.pyplot as plt

# === Remote Sensing & STAC ===
import pystac_client
import odc.stac


def veg_indices_and_c_factor(CHM_Work_Space, Catchment_Shapefile_Path, Sites_Shapefile_Path):
    print("Starting processing vegetation and c factor...")
    catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    # Create folder and sub folders
    #catchment_metrics_folder = os.path.join(CHM_Work_Space, "Catchmnet Health Metrics")
    catchment_name = os.path.splitext(os.path.basename(Catchment_Shapefile_Path))[0].replace('_', ' ')
    catchment_folder = os.path.join(CHM_Work_Space, catchment_name)
    catch_datasets = os.path.join(catchment_folder, "Catchment Datasets")
    catch_plots = os.path.join(catchment_folder, "Catchment Plots and Maps")
    sites_datasets = os.path.join(catchment_folder, "Sites Datasets")
    sites_plots = os.path.join(catchment_folder, "Sites Plots and Maps")
    veg_folder = os.path.join(catch_datasets, "Vegetation")
    satellite_output = os.path.join(veg_folder, "Satellite data")
    indices_output = os.path.join(veg_folder, "Indices")
    ndvi_output = os.path.join(indices_output, "NDVI")
    c_factor_output = os.path.join(indices_output, "C Factor")

    # List of folders to create
    folders_to_create = [catchment_folder, catch_datasets, catch_plots, sites_datasets, sites_plots, veg_folder,
                         indices_output, ndvi_output, c_factor_output, satellite_output]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
    all_sites_gpkg = os.path.join(sites_datasets, "All Sites Data.gpkg")
    # --- Create Monthly, Seasonal, and Annual folders ---
    monthly_ndvi_dir = os.path.join(ndvi_output, "Monthly")
    seasonal_ndvi_dir = os.path.join(ndvi_output, "Seasonal")
    annual_ndvi_dir = os.path.join(ndvi_output, "Annual")
    monthly_c_dir = os.path.join(c_factor_output, "Monthly")
    seasonal_c_dir = os.path.join(c_factor_output, "Seasonal")
    annual_c_dir = os.path.join(c_factor_output, "Annual")

    for folder in [monthly_ndvi_dir, seasonal_ndvi_dir, annual_ndvi_dir, monthly_c_dir, seasonal_c_dir, annual_c_dir]:
        os.makedirs(folder, exist_ok=True)
    # Load the shapefile
    gdf = gpd.read_file(Catchment_Shapefile_Path)
    crs = gdf.crs
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    bbox = gdf_wgs84.total_bounds
    # Load DEM metadata for alignment (reclassifying layers)
    topo_folder = os.path.join(catch_datasets, 'Topography')
    dem_projected_file = os.path.join(topo_folder, "DEM.tif")
    with rio.open(dem_projected_file) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_width = dem_src.width
        dem_height = dem_src.height
        dem_nodata = dem_src.nodata
        dem_bounds = dem_src.bounds
        dem_resolution = abs(dem_src.transform.a)

    # make a query to get data
    """
    query = catalog.search(bbox=bbox, collections=collections, datetime=Datetime, filter=filter_query, sortby="eo:cloud_cover")
    items = list(query.items())
    ds = odc.stac.load(items, bands=Bands, crs=crs, resolution=dem_resolution, groupby="solar_day", bbox=bbox,)
    """ 
    # Split the requested Datetime at the Sentinel/Landsat handover
    SPLIT_DATE = pd.Timestamp("2016-01-01")
    _dt_start, _dt_end = Datetime.split("/")
    dt_start = pd.Timestamp(_dt_start)
    dt_end   = pd.Timestamp(_dt_end)
    ranges = []
    if dt_start < SPLIT_DATE:
        ranges.append((max(dt_start, pd.Timestamp.min), min(dt_end,   SPLIT_DATE - pd.Timedelta(seconds=1)), ["ga_ls7e_ard_3"], ["nbart_red", "nbart_nir"]))
    if dt_end >= SPLIT_DATE:
        ranges.append((max(dt_start, SPLIT_DATE), dt_end, ["ga_s2am_ard_3", "ga_s2bm_ard_3"], ["nbart_red", "nbart_nir_1"]))
    loaded = []
    for dt0, dt1, collections_sel, bands_sel in ranges:
        dt_str = f"{dt0:%Y-%m-%d}/{dt1:%Y-%m-%d}"
        # DEA STAC supports CQL2; fall back to "query" if needed
        try:
            query = catalog.search(bbox=bbox, collections=collections_sel, datetime=dt_str, filter=filter_query  # e.g. "eo:cloud_cover < 20"
            )
        except TypeError:
            query = catalog.search( bbox=bbox, collections=collections_sel, datetime=dt_str, query={"eo:cloud_cover": {"lt": 20}})
    
        items = list(query.items())
        if not items:
            print(f"No items for {collections_sel} in {dt_str}")
            continue
        # IMPORTANT: align to the DEM grid from the start
        ds_part = odc.stac.load(items, bands=bands_sel, crs=dem_crs,  resolution=dem_resolution, groupby="solar_day", bbox=bbox)
        # Harmonise band names so downstream always uses 'nbart_nir'
        if "nbart_nir_1" in ds_part.data_vars:
            ds_part = ds_part.rename({"nbart_nir_1": "nbart_nir"})
        loaded.append(ds_part)
    # Merge the time ranges into a single dataset
    if not loaded:
        raise RuntimeError("No imagery found in the requested time range.")
    ds = xr.concat(loaded, dim="time").sortby("time") if len(loaded) > 1 else loaded[0]
    
    # --- DAILY NDVI and C-factor with DEM alignment ---
    ts_ndvi_output = os.path.join(ndvi_output, "Time step")
    ts_c_factor_output = os.path.join(c_factor_output, "Time step")
    os.makedirs(ts_ndvi_output, exist_ok=True)
    os.makedirs(ts_c_factor_output, exist_ok=True)
    for time_index, timestamp in enumerate(ds.time.values):
        formatted_time = pd.to_datetime(timestamp).strftime('%Y%m%d')
        red = ds['nbart_red'].isel(time=time_index)
        nir = ds['nbart_nir'].isel(time=time_index)
        ndvi = (nir - red) / (nir + red)    
        # Clip NDVI to catchment
        ndvi_clipped = ndvi.rio.clip(gdf.geometry, gdf.crs, drop=True)
        # Align to DEM
        ndvi_aligned = ndvi_clipped.rio.reproject_match(
            xr.open_rasterio(dem_projected_file).squeeze())
        # Save aligned NDVI
        ndvi_raster_path = os.path.join(ts_ndvi_output, f"NDVI_{formatted_time}.tif")
        ndvi_aligned.rio.write_nodata(np.nan, inplace=True)
        ndvi_aligned.rio.to_raster(ndvi_raster_path)
        # Compute and save aligned C-factor
        c_factor = np.clip(np.exp(-2 * ndvi_aligned), 0, 1)
        c_factor_da = xr.DataArray(
            c_factor,
            coords=ndvi_aligned.coords,
            dims=ndvi_aligned.dims)
        c_factor_da.rio.write_crs(gdf.crs, inplace=True)
        c_factor_da.rio.write_nodata(np.nan, inplace=True)
        c_raster_path = os.path.join(ts_c_factor_output, f"C_Factor_{formatted_time}.tif")
        os.makedirs(os.path.dirname(c_raster_path), exist_ok=True)
        c_factor_da.rio.to_raster(c_raster_path)

        # Save raw bands aligned to DEM
        for band_name in ds.data_vars:
            band = ds[band_name].isel(time=time_index).rio.clip(gdf.geometry, gdf.crs, drop=True)
            band_aligned = band.rio.reproject_match(
                rxr.open_rasterio(dem_projected_file).squeeze())
            band_output = os.path.join(satellite_output, f"{band_name}_{formatted_time}.tif")
            band_aligned.rio.to_raster(band_output)

    # --- NDVI and C-factor AGGREGATIONS ---
    ndvi = (ds['nbart_nir'] - ds['nbart_red']) / (ds['nbart_nir'] + ds['nbart_red'])
    ndvi_clipped = ndvi.rio.clip(gdf.geometry, gdf.crs, drop=True)
    #dem_ref = rxr.open_rasterio(dem_projected_file, masked=True).squeeze()
    with rxr.open_rasterio(dem_projected_file, masked=True) as ds:
        dem_ref = ds.squeeze().copy()  # copy data to memory

   # === Monthly Aggregation ===
    ndvi_clipped.coords['year_month'] = ndvi_clipped['time'].dt.strftime('%Y-%m')
    
    ndvi_monthly = ndvi_clipped.groupby('year_month').median(dim='time')
    ndvi_monthly.coords['year'] = ndvi_monthly['year_month'].str.slice(0, 4)
    ndvi_monthly.coords['month'] = ndvi_monthly['year_month'].str.slice(5, 7)
    
    # Calculate C-factor using exponential transformation
    c_monthly = xr.apply_ufunc(lambda x: np.clip(np.exp(-2 * x), 0, 1), ndvi_monthly)
    for ym in ndvi_monthly['year_month'].values:
        year = str(ym)[:4]
        month = str(ym)[5:]
        try:
            ndvi_layer = ndvi_monthly.sel(year_month=ym)
            c_layer = c_monthly.sel(year_month=ym)
            # Align to DEM
            ndvi_layer = ndvi_layer.rio.reproject_match(dem_ref)
            c_layer = c_layer.rio.reproject_match(dem_ref)
    
            ndvi_layer.rio.to_raster(os.path.join(monthly_ndvi_dir, f"NDVI_{year}_{month}.tif"))
            c_layer.rio.to_raster(os.path.join(monthly_c_dir, f"C_Factor_{year}_{month}.tif"))
        except Exception as e:
            print(f"Skipping {year}-{month}: {e}")
    
    # === Seasonal Aggregation ===
    # Add season info using .dt.season (xarray provides it via datetime accessor)
    ndvi_clipped.coords['year_season'] = ndvi_clipped['time'].dt.strftime('%Y') + "-" + ndvi_clipped['time'].dt.season.astype(str)
    
    ndvi_seasonal = ndvi_clipped.groupby('year_season').median(dim='time')
    ndvi_seasonal.coords['year'] = ndvi_seasonal['year_season'].str.slice(0, 4)
    ndvi_seasonal.coords['season'] = ndvi_seasonal['year_season'].str.slice(5)
    
    c_seasonal = xr.apply_ufunc(lambda x: np.clip(np.exp(-2 * x), 0, 1), ndvi_seasonal)
    
    for ys in ndvi_seasonal['year_season'].values:
        year = str(ys)[:4]
        season = str(ys)[5:]
        try:
            ndvi_layer = ndvi_seasonal.sel(year_season=ys)
            c_layer = c_seasonal.sel(year_season=ys)
            # Align to DEM
            ndvi_layer = ndvi_layer.rio.reproject_match(dem_ref)
            c_layer = c_layer.rio.reproject_match(dem_ref)
    
            ndvi_layer.rio.to_raster(os.path.join(seasonal_ndvi_dir, f"NDVI_{season}_{year}.tif"))
            c_layer.rio.to_raster(os.path.join(seasonal_c_dir, f"C_Factor_{season}_{year}.tif"))
        except Exception as e:
            print(f"Skipping {season} {year}: {e}")

    # Annual
    ndvi_annual = ndvi.groupby('time.year').median(dim='time')
    # Then clip each year and save
    for year in ndvi_annual.year.values:
        ndvi_year = ndvi_annual.sel(year=year).rio.clip(gdf.geometry, gdf.crs, drop=True)
        c_factor = xr.apply_ufunc(lambda x: np.clip(np.exp(-2 * x), 0, 1), ndvi_year)
        # Align to DEM
        ndvi_year = ndvi_year.rio.reproject_match(dem_ref)
        c_factor = c_factor.rio.reproject_match(dem_ref)
    
        ndvi_year.rio.to_raster(os.path.join(annual_ndvi_dir, f"NDVI_{year}.tif"))
        c_factor.rio.to_raster(os.path.join(annual_c_dir, f"C_Factor_{year}.tif"))
    print("NDVI and C-factor rasters have been saved: timestep, monthly, seasonal, and annual.")
    # =======================================================================================================================
    # === Load site polygons
    sites_gdf = gpd.read_file(all_sites_gpkg)
    sites_point = gpd.read_file(Sites_Shapefile_Path)
    WH_df = []
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

            for set1_folder in os.listdir(indices_output):
                set1_path = os.path.join(indices_output, set1_folder)
                if not os.path.isdir(set1_path):
                    continue
                for set2_folder in os.listdir(set1_path):
                    if set2_folder not in ["Annual", "Seasonal"]:
                        continue
                    set2_path = os.path.join(set1_path, set2_folder)
                    for file in os.listdir(set2_path):
                        if not file.lower().endswith(".tif"):
                            continue
                        raster_path = os.path.join(set2_path, file)
                        short_name = os.path.splitext(file)[0]
                        site_gdf[f"{short_name} (mean)"] = np.nan
                        site_gdf[f"{short_name} (median)"] = np.nan
                        site_gdf[f"{short_name} (at site)"] = np.nan
                        try:
                            with rio.open(raster_path) as src:
                                # === Plot full raster with all sites for context
                                full_data = src.read(1)
                                full_data = np.where(full_data == src.nodata, np.nan, full_data)
                                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                                fig, ax = plt.subplots(figsize=(5, 5))
                                im = ax.imshow(full_data, cmap='viridis', extent=extent, origin='upper')
                                cbar = plt.colorbar(im, ax=ax, shrink=0.9)
                                cbar.set_label(f"{short_name}")
                                # Plot all sites points
                                if sites_point.crs != src.crs:
                                    sites_point = sites_point.to_crs(src.crs)
                                sites_point.plot(ax=ax, markersize=15, color='red')  
                                # Annotate all sites
                                for s_idx, s_row in sites_point.iterrows():
                                    ax.annotate(text=f"Site {s_row['id']}", xy=(s_row.geometry.x, s_row.geometry.y),
                                        xytext=(3, 3), textcoords='offset points', fontsize=7, color='red')
                                ax.set_title(f"{short_name}", fontsize=10)
                                ax.set_xlabel("Longitude", fontsize=9)
                                ax.set_ylabel("Latitude", fontsize=9)
                                ax.tick_params(axis='both', labelsize=9)
                                plt.tight_layout()
                                plot_path = os.path.join(catch_plots, f"{short_name}.png")
                                plt.savefig(plot_path, dpi=300)
                                plt.close("all") 
                                if site_gdf.crs != src.crs:
                                    site_gdf = site_gdf.to_crs(src.crs)
                                geom = [site_gdf.geometry.iloc[0]]
                                out_image, out_transform = rio.mask.mask(src, geom, crop=True, filled=False)
                                # out_image is a MaskedArray (bands, rows, cols)
                                masked_band = out_image[0]
                                data = masked_band.filled(np.nan).astype(float) # replace masked pixels with np.nan
                                site_gdf.at[0, f"{short_name} (mean)"] = round(np.nanmean(data), 2)
                                site_gdf.at[0, f"{short_name} (median)"] = round(np.nanmedian(data), 2)

                                site_x, site_y = site_gdf['X_site'].iloc[0], site_gdf['Y_site'].iloc[0]
                                rowcol = src.index(site_x, site_y)
                                band = src.read(1)
                                site_value = band[rowcol[0], rowcol[1]]
                                if site_value == src.nodata:
                                    site_value = np.nan
                                site_gdf.at[0, f"{short_name} (at site)"] = round(site_value, 2) if not np.isnan(site_value) else np.nan
                                # Save clipped raster
                                clipped_meta = src.meta.copy()
                                clipped_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], 
                                                     "transform": out_transform, "crs": src.crs, "nodata": src.nodata})
                                clipped_raster_path = os.path.join(site_data, f"{short_name}.tif")
                                with rio.open(clipped_raster_path, "w", **clipped_meta) as dest:
                                    dest.write(out_image)
                                # === Plot clipped raster dynamically ===
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
                                    im = ax.imshow(out_image[0], extent=extent, cmap="viridis", origin='upper')
                                    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
                                    cbar.ax.tick_params(labelsize=9)
                                    cbar.set_label(f'{short_name}', fontsize=9)
                                    # === Overlay catchment boundary
                                    site_gdf.boundary.plot(ax=ax, color='black', linewidth=1.2)
                                    # === Plot site point (cyan circle)
                                    gpd.GeoSeries([Point(site_x, site_y)], crs=site_gdf.crs).plot(ax=ax, markersize=15,
                                        color='red', marker='o', label='Site location')   
                                    # === Annotate site
                                    ax.annotate(f"Site {site_id}", xy=(site_x, site_y), xytext=(5, 5), textcoords='offset points',
                                        fontsize=7, color='red')
                                    ax.set_title(f"{short_name} - Site {site_id}", fontsize=10)
                                    ax.set_xlabel("Longitude", fontsize=9)
                                    ax.set_ylabel("Latitude", fontsize=9)
                                    ax.tick_params(axis='both', labelsize=9)
                                    plt.tight_layout()
                                    # === Save plot
                                    plot_path = os.path.join(site_plot, f"{short_name}.png")
                                    plt.savefig(plot_path, dpi=200)
                                    plt.close("all")
                                except Exception as e:
                                    print(f"Error plotting {short_name} for site {site_id}: {e}")
                        except Exception as e:
                            print(f"Error processing raster {file} for site {site_id}: {e}")
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
    gc.collect() 
    return all_sites_gpkg, dem_crs, sites_datasets, indices_output, annual_ndvi_dir, annual_c_dir