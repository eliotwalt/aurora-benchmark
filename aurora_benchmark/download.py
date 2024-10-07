import dask
import xarray as xr
import os, sys 
import logging 
from scipy.stats import norm

from aurora_benchmark.utils import compute_climatology, resample_dataset, xr_to_netcdf

logger = logging.getLogger(__name__)

AURORA_VARIABLE_NAMES_MAP = {
    'surface': {
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        '2m_temperature': 't2m',
        'mean_sea_level_pressure': 'msl',
        'sea_surface_temperature': 'sst',
        'total_precipitation_6hr': 'tp',
    },
    'atmospheric': {
        'temperature': 't',
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        'specific_humidity': 'q',
        'geopotential': 'z',
    },
    'static': {
        'geopotential': 'z',
        'land_sea_mask': 'lsm',
        'soil_type': 'stype',
    }
}

def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        logger.info(message)

def download_era5_wb2(
    gs_url: str,
    output_dir: str,
    output_dir_climatology: str,
    static_variables: list[str],
    surface_variables: list[str],
    atmospheric_variables: list[str],
    pressure_levels: list[int],
    resampling_frequencies: list[str],
    climatology_years: list[int] = [1979, 2020],
    eval_years: list[int] = [2021, 2022],
    compute_quantile: bool = True,
    quantile_variables: list[str] = ['t2m', 'tp'],
    verbose: bool = True
) -> None:    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_climatology, exist_ok=True)
    
    # open era5 zarr
    verbose_print(verbose, "Opening ERA5 dataset...")
    era5_ds = xr.open_zarr(gs_url, consolidated=True)
    
    # extract base_frequency and spatial_resolution from 
    # gs_url = gs://weatherbench2/datasets/era5/{year1}-{year2}-{base_frequency}-{spatial_resolution}_{other_things}.zarr
    base_frequency = gs_url.split('/')[-1].split('-')[2]
    spatial_resolution = gs_url.split('/')[-1].split('-')[3].split('_')[0] 
    
    # split years
    climatology_ds = era5_ds.sel(time=slice(f'{climatology_years[0]}', f'{climatology_years[1]}'))
    eval_ds = era5_ds.sel(time=slice(f'{eval_years[0]}', f'{eval_years[1]}'))
    
    # create sub datasets for each set of variables
    verbose_print(verbose, "Creating sub datasets for surface, atmospheric, and static variables...")
    surface_clim_ds = climatology_ds[surface_variables]
    atmospheric_clim_ds = climatology_ds[atmospheric_variables].sel(level=pressure_levels)
    
    surface_eval_ds = eval_ds[surface_variables]
    atmospheric_eval_ds = eval_ds[atmospheric_variables].sel(level=pressure_levels)
    
    static_ds = era5_ds[static_variables].isel(time=0, drop=False)
    
    # rename variables to match aurora
    verbose_print(verbose, "Renaming variables to match Aurora...")
    surface_clim_ds = surface_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface'])
    atmospheric_clim_ds = atmospheric_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
    surface_eval_ds = surface_eval_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface'])
    atmospheric_eval_ds = atmospheric_eval_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
    static_ds = static_ds.rename(AURORA_VARIABLE_NAMES_MAP['static'])
    if not isinstance(static_ds, xr.Dataset):
        static_ds = static_ds.to_dataset()

    # merge the eval, climatology, and static datasets
    verbose_print(verbose, "Merging datasets...")
    climatology_ds = xr.concat([surface_clim_ds, atmospheric_clim_ds], dim='variable')
    eval_ds = xr.concat([surface_eval_ds, atmospheric_eval_ds], dim='variable')
    
    # print datasets types and sizes
    verbose_print(verbose, f"Climatology dataset: {type(climatology_ds)} {climatology_ds.sizes}")
    verbose_print(verbose, f"Evaluation dataset: {type(eval_ds)} {eval_ds.sizes}")
    verbose_print(verbose, f"Static dataset: {type(eval_ds)} {static_ds.sizes}")
    
    # save the static variables to disk
    # path: {output_dir}/{variable_name}_static-{spatial_resolution}.nc
    for variable_name in static_ds.data_vars:
        xr_to_netcdf(    
            static_ds[variable_name],
            os.path.join(
                output_dir, 
                f'{variable_name}_static-{spatial_resolution}.nc'
            ),
            precision="float16",
            compression_level=0,
            sort_time=False
        )
    
    # compute climatologies at different frequencies
    for resampling_frequency in resampling_frequencies:
        # reasample eval and climatology datasets
        verbose_print(verbose, f"Resampling to {resampling_frequency} frequency...")
        resampled_climatology_ds = resample_dataset(climatology_ds, resampling_frequency)
        resampled_eval_ds = resample_dataset(eval_ds, resampling_frequency)
        
        # climato
        verbose_print(verbose, f"Computing climatologies at {resampling_frequency} frequency...")
        resampled_climatology_ds = compute_climatology(
            resampled_climatology_ds, resampling_frequency,
            resample=False
        )
        
        # print sizes
        verbose_print(verbose, f"Climatology at {resampling_frequency} frequency: {resampled_climatology_ds.sizes}")
        
        # normal quantiles
        # TODO: use bootstraped quantiles from the climatologies
        if compute_quantile:
            verbose_print(verbose, "Computing quantiles...")
            for variable_name in quantile_variables:
                # search equivalent aurora variable name
                ok = False
                for sub in AURORA_VARIABLE_NAMES_MAP.values():
                    for aurora_variable_name, era5_variable_name in sub.items():
                        if aurora_variable_name == variable_name:
                            variable_name == era5_variable_name
                            ok = True
                            break
                if not ok: raise ValueError(f"Variable {variable_name} not found in AURORA_VARIABLE_NAMES_MAP")
                # add percentiles to resampled eval dataset
                standardised = (resampled_eval_ds[variable_name] - resampled_climatology_ds["mean_"+variable_name]) / resampled_climatology_ds["std_"+variable_name]
                resampled_eval_ds[f"quantile_{variable_name}"] = xr.apply_ufunc(
                    norm.cdf,
                    standardised,
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float]
                )
        
        # save resampled eval dataset to disk
        # pattern: {output_dir}/{variable_name}_{eval_years[0]}-{eval_years[1]}-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc
        verbose_print(verbose, f"Saving resampled evaluation dataset at {resampling_frequency} frequency to disk...")
        for variable_name in resampled_eval_ds.data_vars:
            xr_to_netcdf(
                resampled_eval_ds[variable_name],
                os.path.join(
                    output_dir, 
                    f'{variable_name}_{eval_years[0]}-{eval_years[1]}-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc'
                ),
                precision="float16",
                compression_level=0,
                sort_time=True
            )
        
        # save climatology to disk
        # pattern: {output_dir_climatology}/climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-{mean|std}-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc
        verbose_print(verbose, f"Saving climatologies at {resampling_frequency} frequency to disk...")
        for variable_name in resampled_climatology_ds.data_vars:
            xr_to_netcdf(    
                resampled_climatology_ds[variable_name],
                os.path.join(
                    output_dir_climatology, 
                    f'climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc'
                ),
                precision="float16",
                compression_level=0,
                sort_time=True
            )