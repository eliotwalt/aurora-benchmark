import dask
import xarray as xr
import os, sys 
import logging 

from aurora_benchmark.utils import compute_climatology, resample_dataset

logger = logging.getLogger(__name__)
# set logger level to INFO
logger.setLevel(logging.INFO)

AURORA_VARIABLE_NAMES_MAP = {
    'surface': {
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        '2m_temperature': 't2m',
        'mean_sea_level_pressure': 'msl',
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
    compute_std_to_mean_climatology: bool = True,
    verbose: bool = True
) -> None:    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_climatology, exist_ok=True)
    
    # open era5 zarr
    verbose_print(verbose, "Opening ERA5 dataset...")
    era5_ds = xr.open_zarr(
        gs_url, 
        consolidated=True, 
        chunks={'time': 30, 'latitude': 100, 'longitude': 100}
    )
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
    
    static_ds = climatology_ds[static_variables].isel(time=0).drop_vars('time')
    
    # verbose print datasets sizes
    verbose_print(verbose, f"Surface Climatology Dataset: {surface_clim_ds.sizes}")
    verbose_print(verbose, f"Atmospheric Climatology Dataset: {atmospheric_clim_ds.sizes}")
    verbose_print(verbose, f"Surface Evaluation Dataset: {surface_eval_ds.sizes}")
    verbose_print(verbose, f"Atmospheric Evaluation Dataset: {atmospheric_eval_ds.sizes}")
    verbose_print(verbose, f"Static Dataset: {static_ds.sizes}")
    
    # rename variables to match aurora
    verbose_print(verbose, "Renaming variables to match Aurora...")
    surface_clim_ds = surface_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface'])
    atmospheric_clim_ds = atmospheric_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
    surface_eval_ds = surface_eval_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface'])
    atmospheric_eval_ds = atmospheric_eval_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
    static_ds = static_ds.rename(AURORA_VARIABLE_NAMES_MAP['static'])
    
    # merge the eval, climatology, and static datasets
    verbose_print(verbose, "Merging datasets...")
    climatology_ds = xr.merge([surface_clim_ds, atmospheric_clim_ds, static_ds])
    eval_ds = xr.merge([surface_eval_ds, atmospheric_eval_ds])
    
    # save the static variables to disk
    # path: {output_dir}/aurora_static_static_{climatology_years[0]}-{climatology_years[1]}-{spatial_resolution}.nc
    
    # compute climatologies at different frequencies
    for resampling_frequency in resampling_frequencies:
        verbose_print(verbose, f"Computing climatologies at {resampling_frequency} frequency...")
        resampled_climatology_ds = resample_dataset(climatology_ds, resampling_frequency)
        resampled_eval_ds = resample_dataset(eval_ds, resampling_frequency)
        
        mean_ds, std_ds = compute_climatology(
            resampled_climatology_ds, resampling_frequency,
            resample=False
        )
        
        # print sizes
        verbose_print(verbose, f"Mean climatology at {resampling_frequency} frequency: {mean_ds.sizes}")
        verbose_print(verbose, f"Std climatology at {resampling_frequency} frequency: {std_ds.sizes}")
        
        # save climatology to disk
        # pattern: {output_dir_climatology}/climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-{mean|std}-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc
        for variable_name in mean_ds.data_vars:
            mean_ds.to_netcdf(
                os.path.join(
                    output_dir_climatology, 
                    f'climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-mean-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc'
                )
            )
            std_ds.to_netcdf(
                os.path.join(
                    output_dir_climatology, 
                    f'climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-std-{base_frequency}-{resampling_frequency}-{spatial_resolution}.nc'
                )
            )
    
    

# print("Opening ERA5 dataset...")
# era5_ds = xr.open_zarr(
#     'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr',
#     chunks={'time': 30, 'latitude': 100, 'longitude': 100},
#     consolidated=True
# )

# print("Selecting climatology and test datasets...")
# era5_clim_ds = era5_ds.sel(time=slice(f'{CLIMATOLOGY_YEARS[0]}', f'{CLIMATOLOGY_YEARS[1]}'))
# era5_test_ds = era5_ds.sel(time=slice(f'{TEST_YEARS[0]}', f'{TEST_YEARS[1]}'))

# print("Creating sub datasets for surface, atmospheric, and static variables...")
# # Create sub datasets for surface, atmospheric, and static variables
# surface_clim_ds = era5_clim_ds[DEFAULT_AURORA_VARIABLES['surface_level']]
# atmospheric_clim_ds = era5_clim_ds[DEFAULT_AURORA_VARIABLES['atmospheric']].sel(level=[int(pl) for pl in DEFAULT_AURORA_PRESSURE_LEVELS])
# static_ds = era5_clim_ds[DEFAULT_AURORA_VARIABLES['static']].isel(time=0).drop_vars('time').sel(level=1000)

# surface_test_ds = era5_test_ds[DEFAULT_AURORA_VARIABLES['surface_level']]
# atmospheric_test_ds = era5_test_ds[DEFAULT_AURORA_VARIABLES['atmospheric']].sel(level=[int(pl) for pl in DEFAULT_AURORA_PRESSURE_LEVELS])

# # Print the datasets to verify
# print("Surface Climatology Dataset:", surface_clim_ds.sizes)
# print("Atmospheric Climatology Dataset:", atmospheric_clim_ds.sizes)
# print("Surface Test Dataset:", surface_test_ds.sizes)
# print("Atmospheric Test Dataset:", atmospheric_test_ds.sizes)
# print("Static Dataset:", static_ds.sizes)

# print("Renaming variables to match Aurora...")
# # rename the variables to match aurora
# surface_clim_ds = surface_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface_level'])
# atmospheric_clim_ds = atmospheric_clim_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
# surface_test_ds = surface_test_ds.rename(AURORA_VARIABLE_NAMES_MAP['surface_level'])
# atmospheric_test_ds = atmospheric_test_ds.rename(AURORA_VARIABLE_NAMES_MAP['atmospheric'])
# static_ds = static_ds.rename(AURORA_VARIABLE_NAMES_MAP['static'])

# print("Computing climatologies...")
# # Compute 6h, daily, weekly, and monthly climatology
# # Compute 6h climatology
# surface_clim_6h = surface_clim_ds.groupby([surface_clim_ds.time.dt.dayofyear, surface_clim_ds.time.dt.time]).mean('time')
# atmospheric_clim_6h = atmospheric_clim_ds.groupby([atmospheric_clim_ds.time.dt.dayofyear, atmospheric_clim_ds.time.dt.time]).mean('time')
# print("Surface Climatology 6h Dataset:", surface_clim_6h.sizes)
# print("Atmospheric Climatology 6h Dataset:", atmospheric_clim_6h.sizes)

# # Compute daily climatology
# surface_clim_daily = surface_clim_ds.groupby(surface_clim_ds.time.dt.dayofyear).mean('time')
# atmospheric_clim_daily = atmospheric_clim_ds.groupby(atmospheric_clim_ds.time.dt.dayofyear).mean('time')
# print("\nSurface Climatology Daily Dataset:", surface_clim_daily.sizes)
# print("Atmospheric Climatology Daily Dataset:", atmospheric_clim_daily.sizes)

# # Compute weekly climatology
# surface_clim_weekly = surface_clim_ds.groupby(surface_clim_ds.time.dt.weekofyear).mean('time')
# atmospheric_clim_weekly = atmospheric_clim_ds.groupby(atmospheric_clim_ds.time.dt.weekofyear).mean('time')
# print("\nSurface Climatology Weekly Dataset:", surface_clim_weekly.sizes)
# print("Atmospheric Climatology Weekly Dataset:", atmospheric_clim_weekly.sizes)

# # Compute monthly climatology
# surface_clim_monthly = surface_clim_ds.groupby(surface_clim_ds.time.dt.month).mean('time')
# atmospheric_clim_monthly = atmospheric_clim_ds.groupby(atmospheric_clim_ds.time.dt.month).mean('time')
# print("\nSurface Climatology Monthly Dataset:", surface_clim_monthly.sizes)
# print("Atmospheric Climatology Monthly Dataset:", atmospheric_clim_monthly.sizes)

# print("Saving climatologies to disk...")
# # Save the climatologies to disk
# surface_clim_daily.to_netcdf(f'{output_dir}/era5_surface_clim_daily_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc')
# atmospheric_clim_daily.to_netcdf(os.path.join(output_dir, f'era5_atmospheric_clim_daily_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))
# surface_clim_weekly.to_netcdf(os.path.join(output_dir, f'era5_surface_clim_weekly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))
# atmospheric_clim_weekly.to_netcdf(os.path.join(output_dir, f'era5_atmospheric_clim_weekly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))
# surface_clim_monthly.to_netcdf(os.path.join(output_dir, f'era5_surface_clim_monthly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))
# atmospheric_clim_monthly.to_netcdf(os.path.join(output_dir, f'era5_atmospheric_clim_monthly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))

# print("Saving test datasets to disk...")
# # Save the datasets to disk
# surface_test_ds.to_netcdf(os.path.join(output_dir, f'era5_surface_test_{TEST_YEARS[0]}-{TEST_YEARS[1]}.nc'))
# atmospheric_test_ds.to_netcdf(os.path.join(output_dir, f'era5_atmospheric_test_{TEST_YEARS[0]}-{TEST_YEARS[1]}.nc'))
# static_ds.to_netcdf(os.path.join(output_dir, f'era5_static_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc'))

# print("Data saved successfully!")
# # print all the paths
# print(f"Surface Climatology Daily: {output_dir}/era5_surface_clim_daily_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")
# print(f"Atmospheric Climatology Daily: {output_dir}/era5_atmospheric_clim_daily_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")
# print(f"Surface Climatology Weekly: {output_dir}/era5_surface_clim_weekly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")
# print(f"Atmospheric Climatology Weekly: {output_dir}/era5_atmospheric_clim_weekly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")
# print(f"Surface Climatology Monthly: {output_dir}/era5_surface_clim_monthly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")
# print(f"Atmospheric Climatology Monthly: {output_dir}/era5_atmospheric_clim_monthly_{CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}.nc")