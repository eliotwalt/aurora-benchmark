import dask
import xarray as xr
import os
import logging 
import requests
import pickle

from aurora_benchmark.utils import (
    compute_climatology, 
    xr_to_netcdf,
    rename_xr_variables,
    verbose_print
)

dask.config.set(scheduler='threads')

logger = logging.getLogger(__name__)

AURORA_VARIABLE_NAMES_MAP = {
    # surface
    '10m_u_component_of_wind': '10u',
    '10m_v_component_of_wind': '10v',
    '2m_temperature': '2t',
    'mean_sea_level_pressure': 'msl',
    'sea_surface_temperature': 'sst',
    'total_precipitation_6hr': 'tp',
    # atmospheric
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'specific_humidity': 'q',
    'geopotential': 'z',
    # static
    'geopotential': 'z',
    'land_sea_mask': 'lsm',
    'soil_type': 'slt',
}

# Create a reverse dictionary for inverse lookup
INVERSE_AURORA_VARIABLE_NAMES_MAP = {v: k for k, v in AURORA_VARIABLE_NAMES_MAP.items()}

def download_era5_wb2(
    gs_url: str,
    output_dir: str,
    output_dir_climatology: str,
    static_variables: list[str],
    surface_variables: list[str],
    atmospheric_variables: list[str],
    pressure_levels: list[int],
    climatology_frequencies: list[str],
    climatology_years: list[int] = [1979, 2020],
    eval_years: list[int] = [2021, 2022],
    percentiles: list[float]|None=None,
    percentile_variables: list[str]|None=None,
    verbose: bool = True
) -> None:    
    
    verbose_print(verbose, "Downloading ERA5 data from WeatherBench2...")
    verbose_print(verbose, f" * static_variables: {static_variables}")
    verbose_print(verbose, f" * surface_variables: {surface_variables}")
    verbose_print(verbose, f" * atmospheric_variables: {atmospheric_variables}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_climatology, exist_ok=True)
    
    # extract base_frequency and spatial_resolution from 
    # gs_url = gs://weatherbench2/datasets/era5/{year1}-{year2}-{base_frequency}-{spatial_resolution}_{other_things}.zarr
    base_frequency = gs_url.split('/')[-1].split('-')[2]
    spatial_resolution = gs_url.split('/')[-1].split('-')[3].split('_')[0].split(".")[0] 
    
    # open era5 zarr
    verbose_print(verbose, "Opening ERA5 dataset...")
    if len(atmospheric_variables) > 0:
        chunks = {"time": 28*25, "latitude": 721, "longitude": 1440, "level": 7}
    else:
        chunks = {"time": 28*75, "latitude": 721, "longitude": 1440}
    era5_ds = xr.open_zarr(
        gs_url, consolidated=True,
        chunks=chunks
    )
    
    # split years
    verbose_print(verbose, "Splitting years...")
    climatology_ds = era5_ds.sel(time=slice(f'{climatology_years[0]}', f'{climatology_years[1]}'))
    eval_ds = era5_ds.sel(time=slice(f'{eval_years[0]}', f'{eval_years[1]}'))
    
    # create sub datasets for each set of variables
    verbose_print(verbose, "Creating sub datasets for surface, atmospheric, and static variables...")
    if len(static_variables) > 0:
        static_ds = era5_ds[static_variables]
        if static_ds.sizes.get("time", 0) > 0:
            static_ds = static_ds.isel(time=0, drop=True)
        if isinstance(static_ds, xr.DataArray):
            static_ds = static_ds.to_dataset()
    else:
        static_ds = xr.Dataset()
    if len(surface_variables) > 0:
        surface_clim_ds = climatology_ds[surface_variables]
        surface_eval_ds = eval_ds[surface_variables]
    else:
        surface_clim_ds = xr.Dataset()
        surface_eval_ds = xr.Dataset()
    if len(atmospheric_variables) > 0: 
        atmospheric_clim_ds = climatology_ds[atmospheric_variables].sel(level=pressure_levels)
        atmospheric_eval_ds = eval_ds[atmospheric_variables].sel(level=pressure_levels)
    else:
        atmospheric_clim_ds = xr.Dataset()
        atmospheric_eval_ds = xr.Dataset()

    # merge the eval, climatology, and static datasets
    verbose_print(verbose, "Merging datasets...")
    climatology_ds = xr.merge([surface_clim_ds, atmospheric_clim_ds])
    eval_ds = xr.merge([surface_eval_ds, atmospheric_eval_ds])
    
    # rename variables to match aurora
    verbose_print(verbose, "Renaming variables to match Aurora...")
    climatology_ds = rename_xr_variables(climatology_ds, AURORA_VARIABLE_NAMES_MAP)
    eval_ds = rename_xr_variables(eval_ds, AURORA_VARIABLE_NAMES_MAP)
    static_ds = rename_xr_variables(static_ds, AURORA_VARIABLE_NAMES_MAP)
    if percentile_variables:
        verbose_print(verbose, "Renaming percentile variables to match Aurora...")
        percentile_variables = [AURORA_VARIABLE_NAMES_MAP[var] for var in percentile_variables]
    
    # print datasets types and sizes
    verbose_print(verbose, f"Climatology dataset: {type(climatology_ds)} {climatology_ds.sizes}")
    verbose_print(verbose, f"Evaluation dataset: {type(eval_ds)} {eval_ds.sizes}")
    verbose_print(verbose, f"Static dataset: {type(eval_ds)} {static_ds.sizes}")
    
    if len(static_ds.data_vars) > 0:
        # save the static variables to disk
        # pattern: {output_dir}/{variable_name}_static-{spatial_resolution}.nc
        for variable_name in static_ds.data_vars:
            verbose_print(verbose, f"Saving {variable_name} static dataset to disk...")
            xr_to_netcdf(    
                static_ds[variable_name],
                os.path.join(
                    output_dir, 
                    f'{variable_name}_static-{spatial_resolution}.nc'
                ),
                precision="float32",
                compression_level=1,
                sort_time=False,
                exist_ok=True
            )
        
    if len(eval_ds.data_vars) > 0:
        # save the eval dataset to disk
        # pattern: {variable_name}_{eval_years[0]}-{eval_years[1]}-{base_frequency}-{spatial_resolution}.nc
        for variable_name in eval_ds.data_vars:
            verbose_print(verbose, f"Saving {variable_name} from evaluation dataset to disk...")
            xr_to_netcdf(    
                eval_ds[variable_name],
                os.path.join(
                    output_dir, 
                    f'{variable_name}_{eval_years[0]}-{eval_years[1]}-{base_frequency}-{spatial_resolution}.nc'
                ),
                precision="float32",
                compression_level=1,
                sort_time=False,
                exist_ok=True
            )
    
    # compute climatologies at different frequencies
    if len(climatology_ds.data_vars) > 0:
        for climatology_frequency in climatology_frequencies:
            # reasample eval and climatology datasets
            verbose_print(verbose, f"Computing {climatology_frequency.lower()} climatology...")
            resampled_climatology_ds = compute_climatology(
                climatology_ds, 
                climatology_frequency,
                percentiles=percentiles,
                percentile_variables=percentile_variables,
                resample=True,
            )
            
            # print sizes
            verbose_print(verbose, f"{climatology_frequency.lower()} climatology sizes: {resampled_climatology_ds.sizes}")
            verbose_print(verbose, f"{climatology_frequency.lower()} climatology variables: {resampled_climatology_ds.data_vars}")
            
            # save climatology to disk
            # pattern: {output_dir_climatology}/climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-{mean|std}-{base_frequency}-{climatology_frequency}-{spatial_resolution}.nc
            for variable_name in resampled_climatology_ds.data_vars:
                verbose_print(verbose, f"Saving {variable_name} from {climatology_frequency.lower()} climatology to disk...")
                xr_to_netcdf(    
                    resampled_climatology_ds[variable_name],
                    os.path.join(
                        output_dir_climatology, 
                        f'climatology_{variable_name}_{climatology_years[0]}-{climatology_years[1]}-{base_frequency}-{climatology_frequency.lower()}-{spatial_resolution}.nc'
                    ),
                    precision="float32",
                    compression_level=1,
                    sort_time=False,
                    exist_ok=True
                )
    else:
        verbose_print(verbose, "No climatology data to compute.")
                
    verbose_print(verbose, "Download complete.")
    
def download_static_hf(
    output_dir: str,
    verbose: bool = True
):
    verbose_print(verbose, "Downloading static data from HuggingFace...")
    url = "https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-static.pickle"
    response = requests.get(url)
    with open("/projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/static.pickle", "wb") as f:
        f.write(response.content)
        
    with open("/projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/static.pickle", "rb") as f:
        static = pickle.load(f)
        
    static_ds = xr.Dataset({var: (("latitude", "longitude"), data) for var, data in static.items()})
    
    for var in static_ds.data_vars:
        verbose_print(verbose, f"Saving {var} static dataset to disk...")
        xr_to_netcdf(    
            static_ds[var],
            os.path.join(
                output_dir, 
                f'{var}_static-1440x721.nc'
            ),
            precision="float32",
            compression_level=1,
            sort_time=False,
            exist_ok=True
        )