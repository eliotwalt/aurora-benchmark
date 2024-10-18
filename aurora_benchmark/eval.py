import os
import xarray as xr
import dask
import numpy as np
import pandas as pd
import logging

from aurora_benchmark.utils import Statistics, verbose_print
from aurora_benchmark.plots import (
    rmse_curves, 
    signed_difference_maps, 
    prediction_maps, 
    find_closest_files
)

logger = logging.getLogger(__name__)


np.random.seed(0)

dask.config.set(scheduler='threads')

def open_era5_files(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
):
    surface_ds = xr.merge([
        xr.open_dataset(p, engine="netcdf4", chunks={"time": 50, "latitude": 721, "longitude": 1440})
        for p in era5_surface_paths
    ])

    atmospheric_ds = xr.merge([
        xr.open_dataset(p, engine="netcdf4", chunks={"time": 50, "latitude": 721, "longitude": 1440, "level": 1})
        for p in era5_atmospheric_paths
    ])

    static_ds = xr.merge([
        xr.open_dataset(p, engine="netcdf4", chunks={"latitude": 721, "longitude": 1440})
        for p in era5_static_paths
    ])

    return surface_ds, atmospheric_ds, static_ds

def accumulate_forecast_metrics(
    forecast_dir: str,
    surface_ds: xr.Dataset,
    atmospheric_ds: xr.Dataset,
    verbose: bool=True,
):
    
    # TODO: Add support for different regions and periods ...
    
    global_statistics = {
        "surface_vars": Statistics(),
        "atmospheric_vars": Statistics(),
    }
    med_statistics = {
        "surface_vars": Statistics(),
        "atmospheric_vars": Statistics(),
    }
    med_dry_statistics = {
        "surface_vars": Statistics(),
        "atmospheric_vars": Statistics(),
    }
    med_wet_statistics = {
        "surface_vars": Statistics(),
        "atmospheric_vars": Statistics(),
    }

    # define wet (oct-mar) and dry (apr-sep) seasons
    wet_months = [10, 11, 12, 1, 2, 3]
    dry_months = [4, 5, 6, 7, 8, 9]

    # define med
    med_region = {    
        "latitude": slice(47, 29), 
        "longitude": slice(-8, 38) 
    }
    
    # loop over files
    for i, file in enumerate(os.scandir(forecast_dir)):
        # parse file info
        file_info = file.name.replace(".nc", "").split("_")
        variable_name = file_info[1]
        file_info = file_info[2].split("-")
        init_time = pd.Timestamp(file_info[0])
        base_frequency = file_info[1]
        eval_aggregation = file_info[2]
        eval_start = file_info[3]
        forecast_horizon = file_info[4]    

        # feedback
        verbose_print(verbose, f"Processing {file.name}")

        # load forecast
        pred_trajectory = xr.open_dataset(file.path, engine="netcdf4")
        assert pd.Timedelta((pred_trajectory.lead_time[1]-pred_trajectory.lead_time[0]).values) == pd.Timedelta(eval_aggregation)

        # load ERA5 gt from surface_ds and atmospheric_ds
        true_ds = atmospheric_ds if variable_name in atmospheric_ds.data_vars else surface_ds
        true_trajectory = true_ds[variable_name]\
                .sel(time=slice(init_time+pd.Timedelta(eval_start), init_time+pd.Timedelta(forecast_horizon)))
                

        # resample gt to eval_aggregation
        true_trajectory = true_trajectory.resample(time=pd.Timedelta(eval_aggregation), origin=init_time).mean()
        assert pd.Timedelta((true_trajectory.time[1]-true_trajectory.time[0]).values) == pd.Timedelta(eval_aggregation)
        
        # get the index of times that are in wet and dry months respectively
        dry_times = (init_time + pred_trajectory.lead_time).dt.month.isin(dry_months)
        wet_times = (init_time + pred_trajectory.lead_time).dt.month.isin(wet_months)
        
        # rename true time to lead time
        true_trajectory = true_trajectory.rename({"time": "lead_time"})
        true_trajectory["lead_time"] = true_trajectory["lead_time"] - np.datetime64(init_time)

        # shape
        sizes = pred_trajectory.sizes
        nlt = len(np.unique(pred_trajectory.lead_time.values))
        if variable_name in atmospheric_ds.data_vars:
            stat_key = "atmospheric_vars"
        else:
            stat_key = "surface_vars"
            
        # compute signed error
        signed_error_ds = (pred_trajectory - true_trajectory)
        signed_error_ds = signed_error_ds.assign_coords({"longitude": signed_error_ds.longitude.values-180, "time": init_time})

        # accumulate statistics
        global_statistics[stat_key].update(signed_error_ds)
        med_statistics[stat_key].update(signed_error_ds.sel(med_region))    
        
        # accumulate seasonal statistics
        # !!! We only take the ones that are FULLY in the season (otherwise we cannot concatenate...)
        if dry_times.all().item():
            med_dry_statistics[stat_key].update(signed_error_ds.sel(med_region).isel(lead_time=dry_times))
        if wet_times.all().item():
            med_wet_statistics[stat_key].update(signed_error_ds.sel(med_region).isel(lead_time=wet_times))
            
    return global_statistics, med_statistics, med_dry_statistics, med_wet_statistics

def average_statistics_plots(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
    forecast_dir: str,
    eval_dir: str,
    variables: list[str]=["2t", "msl", "u_250", "v_250"],
    verbose: bool=True,
    **kwargs
):
    
    assert os.path.exists(forecast_dir), f"Forecast directory {forecast_dir} does not exist"
    os.makedirs(eval_dir, exist_ok=True)

    # load ERA5
    verbose_print(verbose, f"Loading ERA5 ...")   # open ERA5 fil
    surface_ds, atmospheric_ds, _ = open_era5_files(
        era5_surface_paths, era5_atmospheric_paths, era5_static_paths
    )
    
    # get dataset properties
    file = list(os.scandir(forecast_dir))[0]
    file_info = file.name.replace(".nc", "").split("_")[2].split("-")
    base_frequency = file_info[1]
    eval_aggregation = file_info[2]
    eval_start = file_info[3]
    forecast_horizon = file_info[4]
    
    # get statistics
    verbose_print(verbose, f"Accumulating forecast metrics globally and for the Mediterranean...")
    global_statistics, med_statistics, med_dry_statistics, med_wet_statistics = accumulate_forecast_metrics(
        forecast_dir, surface_ds, atmospheric_ds, verbose
    )
    
    rmse_curves(
        global_statistics,
        med_statistics,
        med_wet_statistics,
        med_dry_statistics,
        fig_title=f"RMSE for base_frequency={base_frequency}, eval_aggregation={eval_aggregation}, eval_start={eval_start}, forecast_horizon={forecast_horizon}",
        std_fig_title=f"RMSE (with std shading) for base_frequency={base_frequency}, eval_aggregation={eval_aggregation}, eval_start={eval_start}, forecast_horizon={forecast_horizon}",
        eval_dir=eval_dir,
        nrows=4,
        std_plot=True
    )
    
    signed_difference_maps(
        global_statistics,
        med_statistics,
        med_wet_statistics,
        med_dry_statistics,
        eval_dir=eval_dir,
        variables=variables,
    )
    
def predictions_plots(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
    forecast_dir: str,
    eval_dir: str,
    dates: list[pd.Timestamp|str]=[pd.Timestamp("2021-11-15"), pd.Timestamp("2021-06-10")],
    variables: list[str]=["2t", "msl", "u", "v"],
    level: int=250,    
    verbose: bool=True,
    **kwargs
):
    
    # TODO: Add support for multiple levels ...
    
    assert os.path.exists(forecast_dir), f"Forecast directory {forecast_dir} does not exist"
    os.makedirs(eval_dir, exist_ok=True)
    
    # load ERA5
    verbose_print(verbose, f"Loading ERA5 ...")   # open ERA5 
    surface_ds, atmospheric_ds, _ = open_era5_files(
        era5_surface_paths, era5_atmospheric_paths, era5_static_paths
    )
    
    # make the plots 
    for _date in dates:
        if not isinstance(_date, pd.Timestamp):
            _date = pd.Timestamp(_date)
        indexes, _ = find_closest_files(forecast_dir, _date, variables=variables)
        verbose_print(verbose, f"Processing files for date {_date} ...")
        for index in indexes:
            prediction_maps(
                forecast_dir,
                atmospheric_ds,
                surface_ds,
                file_index=index,
                eval_dir=eval_dir,
                lead_times=None,
                level=level
            )