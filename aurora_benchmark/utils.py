import xarray as xr
import pandas as pd
import numpy as np
import os
import gcsfs

fs = gcsfs.GCSFileSystem(anon=True)

def resample_dataset(ds: xr.Dataset, frequency: str) -> xr.Dataset:
    """
    Resample a dataset to a given frequency.
    
    Args:
        ds: xarray.Dataset
            The dataset to resample.
        frequency: str
            The frequency to resample the dataset to.
    
    Returns:
        resampled_ds: xarray.Dataset
            The resampled dataset.
    """
    # resample the dataset to the given frequency
    if pd.infer_freq(ds.time) == frequency:
        resampled_ds = ds
    else:
        resampled_ds = ds.resample(time=frequency).mean()
    
    return resampled_ds

def compute_climatology(ds: xr.Dataset, frequency: str, resample: bool=False) -> xr.Dataset:
    """
    Compute the climatology of a dataset at a given frequency.
    
    Args:
        ds: xarray.Dataset
            The dataset to compute the climatology on.
        frequency: str
            The frequency to compute the climatology at.
    
    Returns
        mean_ds: xarray.Dataset
            The mean climatology dataset.
        std_ds: xarray.Dataset
            The standard deviation climatology dataset.
    """
    # split the time dimension into month/week/day of year, hour of day
    ds = ds.assign_coords(
        week_of_year=ds.time.dt.isocalendar().week,
        day_of_year=ds.time.dt.dayofyear,
        hour_of_day=ds.time.dt.hour,
        month_of_year=ds.time.dt.month
    )
    
    # TODO: ensure that all days have 4x6h, weeks have 7 days, etc.
    
    # resample the dataset to the given frequency
    if resample:
        resampled_ds = resample_dataset(ds, frequency)
    else:
        resampled_ds = ds
    
    # groupby operation differs based on the type of frequency
    if frequency.endswith('H'):
        group_ds = resampled_ds.groupby(['day_of_year', 'hour_of_day'])
    elif frequency.endswith('D'):
        group_ds = resampled_ds.groupby('day_of_year')
    elif frequency.endswith('W'):
        group_ds = resampled_ds.groupby('week_of_year')
    elif frequency.endswith('M'):
        group_ds = resampled_ds.groupby('month_of_year')
    
    # compute stats and get into single dataset
    clim_ds = group_ds.mean('time')
    std_ds = group_ds.std('time')
    for var in clim_ds.data_vars:
        clim_ds["std_" + var] = std_ds[var]
        clim_ds["mean_" + var] = clim_ds[var]
        clim_ds.drop_vars(var)
    
    return clim_ds

def xr_to_netcdf(
    dataset: xr.Dataset | xr.DataArray, 
    path: str, precision: str, 
    compression_level: int=0, 
    sort_time: bool=True
) -> None:
    encode_cfg = {"dtype": precision, "zlib": True}
    if compression_level > 0:
        encode_cfg["complevel"] = compression_level
    
    # Sort by time if required
    if sort_time and "time" in dataset.dims:
        dataset = dataset.sortby("time")
    
    # Update encoding
    encoding = {}
    if isinstance(dataset, xr.Dataset):
        variables = list(dataset.data_vars) + list(dataset.coords)
    elif isinstance(dataset, xr.DataArray):
        variables = [dataset.name] + list(dataset.coords)
    
    for var in variables:
        # Do not touch temporal dimensions
        if np.issubdtype(dataset[var].dtype, np.number):
            if var in encoding:
                encoding[var].update(encode_cfg)
            else:
                encoding[var] = encode_cfg
    
    # Remove existing file if it exists
    if os.path.exists(path):
        os.remove(path)
    
    # Save to NetCDF
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)