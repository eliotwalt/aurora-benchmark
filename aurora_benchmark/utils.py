import xarray as xr
import pandas as pd
import numpy as np
import os

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
    
    if not frequency.endswith("H") and int(frequency[:-1]) > 1:
        raise NotImplementedError(f"{frequency} is not supported.")
    
    # resample the dataset to the given frequency
    if resample:
        ds = resample_dataset(ds, frequency)
    
    # split the time dimension into month/week/day of year, hour of day
    ds = ds.assign_coords(
        week_of_year=ds.time.dt.isocalendar().week,
        day_of_year=ds.time.dt.dayofyear,
        hour_of_day=ds.time.dt.hour,
        month_of_year=ds.time.dt.month
    )
    
    # groupby operation differs based on the type of frequency
    if frequency.endswith('H'):
        group_ds = ds.groupby(['day_of_year', 'hour_of_day'])
    elif frequency.endswith('D'):
        group_ds = ds.groupby('day_of_year')
    elif frequency.endswith('W'):
        group_ds = ds.groupby('week_of_year')
    elif frequency.endswith('M'):
        group_ds = ds.groupby('month_of_year')
    
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
    """
    Write an xarray dataset to a netCDF file.
    
    Args:
        dataset: xarray.Dataset | xarray.DataArray
            The dataset to write to disk.
        path: str
            The path to write the dataset to.
        precision: str
            The precision to write the dataset with.
        compression_level: int
            The compression level to write the dataset with. Defaults to 0.
        sort_time: bool
            Whether to sort the dataset by time before writing. Defaults to True.
    """
    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset()
    
    encode_cfg = {"dtype": precision, "zlib": True}
    if compression_level > 0: encode_cfg["complevel"] = compression_level
    # sort
    if sort_time:
        dataset = dataset.sortby("time")
    # update encoding
    encoding = {}
    for dims in [dataset.data_vars, dataset.coords]:
        for var in dims:
            # do not touch temporal dimensions
            if np.issubdtype(dataset[var].dtype, np.number):
                if var in encoding: encoding[var].update(encode_cfg)
                else: encoding[var] = encode_cfg
    if os.path.exists(path): os.remove(path)
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    
def rename_xr_variables(ds: xr.Dataset, variable_names_map: dict[str, str]) -> xr.Dataset:
    """
    Rename the variables of a dataset according to a mapping.
    
    Args:
        ds: xarray.Dataset
            The dataset to rename the variables of.
        variable_names_map: dict[str, str]
            A mapping of the old variable names to the new variable names.
    
    Returns:
        renamed_ds: xarray.Dataset
            The dataset with the variables renamed.
    """
    
    # intersect variables
    variable_names_map = {k: v for k, v in variable_names_map.items() if k in ds.data_vars}
    print(variable_names_map)
    
    # rename
    renamed_ds = ds.rename(variable_names_map)
        
    return renamed_ds