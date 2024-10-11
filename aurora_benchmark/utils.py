import xarray as xr
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        logger.info(message)
        
def dask_percentile(x, p, dim="time"):
    np_axis = list(x.dims).index(dim)
    # ensure no chunk along calculation dim 
    x = x.chunk({dim: -1})
    return xr.apply_ufunc(
        np.percentile,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[["percentile"]],
        kwargs={"q": p, "axis": np_axis},  # Ensure axis is specified correctly
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"percentile": len(p)},
        vectorize=True,
    )

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

def compute_climatology(
    ds: xr.Dataset, 
    frequency: str, 
    percentiles: list[float]=[66, 95, 5, 33],
    percentile_variables: list[str]|None=None,
    resample: bool=False,
) -> xr.Dataset:
    """
    Compute the climatology of a dataset at a given frequency.
    
    Args:
        ds: xarray.Dataset
            The dataset to compute the climatology on.
        frequency: str
            The frequency to compute the climatology at.
        percentiles: list[float]
            The percentiles to compute. Default is [66, 95, 5, 33].
        percentile_variables: list[str]
            The variables to compute the percentiles for. Default is None.
        resample: bool
            Whether to resample the dataset to the given frequency. Default is False.
    
    Returns
        clim_ds: xarray.Dataset
            The climatology dataset with required statistics.
    """
    
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
    
    # compute climatology per variable
    clim_ds = xr.Dataset()
    for var in ds.data_vars:
        # group according to frequency
        if frequency.upper().endswith('H'):
            group_ds = ds[var].groupby(['day_of_year', 'hour_of_day'])
        elif frequency.upper().endswith('D'):
            group_ds = ds[var].groupby('day_of_year')
        elif frequency.upper().endswith('W'):
            group_ds = ds[var].groupby('week_of_year')
        elif frequency.upper().endswith('M'):
            group_ds = ds[var].groupby('month_of_year')
        else:
            raise NotImplementedError(f"{frequency} is not supported.")
        clim_ds[f"mean_{var}"] = group_ds.mean(dim="time")
        clim_ds[f"std_{var}"] = group_ds.std(dim="time")
        if var in percentile_variables and len(percentiles) > 0:
            qds = group_ds.map(dask_percentile, p=percentiles, dim="time")
            for i, q in enumerate(percentiles):
                clim_ds[f"p{q}_{var}"] = qds.sel(percentile=i)
        
    return clim_ds

def xr_to_netcdf(
    dataset: xr.Dataset | xr.DataArray, 
    path: str, precision: str, 
    compression_level: int=0, 
    sort_time: bool=True,
    exist_ok: bool=False
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
    if os.path.exists(path):
        if exist_ok:
            os.remove(path)
        else:
            raise FileExistsError(f"File already exists: {path}")
    
    # sort
    if sort_time:
        dataset = dataset.sortby("time")
    
    # encoding 
    enc = {"dtype": precision, "zlib": compression_level > 0}
    if compression_level > 0: enc["complevel"] = compression_level
    if isinstance(dataset, xr.Dataset):
        encoding = {k: enc for k in dataset.data_vars}
    elif isinstance(dataset, xr.DataArray):
        encoding = {dataset.name: enc}

    # write
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
    
    # rename
    renamed_ds = ds.rename(variable_names_map)
        
    return renamed_ds