import xarray as xr
import torch
import dask
from torch.utils.data import DataLoader
from aurora import Batch, Metadata
import logging

from aurora_benchmark.utils import verbose_print

dask.config.set(scheduler='threads')

from aurora_benchmark.data import (
    XRAuroraDataset,
    aurora_batch_to_xr,
    aurora_batch_collate_fn
)

def aurora_forecast(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
    #era5_climatology_paths: str,
    interest_variables: list[str],
    output_dir: str,
    batch_size: int=4,
    era5_base_frequency: str="6H",
    init_frequency: str="1D",
    forecast_horizon: str="6W",
    forecast_aggregation: str="1W",
    no_write_period: str="1W", # 1 week
    device: str|torch.device="cuda:0"
): 
    chunks = {"time": 100, "latitude": 721, "longitude": 1440}
    # load xr data
    era5_surface_ds = xr.concat(
        [xr.open_dataset(
            path, engine="netcdf4",
            consolidated=True, 
            chunks=chunks) 
         for path in era5_surface_paths],
        dim="variable"
    )
    era5_atmospheric_ds = xr.concat(
        [xr.open_dataset(
            path, engine="netcdf4",
            consolidated=True, 
            chunks=chunks) 
         for path in era5_atmospheric_paths],
        dim="variable"
    )
    era5_static_ds = xr.concat(
        [xr.open_dataset(
            path, engine="netcdf4",
            consolidated=True, 
            chunks=chunks) 
         for path in era5_static_paths],
        dim="variable"
    )
    # merge into 1 dataset
    # create dataset XRAuroraDataset
    # create dataloader
    # load model, put on device
    # loop over batches
    #   sample batch
    #   put batch on device
    #   forecast rollout
    #   put preds on cpu
    #   convert to xr.Dataset
    #   rename replaced variables (if any)
    #   aggregate & write
    pass