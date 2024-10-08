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
    era5_paths: list[str],
    #era5_climatology_paths: str,
    interest_variables: list[str],
    output_dir: str,
    batch_size: int=4,
    era5_base_frequency: str="6H",
    init_frequency: str="1D",
    forecast_horizon: str="6W",
    forecast_aggregation: str="1W",
    no_write_period: str="1W", # 1 week
): 
    # load xr data
    era_ds = xr.concat(
        [xr.open_dataset(path, engine="netcdf4") 
         for path in era5_paths],
        dim="variable"
    )
    # merge into 1 dataset
    # create dataset XRAuroraDataset
    # create dataloader
    # loop over batches
    #   sample batch
    #   forecast rollout
    #   convert to xr.Dataset
    #   rename replaced variables (if any)
    #   aggregate & write
    pass