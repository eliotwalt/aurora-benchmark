import os 
import xarray as xr
import torch
import dask
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import logging

from aurora import Aurora, AuroraSmall, rollout

from aurora_benchmark.utils import verbose_print, xr_to_netcdf

dask.config.set(scheduler='threads')
logger = logging.getLogger(__name__)

from aurora_benchmark.data import (
    XRAuroraDataset, 
    aurora_batch_collate_fn, 
    aurora_batch_to_xr, 
    unpack_aurora_batch
)

AURORA_MODELS_HF = [
    "aurora-0.25-small-pretrained.ckpt",
    "aurora-0.25-finetuned.ckpt", # on IFS HRES T0
    "aurora-0.25-pretrained.ckpt"
]

AURORA_VARAIBLE_RENAMES = {
    "surface": {
        "u10": "10u",
        "v10": "10v",
        "t2m": "2t",
    },
    "atmospheric": {},
    "static": {
        "stype": "slt"
    },
}
INVERTED_AURORA_VARAIBLE_RENAMES = {
    "surface": {v: k for k, v in AURORA_VARAIBLE_RENAMES["surface"].items()},
    "atmospheric": {v: k for k, v in AURORA_VARAIBLE_RENAMES["atmospheric"].items()},
    "static": {v: k for k, v in AURORA_VARAIBLE_RENAMES["static"].items()},
}

def get_path_metadata(path: str):
    parts = os.path.basename(path).split("-")
    start_year, end_year = parts[0].split("_")[1], parts[1]
    resolution = parts[-1].split(".")[0]
    return start_year, end_year, resolution
    

def aurora_forecast(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
    interest_variables: list[str],
    interest_levels: list[str],
    output_dir: str,
    batch_size: int=4,
    replacement_variables: dict[str, str]=dict(),
    era5_base_frequency: str="6h",
    init_frequency: str="1d",
    forecast_horizon: str="6W",
    eval_aggregation: str|None=None,
    eval_start: str="1W", # 1 week
    aurora_model: str="aurora-0.25-pretrained.ckpt",
    device: str|torch.device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool=True,
): 
    
    os.makedirs(output_dir, exist_ok=True)
    
    # compute additional arguments
    warmup_steps = pd.Timedelta(eval_start) / pd.Timedelta(era5_base_frequency) if eval_start is not None else 0.0
    forecast_steps = pd.Timedelta(forecast_horizon) / pd.Timedelta(era5_base_frequency)
    start_year, end_year, resolution = get_path_metadata(era5_surface_paths[0])
    
    # validate arguments
    assert aurora_model in AURORA_MODELS_HF, f"Model {aurora_model} not found in {AURORA_MODELS_HF}"
    assert forecast_steps.is_integer(), f"forecast_horizon not a multiple of frequency"
    assert warmup_steps.is_integer(), f"eval_start not a multiple of frequency"
    forecast_steps = int(forecast_steps)
    warmup_steps = int(warmup_steps)    
    
    # not implementated protection
    if eval_aggregation is not None:
        raise NotImplementedError("Aggregation not yet implemented.")
    if len(replacement_variables) > 0:
        raise NotImplementedError("Replacement variables not yet implemented.")
    
    # dask
    chunks = {"time": 4*batch_size, "latitude": 721, "longitude": 1440}
    
    # load xr data
    verbose_print(verbose, "Reading data ...")
    surface_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4", 
                         chunks={"time": 4*batch_size, "latitude": 721, "longitude": 1440}) 
         for path in era5_surface_paths],
    ).rename(AURORA_VARAIBLE_RENAMES["surface"])
    atmospheric_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4",
                         chunks={"time": 4*batch_size, "latitude": 721, "longitude": 1440, "level": 7}) 
         for path in era5_atmospheric_paths],
    ).rename(AURORA_VARAIBLE_RENAMES["atmospheric"])
    static_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4")
         for path in era5_static_paths],
    ).rename(AURORA_VARAIBLE_RENAMES["static"])
    
    # create dataset XRAuroraDataset
    dataset = XRAuroraDataset(
        surface_ds=surface_ds,
        atmospheric_ds=atmospheric_ds,
        static_ds=static_ds,
        init_frequency=init_frequency,
        forecast_horizon=forecast_horizon,
        num_time_samples=2, # Aurora has fixed history length of 2...
    )
    verbose_print(verbose, f"Loaded dataset of length {len(dataset)}")
    
    # create dataloader
    eval_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=aurora_batch_collate_fn,
        num_workers=2
    )
    
    # model
    if "small" in aurora_model:
        verbose_print(verbose, "Loading AuroraSmall model ...")
        model = AuroraSmall()
    else:
        verbose_print(verbose, "Loading Aurora model ...")
        model = Aurora(use_lora=False)
    model.load_checkpoint("microsoft/aurora", aurora_model)
    model = model.to(device)
    
    # evaluation loop
    verbose_print(verbose, "Starting evaluation ...")
    xr_preds = {"surface_ds": [], "atmospheric_ds": []}
    with torch.inference_mode() and torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch = batch.to(device)
            # rollout until for forecast_steps
            verbose_print(verbose, f"Rollout prediction on batch {i} ...")
            trajectories = [[] for _ in range(batch_size)]
            for s, batch_pred in enumerate(rollout(model, batch, steps=forecast_steps)):
                if s < warmup_steps:
                    verbose_print(verbose, f" * Rollout step {s+1}: skipping warmup period")
                    continue            
                # separate batched batches
                sub_batch_preds = unpack_aurora_batch(batch_pred.to("cpu"))
                verbose_print(verbose, f" * Rollout step {s+1}: unpacked {len(sub_batch_preds)} sub-batches")
                assert len(sub_batch_preds) == batch_size
                # accumulate
                for b, sub_batch_pred in enumerate(sub_batch_preds):
                    trajectories[b].append(sub_batch_pred)
            verbose_print(verbose, f"Processing trajectories ...")
            # convert to xr 
            for init_time, trajectory in zip(batch.metadata.time, trajectories):
                verbose_print(verbose, f" * init_time={init_time}: combining {len(trajectory)} steps")
                assert len(trajectory) == forecast_steps-warmup_steps
                # collate trajectory batches
                trajectory = aurora_batch_collate_fn(trajectory)
                # convert to xr.Dataset
                trajectory = aurora_batch_to_xr(trajectory, frequency=era5_base_frequency)
                # add lead time
                for var_type, vars_ds in trajectory.items():
                    # ensure processing is necessary
                    if var_type == "static_ds":
                        verbose_print(verbose, f" * Skipping static variables")
                        continue # we do not care about static variables for the forecast
                    if not any([var in vars_ds.data_vars for var in interest_variables]):
                        verbose_print(verbose, f" * Skipping {var_type} variables as no interest variables are present")
                        continue # don't bother processing variables we are not interested in
                    if var_type == "atmospheric_ds" and (interest_levels is None or len(interest_levels)==0):
                        verbose_print(verbose, f" * Skipping atmospheric variables as no interest levels have been requested")
                        continue # we do not care about atmospheric variables if no levels are of interest
                    
                    # select interest variables and levels
                    vars_interest_variables = [var for var in vars_ds.data_vars if var in interest_variables]
                    if var_type == "atmospheric_ds":
                        vars_ds = vars_ds[vars_interest_variables].sel(level=interest_levels)
                    else:
                        vars_ds = vars_ds[vars_interest_variables]
                    
                    # add lead time
                    vars_ds = vars_ds.assign_coords({"lead_time": vars_ds.time.values - np.datetime64(init_time)})
                    vars_ds = vars_ds.set_index({"lead_time": "lead_time"})
                    
                    # TODO: aggregate desired timesteps to agg freq  
                    # vars_ds = vars_ds.resample(time=eval_aggregation).mean()  # not enough as it messes with "lead time"                    
                    
                    # append to predictions
                    xr_preds[var_type].append(vars_ds)
            
    # merge predictions and save
    for var_type, var_ds_list in xr_preds.items():
        ds = xr.merge(var_ds_list).rename(INVERTED_AURORA_VARAIBLE_RENAMES[var_type])
        verbose_print(verbose, f"Writing {var_type} predictions ...")
        for var in ds.data_vars:
            for lead_time in np.unique(ds.lead_time.values).astype("timedelta64[h]"):
                
                # TODO: lead time based on eval_aggregation??
                lead_time = f"{lead_time}h"
                
                xr_to_netcdf(
                    ds[var].sel(lead_time=lead_time),
                    os.path.join(
                        output_dir,
                        f"{var}-{start_year}-{end_year}-{era5_base_frequency}-{init_frequency}-{forecast_horizon}-{lead_time}-{resolution}.nc"
                    ),
                    precision="float32",
                    compression_level=0,
                    sort_time=False,
                    exist_ok=True
                )
    
if __name__ == "__main__":
    aurora_forecast(
        era5_surface_paths=[
            "data/era5/era5_surface_2010-2011_0.25.nc",
            "data/era5/era5_surface_2012-2013_0.25.nc",
        ],
        era5_atmospheric_paths=[
            "data/era5/era5_atmospheric_2010-2011_0.25.nc",
            "data/era5/era5_atmospheric_2012-2013_0.25.nc",
        ],
        era5_static_paths=[
            "data/era5/era5_static_2010-2011_0.25.nc",
            "data/era5/era5_static_2012-2013_0.25.nc",
        ],
        interest_variables=["u", "v", "msl", "2t"],
        interest_levels=[250],
        output_dir="output",
        batch_size=4,
        replacement_variables=dict(),
        era5_base_frequency="6h",
        init_frequency="1d",
        forecast_horizon="6W",
        eval_aggregation=None,
        eval_start="1W",
        aurora_model="aurora-0.25-pretrained.ckpt",
        device="cpu",
        verbose=True
    )