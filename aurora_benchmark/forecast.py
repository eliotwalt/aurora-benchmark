import os
import xarray as xr
import torch
import dask
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import logging
import dataclasses

from aurora import Batch, Aurora, AuroraSmall

from aurora_benchmark.utils import verbose_print, xr_to_netcdf

from aurora_benchmark.parallel import AuroraBatchDataParallel, rollout, ParallelAurora, ParallelAuroraSmall
from aurora_benchmark.data import (
    XRAuroraDataset, 
    XRAuroraBatchedDataset,
    aurora_batch_collate_fn, 
    aurora_batch_to_xr, 
    unpack_aurora_batch
)

logger = logging.getLogger(__name__)

AURORA_MODELS_HF = [
    "aurora-0.25-small-pretrained.ckpt",
    "aurora-0.25-finetuned.ckpt", # on IFS HRES T0
    "aurora-0.25-pretrained.ckpt"
]

AURORA_VARIABLE_RENAMES = {
    "surface": {
        "u10": "10u",
        "v10": "10v",
        "t2m": "2t",
    },
    "atmospheric": {},
    "static": {},
}
INVERTED_AURORA_VARIABLE_RENAMES = {
    "surface": {v: k for k, v in AURORA_VARIABLE_RENAMES["surface"].items()},
    "atmospheric": {v: k for k, v in AURORA_VARIABLE_RENAMES["atmospheric"].items()},
    "static": {v: k for k, v in AURORA_VARIABLE_RENAMES["static"].items()},
}    

def aurora_forecast(
    era5_surface_paths: list[str],
    era5_atmospheric_paths: list[str],
    era5_static_paths: list[str],
    interest_variables: list[str],
    interest_levels: list[str],
    output_dir: str,
    batch_size: int=1,
    replacement_variables: dict[str, str]=dict(),
    era5_base_frequency: str="6h",
    init_frequency: str="1d",
    forecast_horizon: str="6W",
    eval_aggregation: str|None=None,
    eval_start: str="1W", # 1 week
    aurora_model: str="aurora-0.25-pretrained.ckpt",
    device: str|torch.device="cuda" if torch.cuda.is_available() else "cpu",
    rechunk: bool=False,
    drop_timestamps: bool=False,
    persist: bool=False,
    verbose: bool=True,
    use_dataloader: bool=False,
    data_parallel: bool=False,
):     
    # no override!
    os.makedirs(output_dir, exist_ok=False)
    verbose_print(verbose, f"Forecasts output dir: {output_dir}")
    
    # compute additional arguments
    warmup_steps = pd.Timedelta(eval_start) / pd.Timedelta(era5_base_frequency) if eval_start is not None else 0.0
    forecast_steps = pd.Timedelta(forecast_horizon) / pd.Timedelta(era5_base_frequency)
    
    # setup dask scheduler
    if use_dataloader:
        raise NotImplementedError("Dataloader not yet implemented because netcdf+dask is not thread safe. Use_dataloader=False.")
        # dask.config.set(scheduler='synchronous')
    else:
        dask.config.set(scheduler='threads')
    verbose_print(verbose, f"Using dask scheduler: {dask.config.get('scheduler')}")
        
    # validate arguments
    assert aurora_model in AURORA_MODELS_HF, f"Model {aurora_model} not found in {AURORA_MODELS_HF}"
    assert forecast_steps.is_integer(), f"forecast_horizon not a multiple of frequency"
    assert warmup_steps.is_integer(), f"eval_start not a multiple of frequency"
    assert (forecast_steps-warmup_steps) * pd.Timedelta(era5_base_frequency) >= pd.Timedelta(eval_aggregation), "Evaluation steps must be at least as long as eval_aggregation" 
    if data_parallel and device == "cpu":
        verbose_print(verbose, "DataParallel requires GPUs. Using single CPU ...")
        data_parallel = False
    if data_parallel and torch.cuda.device_count()<=1:
        verbose_print(verbose, "DataParallel requires multiple GPUs. Using single GPU ...")
        data_parallel = False
    forecast_steps = int(forecast_steps)
    warmup_steps = int(warmup_steps)
    
    # not implementated protection
    if len(replacement_variables) > 0:
        raise NotImplementedError("Replacement variables not yet implemented.")
    
    # dask
    time_chunk = 50 * batch_size
    
    # batch_size safety
    if batch_size > 1:
        # TODO: Fix batch processing        
        
        raise NotImplementedError("Batch size > 1 not yet implemented.")
    
    # model
    if data_parallel:
        if "small" in aurora_model:
            verbose_print(verbose, f"Loading ParallelAuroraSmall model with {torch.cuda.device_count()} GPUs...")
            model = ParallelAuroraSmall()
        else:
            verbose_print(verbose, f"Loading ParallelAurora model with {torch.cuda.device_count()} GPUs...")
            model = ParallelAurora(use_lora=False)
        model.load_checkpoint("microsoft/aurora", aurora_model)
        model = AuroraBatchDataParallel(model)
        verbose_print(verbose, f"Adjusting batch size for DataParallel ({batch_size} batch(es)/GPU)...")
        batch_size *= torch.cuda.device_count()
    else:
        if "small" in aurora_model:
            verbose_print(verbose, "Loading AuroraSmall model ...")
            model = AuroraSmall()
        else:
            verbose_print(verbose, "Loading Aurora model ...")
            model = Aurora(use_lora=False)
        model.load_checkpoint("microsoft/aurora", aurora_model)
    model = model.to(device)
    
    # load xr data
    verbose_print(verbose, "Reading data ...")
    surface_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4", 
                         chunks={"time": time_chunk, "latitude": 721, "longitude": 1440})#backend_kwargs={'diskless': True, 'persist': False}) 
         for path in era5_surface_paths],
    )
    atmospheric_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4",
                         chunks={"time": time_chunk, "latitude": 721, "longitude": 1440, "level": 1})#backend_kwargs={'diskless': True, 'persist': False}) 
         for path in era5_atmospheric_paths],
    )
    static_ds = xr.merge(
        [xr.open_dataset(path, engine="netcdf4")#backend_kwargs={'diskless': True, 'persist': False})
         for path in era5_static_paths],
    )
    
    # feedback dims
    verbose_print(verbose, f"surface_ds: {surface_ds.dims}")
    verbose_print(verbose, f"atmospheric_ds: {atmospheric_ds.dims}")
    verbose_print(verbose, f"static_ds: {static_ds.dims}")
    
    # create dataset XRAuroraDataset and DataLoader
    if use_dataloader:
        verbose_print(verbose, f"Creating XRAuroraDataset and DataLoader...")
        dataset = XRAuroraDataset(
            surface_ds=surface_ds,
            atmospheric_ds=atmospheric_ds,
            static_ds=static_ds,
            init_frequency=init_frequency,
            forecast_horizon=forecast_horizon,
            num_time_samples=2, # Aurora has fixed history length of 2...
            drop_timestamps=drop_timestamps,
            persist=persist,
            rechunk=rechunk
        )
        
        num_workers = 1 #int(os.getenv('SLURM_CPUS_PER_TASK', 1))+2 if os.getenv('SLURM_CPUS_PER_TASK') is not None else os.cpu_count()+2
        verbose_print(verbose, f"Creating DataLoader with {num_workers} workers ...")
        eval_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=aurora_batch_collate_fn,
            num_workers=num_workers,
        )
    else:
        # This is done to avoid the issue with torch DataLoader and dask
        # when using netcdf files (i.e. netcdf backend is not thread safe)
        verbose_print(verbose, f"Creating XRAuroraBatchedDataset ...")
        dataset = XRAuroraBatchedDataset(
            batch_size=batch_size,
            surface_ds=surface_ds,
            atmospheric_ds=atmospheric_ds,
            static_ds=static_ds,
            init_frequency=init_frequency,
            forecast_horizon=forecast_horizon,
            num_time_samples=2, # Aurora has fixed history length of 2...
            drop_timestamps=drop_timestamps,
            persist=persist,
            rechunk=rechunk
        )
        eval_loader = dataset
        
    verbose_print(verbose, f"Dataset length: {dataset.flat_length() if hasattr(dataset, 'flat_length') else len(dataset)}")
    verbose_print(verbose, f"Dataloader length: {len(eval_loader)} (type: {type(eval_loader)}, batch_size: {batch_size})")  
    
    # evaluation loop
    devices = device if not data_parallel else [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    verbose_print(verbose, f"Starting evaluation on {devices}...")
    with torch.inference_mode() and torch.no_grad():
        
        for i, batch in enumerate(eval_loader):
            verbose_print(verbose,f"Rollout prediction on batch {i} ...")
            if batch is None: continue
            batch = batch.to(device)
            
            trajectories = [[] for _ in range(batch_size)]
            for s, batch_pred in enumerate(rollout(model, batch, steps=forecast_steps)):
                if s < warmup_steps:
                    verbose_print(verbose, f" * Rollout step {s+1}: skipping warmup period")
                    continue            
                # separate batched batches
                sub_batch_preds = unpack_aurora_batch(batch_pred.to("cpu"))
                verbose_print(verbose, f" * Rollout step {s+1}: unpacked {len(sub_batch_preds)} sub-batches")
                if i != len(eval_loader) - 1: # the last batch may not be full
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
                
                # process individual trajectory elements (i.e. variable types)
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
                        
                    # override time coordinates using the era5_base_frequency
                    # this is necessary because Aurora assumes timesteps of 6h
                    # but we can theoretically use any frequency
                    vars_ds = vars_ds.assign_coords(
                        {"time": pd.date_range(init_time+warmup_steps*pd.Timedelta(era5_base_frequency), 
                                            periods=vars_ds.sizes["time"], 
                                            freq=era5_base_frequency)})    
                        
                    # aggregate at eval_agg frequency
                    # use pd.Timedelta to avoid xarray automatically starting the resampling 
                    # on Mondays for weekly etc.
                    # Note that resulting 'time' will be the first timestamp in the aggregated period
                    vars_ds = vars_ds.resample(time=pd.Timedelta(eval_aggregation), origin=init_time).mean()
                    vars_ds = vars_ds.rename({"time": "lead_time"})
                    vars_ds["lead_time"] = vars_ds["lead_time"] - np.datetime64(init_time)
                    
                    # per-variable processing
                    for var in vars_ds.data_vars:
                        # add lead time
                        var_ds = vars_ds[var]
                        
                        # save
                        path = f"forecast_{var}_" + "-".join([
                            init_time.strftime("%Y%m%dT%H%M%S"),
                            str(era5_base_frequency),
                            str(eval_aggregation),
                            str(eval_start),
                            str(forecast_horizon),
                            str(var_ds.sizes["longitude"])+ "x" +str(var_ds.sizes["latitude"]),
                        ]) + ".nc"
                        path = os.path.join(output_dir, path)
                        verbose_print(verbose, f"   * Saving new {var_type} forecast: {path}")
                        xr_to_netcdf(
                            var_ds, path, 
                            precision="float32", 
                            compression_level=1, 
                            sort_time=False, 
                            exist_ok=True
                        )