import xarray as xr
import torch
import dask
import pandas as pd
import numpy as np
from torch.utils.data import Dataset 
from aurora import Batch, Metadata
import logging
import warnings
import math
from copy import deepcopy
from torch.nn.parallel._functions import Scatter, Gather

from aurora_benchmark.utils import verbose_print

logger = logging.getLogger(__name__)

AURORA_VARIABLE_NAMES = {
    "surface": [
        '10u',
        '10v',
        '2t',
        'msl',
        #'sst', # additional
        #'tp', # additional
    ],
    "atmospheric": [
        't',
        'u',
        'v',
        'q',
        'z',
    ],
    "static": [
        'z',
        'lsm',
        'slt',
    ]
}
AURORA_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def xr_to_aurora_batch(
    surface_ds: xr.Dataset,
    atmospheric_ds: xr.Dataset,
    static_ds: xr.Dataset,
    surface_variables: list[str]=AURORA_VARIABLE_NAMES["surface"],
    static_variables: list[str]=AURORA_VARIABLE_NAMES["static"],
    atmospheric_variables: list[str]=AURORA_VARIABLE_NAMES["atmospheric"],
) -> Batch:
    """ 
    Create an Aurora Batch from XR Datasets.
    
    inspired by https://microsoft.github.io/aurora/example_era5.html
    and https://microsoft.github.io/aurora/example_hres_t0.html
    """ 
    def prepare_array(x: np.ndarray, shape: tuple[int], flip: bool) -> torch.Tensor:
        if flip:
            return torch.from_numpy(x.reshape(shape)[...,::-1,:].copy())
        else:
            return torch.from_numpy(x.reshape(shape).copy())
    # ensure only 2 timesteps
    # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
    # `datetime.datetime`s. Note that this needs to be a tuple of length one:
    # one value for every batch element.
    times = surface_ds.time.values.astype("datetime64[s]").tolist()
    assert len(times) == 2, f"Aurora requires 2 time samples, got {len(times)}."
    _time = (times[1],) # only the second time sample (i.e. current step)
    # ensure longitudes start at 0
    longitudes = torch.Tensor(sorted(atmospheric_ds.longitude.values, reverse=False))
    if longitudes.min() == -180.0:
        longitudes += 180.0
    # get shapes for explicit reshaping
    C = atmospheric_ds.sizes["level"]
    T = atmospheric_ds.sizes["time"]
    H = atmospheric_ds.sizes["latitude"]
    W = atmospheric_ds.sizes["longitude"]
    # get lats in decreasing order
    if surface_ds.latitude.values[0] < surface_ds.latitude.values[1]: # i.e. increasing
        lats = torch.from_numpy(surface_ds.latitude.values[::-1].copy())
    else:
        lats = torch.from_numpy(surface_ds.latitude.values.copy())
    # get lons in increasing order
    if atmospheric_ds.longitude.values[0] < atmospheric_ds.longitude.values[1]: # i.e. increasing
        lons = torch.from_numpy(atmospheric_ds.longitude.values.copy())
    else:
        lons = torch.from_numpy(atmospheric_ds.longitude.values[::-1].copy())
    return Batch(
        surf_vars = {
            var: prepare_array(
                surface_ds[var].values, 
                (1, T, H, W),
                surface_ds.latitude.values[0] < surface_ds.latitude.values[1]) #torch.from_numpy(surface_ds[var].values).reshape(1, T, H, W)
            for var in surface_variables
        },
        atmos_vars = {
            var: prepare_array(
                atmospheric_ds[var].values, 
                (1, T, C, H, W),
                atmospheric_ds.latitude.values[0] < atmospheric_ds.latitude.values[1]) # torch.from_numpy(atmospheric_ds[var].values).reshape(1, T, C, H, W)
            for var in atmospheric_variables
        },
        static_vars = {
            var: prepare_array(
                static_ds[var].values, 
                (H, W),
                static_ds.latitude.values[0] < static_ds.latitude.values[1]) #torch.from_numpy(static_ds[var].values).reshape(H, W)
            for var in static_variables
        },
        metadata=Metadata(
            lat=lats,
            lon=lons,
            time=_time,
            atmos_levels=tuple(int(level) for level in atmospheric_ds.level.values),
        )
    )

def aurora_batch_to_xr(batch: Batch, frequency: str) -> dict[str, xr.Dataset]:
    """
    Retrieve xr Dataset structures from an Aurora batch.
    
    Args:
        batch: Batch
            The Aurora batch.
        frequency: str
            The frequency of the time samples. This is used to create the time
            coordinates. For example, if the frequency is 6 hours, then the time
            coordinates will be 6 hours apart.
    """
    
    # TODO
    # 1. As manty time coordinates as time samples!
    # 2. Ensure staic variables are oki
    # 3. Remove the batch dimension to retrieve time
    
    def flatten_batch_dim(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if len(shape) == 4: # B, T, H, W
            x = x.reshape(shape[0] * shape[1], shape[2], shape[3])
        elif len(shape) == 5: # B, T, C, H, W
            x = x.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4])
        else:
            raise ValueError(f"Expected 4 or 5 dimensions, got {len(shape)}")
        return x
        
    def expand_timestamps(timestamps: tuple, frequency: str, num_time_samples: int) -> np.ndarray:
        """
        Expand the timestamps metadata to include all the time samples.
        """
        timestamps = np.array(timestamps, dtype="datetime64").reshape(-1, 1)
        timestamps = np.concatenate([
            timestamps + k * pd.Timedelta(frequency)
            for k in range(0, num_time_samples)
        ], axis=1).reshape(-1)
        return timestamps
    
    num_time_samples = batch.surf_vars[list(batch.surf_vars.keys())[0]].shape[1]
    
    surface_ds = xr.Dataset({
        var: xr.DataArray(
            flatten_batch_dim(batch.surf_vars[var].numpy()),
            dims=["time", "latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon,
                    "time": expand_timestamps(batch.metadata.time, frequency, num_time_samples),}
        )
        for var in batch.surf_vars
    })
    atmospheric_ds = xr.Dataset({
        var: xr.DataArray(
            flatten_batch_dim(batch.atmos_vars[var].numpy()),
            dims=["time", "level", "latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon,
                    "time": expand_timestamps(batch.metadata.time, frequency, num_time_samples),
                    "level": list(batch.metadata.atmos_levels),},
        )
        for var in batch.atmos_vars
    })
    static_ds = xr.Dataset({
        var: xr.DataArray(
            batch.static_vars[var].numpy(),
            dims=["latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon},
        )
        for var in batch.static_vars
    })
    return {
        "surface_ds": surface_ds,
        "atmospheric_ds": atmospheric_ds,
        "static_ds": static_ds,
    }

def aurora_batch_collate_fn(batches: list[Batch]|None) -> Batch:
    # check input
    _batches = batches.copy()
    for i, batch in enumerate(_batches):
        if batch is None: batches.pop(i)
        elif not isinstance(batch, Batch):
            raise ValueError(f"Expected a list of Aurora batches or NoneType, got {type(batch)}")
    if len(batches) == 0:
        return # nothing to batch return None
    elif len(batches) == 1:
        return batches[0] # nothing to batch return the single batch    
    # Prediction batches have a single time sample apparently
    times = []
    for batch in batches:
        time = batch.metadata.time
        if isinstance(time, tuple):
            times.append(time[0])
        else:
            times.append(time)        
    return Batch(
        surf_vars={
            var: torch.cat([batch.surf_vars[var] for batch in batches], dim=0)
            for var in batches[0].surf_vars
        },
        atmos_vars={
            var: torch.cat([batch.atmos_vars[var] for batch in batches], dim=0)
            for var in batches[0].atmos_vars
        },
        static_vars={
            var: batches[0].static_vars[var]
            for var in batches[0].static_vars
        },
        metadata=Metadata(
            lat=batches[0].metadata.lat,
            lon=batches[0].metadata.lon,
            time=tuple(times),
            atmos_levels=batches[0].metadata.atmos_levels,
        )
    )

def unpack_aurora_batch(batch: Batch) -> list[Batch]:
    """
    Unpack Aurora batch into a list of batches
    """
    # compute batch size
    batch_size = len(batch.metadata.time)
    # unpack the batch
    batches = [
        Batch(
            surf_vars={k: batch.surf_vars[k][b].unsqueeze(0) for k in batch.surf_vars},
            atmos_vars={k: batch.atmos_vars[k][b].unsqueeze(0) for k in batch.atmos_vars},
            static_vars=batch.static_vars,
            metadata=Metadata(
                lat=batch.metadata.lat,
                lon=batch.metadata.lon,
                time=(batch.metadata.time[b],),
                atmos_levels=batch.metadata.atmos_levels,
            )
        )
        for b in range(batch_size)
    ]
    return batches

class XRAuroraDataset(Dataset):
    """
    A Dataset class to sample Aurora batches from an xarray Dataset.
    
    WARNING: only works for 6-hourly data and init_frequency of 1 day. 
        To go around that, we would need to make changes to the 
        _get_init_times method which explicitly uses these parameters.
    """
    def __init__(
        self,
        surface_ds: xr.Dataset,
        atmospheric_ds: xr.Dataset,
        static_ds: xr.Dataset,
        init_frequency: str="1D",
        forecast_horizon: str="6W",
        num_time_samples: int=2,
        surface_variables: list[str]=AURORA_VARIABLE_NAMES["surface"],
        atmospheric_variables: list[str]=AURORA_VARIABLE_NAMES["atmospheric"],
        static_variables: list[str]=AURORA_VARIABLE_NAMES["static"],
        pressure_levels: list[int]=AURORA_PRESSURE_LEVELS,
        replacement_variables: dict[str, str]={},
        drop_timestamps: bool=False,
        rechunk: bool=False,
        persist: bool=False,
        shuffle: bool=False,
    ) -> None:
        """
        Initialise the XRAuroraDataset.
        
        Args:
            surface_ds: xr.Dataset
                The surface dataset.
            atmospheric_ds: xr.Dataset
                The atmospheric dataset.
            static_ds: xr.Dataset
                The static dataset.
            init_frequency: str
                The frequency of the initialisation times. Defaults to 1 day.
            forecast_horizon: str
                The forecast horizon. Defaults to 6 weeks.
            num_time_samples: int
                The number of time samples to keep at each initialisation time. Defaults to 2 (as in Aurora).
            surface_variables: list[str]
                The surface variables to include. Defaults to Aurora's surface variables.
            atmospheric_variables: list[str]    
                The atmospheric variables to include. Defaults to Aurora's atmospheric variables.
            static_variables: list[str] 
                The static variables to include. Defaults to Aurora's static variables.
            pressure_levels: list[int]
                The pressure levels to include. Defaults to Aurora's pressure levels.
            replacement_variables: dict[str, str]   
                A dictionary of replacement variables, i.e. {'q_850': 'tp'}
                means that `q_850` will be replaced with `tp` without telling
                the model. Defaults to an empty dictionary.
            drop_timestamps: bool
                Whether to drop the timestamps that are not used for initialisation. Defaults to False.
            rechunk: bool
                Whether to rechunk the dataset. Defaults to False.
            persist: bool
                Whether to persist the datasets. Defaults to False.
        """
        super().__init__()
        self.surface_ds = surface_ds
        self.atmospheric_ds = atmospheric_ds
        self.static_ds = static_ds
        self.init_frequency = init_frequency
        self.forecast_horizon = forecast_horizon
        self.num_time_samples = num_time_samples
        self.surface_variables = surface_variables
        self.atmospheric_variables = atmospheric_variables
        self.static_variables = static_variables
        self.pressure_levels = pressure_levels
        self.replacement_variables = replacement_variables        
        self.init_timestamps = self._get_init_timestamps()
        if shuffle:
            np.random.shuffle(self.init_timestamps)
        
        if drop_timestamps:
            flat_init_timestamps = np.unique(self.init_timestamps.flatten()).tolist()
            self.surface_ds = self.surface_ds.sel(time=flat_init_timestamps).assign_coords(time=flat_init_timestamps)
            self.atmospheric_ds = self.atmospheric_ds.sel(time=flat_init_timestamps).assign_coords(time=flat_init_timestamps)
        
        if rechunk:
            # check if the dataset is already chunked
            if not self.surface_ds.chunks and not self.atmospheric_ds.chunks:
                warnings.warn("The datasets are not chunked. Chunking the dataset may reduce performance. Consider setting rechunk to `False`.")
            # compute the closest multitple of self.num_time_samples from the current time chunk
            current_time_chunk = self.surface_ds.chunks["time"][0]
            new_time_chunk = current_time_chunk - (current_time_chunk % self.num_time_samples)
            self.surface_ds = self.surface_ds.chunk({"time": new_time_chunk})
            self.atmospheric_ds = self.atmospheric_ds.chunk({"time": new_time_chunk})
            
        # Persist the datasets to avoid recomputation and manage memory efficiently
        if persist:
            self.surface_ds = self.surface_ds.persist()
            self.atmospheric_ds = self.atmospheric_ds.persist()
            self.static_ds = self.static_ds.persist()
        
        if len(replacement_variables) > 0:
            raise NotImplementedError("Replacement variables are not yet implemented.")
        
    def _get_init_timestamps(self) -> np.ndarray:
        """
        Compute the timestamps of all model initialisations
        
        Returns:
            valid_timestamps: np.ndarray
                The timestamps of the model initialisations of shape (N, self.num_time_samples), 
                such that valid_timestamps[i] contains all the timestamps required for the 
                i-th batch. dtype=datetime64[s]
        """
        assert (self.surface_ds.time == self.atmospheric_ds.time).all(), f"got different timestamps for surface and atmospheric data."
        
        # extract initial timestamps
        timestamps = self.surface_ds.time.values.astype("datetime64[s]")
        
        # remove samples that do not allow for at least forecast horizon supervised steps
        timestamps = timestamps[timestamps < timestamps[-1] - pd.Timedelta(self.forecast_horizon)]

        # reshape according to num_time_samples
        valid_timestamps = np.concatenate([
            pd.date_range(
                start=timestamps[k],
                end=timestamps[-(self.num_time_samples-k)]
            ).values.astype("datetime64[s]").reshape(-1, 1)
            for k in range(self.num_time_samples)
        ], axis=1)
        
        # "resample" to init freqency
        # TODO: this could be done more efficiently, e.g. directly in the
        #      pd.date_range function
        init_timestamps = []
        for index, t in enumerate(valid_timestamps):
            if index == 0:
                init_timestamps.append(t)
            elif pd.Timedelta(t[0]-init_timestamps[-1][0]) >= pd.Timedelta(self.init_frequency):
                init_timestamps.append(t)
            else:
                continue
        init_timestamps = np.vstack(init_timestamps)
        
        return init_timestamps
    
    @classmethod
    def from_cloud_storage(cls, zarr_url: str, **kwargs) -> None:
        """
        No need to download anything locally
        """
        raise NotImplementedError
        
    def __len__(self) -> int:
        return self.init_timestamps.shape[0]
    
    def __getitem__(self, k: int) -> Batch:
        batch_timestamps = self.init_timestamps[k]
        return xr_to_aurora_batch(
            self.surface_ds.sel(time=batch_timestamps).compute(),
            self.atmospheric_ds.sel(time=batch_timestamps).compute(),
            self.static_ds.compute(),
            surface_variables=self.surface_variables,
            static_variables=self.static_variables,
            atmospheric_variables=self.atmospheric_variables
        )
        
class XRAuroraBatchedDataset(XRAuroraDataset):
    def __init__(
        self,
        batch_size: int,
        *args, **kwargs
    ) -> None:
        """
        Initialise the XRAuroraBatchedDataset.
        
        Args:
            batch_size: int
                The batch size.
            *args, **kwargs:
                Additional arguments to pass to the XRAuroraDataset.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.flat_init_timestamps = self.init_timestamps.copy()
        # add None to the init_timestamps to ensure that the last batch is not cut off
        if len(self.init_timestamps) % self.batch_size != 0:
            self.init_timestamps = np.concatenate([
                self.init_timestamps,
                np.full((self.batch_size - len(self.init_timestamps) % self.batch_size, self.num_time_samples), None, dtype="datetime64[s]")
            ])
        self.init_timestamps = self.init_timestamps.reshape(-1, self.batch_size, self.num_time_samples)
        
    def flat_length(self) -> int:
        return self.flat_init_timestamps.shape[0]
        
    def __len__(self) -> int:
        return self.init_timestamps.shape[0]
        
    def __getitem__(self, k: int) -> Batch:
        batch_timestamps = self.init_timestamps[k]
        batches = []
        
        # TODO: xr_to_aurora_batch for batch_size > 1 would allow for parallelisation
        #   i.e. no for loop.
        
        for bts in batch_timestamps:
            if pd.isnull(bts).any():
                continue
            batches.append(xr_to_aurora_batch(
                self.surface_ds.sel(time=bts).compute(),
                self.atmospheric_ds.sel(time=bts).compute(),
                self.static_ds.compute(),
                surface_variables=self.surface_variables,
                static_variables=self.static_variables,
                atmospheric_variables=self.atmospheric_variables
            ))
        return aurora_batch_collate_fn(batches)
    
def aurora_batch_scatter(batch: Batch, kwargs: dict|None, device_ids: list[int|torch.device]) -> list[Batch]:
    B = batch.surf_vars[list(batch.surf_vars.keys())[0]].shape[0]
    # scatter every sub tensor
    surf_vars = {
        var: Scatter.apply(device_ids, None, 0, batch.surf_vars[var])
        for var in batch.surf_vars.keys()
    }
    atmos_vars = {
        var: Scatter.apply(device_ids, None, 0, batch.atmos_vars[var])
        for var in batch.atmos_vars.keys()
    }
    static_vars = {
        var: Scatter.apply(device_ids, None, 0, batch.static_vars[var].unsqueeze(0).repeat(B, 1, 1))
        for var in batch.static_vars.keys()
    }
    scattered_batches = []
    n_batches = len(surf_vars[list(surf_vars.keys())[0]])
    current_index = 0
    for i in range(n_batches):
        # get variables
        sub_surf_vars = {var: surf_vars[var][i] for var in surf_vars.keys()}
        sub_atmos_vars = {var: atmos_vars[var][i] for var in atmos_vars.keys()}
        sub_static_vars = {var: static_vars[var][i].squeeze(0) for var in static_vars.keys()}
        # get sub batch size
        sub_batch_size = sub_surf_vars[list(sub_surf_vars.keys())[0]].shape[0]
        # get metadata
        metadata = deepcopy(batch.metadata)
        # ensure get the correct timesteps
        metadata.time = metadata.time[current_index:current_index+sub_batch_size]
        current_index += sub_batch_size
        # create new batch
        new_batch = Batch(
            surf_vars=sub_surf_vars,
            atmos_vars=sub_atmos_vars,
            static_vars=sub_static_vars,
            metadata=metadata,
        )
        scattered_batches.append(new_batch)
    scattered_kwargs = [{} for _ in scattered_batches]
    return scattered_batches, scattered_kwargs

def aurora_batch_gather(outputs: list[Batch], output_device: int|torch.device) -> Batch:
    # gather every sub tensor
    surf_vars = {
        var: Gather.apply(
            output_device, 
            0, 
            *[batch.surf_vars[var] for batch in outputs])
        for var in outputs[0].surf_vars.keys()
    }
    atmos_vars = {
        var: Gather.apply(
            output_device, 
            0, 
            *[batch.atmos_vars[var] for batch in outputs])
        for var in outputs[0].atmos_vars.keys()
    }
    static_vars = { # pass through Gather to ensure correct device then select the first "batch"
        var: Gather.apply(
            output_device, 
            0, 
            *[batch.static_vars[var].unsqueeze(0) for batch in outputs])[0].squeeze(0)
        for var in outputs[0].static_vars.keys()
    }
    # gather metadata
    times = []
    for batch in outputs:
        time = batch.metadata.time
        if isinstance(time, tuple):
            times.append(time[0])
        else:
            times.append(time)        
    metadata = deepcopy(outputs[0].metadata)
    metadata.time = tuple(times)
    metadata.lat = metadata.lat.to(output_device)
    metadata.lon = metadata.lon.to(output_device)
    # build Batch
    gathered_batch = Batch(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        metadata=metadata,
    )
    return gathered_batch