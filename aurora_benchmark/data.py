import xarray as xr
import torch
import dask
import pandas as pd
import numpy as np
from torch.utils.data import Dataset 
from aurora import Batch, Metadata
import logging
import warnings

from aurora_benchmark.utils import verbose_print

logger = logging.getLogger(__name__)
dask.config.set(scheduler='threads')

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
    inspired by https://microsoft.github.io/aurora/example_era5.html
    """ 
    # ensure only 2 timesteps
    times = surface_ds.time.values.astype("datetime64[s]").tolist()
    assert len(times) == 2, f"Aurora requires 2 time samples, got {len(times)}."
    _time = times[1] # only the second time sample (i.e. current step)
    # ensure longitudes start at 0
    longitudes = torch.Tensor(sorted(atmospheric_ds.longitude.values, reverse=False))
    if longitudes.min() == -180.0:
        longitudes += 180.0
    return Batch(
        surf_vars = {
            var: torch.from_numpy(surface_ds[var].values).unsqueeze(0)
            for var in surface_variables
        },
        atmos_vars = {
            var: torch.from_numpy(atmospheric_ds[var].values).unsqueeze(0)
            for var in atmospheric_variables
        },
        static_vars = {
            var: torch.from_numpy(static_ds[var].values)
            for var in static_variables
        },
        metadata=Metadata(
            lat=torch.Tensor(sorted(atmospheric_ds.latitude.values, reverse=True)),
            lon=torch.Tensor(sorted(atmospheric_ds.longitude.values, reverse=False)),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
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

def aurora_batch_collate_fn(batches: list[Batch]) -> Batch:
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
            time=tuple(batch.metadata.time for batch in batches),
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
                time=batch.metadata.time[b],
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
        """
        super().__init__()
        self.surface_ds = surface_ds.sortby("time")
        self.atmospheric_ds = atmospheric_ds.sortby("time")
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
        
        if init_frequency != "1D":
            warnings.warn("The init_frequency is not 1 day. This has not been tested.")
        
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

        # keep only the timesteps at every [init_frequency + k] for k = 1 to num time samples
        valid_timestamps = np.concatenate([
            pd.date_range(
                start=timestamps[k],
                end=timestamps[-(self.num_time_samples-k)]
            ).values.astype("datetime64[s]").reshape(-1, 1)
            for k in range(self.num_time_samples)
        ], axis=1)
        
        return valid_timestamps
    
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
            self.surface_ds.sel(time=batch_timestamps),
            self.atmospheric_ds.sel(time=batch_timestamps),
            self.static_ds,
            surface_variables=self.surface_variables,
            static_variables=self.static_variables,
            atmospheric_variables=self.atmospheric_variables
        )
        
if __name__ == "__main__":
    # toy data to test the dataset
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from xarray import DataArray, Dataset
    from aurora import Batch, Metadata
    
    # create a toy dataset
    time = pd.date_range(start="2020-01-01", end="2020-12-31", freq="6h")
    num_longitudes = 1440 // 4
    num_latitudes = 721 // 4
    
    surface_ds = Dataset(
        {
            "u10": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "v10": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "t2m": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "msl": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "sst": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "tp": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
        },
        coords={"time": time, "latitude": np.linspace(-90, 90, num_latitudes), "longitude": np.linspace(-180, 180, num_longitudes)}
    )
    
    atmospheric_ds = Dataset(
        {
            "t": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "u": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "v": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "q": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
            "z": DataArray(np.random.rand(len(time), num_latitudes, num_longitudes), dims=["time", "latitude", "longitude"]),
        },
        coords={"time": time, "latitude": np.linspace(-90, 90, num_latitudes), "longitude": np.linspace(-180, 180, num_longitudes)}
    )
    
    static_ds = Dataset(
        {
            "z": DataArray(np.random.rand(num_latitudes, num_longitudes), dims=["latitude", "longitude"]),
            "lsm": DataArray(np.random.rand(num_latitudes, num_longitudes), dims=["latitude", "longitude"]),
            "stype": DataArray(np.random.rand(num_latitudes, num_longitudes), dims=["latitude", "longitude"]),
        },
        coords={"latitude": np.linspace(-90, 90, num_latitudes), "longitude": np.linspace(-180, 180, num_longitudes)}
    )
    
    # create the dataset
    dataset = XRAuroraDataset(
        surface_ds=surface_ds,
        atmospheric_ds=atmospheric_ds,
        static_ds=static_ds,
        init_frequency="1D",
        forecast_horizon="2W",
        num_time_samples=2, 
    )
    
    print(f"Number of batches: {len(dataset)}")
    
    # sample a batch
    batch = dataset[0]
    
    print(f"Batch metadata: {batch.metadata}")
    
    
    
    