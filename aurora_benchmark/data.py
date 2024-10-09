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
        'u10',
        'v10',
        't2m',
        'msl',
        'sst',
        'tp',
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
        'stype',
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
    # ensure longitudes start at 0
    longitudes = torch.Tensor(sorted(atmospheric_ds.longitude.values, reverse=False))
    if longitudes.min() == -180.0:
        longitudes += 180.0
    return Batch(
        surf_vars = {
            var: torch.from_numpy(surface_ds[var].values)
            for var in surface_variables
        },
        atmos_vars = {
            var: torch.from_numpy(atmospheric_ds[var].values)
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
            time=(surface_ds.time.values.astype("datetime64[s]").tolist(),),
            atmos_levels=tuple(int(level) for level in atmospheric_ds.level.values),
        )
    )

def aurora_batch_to_xr(batch: Batch) -> dict[str, xr.Dataset]:
    surface_ds = {
        var: xr.Dataset(
            batch.surf_vars[var].numpy(),
            dims=["latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon,
                    "time": batch.metadata.time},
        )
        for var in batch.surf_vars
    }
    atmospheric_ds = {
        var: xr.Dataset(
            batch.atmos_vars[var].numpy(),
            dims=["latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon,
                    "time": batch.metadata.time,
                    "level": batch.metadata.atmos_levels},
        )
        for var in batch.atmos_vars
    }
    static_ds = {
        var: xr.Dataset(
            batch.static_vars[var].numpy(),
            dims=["latitude", "longitude"],
            coords={"latitude": batch.metadata.lat, 
                    "longitude": batch.metadata.lon},
        )
        for var in batch.static_vars
    }
    return {
        "surface": xr.merge(surface_ds),
        "atmospheric": xr.merge(atmospheric_ds),
        "static": xr.merge(static_ds),
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
            var: torch.cat([batch.static_vars[var] for batch in batches], dim=0)
            for var in batches[0].static_vars
        },
        metadata=Metadata(
            lat=batches[0].metadata.lat,
            lon=batches[0].metadata.lon,
            time=tuple(batch.metadata.time for batch in batches),
            atmos_levels=batches[0].metadata.atmos_levels,
        )
    )

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
        self.init_frequency = init_frequency
        self.forecast_horizon = forecast_horizon
        self.num_time_samples = num_time_samples
        self.surface_variables = surface_variables
        self.atmospheric_variables = atmospheric_variables
        self.static_variables = static_variables
        self.pressure_levels = pressure_levels
        self.replacement_variables = replacement_variables
        
        self.surface_ds = self._get_init_times(surface_ds)
        self.atmospheric_ds = self._get_init_times(atmospheric_ds)
        self.atmospheric_ds = self.atmospheric_ds.sel(level=pressure_levels)
        self.static_ds = static_ds
        
        assert len(self.surface_ds.time) == len(self.atmospheric_ds.time)
        
        if init_frequency != "1D":
            raise NotImplementedError("Only daily initialisation times are supported.")
        
        if len(replacement_variables) > 0:
            raise NotImplementedError("Replacement variables are not yet implemented.")
        
    def _get_init_times(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Keep only the valid initialisation timestamps, i.e., at init_frequency
        from one another and with enough ground truth data to make a forecast
        of forecast_horizon.
        """
        def first_n(group: xr.Dataset, n: int) -> xr.Dataset:
            if len(group.time) < n:
                # return an empty Dataset with the same structure
                # that will be discarded by the resample method
                return group.isel(time=slice(0, 0))
            return group.isel(time=slice(0, n))
        # ensure sorted
        ds = ds.sortby("time")
        # resample to keep only the first num_time_samples at each init_frequency
        timestamps = ds.indexes["time"]
        ds = ds.resample(time=self.init_frequency).map(first_n, n=self.num_time_samples)
        ds['time'] = timestamps[ds.time.values]
        # select the timesteps that are at least forecast_horizon away from the last timestep
        ds = ds.sel(time=ds.time <= ds.time.max() - pd.Timedelta(self.forecast_horizon))
        return ds
        
    def __len__(self) -> int:
        return len(self.sruface_ds.time)-1
    
    def __getitem__(self, idx: int) -> Batch:
        time_slice = slice(self.surface_ds.time.values[idx], self.surface_ds.time.values[idx-1])     
        return xr_to_aurora_batch(
            self.surface_ds.sel(time=time_slice),
            self.atmospheric_ds.sel(time=time_slice),
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
    
    
    
    