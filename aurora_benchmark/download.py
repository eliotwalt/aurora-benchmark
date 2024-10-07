import dask
import xarray as xr

def download_era5_wb2(
    gs_url: str,
    output_dir: str,
    resampling_frequencies: list[str] = ["1D", "1W"],
    climatology_years: list[int] = [1979, 2020],
    eval_years: list[int] = [2021, 2022],
    compute_std_to_climatology: bool = True
):
    pass