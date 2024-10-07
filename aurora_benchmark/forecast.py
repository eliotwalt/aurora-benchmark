import aurora

from utils import XRBatch 

def aurora_forecast(
    era5_path: str,
    era5_climatology_path: str,
    era5_base_frequency: str,
    init_frequency: str,
    num_forecast_steps: int,
    num_aggregate_steps: int,
    batch_size: int,
    start_write_step: int,
    interest_variables: list[str],
    output_dir: str,
): 
    pass