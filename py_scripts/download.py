import os, sys
import argparse
import yaml
import logging

from aurora_benchmark.download import download_era5_wb2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from py_scripts.task_array import get_job_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def yaml_file(x):
    try:
        with open(x, "r") as f:
            return yaml.safe_load(f) 
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(f"Unable to read input as a yaml file: {x}")

p = argparse.ArgumentParser()
p.add_argument("--download_config", type=yaml_file, required=True)
p.add_argument("--host_config", type=yaml_file, required=True)
args = p.parse_args()

download_config = args.download_config
host_config = args.host_config

# add data root specific to the host
download_config["output_dir"] = os.path.join(host_config["data_root_dir"], download_config["output_dir"])

# check whether we are running in a slurm array job
if os.environ.get("SLURM_ARRAY_TASK_ID", None) is not None:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    download_config = get_job_config(download_config, task_id)
download_era5_wb2(**download_config)