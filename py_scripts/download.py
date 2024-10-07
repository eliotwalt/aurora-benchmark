import os
import argparse
import yaml
import logging

from aurora_benchmark.download import download_era5_wb2

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

download_config["output_dir"] = os.path.join(host_config["data_root_dir"], download_config["output_dir"])

download_era5_wb2(**download_config)