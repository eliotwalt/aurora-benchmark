import os, sys
import argparse
import yaml
import logging

from aurora_benchmark.forecast import aurora_forecast

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Suppress logs from Google libraries
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.cloud').setLevel(logging.ERROR)

def yaml_file(x):
    try:
        with open(x, "r") as f:
            return yaml.safe_load(f) 
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(f"Unable to read input as a yaml file: {x}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forecast_config", type=yaml_file, required=True)
    p.add_argument("--host_config", type=yaml_file, required=True)
    args = p.parse_args()

    forecast_config = args.forecast_config
    host_config = args.host_config

    # add data root specific to the host
    forecast_config["output_dir"] = os.path.join(host_config["data_root_dir"], forecast_config["output_dir"])

    # forecast
    aurora_forecast(**forecast_config)
    
    # # close dask cluster
    # client.close()
    # cluster.close()
    
    # exit without error
    sys.exit(0)