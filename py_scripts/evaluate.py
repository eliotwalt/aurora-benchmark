import os, sys
import argparse
import yaml
import logging

from aurora_benchmark.eval import average_statistics_plots, predictions_plots

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
    p.add_argument("--eval_config", type=yaml_file, required=True)
    p.add_argument("--host_config", type=yaml_file, required=True)
    p.add_argument("--task", type=str, required=True, choices=["average", "predictions"])
    args = p.parse_args()

    eval_config = args.eval_config
    host_config = args.host_config

    # add data root specific to the host
    eval_config["forecast_dir"] = os.path.join(host_config["data_root_dir"], eval_config["forecast_dir"])
    for key in ["era5_surface_paths", "era5_atmospheric_paths", "era5_static_paths",]:
        eval_config[key] = [
            os.path.join(host_config["data_root_dir"], path) 
            for path in eval_config[key]
        ]
    
    # separate levels and variables if predictions
    if args.task == "predictions":
        eval_config["level"] = list(set([
            var.split("_")[1] for var in eval_config["variables"]
            if "_" in var
        ]))
        assert len(eval_config["level"]) == 1, "Only one level is supported for predictions atm"
         
        eval_config["variables"] = list(set([
            var.split("_")[0] if "_" in var else var
            for var in eval_config["variables"]
        ]))
        predictions_plots(**eval_config)
    else:
        average_statistics_plots(**eval_config)
    
    sys.exit(0)