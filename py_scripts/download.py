import os, sys
import argparse
import yaml
import logging
# from dask.distributed import Client, LocalCluster

from aurora_benchmark.download import download_era5_wb2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from py_scripts.download_task_array import get_job_config

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
    p.add_argument("--download_config", type=yaml_file, required=True)
    p.add_argument("--host_config", type=yaml_file, required=True)
    args = p.parse_args()
    
    # # setup slurm with dask
    # if os.environ.get("SLURM_CPUS_PER_TASK", False):
    #     n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    #     cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=2)
    #     client = Client(cluster)
    #     logger.info(f"SLURM detected. Setting up dask with {n_cpus} workers...")
    # else:
    #     # utilise all local cpus (os.cpu_count()
    #     n_cpus = os.cpu_count()
    #     cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=2)
    #     client = Client(cluster)
    #     logger.info(f"Running locally. Setting up dask with {n_cpus} workers...")

    download_config = args.download_config
    host_config = args.host_config

    # add data root specific to the host
    download_config["output_dir"] = os.path.join(host_config["data_root_dir"], download_config["output_dir"])
    download_config["output_dir_climatology"] = os.path.join(host_config["data_root_dir"], download_config["output_dir_climatology"])

    # check whether we are running in a slurm array job
    if os.environ.get("SLURM_ARRAY_TASK_ID", None) is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        download_config = get_job_config(download_config, task_id)
    download_era5_wb2(**download_config)
    
    # # close dask cluster
    # client.close()
    # cluster.close()