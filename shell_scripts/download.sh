#!/bin/bash

DOWNLOAD_CONFIG=./configs/download_era5_wb2_1979-2022-6h-1444x721.yaml
HOST_CONFIG=./configs/snellius.yaml

# compute num jobs
num_jobs=$(python ./py_scripts/task_array.py $DOWNLOAD_CONFIG)

# load modules
./env/modules_cpu.sh

# activate environment
./shell_scripts/activate_cpu.sh --cpu

# download data with python script
sbatch --cpus-per-task=1 --mem=32G --time=6:00:00 --output=./logs/%A_%a.out \
    --error=./logs/%A_%a.err --job-name=dl_era5_wb2 --partition=staging \
    --array=0-$((num_jobs-1))%1 \
    --wrap="python ./py_scripts/download.py --download_config $DOWNLOAD_CONFIG --host_config $HOST_CONFIG"