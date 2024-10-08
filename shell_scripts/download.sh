#!/bin/bash

DOWNLOAD_CONFIG=./configs/download_era5_wb2_1979-2022-6h-1444x721.yaml
HOST_CONFIG=./configs/snellius.yaml

# activate CPU environment
./env/modules_cpu.sh
source ./env/venv_cpu/bin/activate

# compute num jobs
num_jobs=$(python ./py_scripts/task_array.py $DOWNLOAD_CONFIG)

# download data with python script
sbatch --cpus-per-task=1 --mem=32G --time=6:00:00 --output=./logs/download/%A_%a.out \
    --error=./logs/download/%A_%a.out --job-name=dl_era5_wb2 --partition=staging \
    --array=0-$((num_jobs-1))%1 \
    --wrap="./env/modules_cpu.sh && source ./env/venv_cpu/bin/activate  && python ./py_scripts/download.py --download_config $DOWNLOAD_CONFIG --host_config $HOST_CONFIG"
