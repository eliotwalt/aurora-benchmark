#!/bin/bash

FORECAST_CONFIG=./configs/forecast_era5_wb2_2021-2022-6h-6w-1440x721_original_vars.yaml
HOST_CONFIG=./configs/snellius.yaml

# activate CPU environment
./env/modules_gpu.sh
source ./env/venv_gpu/bin/activate

# download data with python script
sbatch --cpus-per-task=16 \
    --gpus=1 \
    --mem=96G \
    --job-name=fc_era5_wb2 \
    --partition=gpu_a100 \
    --time=120:00:00 \
    --output=./logs/forecast/%j.out \
    --error=./logs/forecast/%j.out \
    --wrap="./env/modules_gpu.sh && source ./env/venv_gpu/bin/activate  && python ./py_scripts/forecast.py --forecast_config $FORECAST_CONFIG --host_config $HOST_CONFIG"
