#!/bin/bash

EVAL_CONFIG=./configs/evaluate_era5_wb2_1979-2022-6h-1440x721.yaml
HOST_CONFIG=./configs/snellius.yaml

# activate CPU environment
./env/modules_cpu.sh
source ./env/venv_gpu/bin/activate

# evaluate average stats
sbatch --cpus-per-task=16 --mem=400G --time=42:00:00 --output=./logs/evaluate_avg/%j.out \
    --error=./logs/evaluate_avg/%j.out --job-name=eval_avg --partition=himem_4tb \
    --wrap="./env/modules_gpu.sh && source ./env/venv_gpu/bin/activate  && python ./py_scripts/evaluate.py --eval_config $EVAL_CONFIG --host_config $HOST_CONFIG --task average"

sbatch --cpus-per-task=4 --mem=64G --time=12:00:00 --output=./logs/evaluate_preds/%j.out \
    --error=./logs/evaluate_preds/%j.out --job-name=eval_avg --partition=staging \
    --wrap="./env/modules_gpu.sh && source ./env/venv_gpu/bin/activate  && python ./py_scripts/evaluate.py --eval_config $EVAL_CONFIG --host_config $HOST_CONFIG --task predictions"