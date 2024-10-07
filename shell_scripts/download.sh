#!/bin/bash

DOWNLOAD_CONFIG=./configs/download_era5_wb2_1979-2022-6h-1444x721.yaml
HOST_CONFIG=./configs/snellius.yaml

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/%A_%a.out
#SBATCH --error=./logs/%A_%a.err
#SBATCH --job-name=dl_era5_wb2
#SBATCH --partition=staging

# Update the array size dynamically
num_jobs=$(python ./py_scripts/task_array.py $DOWNLOAD_CONFIG)
scontrol update JobId=$SLURM_JOB_ID ArrayTaskLimit=$num_jobs

# load modules
./env/modules_cpu.sh

# activate environment
./shell_scripts/activate_cpu.sh --cpu

# download data with python script
python ./py_scripts/download.py --download_config $DOWNLOAD_CONFIG \
                                --host_config $HOST_CONFIG