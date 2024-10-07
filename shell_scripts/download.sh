#!/bin/bash

# slurm options: 1 cpu, 32gb memory, 6h, log/error file at ./logs/{timestamp}/logs.log, job-name "dl_era5_wb2" , partition staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/%J/logs.log
#SBATCH --error=./logs/%J/logs.log
#SBATCH --job-name=dl_era5_wb2
#SBATCH --partition=staging

# load modules
./env/modules_cpu.sh

# activate environment
./shell_scripts/activate_cpu.sh --cpu

# download data with python script
python ./py_scripts/download.py --download_config ./configs/download_era5_wb2_1979-2022-6h-1444x721.yaml \
                                --host_config ./configs/snellius.yaml