#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=./logs/toy_era5/%A_%a.out
#SBATCH --error=./logs/toy_era5/%A_%a.out
#SBATCH --job-name=small_era5
#SBATCH --partition=himem_4tb
#SBATCH --array=1-11

module load 2023
module load CDO/2.2.2-gompi-2023a

# find all the variables as /projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/${variable}_2021-2022-6h-1440x721.nc
variables=(msl q sst t t2m tp u10 u v10 v z)

mkdir -p ./toy_data/era5-1d-360x180/

variable=${variables[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "Processing ${variable}"

input=/projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/${variable}_2021-2022-6h-1440x721.nc
output=./toy_data/era5-1d-360x180/${variable}-2021-2022-1d-360x180.nc
cdo -f nc -b F32 -daymean -remapbil,r360x180 $input $output
