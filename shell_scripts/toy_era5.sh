#!/bin/bash
#SBATCH --cpus-per-task=11
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/toy_era5/%j.out
#SBATCH --error=./logs/toy_era5/%j.out
#SBATCH --job-name=small_era5
#SBATCH --partition=himem_4tb

module load 2023
module load CDO/2.2.2-gompi-2023a

# find all the variables as /projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/${variable}_2021-2022-6h-1440x721.nc
variables="msl q sst t t2m tp u10 u v10 v z"

mkdir -p tmp_data

for variable in ${variables}; 
do
    echo "Processing ${variable}"
    input=/projects/prjs0981/ewalt/aurora_benchmark/data/era5_wb2/2021-2022-6h-1444x721/${variable}_2021-2022-6h-1440x721.nc
    output=tmp_data/${variable}-2021-2022-1d-360x180.nc
    cdo -f nc -b F32 -remapbil,r360x180 -daymean $input $output &
done

wait
