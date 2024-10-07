#!/bin/bash

# get flag option
# --cpu: load cpu modules
# --gpu: load gpu modules

while [ "$1" != "" ]; do
    case $1 in
        --cpu )             shift
                            source ./env/modules_cpu.sh
                            ;;
        --gpu )             shift
                            source ./env/modules_gpu.sh
                            ;;
        * )                 echo "Invalid option"
                            exit 1
    esac
    shift
done

# ensure at least --gpu or --cpu is provided
if [ -z "$1" ]; then
    echo "Please provide at least one of the following options: --cpu, --gpu"
    exit 1
fi

# activate environment
source ./env/venv/bin/activate
