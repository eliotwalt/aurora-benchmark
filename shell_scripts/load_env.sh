#!/bin/bash

# get flag option
# --cpu: load cpu modules
# --gpu: load gpu modules

flag_provided=""

while [ "$1" != "" ]; do
    case $1 in
        --cpu )             shift
                            echo "Loading CPU modules"
                            flag_provided="cpu"
                            ./env/modules_cpu.sh
                            source ./env/venv_cpu/bin/activate
                            ;;
        --gpu )             shift
                            echo "Loading GPU modules"
                            flag_provided="gpu"
                            ./env/modules_gpu.sh
                            source ./env/venv_gpu/bin/activate
                            ;;
        * )                 echo "Invalid option"
                            exit 1
    esac
    shift
done

# ensure at least --gpu or --cpu is provided
if [ -z "$flag_provided" ]; then
    echo "Please provide at least one of the following options: --cpu, --gpu"
    exit 1
fi

