#!/bin/bash
#PBS -N ace
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/multi-kernels/pbs-out/output_ace.txt
#PBS -e /home/lcarpent/energy-workspace/multi-kernels/pbs-out/error_ace.txt
#PBS -q debug

# Load required modules or source oneAPI environment
module load geopm-runtime
source /opt/aurora/24.347.0/oneapi/setvars.sh

# /home/lcarpent/energy-workspace/SYnergy/build/samples/matrix_mul  
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

app_name=$(basename "$path_app")
echo "Executing $path_app ..."  # Output: app_name
cat $CONFIG_FILE_PATH | ${path_app} $num_runs > $LOG_DIR/ace_${core_freq}_run${run}.csv 2> $LOG_DIR/ace_${core_freq}_run${run}.log
