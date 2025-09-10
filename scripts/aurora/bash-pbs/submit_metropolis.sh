#!/bin/bash
#PBS -N metropolis
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/multi-kernels/pbs-out/output_metropolis.txt
#PBS -e /home/lcarpent/energy-workspace/multi-kernels/pbs-out/error_metropolis.txt
#PBS -q debug

# Load required modules or source oneAPI environment
module load geopm-runtime
source /opt/aurora/24.347.0/oneapi/setvars.sh

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

app_name=$(basename "$path_app")
echo "Executing $path_app ..."  # Output: app_name

cat $CONFIG_FILE_PATH | ${path_app} "-l" $L $R "-t" $TR $dT "-h" $h "-a" $atrials $ains $apts $ams "-z" $seed > $LOG_DIR/metropolis_${core_freq}_run${run}.csv 2> $LOG_DIR/metropolis_${core_freq}_run${run}.log
