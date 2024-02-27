#!/bin/bash

benches=("ace" "aop" "bh" "metropolis" "mnist" "srad")
curr_benches="ace,aop,bh,metropolis,mnist,srad"
sampling=3
log_dir=""

# define help function
function help {
  echo "Usage: run_profiling.sh [OPTIONS]"
  echo "Options:"
  echo "  --benchmarks=ace,aop,bh,metropolis,mnist,srad"
  echo "  -o, --output-dir"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks=*)
      curr_benches="${1#*=}"
      shift
      ;;
    -o | --output-dir)
      log_dir=$2
      shift
      shift
      ;;
    -h | --help)
      help
      return 0 2>/dev/null
      exit 0
      ;;
    *)
    echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

# check log dir
if [ -z "$log_dir" ]; then
  echo "Output directory not specified"
  exit 1
fi

# check if selected benchmarks are valid
for bench in $(echo $curr_benches | tr "," "\n")
do
  if [[ ! " ${benches[@]} " =~ " ${bench} " ]]; then
    echo "Invalid benchmark: $bench"
    echo "Valid benchmarks: ${benches[@]}"
    exit 1
  fi
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXEC_DIR=$SCRIPT_DIR/build
cd $log_dir
LOG_DIR=$(pwd)

cd $EXEC_DIR

# only for nvidia gpu, for intel gpu we have a min and max 
def_core=""
def_mem=""
mem_frequencies=""
core_frequencies=""

# Get default core and memory frequency 
mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)
nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')
def_mem=$(echo $nvsmi_out | awk '{print $7}')

sampled_freq=()
i=-1
for core_freq in $core_frequencies; do
  i=$((i+1))
  if [ $((i % sampling)) != 0 ]
  then
    continue
  fi
  sampled_freq+=($core_freq)
done
mem_freq=$def_mem
for core_freq in "${sampled_freq[@]}"; do
  
  geopmwrite GPU_CORE_FREQUENCY_MIN_CONTROL gpu 0 "${core_freq}000000"
  geopmwrite GPU_CORE_FREQUENCY_MAX_CONTROL gpu 0 "${core_freq}000000"


  echo "[*] core_freq:  $core_freq"

  # ACE
  mkdir -p $LOG_DIR/ace/
  if [[ $curr_benches == *"ace"* ]]; then
    echo "[*] Running ACE"
    num_runs=1
    ./ace_main $num_runs > $LOG_DIR/ace/ace_${mem_freq}_${core_freq}.csv 2> $LOG_DIR/ace/ace_${mem_freq}_${core_freq}.log
  fi

  # AOP TODO: fix energy consumption
  mkdir -p $LOG_DIR/aop/

  if [[ $curr_benches == *"aop"* ]]; then
    echo "[*] Running AOP"
    timesteps=50 # 100
    num_paths=24576 # 32
    num_runs=1 # 1
    T=1.0 # 1.0
    K=4.0 # 4.0
    S0=3.60 # 3.60
    r=0.06 # 0.06
    sigma=0.2 # 0.2
    price_put="-call"
    ./aop_main -timesteps $timesteps -paths $num_paths -runs $num_runs\
      -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put > $LOG_DIR/aop/aop_${mem_freq}_${core_freq}.csv 2> $LOG_DIR/aop/aop_${mem_freq}_${core_freq}.log
  fi

  mkdir -p $LOG_DIR/metropolis/
  # Metropolis
  if [[ $curr_benches == *"metropolis"* ]]; then
    echo "[*] Running Metropolis"
    L=1024 # 32
    R=1 # 1
    atrials=1 # 1
    ains=1 # 1
    apts=1 # 1
    ams=1 # 1
    seed=2 # 2
    TR=0.1 # 0.1
    dT=0.1 # 0.1
    h=0.1 # 0.1
    ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed > $LOG_DIR/metropolis/metropolis_${mem_freq}_${core_freq}.csv 2> $LOG_DIR/metropolis/metropolis_${mem_freq}_${core_freq}.log
  fi

  mkdir -p $LOG_DIR/mnist
  # Mnist TODO: fix energy consumption
  if [[ $curr_benches == *"mnist"* ]]; then
    echo "[*] Running MNIST"
    num_iters=1 # 1 
    ./mnist_main $num_iters > $LOG_DIR/mnist/mnist_${mem_freq}_${core_freq}.csv 2> $LOG_DIR/mnist/mnist_${mem_freq}_${core_freq}.log
  fi

  mkdir -p $LOG_DIR/srad/
  # Srad
  if [[ $curr_benches == *"srad"* ]]; then
    echo "[*] Running SRAD"
    num_iters=1
    lambda=1
    number_of_rows=1024 #512
    number_of_cols=1024 #512
    ./srad_main $num_iters $lambda $number_of_rows $number_of_cols > $LOG_DIR/srad/srad_${mem_freq}_${core_freq}.csv 2> $LOG_DIR/srad/srad_${mem_freq}_${core_freq}.log
  fi
done
# Finalize
cd $SCRIPT_DIR