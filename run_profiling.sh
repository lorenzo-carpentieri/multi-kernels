#!/bin/bash

ARCHS=("cuda" "rocm" "lz" "geopm")
arch=""
benches=("ace" "aop" "bh" "metropolis" "mnist" "srad")
curr_benches="ace,aop,bh,metropolis,mnist,srad"
sampling=5
log_dir=""

# define help function
function help {
  echo "Usage: run_profiling.sh -o out_dir [OPTIONS]"
  echo "Options:"
  echo "  --benchmarks=ace,aop,bh,metropolis,mnist,srad"
  echo "  -o, --output-dir"
  echo "  -h, --help"
}

# Function to get index of an element in a list
function get_index() {
    local element=$1
    shift
    local list_string=("$@")
    IFS=' ' read -r -a list <<< "$list_string"
    index=-1

    for i in "${!list[@]}"; do
        if [ "${list[$i]}" = "$element" ]; then
	    index=$i
            break
        fi
    done
    echo "$index"
}

# Function to set gpu frequency
function set_gpu_frequency() {
  local frequency=$1
  echo "[*] Setting Core Freq:  $frequency MHz"

  if [ "$arch" = "rocm" ]; then
    idx=$(get_index "$frequency" "${core_frequencies[@]}")
    if [ "$frequency" = "-1" ]; then
      echo "Invalid frequency: $frequency"
      exit 1
    fi
    rocm-smi --setsclk ${idx} > /dev/null
  elif [ "$arch" = "cuda" ]; then
    nvidia-smi -lgc $frequency > /dev/null
  elif [ "$arch" = "lz" ]; then
    intel_gpu_frequency -s $frequency
  fi
}

# Function to reset gpu frequency
function reset_gpu_frequency() {
  if [ "$arch" = "rocm" ]; then
    rocm-smi -r > /dev/null
  elif [ "$arch" = "cuda" ]; then
    nvidia-smi -rgc > /dev/null
  elif [ "$arch" = "lz" ]; then
    intel_gpu_frequency -d
  fi
}

# Function to get core frequencies
function get_core_frequencies {
  if [ "$arch" = "cuda" ]; then
    core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)
  elif [ "$arch" = "rocm" ]; then
    core_frequencies="300 495 731 962 1029 1087 1147 1189 1235 1283 1319 1363 1404 1430 1472 1502" # TODO: make it dynamic
  elif [ "$arch" = "lz" ]; then
    core_frequencies="200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550"
  fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      arch=$2
      shift
      shift
      ;;
    --benchmarks=*)
      curr_benches="${1#*=}"
      shift
      ;;
    --sampling=*)
      sampling="${1#*=}"
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

# check if selected arch is valid
for a in "${ARCHS[@]}"; do
  if [ "$a" == "$arch" ]; then
    valid_arch=1
    break
  fi
done
if [ -z "$valid_arch" ]; then
  echo "Invalid architecture: $arch"
  echo "Valid architectures: ${ARCHS[@]}"
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
core_frequencies=""

# Get default core and memory frequency 
get_core_frequencies

# Sample core frequencies
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

# Run benchmarks
for core_freq in "${sampled_freq[@]}"; do

  set_gpu_frequency $core_freq

  # ACE
  if [[ $curr_benches == *"ace"* ]]; then
    mkdir -p $LOG_DIR/ace/
    echo "[*] Running ACE"
    num_runs=1
    ./ace_main $num_runs > $LOG_DIR/ace/ace_${core_freq}.csv 2> $LOG_DIR/ace/ace_${core_freq}.log
  fi

  # AOP TODO: fix energy consumption
  if [[ $curr_benches == *"aop"* ]]; then
    mkdir -p $LOG_DIR/aop/
    echo "[*] Running AOP"
    timesteps=50 # 100
    num_paths=128 # 32
    num_runs=1 # 1
    T=1.0 # 1.0
    K=4.0 # 4.0
    S0=3.60 # 3.60
    r=0.06 # 0.06
    sigma=0.2 # 0.2
    price_put="-call"
    ./aop_main -timesteps $timesteps -paths $num_paths -runs $num_runs\
      -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put > $LOG_DIR/aop/aop_${core_freq}.csv 2> $LOG_DIR/aop/aop_${core_freq}.log
  fi

  # Metropolis
  if [[ $curr_benches == *"metropolis"* ]]; then
    mkdir -p $LOG_DIR/metropolis/
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
    ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed > $LOG_DIR/metropolis/metropolis_${core_freq}.csv 2> $LOG_DIR/metropolis/metropolis_${core_freq}.log
  fi

  # Mnist TODO: fix energy consumption
  if [[ $curr_benches == *"mnist"* ]]; then
    mkdir -p $LOG_DIR/mnist
    echo "[*] Running MNIST"
    num_iters=1 # 1 
    ./mnist_main $num_iters > $LOG_DIR/mnist/mnist_${core_freq}.csv 2> $LOG_DIR/mnist/mnist_${core_freq}.log
  fi

  # Srad
  if [[ $curr_benches == *"srad"* ]]; then
    mkdir -p $LOG_DIR/srad/
    echo "[*] Running SRAD"
    num_iters=100
    lambda=1
    number_of_rows=2048 #512
    number_of_cols=2048 #512
    ./srad_main $num_iters $lambda $number_of_rows $number_of_cols > $LOG_DIR/srad/srad_${core_freq}.csv 2> $LOG_DIR/srad/srad_${core_freq}.log
  fi
done

# Finalize
cd $SCRIPT_DIR
reset_gpu_frequency