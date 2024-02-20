#!/bin/bash

benches=("ace" "aop" "bh" "metropolis" "mnist" "srad")
curr_benches="ace,aop,bh,metropolis,mnist,srad"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks=*)
      curr_benches="${1#*=}"
      shift
      ;;
    *)
    echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

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
LOG_DIR=$SCRIPT_DIR/logs

cd $EXEC_DIR

mkdir -p $LOG_DIR
# ACE
if [[ $curr_benches == *"ace"* ]]; then
  echo "[*] Running ACE"
  num_runs=1
  ./ace_main $num_runs > $LOG_DIR/ace.csv 2> $LOG_DIR/ace.log
fi

# AOP TODO: fix energy consumption
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
    -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put > $LOG_DIR/aop.csv 2> $LOG_DIR/aop.log
fi

# BH TODO: Fix energy consumption
if [[ $curr_benches == *"bh"* ]]; then
  echo "[*] Running BH"
  number_of_bodies=1
  number_of_timesteps=1
  ./bh_main $number_of_bodies $number_of_timesteps > $LOG_DIR/bh.csv 2> $LOG_DIR/bh.log
fi

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
  ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed > $LOG_DIR/metropolis.csv 2> $LOG_DIR/metropolis.log
fi

# Mnist TODO: fix energy consumption
if [[ $curr_benches == *"mnist"* ]]; then
  echo "[*] Running MNIST"
  num_iters=1 # 1 
  ./mnist_main $num_iters > $LOG_DIR/mnist.csv 2> $LOG_DIR/mnist.log
fi

# Srad
if [[ $curr_benches == *"srad"* ]]; then
  echo "[*] Running SRAD"
  num_iters=1
  lambda=1
  number_of_rows=16384 # 512
  number_of_cols=16384 # 512
  ./srad_main $num_iters $lambda $number_of_rows $number_of_cols > $LOG_DIR/srad.csv 2> $LOG_DIR/srad.log
fi

# Finalize
cd $SCRIPT_DIR