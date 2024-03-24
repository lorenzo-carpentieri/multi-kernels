#!/bin/bash

benches=("ace" "aop" "bh" "metropolis" "mnist" "srad")
curr_benches="ace,aop,bh,metropolis,mnist,srad"
sampling=3
num_runs=5
log_dir=""

function help {
  echo "Usage: run_phase.sh [OPTIONS]"
  echo "Options:"
  echo "  --benchmarks=ace,aop,bh,metropolis,mnist,srad"
  echo "  --num-runs=5"
  echo "  -o, --output-dir"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks=*)
      curr_benches="${1#*=}"
      shift
      ;;
    --num-runs=*)
      num_runs="${1#*=}"
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
CONF_DIR=$SCRIPT_DIR/config

cd $EXEC_DIR

for bench in $(echo $curr_benches | tr "," "\n"); do
  mkdir -p $LOG_DIR/$bench
  rm -f $LOG_DIR/$bench/*.dat
done
for it in $(seq 1 $num_runs); do
  # ACE
  if [[ $curr_benches == *"ace"* ]]; then
  echo "[*] Running ACE"
  num_runs=1
  cat $CONF_DIR/app/ace.conf | (./ace_main $num_runs >> $LOG_DIR/ace/ace_app.dat 2> $LOG_DIR/ace/ace_app$it.log)
  cat $CONF_DIR/phase/ace.conf | (./ace_main $num_runs >> $LOG_DIR/ace/ace_phase.dat 2> $LOG_DIR/ace/ace_phase$it.log)
  cat $CONF_DIR/kernel/ace.conf | (./ace_main $num_runs >> $LOG_DIR/ace/ace_kernel.dat 2> $LOG_DIR/ace/ace_kernel$it.log)
  fi

  if [[ $curr_benches == *"aop"* ]]; then
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
  cat $CONF_DIR/app/aop.conf | ./aop_main -timesteps $timesteps -paths $num_paths -runs $num_runs -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put >> $LOG_DIR/aop/aop_app.dat 2> $LOG_DIR/aop/aop_app$it.log
  cat $CONF_DIR/phase/aop.conf | ./aop_main -timesteps $timesteps -paths $num_paths -runs $num_runs -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put >> $LOG_DIR/aop/aop_phase.dat 2> $LOG_DIR/aop/aop_phase$it.log
  cat $CONF_DIR/kernel/aop.conf | ./aop_main -timesteps $timesteps -paths $num_paths -runs $num_runs -T $T -S0 $S0 -K $K -r $r -sigma $sigma $price_put >> $LOG_DIR/aop/aop_kernel.dat 2> $LOG_DIR/aop/aop_kernel$it.log
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
  cat $CONF_DIR/app/metropolis.conf | ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed >> $LOG_DIR/metropolis/metropolis_app.dat 2> $LOG_DIR/metropolis/metropolis_app$it.log
  cat $CONF_DIR/phase/metropolis.conf | ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed >> $LOG_DIR/metropolis/metropolis_phase.dat 2> $LOG_DIR/metropolis/metropolis_phase$it.log
  cat $CONF_DIR/kernel/metropolis.conf | ./metropolis_main -l $L $R -t $TR $dT -h $h -a $atrials $ains $apts $ams -z $seed >> $LOG_DIR/metropolis/metropolis_kernel.dat 2> $LOG_DIR/metropolis/metropolis_kernel$it.log
  fi

  if [[ $curr_benches == *"mnist"* ]]; then
  echo "[*] Running MNIST"
  num_iters=1 # 1 
  cat $CONF_DIR/app/mnist.conf | ./mnist_main $num_iters >> $LOG_DIR/mnist/mnist_app.dat 2> $LOG_DIR/mnist/mnist_app$it.log
  cat $CONF_DIR/phase/mnist.conf | ./mnist_main $num_iters >> $LOG_DIR/mnist/mnist_phase.dat 2> $LOG_DIR/mnist/mnist_phase$it.log
  cat $CONF_DIR/kernel/mnist.conf | ./mnist_main $num_iters >> $LOG_DIR/mnist/mnist_kernel.dat 2> $LOG_DIR/mnist/mnist_kernel$it.log
  fi

  # Srad
  if [[ $curr_benches == *"srad"* ]]; then
  echo "[*] Running SRAD"
  num_iters=100
  lambda=1
  number_of_rows=16384 #16384 #512
  number_of_cols=16384 #16384 #512
  cat $CONF_DIR/app/srad.conf | ./srad_main $num_iters $lambda $number_of_rows $number_of_cols >> $LOG_DIR/srad/srad_app.dat 2> $LOG_DIR/srad/srad_app$it.log
  cat $CONF_DIR/phase/srad.conf | ./srad_main $num_iters $lambda $number_of_rows $number_of_cols >> $LOG_DIR/srad/srad_phase.dat 2> $LOG_DIR/srad/srad_phase$it.log
  cat $CONF_DIR/kernel/srad.conf | ./srad_main $num_iters $lambda $number_of_rows $number_of_cols >> $LOG_DIR/srad/srad_kernel.dat 2> $LOG_DIR/srad/srad_kernel$it.log
  fi
done

# Finalize
cd $SCRIPT_DIR