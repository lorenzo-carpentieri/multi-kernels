#!/bin/bash

parse_data=false

function help {
  echo "Usage: process_phase.sh [OPTIONS]"
  echo "Options:"
  echo "  --parse"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parse)
      parse_data=true
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $parse_data = true ] ; then
    echo "[*] Parsing data..."
    python3 $SCRIPT_DIR/scripts/parse_phase_logs.py $SCRIPT_DIR/logs/phase/native $SCRIPT_DIR/parsed/phase/phase_results.csv
fi
python3 $SCRIPT_DIR/scripts/plot_phase_results.py kernels $SCRIPT_DIR/parsed/phase/phase_results.csv $SCRIPT_DIR/plot
python3 $SCRIPT_DIR/scripts/plot_phase_results.py total $SCRIPT_DIR/parsed/phase/phase_results.csv $SCRIPT_DIR/plot
