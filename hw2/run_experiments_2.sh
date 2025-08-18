#!/bin/bash

# --- Parallel CartPole Experiments on an 8-GPU Server ---

echo "========== Running Experiment 2 (HalfCheetah) =========="

echo "Starting 2 parallel experiments..."

# Define the root directory for log files
LOG_ROOT="data"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_ROOT"

echo "Starting 2 parallel experiments... Logs will be saved in the '$LOG_ROOT' directory."

# --- Small Batch Experiments (b=1000) ---

# Experiment 1: No RTG, No NA -> on GPU 0
echo "Launching 'HalfCheetah' on GPU 0. Log: $LOG_ROOT/HalfCheetah.log"
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01\
    --exp_name cheetah --which_gpu 0 > "$LOG_ROOT/HalfCheetah.log" 2>&1 &

# Experiment 2: With RTG -> on GPU 1
echo "Launching 'HalfCheetah' on GPU 1. Log: $LOG_ROOT/HalfCheetah_baseline.log"
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01\
    --use_baseline -blr 0.01 -bgs 5 \
    --exp_name cheetah_baseline --which_gpu 1 > "$LOG_ROOT/HalfCheetah_baseline.log" 2>&1 &

# The 'wait' command will pause the script until all background jobs have finished.
wait

echo "All 2 experiments have finished."
echo "Check the .log files in the '$LOG_ROOT' directory for output."

echo "========== Finished Question 2 =========="