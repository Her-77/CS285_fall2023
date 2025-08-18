#!/bin/bash

echo "========== Running Experiment 5: Humanoid-v4 =========="

# --- Logging Setup ---
EXP_NAME="humanoid"
LOG_DIR="data/humanoid_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "Starting Humanoid-v4 experiment..."
echo "Output will be saved to $LOG_FILE"

# --- Run the experiment ---
python cs285/scripts/run_hw2.py \
    --env_name Humanoid-v4 \
    --ep_len 1000 \
    --discount 0.99 \
    -n 1000 \
    -l 3 \
    -s 256  \
    -b 50000 \
    -lr 0.001 \
    --baseline_gradient_steps 50 \
    -na \
    -rtg \
    --use_baseline \
    --gae_lambda 0.97 \
    --exp_name "$EXP_NAME" \
    --video_log_freq 5 \
    --which_gpu 7 > "$LOG_FILE" 2>&1 &

# The 'wait' command will pause the script until the background job has finished.
wait

echo "Experiment 5: Humanoid-v4 experiment has finished."