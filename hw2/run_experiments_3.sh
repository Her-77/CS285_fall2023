#!/bin/bash
# --- Hyperparameter Search for GAE Lambda on LunarLander-v2 ---

echo "========== Running Experiment 3 (LunarLander) =========="

# --- Define Hyperparameter Search Space ---
LAMBDA_VALUES=(0 0.95 0.98 0.99 1)

# --- GPU Management ---
# 假设我们至少有5张卡，如果不够，可以减少并行任务数或让任务在同一张卡上排队
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
GPU_COUNTER=0

# --- Logging Setup ---
LOG_DIR="data/lunar_lander_logs_video"
mkdir -p "$LOG_DIR"
echo "Starting all experiments... Logs will be saved in '$LOG_DIR'"

# --- Main Loop ---
for lambda_val in "${LAMBDA_VALUES[@]}"; do
    # Assign a GPU for this run
    GPU_ID=${GPUS[$((GPU_COUNTER % NUM_GPUS))]}
    
    EXP_NAME="lunar_lander_lambda${lambda_val}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    echo "Launching experiment '$EXP_NAME' on GPU $GPU_ID. Log: $LOG_FILE"
    
    python cs285/scripts/run_hw2.py \
        --env_name LunarLander-v2 \
        --ep_len 1000 \
        --discount 0.99 \
        -n 300 \
        -l 3 \
        -s 128 \
        -b 2000 \
        -lr 0.001 \
        --use_reward_to_go \
        --use_baseline \
        --gae_lambda "$lambda_val" \
        --exp_name "$EXP_NAME" \
        --video_log_freq 5 \
        --which_gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &

    # Increment GPU counter
    GPU_COUNTER=$((GPU_COUNTER + 1))
done

# Wait for all background jobs to finish
wait

echo "All LunarLander experiments have finished."