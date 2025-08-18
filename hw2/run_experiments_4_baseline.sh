#!/bin/bash
# --- Run Baseline Experiments for InvertedPendulum with 5 seeds ---

echo "========== Running Experiment 4: Baseline for InvertedPendulum =========="

# --- Define Seeds ---
SEEDS=(1 2 3 4 5)

# --- GPU Management ---
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# --- Logging Setup ---
LOG_DIR="data/pendulum_baseline_logs"
mkdir -p "$LOG_DIR"
echo "Starting 5 baseline experiments... Logs will be saved in '$LOG_DIR'"

# --- Main Loop ---
for i in "${!SEEDS[@]}"; do
    # Assign a GPU for this run
    GPU_ID=${GPUS[$((i % NUM_GPUS))]}
    SEED=${SEEDS[$i]}
    
    EXP_NAME="pendulum_default_s${SEED}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    echo "Launching experiment '$EXP_NAME' on GPU $GPU_ID. Log: $LOG_FILE"
    
    python cs285/scripts/run_hw2.py \
        --env_name InvertedPendulum-v4 \
        -n 100 \
        --exp_name "$EXP_NAME" \
        -rtg --use_baseline -na \
        --batch_size 5000 \
        --seed "$SEED" \
        --video_log_freq 5 \
        --which_gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &
done

# Wait for all background jobs to finish
wait

echo "All 5 baseline experiments have finished."