#!/bin/bash
# --- Hyperparameter Search for HalfCheetah with Parallel GPU and Logging ---

echo "========== Running Hyperparameter Search for HalfCheetah =========="

# --- Define Hyperparameter Search Space ---
BLR_VALUES=(0.005 0.01 0.02) # Baseline Learning Rates
BGS_VALUES=(1 5 10)         # Baseline Gradient Steps

# --- GPU Management ---
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
GPU_COUNTER=0

# --- Logging Setup ---
LOG_DIR="data/hyperparam_logs"
mkdir -p "$LOG_DIR"
echo "Starting all experiments... Logs will be saved in '$LOG_DIR'"

# --- Main Loop ---
for blr in "${BLR_VALUES[@]}"; do
    for bgs in "${BGS_VALUES[@]}"; do
        
        # --- Experiment 1: Run WITHOUT Advantage Normalization ---
        GPU_ID=${GPUS[$((GPU_COUNTER % NUM_GPUS))]} # Assign a GPU
        EXP_NAME="cheetah_bgs${bgs}_blr${blr}"
        LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
        
        echo "Launching experiment '$EXP_NAME' on GPU $GPU_ID. Log: $LOG_FILE"
        python cs285/scripts/run_hw2.py \
            --env_name HalfCheetah-v4 \
            --ep_len 100 \
            --discount 0.95 \
            -n 100 \
            -b 5000 \
            -lr 0.01 \
            -rtg \
            --use_baseline \
            --baseline_learning_rate "$blr" \
            --baseline_gradient_steps "$bgs" \
            --exp_name "$EXP_NAME" \
            --which_gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &
        
        GPU_COUNTER=$((GPU_COUNTER + 1)) # Increment counter AFTER first job

        # --- Experiment 2: Run WITH Advantage Normalization ---
        GPU_ID=${GPUS[$((GPU_COUNTER % NUM_GPUS))]} # Re-assign GPU for the next experiment
        EXP_NAME_NA="cheetah_bgs${bgs}_blr${blr}_na"
        LOG_FILE_NA="$LOG_DIR/${EXP_NAME_NA}.log"
        
        echo "Launching experiment '$EXP_NAME_NA' on GPU $GPU_ID. Log: $LOG_FILE_NA"
        python cs285/scripts/run_hw2.py \
            --env_name HalfCheetah-v4 \
            --ep_len 100 \
            --discount 0.95 \
            -n 100 \
            -b 5000 \
            -lr 0.01 \
            -rtg \
            --use_baseline \
            -na \
            --baseline_learning_rate "$blr" \
            --baseline_gradient_steps "$bgs" \
            --exp_name "$EXP_NAME_NA" \
            --which_gpu "$GPU_ID" > "$LOG_FILE_NA" 2>&1 &

        GPU_COUNTER=$((GPU_COUNTER + 1)) # Increment counter AFTER second job
    done
done

# Wait for all background jobs to finish
wait

echo "All hyperparameter search experiments have finished."