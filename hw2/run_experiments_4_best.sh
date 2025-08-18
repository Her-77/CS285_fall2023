#!/bin/bash
# --- Run Best Hyperparameters for InvertedPendulum with 5 seeds ---

echo "========== Running Experiment 4: Best Hyperparameters for InvertedPendulum =========="

# --- Define Seeds ---
SEEDS=(1 2 3 4 5)

# --- Best Hyperparameters Found ---
B_SIZE=1000
ACTOR_LR=0.02
CRITIC_LR=0.01
CRITIC_STEPS=10

# --- GPU Management ---
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# --- Logging Setup ---
LOG_DIR="data/pendulum_best_params_logs"
mkdir -p "$LOG_DIR"
echo "Starting 5 'best' experiments... Logs will be saved in '$LOG_DIR'"

# --- Main Loop ---
for i in "${!SEEDS[@]}"; do
    GPU_ID=${GPUS[$((i % NUM_GPUS))]}
    SEED=${SEEDS[$i]}
    
    EXP_NAME="pendulum_best_s${SEED}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    echo "Launching experiment '$EXP_NAME' on GPU $GPU_ID. Log: $LOG_FILE"
    
    python cs285/scripts/run_hw2.py \
        --env_name InvertedPendulum-v4 \
        -n 100 \
        -rtg --use_baseline -na \
        --batch_size "$B_SIZE" \
        --learning_rate "$ACTOR_LR" \
        --baseline_learning_rate "$CRITIC_LR" \
        --baseline_gradient_steps "$CRITIC_STEPS" \
        --seed "$SEED" \
        --exp_name "$EXP_NAME" \
        --which_gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &
done

wait
echo "All 5 'best' experiments have finished."