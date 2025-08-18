#!/bin/bash
# --- Comprehensive Hyperparameter Search for InvertedPendulum ---

echo "========== Running Experiment 4: Comprehensive Hyperparameter Search =========="

# --- Define Hyperparameter Search Space ---
BATCH_SIZES=(1000 5000 10000)
ACTOR_LR_VALUES=(0.005 0.01 0.02)
CRITIC_LR_VALUES=(0.01 0.02)      # Critic learning rate
CRITIC_STEPS_VALUES=(5 10)       # Critic gradient steps

# --- GPU Management ---
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
GPU_COUNTER=0

# --- Logging Setup ---
LOG_DIR="data/pendulum_full_search_logs"
mkdir -p "$LOG_DIR"
echo "Starting comprehensive search... Logs will be saved in '$LOG_DIR'"

# --- Main Loop ---
for b_size in "${BATCH_SIZES[@]}"; do
    for actor_lr in "${ACTOR_LR_VALUES[@]}"; do
        for critic_lr in "${CRITIC_LR_VALUES[@]}"; do
            for critic_steps in "${CRITIC_STEPS_VALUES[@]}"; do
                # Assign a GPU for this run
                GPU_ID=${GPUS[$((GPU_COUNTER % NUM_GPUS))]}
                
                # Create a descriptive experiment name
                EXP_NAME="pendulum_b${b_size}_lr${actor_lr}_blr${critic_lr}_bgs${critic_steps}"
                LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
                
                echo "Launching experiment '$EXP_NAME' on GPU $GPU_ID. Log: $LOG_FILE"
                
                python cs285/scripts/run_hw2.py \
                    --env_name InvertedPendulum-v4 \
                    -n 100 \
                    -rtg --use_baseline -na \
                    --batch_size "$b_size" \
                    --learning_rate "$actor_lr" \
                    --baseline_learning_rate "$critic_lr" \
                    --baseline_gradient_steps "$critic_steps" \
                    --seed 1 \
                    --exp_name "$EXP_NAME" \
                    --which_gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &
                
                # Increment GPU counter
                GPU_COUNTER=$((GPU_COUNTER + 1))

                # Optional: Add a small delay to prevent all processes from starting at the exact same microsecond
                sleep 0.5 
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All hyperparameter search experiments have finished."