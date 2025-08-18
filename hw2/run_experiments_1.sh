#!/bin/bash

# --- Parallel CartPole Experiments on an 8-GPU Server ---

echo "========== Running Experiment 1 (CartPole) =========="

echo "Starting 8 parallel experiments..."

# Define the root directory for log files
LOG_ROOT="data"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_ROOT"

echo "Starting 8 parallel experiments... Logs will be saved in the '$LOG_ROOT' directory."

# --- Small Batch Experiments (b=1000) ---

# Experiment 1: No RTG, No NA -> on GPU 0
echo "Launching 'cartpole' on GPU 0. Log: $LOG_ROOT/cartpole.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    --exp_name cartpole --which_gpu 0 > "$LOG_ROOT/cartpole.log" 2>&1 &

# Experiment 2: With RTG -> on GPU 1
echo "Launching 'cartpole_rtg' on GPU 1. Log: $LOG_ROOT/cartpole_rtg.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg \
    --exp_name cartpole_rtg --which_gpu 1 > "$LOG_ROOT/cartpole_rtg.log" 2>&1 &

# Experiment 3: With NA -> on GPU 2
echo "Launching 'cartpole_na' on GPU 2. Log: $LOG_ROOT/cartpole_na.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na \
    --exp_name cartpole_na --which_gpu 2 > "$LOG_ROOT/cartpole_na.log" 2>&1 &

# Experiment 4: With RTG, With NA -> on GPU 3
echo "Launching 'cartpole_rtg_na' on GPU 3. Log: $LOG_ROOT/cartpole_rtg_na.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na \
    --exp_name cartpole_rtg_na --which_gpu 3 > "$LOG_ROOT/cartpole_rtg_na.log" 2>&1 &


# --- Large Batch Experiments (b=4000) ---

# Experiment 5: No RTG, No NA -> on GPU 4
echo "Launching 'cartpole_lb' on GPU 4. Log: $LOG_ROOT/cartpole_lb.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
    --exp_name cartpole_lb --which_gpu 4 > "$LOG_ROOT/cartpole_lb.log" 2>&1 &

# Experiment 6: With RTG -> on GPU 5
echo "Launching 'cartpole_lb_rtg' on GPU 5. Log: $LOG_ROOT/cartpole_lb_rtg.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg \
    --exp_name cartpole_lb_rtg --which_gpu 5 > "$LOG_ROOT/cartpole_lb_rtg.log" 2>&1 &

# Experiment 7: With NA -> on GPU 6
echo "Launching 'cartpole_lb_na' on GPU 6. Log: $LOG_ROOT/cartpole_lb_na.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na \
    --exp_name cartpole_lb_na --which_gpu 6 > "$LOG_ROOT/cartpole_lb_na.log" 2>&1 &

# Experiment 8: With RTG, With NA -> on GPU 7
echo "Launching 'cartpole_lb_rtg_na' on GPU 7. Log: $LOG_ROOT/cartpole_lb_rtg_na.log"
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na \
    --exp_name cartpole_lb_rtg_na --which_gpu 7 > "$LOG_ROOT/cartpole_lb_rtg_na.log" 2>&1 &


# The 'wait' command will pause the script until all background jobs have finished.
wait

echo "All 8 experiments have finished."
echo "Check the .log files in the '$LOG_ROOT' directory for output."

echo "========== Finished Question 1 =========="