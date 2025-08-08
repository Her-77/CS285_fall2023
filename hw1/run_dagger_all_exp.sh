#!/bin/bash

# --- 全局配置 ---
# 定义所有要运行实验的环境名称
ENVS=("Ant-v4" "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4")
MAX_ITER=20 # DAgger 迭代的最大次数

# --- 主循环：遍历所有环境 ---
for ENV_NAME in "${ENVS[@]}"; do
    # 从 "Ant-v4" 中提取 "Ant"
    ENV_NAME_SIMPLE=$(echo $ENV_NAME | sed 's/-v4//')

    echo "======================================================"
    echo "Starting DAgger Experiment for: ${ENV_NAME_SIMPLE}"
    echo "======================================================"

    # --- 动态文件名配置 ---
    EXPERT_POLICY="cs285/policies/experts/${ENV_NAME_SIMPLE}.pkl"
    EXPERT_DATA="cs285/expert_data/expert_data_${ENV_NAME}.pkl"
    DAGGER_RESULTS_FILE="dagger_${ENV_NAME_SIMPLE}_results.csv"
    BASELINE_RESULTS_FILE="baseline_${ENV_NAME_SIMPLE}_results.csv"
    TEMP_LOG_FILE="dagger_temp_${ENV_NAME_SIMPLE}.log"

    # --- 步骤 1: 计算基准性能 (专家 和 BC) ---
    echo "--- [${ENV_NAME_SIMPLE}] Step 1: Calculating Expert and BC baselines ---"

    # 使用内联Python脚本计算专家回报
    GET_EXPERT_RETURN_PY="
import pickle
import numpy as np
try:
    paths = pickle.load(open('$EXPERT_DATA', 'rb'))
    returns = [path['reward'].sum() for path in paths]
    print(f'{np.mean(returns)},{np.std(returns)}')
except FileNotFoundError:
    print('0,0') # 如果专家数据不存在，返回0
"
    expert_stats=$(conda run -n cs285 python -c "$GET_EXPERT_RETURN_PY")
    expert_return=$(echo $expert_stats | cut -d ',' -f 1)
    expert_std=$(echo $expert_stats | cut -d ',' -f 2)

    # 运行纯BC实验 (n_iter=1)
    conda run -n cs285 python cs285/scripts/run_hw1.py \
        --env_name $ENV_NAME \
        --expert_policy_file $EXPERT_POLICY \
        --expert_data $EXPERT_DATA \
        --exp_name "bc_${ENV_NAME_SIMPLE}_baseline" \
        --n_iter 1 \
        --num_agent_train_steps_per_iter 1000 \
        --eval_batch_size 5000 \
        --ep_len 1000 > $TEMP_LOG_FILE

    bc_return=$(grep "Eval_AverageReturn" $TEMP_LOG_FILE | awk '{print $3}')
    bc_std=$(grep "Eval_StdReturn" $TEMP_LOG_FILE | awk '{print $3}')

    # 保存基准数据到CSV文件
    echo "policy,return,std" > $BASELINE_RESULTS_FILE
    echo "expert,${expert_return},${expert_std}" >> $BASELINE_RESULTS_FILE
    echo "bc,${bc_return},${bc_std}" >> $BASELINE_RESULTS_FILE
    echo "[${ENV_NAME_SIMPLE}] Expert Return: $expert_return, BC Return: $bc_return"

    # --- 步骤 2: 运行 DAgger 实验 ---
    echo "--- [${ENV_NAME_SIMPLE}] Step 2: Running DAgger experiment for ${MAX_ITER} iterations ---"
    conda run -n cs285 python cs285/scripts/run_hw1.py \
        --env_name $ENV_NAME \
        --expert_policy_file $EXPERT_POLICY \
        --expert_data $EXPERT_DATA \
        --exp_name "dagger_${ENV_NAME_SIMPLE}_exp" \
        --do_dagger \
        --n_iter $MAX_ITER \
        --num_agent_train_steps_per_iter 1000 \
        --eval_batch_size 5000 \
        --ep_len 1000 > $TEMP_LOG_FILE

    # --- 步骤 3: 处理 DAgger 结果 ---
    echo "--- [${ENV_NAME_SIMPLE}] Step 3: Processing DAgger results ---"
    echo "iteration,eval_return,eval_std" > $DAGGER_RESULTS_FILE
    
    # 从日志中提取数据并格式化为CSV
    eval_returns=$(grep "Eval_AverageReturn" $TEMP_LOG_FILE | awk '{print $3}')
    eval_stds=$(grep "Eval_StdReturn" $TEMP_LOG_FILE | awk '{print $3}')
    iterations=$(seq 0 $((${MAX_ITER}-1)))

    paste -d ',' <(echo "$iterations") <(echo "$eval_returns") <(echo "$eval_stds") | sed 's/\r//g' >> $DAGGER_RESULTS_FILE

    # --- 步骤 4: 绘图 ---
    echo "--- [${ENV_NAME_SIMPLE}] Step 4: Generating plot ---"
    conda run -n cs285 python plot_dagger_results.py --env $ENV_NAME

    # --- 清理 ---
    rm $TEMP_LOG_FILE

done

# --- 结束 ---
echo "======================================================"
echo "All DAgger experiments and plotting finished successfully!"
echo "======================================================"
