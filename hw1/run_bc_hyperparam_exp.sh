#!/bin/bash

# 1. 配置区 (Configuration Section)
# 定义固定的实验参数，方便修改和阅读
ENV_NAME="Walker2d-v4"
EXPERT_POLICY="cs285/policies/experts/Walker2d.pkl"
EXPERT_DATA="cs285/expert_data/expert_data_Walker2d-v4.pkl"
RESULTS_FILE="bc_walker_hyperparam_results.csv"
TEMP_LOG_FILE="temp_log.txt"

# 2. 超参数定义区 (Hyperparameter Definition)
# 定义一个数组，包含所有要测试的 num_agent_train_steps_per_iter 的值
TRAIN_STEPS_VALUES=(100 500 1000 2000 5000 10000)

# 3. 初始化结果文件 (Initialize Output File)
# 创建或清空结果文件，并写入CSV表头，确保每次运行都是全新的结果
echo "train_steps,eval_return,eval_std" > $RESULTS_FILE

# 4. 主循环 (Main Loop)
# 遍历超参数数组中的每一个值
for steps in "${TRAIN_STEPS_VALUES[@]}"; do
    # a. 构造唯一的实验名称，以防日志文件冲突
    exp_name="bc_walker_steps_${steps}"

    # b. 打印进度信息，告知用户当前实验状态
    echo "--- Running BC on ${ENV_NAME} with ${steps} training steps ---"

    # c. 构造并执行命令
    # 运行Python训练脚本，并将所有输出重定向到临时日志文件
    python cs285/scripts/run_hw1.py \
        --env_name $ENV_NAME \
        --expert_policy_file $EXPERT_POLICY \
        --expert_data $EXPERT_DATA \
        --exp_name $exp_name \
        --num_agent_train_steps_per_iter $steps \
        --n_iter 1 \
        --eval_batch_size 5000 \
        --ep_len 1000 > $TEMP_LOG_FILE

    # d. 解析结果
    # 从临时日志文件中提取关键性能指标
    eval_return=$(grep "Eval_AverageReturn" $TEMP_LOG_FILE | awk '{print $3}')
    eval_std=$(grep "Eval_StdReturn" $TEMP_LOG_FILE | awk '{print $3}')

    # e. 记录结果
    # 将超参数和对应的性能指标追加到CSV文件中
    echo "${steps},${eval_return},${eval_std}" >> $RESULTS_FILE

    # f. 清理
    # 删除临时日志文件
    rm $TEMP_LOG_FILE

done

# 5. 结束
# 打印完成信息
echo "--- All experiments finished. Results saved to ${RESULTS_FILE} ---"
