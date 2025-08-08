
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- 1. 设置命令行参数解析 ---
# 这样脚本就可以从外部接收环境名称，例如 "Ant" 或 "Hopper"
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='The environment name (e.g., Ant, Walker2d) to plot results for.')
args = parser.parse_args()

env_name_simple = args.env.replace("-v4", "")

# --- 2. 定义文件名 ---
# 根据环境名称动态生成输入和输出文件名
dagger_results_file = f"dagger_{env_name_simple}_results.csv"
baseline_results_file = f"baseline_{env_name_simple}_results.csv"
output_plot_file = f"dagger_{env_name_simple}_plot.png"

# --- 3. 读取数据 ---
# 读取 DAgger 实验结果和基准性能数据
try:
    dagger_data = pd.read_csv(dagger_results_file)
    baseline_data = pd.read_csv(baseline_results_file).set_index('policy')
except FileNotFoundError as e:
    print(f"Error: Could not find result file {e.filename}. Please run the experiment script first.")
    exit(1)

# 提取 DAgger 数据
iters = dagger_data["iteration"]
dagger_return = dagger_data["eval_return"]
dagger_std = dagger_data["eval_std"]

# 提取基准数据
expert_return = baseline_data.loc['expert']['return']
bc_return = baseline_data.loc['bc']['return']

# --- 4. 绘图 ---
plt.style.use('seaborn-whitegrid') # 使用一个美观的绘图风格
plt.figure(figsize=(12, 7))

# 绘制 DAgger 学习曲线（带误差棒）
plt.errorbar(iters, dagger_return, yerr=dagger_std, fmt='-o', capsize=5, 
             label=f'DAgger Agent ({env_name_simple})', color='#1f77b4', markersize=8)

# 绘制专家性能基准线
plt.axhline(y=expert_return, color='#2ca02c', linestyle='--', linewidth=2, label=f'Expert Baseline')

# 绘制 BC 性能基准线
plt.axhline(y=bc_return, color='#d62728', linestyle=':', linewidth=2, label='BC Baseline')

# --- 5. 美化图表 ---
plt.title(f"DAgger Performance vs. Iterations on {env_name_simple}", fontsize=16)
plt.xlabel("DAgger Iteration", fontsize=12)
plt.ylabel("Evaluation Average Return", fontsize=12)
plt.xticks(iters) # 确保X轴只显示整数迭代次数
plt.legend(fontsize=10)
plt.tight_layout() # 自动调整布局

# --- 6. 保存图表 ---
plt.savefig(output_plot_file)

print(f"--- Plot for {env_name_simple} successfully generated and saved to {output_plot_file} ---")
