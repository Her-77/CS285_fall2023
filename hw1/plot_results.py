
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
results_file = "bc_walker_hyperparam_results.csv"
data = pd.read_csv(results_file)

# 提取数据
train_steps = data["train_steps"]
eval_return = data["eval_return"]
eval_std = data["eval_std"]

# --- 开始绘图 ---

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制带有误差棒的折线图
# yerr=eval_std 表示使用标准差作为误差范围
# fmt='-o' 表示使用实线和圆点标记
# capsize=5 使得误差棒的顶部和底部有短横线，更易读
plt.errorbar(train_steps, eval_return, yerr=eval_std, fmt='-o', capsize=5, label='Average Return with Std Dev')

# --- 美化图表 ---

# 设置图表标题
plt.title("BC Performance vs. Training Steps on Walker2d-v4")

# 设置X轴和Y轴的标签
plt.xlabel("Number of Training Steps per Iteration")
plt.ylabel("Evaluation Average Return")

# 设置X轴为对数刻度，这样可以更好地观察不同数量级训练步数下的性能变化
plt.xscale('log')

# 添加网格线，方便观察
plt.grid(True, which="both", ls="--")

# 添加图例
plt.legend()

# --- 保存图表 ---

# 定义输出图片的文件名
output_plot_file = "bc_walker_hyperparam_plot.png"

# 保存图表到文件
plt.savefig(output_plot_file)

print(f"--- Plot successfully generated and saved to {output_plot_file} ---")

