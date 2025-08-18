import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse

def plot_comparison(baseline_logdir: str, best_logdir: str, output_path: str = "final_comparison.png"):
    """
    Parses logs from two directories (baseline and best), calculates mean and std
    over seeds, and plots a comparison graph.
    """
    
    # --- Helper function to process one set of experiments (e.g., all baseline runs) ---
    def process_log_dir(logdir):
        print(f"Processing directory: {logdir}")
        reader = SummaryReader(logdir, extra_columns={"dir_name"})
        df = reader.scalars
        
        if df.empty:
            print(f"Warning: No data found in {logdir}")
            return None, None

        all_runs_data = []
        for run_name, group in df.groupby('dir_name'):
            eval_returns = group[group['tag'] == 'Eval_AverageReturn'][['step', 'value']]
            env_steps = group[group['tag'] == 'Train_EnvstepsSoFar'][['step', 'value']]
            merged = pd.merge(eval_returns, env_steps, on='step', suffixes=('_return', '_steps'))
            all_runs_data.append(merged.set_index('value_steps')['value_return'])

        # Combine all runs into a single DataFrame, aligning by env_steps index
        combined_df = pd.concat(all_runs_data, axis=1)
        
        # Interpolate to fill missing values (common when env_steps don't align perfectly)
        combined_df = combined_df.interpolate(method='linear', limit_direction='forward', axis=0)

        mean_returns = combined_df.mean(axis=1)
        std_returns = combined_df.std(axis=1)
        
        return mean_returns, std_returns

    # --- Process both directories ---
    baseline_mean, baseline_std = process_log_dir(baseline_logdir)
    best_mean, best_std = process_log_dir(best_logdir)

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    if baseline_mean is not None:
        plt.plot(baseline_mean.index, baseline_mean, label="Default Hyperparameters", color="blue")
        plt.fill_between(baseline_mean.index, baseline_mean - baseline_std, baseline_mean + baseline_std, color="blue", alpha=0.2)

    if best_mean is not None:
        plt.plot(best_mean.index, best_mean, label="Best Hyperparameters", color="red")
        plt.fill_between(best_mean.index, best_mean - best_std, best_mean + best_std, color="red", alpha=0.2)

    plt.xlabel("Train_EnvstepsSoFar", fontsize=14)
    plt.ylabel("Eval_AverageReturn", fontsize=14)
    plt.title("Comparison of Default vs. Best Hyperparameters on InvertedPendulum", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    
    plt.savefig(output_path)
    print(f"\nFinal comparison plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True, help="Path to the directory with baseline logs (5 seeds)")
    parser.add_argument("--best_dir", required=True, help="Path to the directory with best hyperparameter logs (5 seeds)")
    args = parser.parse_args()
    plot_comparison(args.baseline_dir, args.best_dir)