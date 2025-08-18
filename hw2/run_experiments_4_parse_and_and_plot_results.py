import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse

def analyze_hyperparams(logdir: str, reward_threshold: float = 1000.0):
    """
    Parses all TensorBoard event files in a directory, finds the best
    hyperparameters, and plots all learning curves.
    """
    print(f"Reading logs from: {logdir}")
    
    # tbparse a library that reads all event files and returns a tidy pandas DataFrame
    reader = SummaryReader(logdir, extra_columns={"dir_name"})
    df = reader.scalars
    
    if df.empty:
        print("No scalar data found in the specified log directory.")
        return

    print(f"Found {len(df['dir_name'].unique())} experiment runs.")

    # --- Step 1: Find the best hyperparameter combination ---
    best_run = None
    min_steps_to_reach_threshold = float('inf')

    # Group data by each unique experiment run
    for run_name, group in df.groupby('dir_name'):
        
        # Get the two metrics we care about
        eval_returns = group[group['tag'] == 'Eval_AverageReturn']
        env_steps = group[group['tag'] == 'Train_EnvstepsSoFar']
        
        if eval_returns.empty or env_steps.empty:
            continue

        # Merge them based on the step (iteration)
        merged = pd.merge(eval_returns, env_steps, on='step', suffixes=('_return', '_steps'))

        # Find the first time the return exceeds the threshold
        successful_steps = merged[merged['value_return'] >= reward_threshold]

        if not successful_steps.empty:
            steps_to_reach = successful_steps.iloc[0]['value_steps']
            if steps_to_reach < min_steps_to_reach_threshold:
                min_steps_to_reach_threshold = steps_to_reach
                best_run = run_name

    print("\n--- Analysis Complete ---")
    if best_run:
        print(f"🏆 Best Hyperparameters: {best_run}")
        print(f"   Reached {reward_threshold} avg return in {int(min_steps_to_reach_threshold)} environment steps.")
    else:
        print(f"No run reached the threshold of {reward_threshold} average return.")
    print("-------------------------\n")


    # --- Step 2: Plot all curves on a single graph ---
    plt.figure(figsize=(15, 9))

    for run_name, group in df.groupby('dir_name'):
        eval_returns = group[group['tag'] == 'Eval_AverageReturn']
        env_steps = group[group['tag'] == 'Train_EnvstepsSoFar']

        if eval_returns.empty or env_steps.empty:
            continue
            
        merged = pd.merge(eval_returns, env_steps, on='step', suffixes=('_return', '_steps'))
        
        # Use the experiment name for the plot label, removing the long prefix
        label = os.path.basename(run_name).replace('q2_pg_', '').replace('_InvertedPendulum-v4', '')

        plt.plot(merged['value_steps'], merged['value_return'], label=label, alpha=0.7)

    plt.xlabel("Train_EnvstepsSoFar", fontsize=14)
    plt.ylabel("Eval_AverageReturn", fontsize=14)
    plt.title("Hyperparameter Search for InvertedPendulum", fontsize=16)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    
    # Save the plot
    output_path = "hyperparameter_comparison.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", 
        required=True,
        help="Path to the root directory containing hyperparameter search logs (e.g., data/pendulum_hyperparam_logs)"
    )
    args = parser.parse_args()
    analyze_hyperparams(args.logdir)