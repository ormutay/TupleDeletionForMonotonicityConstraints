import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Plotting ---
def plot_aggregation(df, grouping_column, aggregation_column, title, output_file, agg_func, agg_func_name):
    """Plot the aggregated values of the DataFrame."""
    aggregated_df = df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(aggregated_df[grouping_column], aggregated_df[aggregation_column], color='skyblue')
    plt.xlabel(grouping_column)
    plt.ylabel(f"{agg_func_name}({aggregation_column})")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file, format='pdf')
    plt.close()
    print(f"Plot saved to {output_file}")


def plot_impact_per_iteration(log_file, output_file):
    """Plot the impact per iteration based on the iteration log."""
    log_data = pd.read_csv(log_file)

    if "Impact" not in log_data.columns or "Iteration" not in log_data.columns:
        print("Error: Missing required columns in the log file.")
        return

    colors = log_data["Impact"].apply(lambda x: 'green' if x > 0 else ('yellow' if x == 0 else 'red')).tolist()

    plt.figure(figsize=(10, 6))
    for i, impact in enumerate(log_data["Impact"]):
        plt.plot([log_data["Iteration"].iloc[i]], [impact], marker='o', markersize=8,
                 color='black', markerfacecolor=colors[i], markeredgecolor='black')

    plt.plot(log_data["Iteration"], log_data["Impact"], color='black', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("Impact")
    plt.title("Impact per Iteration")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_file, format="pdf")
    plt.close()
    print(f"Impact per iteration plot saved to {output_file}")

