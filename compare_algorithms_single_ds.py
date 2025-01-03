import os
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt
import argparse
from io import StringIO

def run_algorithm(command):
    """Run an algorithm and return execution time and row indices of removed tuples."""
    start_time = time.time()
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=600)
        execution_time = time.time() - start_time

        # Extract indices of removed tuples from output
        indices = []
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if "The removed tuples are:" in line:
                from io import StringIO
                removed_df = pd.read_csv(StringIO('\n'.join(output_lines[output_lines.index(line) + 1:])), delim_whitespace=True)
                indices = removed_df.index.tolist()
                break

        return execution_time, indices

    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return None, []



def calculate_overlap(dp_indices, greedy_indices):
    """Calculate the overlap between two sets of removed row indices."""
    return len(set(dp_indices).intersection(set(greedy_indices)))


def compare_single_dataset(dataset_path, dp_command, greedy_command, output_folder):
    """Compare DP and Greedy algorithms on a single dataset."""
    num_rows_in_dataset = pd.read_csv(dataset_path).shape[0]

    # DP and Greedy command
    dp_time, dp_indices = run_algorithm(dp_command)
    greedy_time, greedy_indices = run_algorithm(greedy_command)

    # Calculate overlap
    overlap_count = calculate_overlap(dp_indices, greedy_indices)


    results = {
        "dataset": os.path.basename(dataset_path),
        "num_rows": num_rows_in_dataset,
        "dp_time": dp_time,
        "dp_removed_count": len(dp_indices),
        "greedy_time": greedy_time,
        "greedy_removed_count": len(greedy_indices),
        "overlap_count": overlap_count
    }

    # Save results
    os.makedirs(output_folder, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_folder, "comparison_results.csv"), index=False)

    plot_results(results_df, output_folder)

    return results

def plot_results(results_df, output_folder):
    num_rows = results_df["num_rows"].iloc[0]  # Total number of rows in the dataset
    """Plot comparison results."""
    os.makedirs(output_folder, exist_ok=True)

    # Plot Execution Time
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["DP", "Greedy"], [results_df["dp_time"].iloc[0], results_df["greedy_time"].iloc[0]], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "execution_time_comparison.pdf"))
    plt.close()

    # Plot Rows Removed

    # Pie Chart for Rows Removed vs Total Rows
    plt.figure(figsize=(8, 8))
    dp_removed = results_df["dp_removed_count"].iloc[0]
    greedy_removed = results_df["greedy_removed_count"].iloc[0]
    remaining_rows = num_rows - max(dp_removed, greedy_removed)

    plt.pie(
        [dp_removed, greedy_removed, remaining_rows],
        labels=["DP Removed", "Greedy Removed", "Remaining Rows"],
        autopct='%1.1f%%',
        colors=["#FF9999", "#99FF99", "#9999FF"]
    )
    plt.title("Proportion of Rows Removed by Algorithms")
    plt.savefig(os.path.join(output_folder, "rows_removed_pie_chart.pdf"))
    plt.close()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["DP", "Greedy"], [results_df["dp_removed_count"].iloc[0], results_df["greedy_removed_count"].iloc[0]], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')
    plt.ylabel("Rows Removed")
    plt.title("Rows Removed Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "rows_removed_comparison.pdf"))
    plt.close()

    # Plot Overlap
    plt.figure(figsize=(8, 8))
    dp_removed = results_df["dp_removed_count"].iloc[0]
    greedy_removed = results_df["greedy_removed_count"].iloc[0]
    overlap_count = results_df["overlap_count"].iloc[0]
    unique_dp_removed = dp_removed - overlap_count
    unique_greedy_removed = greedy_removed - overlap_count

    plt.pie(
        [unique_dp_removed, unique_greedy_removed, overlap_count],
        labels=["Unique DP Removed", "Unique Greedy Removed", "Overlap"],
        autopct='%1.1f%%',
        colors=["#FF9999", "#99FF99", "#9999FF"]
    )
    plt.title("Proportion of Overlap and Unique Rows Removed")
    plt.savefig(os.path.join(output_folder, "overlap_pie_chart.pdf"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DP and Greedy algorithms on multiple datasets.")
    parser.add_argument("ds", type=str, help="The path to the dataset")
    parser.add_argument("agg_func", type=str, choices=["sum", "max", "avg", "median"], help="The aggregation function to use.")
    parser.add_argument("--gb_col", type=str, default="A", help="The column to group by.")
    parser.add_argument("--agg_col", type=str, default="B", help="The column to aggregate.")
    parser.add_argument("--output_folder", type=str, default="compare_results", help="The folder to save the results.")
    args = parser.parse_args()

    dataset_path = args.ds
    aggregation_column = args.agg_col
    grouping_column = args.gb_col
    aggregation_function = args.agg_func
    output_folder = args.output_folder

    DP_COMMAND_TEMPLATE = [
        "python", "Trendline-Outlier-Detection/main.py",
        aggregation_function.upper(), dataset_path, aggregation_column, grouping_column
    ]
    GREEDY_COMMAND_TEMPLATE = [
        "python", "aggr-main.py", dataset_path, aggregation_function,
        "--grouping_column", grouping_column, "--aggregation_column", aggregation_column,
        "--output_folder", output_folder
    ]
    results = compare_single_dataset(dataset_path, DP_COMMAND_TEMPLATE, GREEDY_COMMAND_TEMPLATE, output_folder)
    print("Comparison Results:", results)
