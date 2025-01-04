import os
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from io import StringIO


def run_algorithm(command, agg_col, gb_col):
    """Run an algorithm and return execution time and rows removed."""
    start_time = time.time()
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=600)
        execution_time = time.time() - start_time

        # Extract removed tuples from output
        rows_removed = pd.DataFrame()
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if "The removed tuples are:" in line:
                raw_data = '\n'.join(output_lines[output_lines.index(line) + 1:])
                rows_removed = pd.read_csv(
                    StringIO(raw_data),
                    delim_whitespace=True,
                    names=["index", gb_col, agg_col],  # Explicitly name columns
                    usecols=[1, 2],  # Ignore the index column
                    skiprows=1,  # Skip the repeated header
                )
                break

        # Drop NaN rows if they exist
        rows_removed = rows_removed.dropna()

        # Debugging output
        print("Parsed Rows Removed DataFrame:\n", rows_removed)

        return execution_time, rows_removed

    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return None, pd.DataFrame()


def calculate_overlap_by_columns(dp_removed: pd.DataFrame, greedy_removed: pd.DataFrame, gb_col: str, agg_col: str) -> int:
    """
    Calculate the overlap between DP and Greedy removed tuples by comparing the grouping and aggregation columns.

    Args:
        dp_removed (pd.DataFrame): Removed tuples from the DP algorithm.
        greedy_removed (pd.DataFrame): Removed tuples from the Greedy algorithm.
        gb_col (str): The name of the grouping column.
        agg_col (str): The name of the aggregation column.

    Returns:
        int: The count of overlapping tuples based on gb_col and agg_col values.
    """
    overlap_count = 0
    greedy_removed_copy = greedy_removed.copy()  # Work on a copy to avoid modifying the original DataFrame

    for _, dp_row in dp_removed.iterrows():
        match_index = greedy_removed_copy[
            (greedy_removed_copy[gb_col] == dp_row[gb_col]) &
            (greedy_removed_copy[agg_col] == dp_row[agg_col])
        ].index

        if not match_index.empty:
            # Increment overlap count and remove the matched row
            overlap_count += 1
            greedy_removed_copy = greedy_removed_copy.drop(match_index[0])  # Drop only the first match

    return overlap_count


def compare_single_dataset(dataset_path, dp_command, greedy_command, output_folder, agg_col, gb_col):
    """Compare DP and Greedy algorithms on a single dataset."""
    num_rows_in_dataset = pd.read_csv(dataset_path).shape[0]

    # DP and Greedy command
    dp_time, dp_removed = run_algorithm(dp_command, agg_col, gb_col)
    greedy_time, greedy_removed = run_algorithm(greedy_command, agg_col, gb_col)

    # Calculate overlap
    overlap_count = calculate_overlap_by_columns(dp_removed, greedy_removed, gb_col, agg_col)

    results = {
        "dataset": os.path.basename(dataset_path),
        "num_rows": num_rows_in_dataset,
        "dp_time": dp_time,
        "dp_removed_count": len(dp_removed),
        "greedy_time": greedy_time,
        "greedy_removed_count": len(greedy_removed),
        "overlap_count": overlap_count
    }

    # Save results
    os.makedirs(output_folder, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_folder, "comparison_results.csv"), index=False)

    plot_results(results_df, output_folder)

    return results


def plot_overlap_pie_chart(dp_removed_count, greedy_removed_count, overlap_count, output_folder):
    """
    Plot the proportion of overlap and unique rows removed as a pie chart with numbers inside
    and labels outside.

    Args:
        dp_removed_count (int): Number of rows removed by DP.
        greedy_removed_count (int): Number of rows removed by Greedy.
        overlap_count (int): Number of overlapping rows.
        output_folder (str): Path to save the chart.
    """
    # Calculate unique counts
    unique_dp_removed = dp_removed_count - overlap_count
    unique_greedy_removed = greedy_removed_count - overlap_count

    # Data and labels
    sizes = [unique_dp_removed, unique_greedy_removed, overlap_count]
    labels = ["Unique DP Removed", "Unique Greedy Removed", "Overlap"]

    # Plot
    plt.figure(figsize=(8, 8))
    wedges, texts = plt.pie(
        sizes,
        labels=labels,  # Add labels outside the pie chart
        colors=["#FF9999", "#99FF99", "#9999FF"],
        textprops={'fontsize': 14},
        startangle=90  # Optional for better alignment
    )

    # Add numbers inside the pie chart
    for i, wedge in enumerate(wedges):
        theta = (wedge.theta1 + wedge.theta2) / 2  # Midpoint angle in degrees
        theta_rad = np.radians(theta)  # Convert to radians for trigonometric functions
        x_text = 0.5 * np.cos(theta_rad)  # X-coordinate (closer to the center)
        y_text = 0.5 * np.sin(theta_rad)  # Y-coordinate (closer to the center)
        plt.text(
            x_text, y_text,
            f"{sizes[i]}",  # Display the number
            ha="center", va="center", fontsize=14, color="black"  # Black text for visibility
        )

    # Title
    plt.title("Proportion of Overlap and Unique Rows Removed", fontsize=16)
    plt.tight_layout()

    # Save the chart
    plt.savefig(os.path.join(output_folder, "overlap_pie_chart.pdf"))
    plt.close()


def plot_results(results_df, output_folder):
    """Plot comparison results."""

    os.makedirs(output_folder, exist_ok=True)
    num_rows = results_df["num_rows"].iloc[0]  # Total number of rows in the dataset
    dp_removed_count = results_df["dp_removed_count"].iloc[0]
    greedy_removed_count = results_df["greedy_removed_count"].iloc[0]
    overlap_count = results_df["overlap_count"].iloc[0]

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
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["DP", "Greedy"],
                   [results_df["dp_removed_count"].iloc[0], results_df["greedy_removed_count"].iloc[0]], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')
    plt.ylabel("Rows Removed")
    plt.title("Rows Removed Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "rows_removed_comparison.pdf"))
    plt.close()

    # Plot Overlap
    plot_overlap_pie_chart(dp_removed_count, greedy_removed_count, overlap_count, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DP and Greedy algorithms on multiple datasets.")
    parser.add_argument("ds", type=str, help="The path to the dataset")
    parser.add_argument("agg_func", type=str, choices=["sum", "max", "avg", "median"],
                        help="The aggregation function to use.")
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
    results = compare_single_dataset(dataset_path, DP_COMMAND_TEMPLATE, GREEDY_COMMAND_TEMPLATE, output_folder,
                                     aggregation_column, grouping_column)
    print("Comparison Results:", results)
