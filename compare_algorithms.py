import os
import sys
import pandas as pd
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import argparse

# Function to run an algorithm and measure rows removed and execution time
def run_algorithm(command, is_dp=False, timeout=600):
    start_time = time.time()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=timeout)
        end_time = time.time()
        execution_time = end_time - start_time

        rows_removed = None
        if is_dp:
            # DP algorithm: parse "The removed tuples are:" section
            output_lines = result.stdout.splitlines()
            in_removed_section = False
            rows_removed = 0
            for line in output_lines:
                if "The removed tuples are:" in line:
                    in_removed_section = True
                    continue
                if in_removed_section:
                    if line.strip() == "":  # End of section
                        break
                    rows_removed += 1  # Count each row in the removed tuples section
        else:
            # Greedy algorithm: parse "Total tuple removals"
            for line in result.stdout.splitlines():
                if "Total tuple removals" in line:
                    rows_removed = int(float(line.split(":")[-1].strip()))

        return {"time": execution_time, "rows_removed": rows_removed}

    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return {"time": None, "rows_removed": None}  # Return None for timeout

# Helper function to process a single dataset
def process_single_dataset(dataset_path, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, group_column, agg_column, output_folder):
    print(f"Processing dataset: {dataset_path}")
    filename = os.path.basename(dataset_path)

    # Extract parameters from filename
    num_rows = int(filename.split("_")[3][1:])
    num_groups = int(filename.split("_")[2][1:]) if "_g" in filename else None
    violation_percentage = int(filename.split("_")[4][1:]) if "_v" in filename else None
    index = int(filename.split("_")[-1].split(".")[0])  # Extract index from filename

    # Run greedy algorithm
    greedy_command = [
        "python", greedy_algo_path, dataset_path,
        "--grouping_column", group_column,
        "--aggregation_column", agg_column,
        "--output_csv", f"{results_folder}/greedy_{filename}_output.csv"
    ]
    print(f"Running greedy algorithm: {' '.join(greedy_command)}")
    greedy_results = run_algorithm(greedy_command, is_dp=False, timeout=timeout)

    # Run DP algorithm
    dp_command = [
        "python", dp_algo_path, agg_function, dataset_path, agg_column, group_column
    ]
    print(f"Running DP algorithm: {' '.join(dp_command)}")
    dp_results = run_algorithm(dp_command, is_dp=True, timeout=timeout)

    return {
        "dataset": filename,
        "num_rows": num_rows,
        "num_groups": num_groups,
        "violation_percentage": violation_percentage,
        "index": index,
        "greedy_time": greedy_results["time"],
        "greedy_rows_removed": greedy_results["rows_removed"],
        "dp_time": dp_results["time"],
        "dp_rows_removed": dp_results["rows_removed"],
    }


def process_datasets_parallel(dataset_folder, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, grouping_column="A", aggregation_column="B", output_folder="output"):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []

    print(f"Found {len(dataset_paths)} datasets. Processing in parallel...")
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(process_single_dataset, dataset, greedy_algo_path, dp_algo_path, agg_function, timeout,
                            results_folder, grouping_column, aggregation_column, output_folder): dataset
            for dataset in dataset_paths
        }
        for future in as_completed(future_to_dataset):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing dataset {future_to_dataset[future]}: {e}")

    return pd.DataFrame(results)

def plot_results(results, output_folder, x_axis, x_label):
    print(f"Preparing data for plots with {x_axis} as x-axis...")

    # Ensure numeric data for aggregation
    numeric_columns = ["num_rows", "num_groups", "violation_percentage", "greedy_rows_removed", "dp_rows_removed", "greedy_time", "dp_time"]
    results = results[numeric_columns].dropna(subset=[x_axis])

    # Reset index to avoid duplicate index issues
    results = results.reset_index(drop=True)

    # Calculate the percentage of rows removed
    results["greedy_rows_removed_pct"] = (results["greedy_rows_removed"] / results["num_rows"]) * 100
    results["dp_rows_removed_pct"] = (results["dp_rows_removed"] / results["num_rows"]) * 100

    # Group by the specified x-axis and calculate the mean
    grouped = results.groupby(x_axis).mean().sort_index()

    # Plot 1: Percentage of Rows Removed vs. x_axis
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped["greedy_rows_removed_pct"], marker='o', label="Greedy Algorithm")
    plt.plot(grouped.index, grouped["dp_rows_removed_pct"], marker='o', label="DP Algorithm")
    plt.xlabel(x_label)
    plt.ylabel("Mean Percentage of Rows Removed (%)")
    plt.title(f"Percentage of Rows Removed by Algorithms ({x_label})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"rows_removed_percentage_comparison_{x_axis}.pdf"), format='pdf')
    plt.show()

    # Plot 2: Execution Time vs. x_axis
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped["greedy_time"], marker='o', label="Greedy Algorithm")
    plt.plot(grouped.index, grouped["dp_time"], marker='o', label="DP Algorithm")
    plt.xlabel(x_label)
    plt.ylabel("Mean Execution Time (seconds)")
    plt.title(f"Execution Time of Algorithms ({x_label})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"execution_time_comparison_{x_axis}.pdf"), format='pdf')
    plt.show()

    print(f"Plots saved successfully under {output_folder} folder.")


#python/py -3.13 compare_algorithms.py --agg_function <agg_function> --dataset_folder <dataset_folder> --results_folder <results_folder> --timeout_min <timeout_min> --grouping_column <grouping_column> --aggregation_column <aggregation_column>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Greedy and DP Algorithms")
    parser.add_argument("--agg_function", type=str, required=True, help="The aggregation function to use")
    parser.add_argument("--dataset_folder", type=str, default="datasets", help="The folder containing the datasets")
    parser.add_argument("--results_folder", type=str, default="results", help="The folder to store results")
    parser.add_argument("--timeout_min", type=int, default=600, help="The timeout for each algorithm run in minutes")
    parser.add_argument("--grouping_column", type=str, default="A", help="The column to group by")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The column to aggregate")
    args = parser.parse_args()

    agg_function = args.agg_function
    results_base_folder = args.results_folder
    timeout = args.timeout_min * 60
    grouping_column = args.grouping_column
    aggregation_column = args.aggregation_column

    base_folder = args.dataset_folder
    result_folder = os.path.join(results_base_folder, agg_function)

    for folder in ["rows", "groups", "violations"]:
        dataset_folder = os.path.join(base_folder, folder, "datasets")
        output_folder = os.path.join(result_folder, folder)
        os.makedirs(output_folder, exist_ok=True)

        results = process_datasets_parallel(dataset_folder, "max-main.py", "Trendline-Outlier-Detection/main.py",
                                            agg_function, timeout=timeout, results_folder=result_folder,
                                            grouping_column=grouping_column, aggregation_column=aggregation_column,
                                            output_folder=output_folder)
        if not results.empty:
            results.to_csv(os.path.join(output_folder, "comparison_results.csv"), index=False)
            if folder == "rows":
                plot_results(results, output_folder, "num_rows", "Number of Rows")
            elif folder == "groups":
                plot_results(results, output_folder, "num_groups", "Number of Groups")
            elif folder == "violations":
                plot_results(results, output_folder, "violation_percentage", "Violation Percentage")
