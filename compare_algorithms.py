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
    print(command)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=timeout)
        end_time = time.time()
        execution_time = end_time - start_time

        rows_removed = None
        for line in result.stdout.splitlines():
            if "Total tuple removals" in line:
                rows_removed = int(float(line.split(":")[-1].strip()))
            elif "Num removed tuples" in line:
                rows_removed = int(float(line.split(":")[-1].strip().split("/")[0]))
        if rows_removed is None:
            return {"time": None, "rows_removed": None}  # Return None for failure

        return {"time": execution_time, "rows_removed": rows_removed}

    except (subprocess.TimeoutExpired, ValueError) as e:
        print(f"Error running command: {e}")
        return {"time": None, "rows_removed": None}  # Return None for failure


# Helper function to process a single dataset
def process_single_dataset(dataset_path, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, group_column, agg_column, output_folder):
    print(f"Processing dataset: {dataset_path}")
    filename = os.path.basename(dataset_path)

    try:
        # Extract parameters from filename
        num_rows = int(filename.split("_")[3][1:])
        num_groups = int(filename.split("_")[2][1:]) if "_g" in filename else None
        violation_percentage = int(filename.split("_")[4][1:]) if "_v" in filename else None
        index = int(filename.split("_")[-1].split(".")[0])  # Extract index from filename
    except:
        df = pd.read_csv(dataset_path, index_col=0)
        num_rows = len(df)
        num_groups = df[group_column].nunique()
        violation_percentage = "nat"
        index = filename.split(".")[0].split("index")[1]

    output_csv = os.path.join(results_folder, f"greedy_{filename}_output.csv")
    # Run greedy algorithm
    greedy_command = [
        "python", greedy_algo_path, dataset_path,
        agg_function.lower(),
        "--grouping_column", group_column,
        "--aggregation_column", agg_column,
        "--output_folder", output_folder,
    ]
    greedy_results = run_algorithm(greedy_command, is_dp=False, timeout=timeout)

    # Run DP algorithm
    dp_results = {'time': 'NA', 'rows_removed': 'NA'}
    if agg_function.lower() != 'avg' and num_rows < 100000:
        dp_command = [
            "python", dp_algo_path, agg_function.upper(), dataset_path, agg_column,
            group_column
        ]
        dp_results = run_algorithm(dp_command, is_dp=True, timeout=timeout)
        print(dp_results)

    # Run DP + pruning algorithm
    dp_prune_results = {'time': 'NA', 'rows_removed': 'NA'}
    if agg_function.lower() in ('sum', 'avg', 'median'):
        dp_prune_command = [
            "python", dp_algo_path, agg_function.upper(), dataset_path, agg_column,
            group_column, '--prune' , str(greedy_results["rows_removed"])
        ]
        if agg_function.lower() == 'avg':
            dp_prune_command.append('--mem_opt')
        dp_prune_results = run_algorithm(dp_prune_command, is_dp=True, timeout=timeout)

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
        "dp_prune_time": dp_prune_results["time"],
        "dp_prune_rows_removed": dp_prune_results["rows_removed"],
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


def process_datasets_serial(dataset_folder, greedy_algo_path, dp_algo_path, agg_function, timeout, results_folder, grouping_column="A", aggregation_column="B", output_folder="output"):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []
    output_path = os.path.join(output_folder, "comparison_results.csv")
    # set the file to contain only headers.
    headers =  ["dataset", "num_rows", "num_groups", "violation_percentage",
                "index", "greedy_time", "greedy_rows_removed", "dp_time",
                "dp_rows_removed", "dp_prune_time", "dp_prune_rows_removed"]
    pd.DataFrame(columns=headers).to_csv(output_path, index=False)

    print(f"Found {len(dataset_paths)} datasets. Processing serially...")
    for path in dataset_paths:
        result = process_single_dataset(path, greedy_algo_path, dp_algo_path, agg_function, timeout,
                            results_folder, grouping_column, aggregation_column, output_folder)
        pd.DataFrame([result]).to_csv(os.path.join(output_folder, "comparison_results.csv"), mode='a', index=False, header=False)
        results.append(result)
    return pd.DataFrame(results)

def plot_results(results, output_folder, x_axis, x_label, agg_function):
    print(f"Preparing data for plots with {x_axis} as x-axis...")

    # Ensure numeric data for aggregation
    numeric_columns = ["num_rows", "num_groups", "violation_percentage", "greedy_rows_removed", "dp_rows_removed", "greedy_time", "dp_time"]
    results = results[numeric_columns].dropna(subset=[x_axis])

    # Reset index to avoid duplicate index issues
    results = results.reset_index(drop=True)

    # Calculate the percentage of rows removed
    results["greedy_rows_removed_pct"] = (results["greedy_rows_removed"] / results["num_rows"]) * 100
    results["dp_rows_removed_pct"] = (results["dp_rows_removed"] / results[
        "num_rows"]) * 100 if "dp_rows_removed" in results else None

    # Group by the specified x-axis and calculate the mean
    grouped = results.groupby(x_axis).mean().sort_index()

    # Plot 1: Percentage of Rows Removed vs. x_axis
    plt.figure(figsize=(24, 14))
    plt.plot(grouped.index, grouped["greedy_rows_removed_pct"], marker='o', label="Greedy Algorithm", linewidth=6)
    if "dp_rows_removed_pct" in grouped and not grouped["dp_rows_removed_pct"].isna().all():
        plt.plot(grouped.index, grouped["dp_rows_removed_pct"], marker='o', linestyle="--", label="DP Algorithm",
                 linewidth=6)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel("Mean Percentage of Rows Removed (%)", fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.title(f"Percentage of Rows Removed by Algorithms ({x_label})", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, linewidth=4)
    plt.savefig(os.path.join(output_folder, f"rows_removed_percentage_comparison_{agg_function}_{x_axis}.pdf"), format='pdf')
    plt.show()

    # Plot 2: Execution Time vs. x_axis (Logarithmic Scale)
    plt.figure(figsize=(24, 14))
    plt.plot(grouped.index, grouped["greedy_time"], marker='o', label="Greedy Algorithm", linewidth=6)
    plt.plot(grouped.index, grouped["dp_time"], marker='o', linestyle="--", label="DP Algorithm", linewidth=6)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel("Mean Execution Time (seconds)", fontsize=28)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.title(f"Execution Time of Algorithms ({x_label}) [Log Scale]", fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True, linewidth=4)
    plt.savefig(os.path.join(output_folder, f"execution_time_comparison_{agg_function}_{x_axis}(log).pdf"), format='pdf')
    plt.show()

    print(f"Plots saved successfully under {output_folder} folder.")

#python/py -3.13 compare_algorithms.py --agg_function <agg_function> --dataset_folder <dataset_folder> --results_folder <results_folder> --timeout_min <timeout_min> --grouping_column <grouping_column> --aggregation_column <aggregation_column>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Greedy and DP Algorithms")
    parser.add_argument("--agg_function", type=str, choices=["max", "sum", "avg", "median"], required=True, help="The aggregation function to use")
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


    dataset_folder = args.dataset_folder
    output_folder = results_base_folder
    os.makedirs(output_folder, exist_ok=True)
    results = process_datasets_serial(dataset_folder, f"aggr-main.py", "Trendline-Outlier-Detection/main.py",
                                        agg_function, timeout=timeout, results_folder=output_folder,
                                        grouping_column=grouping_column, aggregation_column=aggregation_column,
                                        output_folder=output_folder)
    results.to_csv(os.path.join(output_folder, "comparison_results.csv"), index=False)
    sys.exit()



    base_folder = args.dataset_folder
    result_folder = os.path.join(results_base_folder, agg_function)

    for folder in ["rows", "groups", "violations"]:
        dataset_folder = os.path.join(base_folder, folder, "datasets")
        output_folder = os.path.join(result_folder, folder)
        os.makedirs(output_folder, exist_ok=True)

        results = process_datasets_parallel(dataset_folder, f"aggr-main.py", "Trendline-Outlier-Detection/main.py",
                                            agg_function, timeout=timeout, results_folder=result_folder,
                                            grouping_column=grouping_column, aggregation_column=aggregation_column,
                                            output_folder=output_folder)
        if results.empty:
            print(f"No results found for {folder} datasets. Skipping plotting results...")
            continue

        results.to_csv(os.path.join(output_folder, "comparison_results.csv"), index=False)
        if folder == "rows":
            plot_results(results, output_folder, "num_rows", "Number of Rows", agg_function)
        elif folder == "groups":
            plot_results(results, output_folder, "num_groups", "Number of Groups", agg_function)
        elif folder == "violations":
            plot_results(results, output_folder, "violation_percentage", "Violation Percentage", agg_function)
