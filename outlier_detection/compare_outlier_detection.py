import subprocess
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
import concurrent.futures

def run_greedy(dataset_path, agg_func, grouping_col, agg_col, output_folder):
    greedy_command = [
        "python", "../aggr-main.py", dataset_path, agg_func,
        "--grouping_column", grouping_col,
        "--aggregation_column", agg_col,
        "--output_folder", output_folder,
    ]

    start_time = time.time()
    process = subprocess.run(greedy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    greedy_time = time.time() - start_time

    rows_removed = 0
    for line in process.stdout.splitlines():
        if "Total tuple removals" in line:
            rows_removed = int(float(line.split(":")[-1].strip()))

    return {
        "dataset": dataset_path,
        "rows_removed": rows_removed,
        "greedy_time": greedy_time,
    }

def run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, method_name, param, max_removal_pct):
    print(f"Running outlier detection on {dataset_path} with {method_name} and param {param}")

    outlier_command = [
        "python", "outlier_detection.py", dataset_path,
        "--agg_func", agg_func,
        "--group_col", grouping_col,
        "--agg_col", agg_col,
        "--max_removal_pct", str(max_removal_pct),
        "--output_folder", output_folder,
        "--method", method_name,
        "--param", str(param),
    ]

    print("# " + " ".join(outlier_command))

    start_time = time.time()
    process = subprocess.run(outlier_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stdout
    outlier_time = time.time() - start_time

    outlier_rows_removed = 0
    df_after_outlier_removal_path = None
    for line in output.splitlines():
        if "Removed" in line and "outliers" in line:
            outlier_rows_removed += int(line.split()[1])
        if "Outlier removal results saved to" in line:
            df_after_outlier_removal_path = line.split("to ")[-1].strip()

    # Ensure we have a valid filtered dataset to run the greedy algorithm
    if not df_after_outlier_removal_path or not os.path.exists(df_after_outlier_removal_path):
        raise ValueError("Outlier detection did not produce a valid filtered dataset")
        return

    # Run the greedy algorithm on the filtered dataset
    greedy_results = run_greedy(df_after_outlier_removal_path, agg_func, grouping_col, agg_col, output_folder)

    results.append({
        "dataset": dataset_path,
        "max_removal_pct": max_removal_pct,
        "method": method_name,
        "param": param,
        "outlier_rows_removed": outlier_rows_removed,
        "outlier_time": outlier_time,
        "total_rows_removed": outlier_rows_removed + greedy_results["rows_removed"],
        "total_time": outlier_time + greedy_results["greedy_time"],
    })

"""
def run_outlier_methods_parallels(dataset_path, agg_func, grouping_col, agg_col, output_folder, results):
    max_removal_pcts = [0.005, 0.01, 0.05]

    z_score_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    knn_neighbors_values = [3, 5, 8, 10]
    isolation_contamination_values = [0.01, 0.05, 0.1, 0.2]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # z_score
        for param in tqdm(z_score_values, desc="z_score"):
            for max_removal_pct in max_removal_pcts:
                futures.append(executor.submit(run_combination, dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "z_score", param, max_removal_pct))

        # knn
        for param in tqdm(knn_neighbors_values, desc="knn"):
            for max_removal_pct in max_removal_pcts:
                futures.append(executor.submit(run_combination, dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "knn", param, max_removal_pct))

        # isolation_forest
        for param in tqdm(isolation_contamination_values, desc="isolation_forest"):
            for max_removal_pct in max_removal_pcts:
                futures.append(executor.submit(run_combination, dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "isolation_forest", param, max_removal_pct))

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
         future.result()  # Ensure any exceptions are raised
"""

def run_outlier_methods(dataset_path, agg_func, grouping_col, agg_col, output_folder, results):
    max_removal_pcts = [0.005, 0.01, 0.05]

    z_score_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    knn_neighbors_values = [3, 5, 8, 10]
    isolation_contamination_values = [0.01, 0.05, 0.1, 0.2]

    # z_score
    for param in tqdm(z_score_values, desc="z_score"):
        for max_removal_pct in max_removal_pcts:
            run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "z_score",
                            param, max_removal_pct)

    # knn
    for param in tqdm(knn_neighbors_values, desc="knn"):
        for max_removal_pct in max_removal_pcts:
            run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "knn",
                            param, max_removal_pct)

    # isolation_forest
    for param in tqdm(isolation_contamination_values, desc="isolation_forest"):
        for max_removal_pct in max_removal_pcts:
            run_combination(dataset_path, agg_func, grouping_col, agg_col, output_folder, results, "isolation_forest",
                            param, max_removal_pct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier Detection compare")
    parser.add_argument("dataset_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--agg_func", type=str, choices=["max", "sum", "avg", "median"],
                        help="Aggregation function to check monotonicity")
    parser.add_argument("--grouping_col", type=str, default="A", help="Column to group by")
    parser.add_argument("--agg_col", type=str, default="B", help="The column to aggregate")
    parser.add_argument("--output_folder", type=str, default="compare_outlier_results", help="Output folder")

    args = parser.parse_args()

    outlier_results = []

    os.makedirs(args.output_folder, exist_ok=True)
    run_outlier_methods(args.dataset_path, args.agg_func, args.grouping_col, args.agg_col, args.output_folder,
                       outlier_results)

    greedy_results = run_greedy(args.dataset_path, args.agg_func, args.grouping_col, args.agg_col, args.output_folder)

    # Convert results to DataFrame and save
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    outlier_results_df = pd.DataFrame(outlier_results)
    outlier_results_df.to_csv(os.path.join(args.output_folder, f"outlier_results-{dataset_name}-{args.agg_func}.csv"), index=False)
    print(greedy_results)
    # greedy_results_df = pd.DataFrame(greedy_results)
    # greedy_results_df.to_csv(os.path.join(args.output_folder, "greedy_results.csv"), index=False)

    #
    # # Print best configuration (least rows removed with fastest time)
    # best_result = results_df.sort_values(by=["rows_removed", "total_time"]).iloc[0]
    # print("Best Configuration:")
    # print(best_result)
