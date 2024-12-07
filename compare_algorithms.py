import os
import sys
import pandas as pd
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Function to run an algorithm and measure rows removed and execution time
def run_algorithm(command, is_external=False):
    start_time = time.time()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=600)  # 600-second timeout
        end_time = time.time()
        execution_time = end_time - start_time

        rows_removed = None
        if is_external:
            # External algorithm: parse "The removed tuples are:" section
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
            # My algorithm: parse "Total tuple removals"
            for line in result.stdout.splitlines():
                if "Total tuple removals" in line:
                    rows_removed = int(float(line.split(":")[-1].strip()))

        return {"time": execution_time, "rows_removed": rows_removed}

    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return {"time": None, "rows_removed": None}  # Return None for timeout

# Helper function to process a single dataset
def process_single_dataset(dataset_path, my_algo_path, external_algo_path, agg_function):
    print(f"Processing dataset: {dataset_path}")
    filename = os.path.basename(dataset_path)

    # Extract parameters from filename
    num_rows = int(filename.split("_")[3][1:])
    num_groups = int(filename.split("_")[2][1:]) if "_g" in filename else None
    violation_percentage = int(filename.split("_")[4][1:]) if "_v" in filename else None
    index = int(filename.split("_")[-1].split(".")[0])  # Extract index from filename

    # Run my algorithm
    my_command = ["python", my_algo_path, dataset_path, agg_function]
    my_results = run_algorithm(my_command)

    # Run external algorithm
    external_command = ["python", external_algo_path, agg_function, dataset_path, "B", "A"]
    external_results = run_algorithm(external_command, is_external=True)

    return {
        "dataset": filename,
        "num_rows": num_rows,
        "num_groups": num_groups,
        "violation_percentage": violation_percentage,
        "index": index,
        "my_time": my_results["time"],
        "my_rows_removed": my_results["rows_removed"],
        "external_time": external_results["time"],
        "external_rows_removed": external_results["rows_removed"],
    }

# Function to process datasets in parallel
def process_datasets_parallel(dataset_folder, my_algo_path, external_algo_path, agg_function):
    print("Collecting datasets...")
    dataset_paths = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")
    ]
    results = []

    print(f"Found {len(dataset_paths)} datasets. Processing in parallel...")
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(process_single_dataset, dataset, my_algo_path, external_algo_path, agg_function): dataset
            for dataset in dataset_paths
        }
        for future in as_completed(future_to_dataset):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing dataset {future_to_dataset[future]}: {e}")

    return pd.DataFrame(results)

# Function to plot results based on different x-axes
def plot_results(results, output_folder, x_axis, x_label):
    print(f"Preparing data for plots with {x_axis} as x-axis...")

    # Ensure numeric data for aggregation
    numeric_columns = ["num_rows", "num_groups", "violation_percentage", "my_rows_removed", "external_rows_removed", "my_time", "external_time"]
    results = results[numeric_columns].dropna(subset=[x_axis])

    # Group by the specified x-axis and calculate the mean
    grouped = results.groupby(x_axis).mean()

    # Plot 1: Rows removed vs. x_axis
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped["my_rows_removed"], marker='o', label="My Algorithm")
    plt.plot(grouped.index, grouped["external_rows_removed"], marker='o', label="External Algorithm")
    plt.xlabel(x_label)
    plt.ylabel("Mean Rows Removed")
    plt.title(f"Rows Removed by Algorithms ({x_label})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"rows_removed_comparison_{x_axis}.png"))
    plt.show()

    # Plot 2: Execution time vs. x_axis
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped["my_time"], marker='o', label="My Algorithm")
    plt.plot(grouped.index, grouped["external_time"], marker='o', label="External Algorithm")
    plt.xlabel(x_label)
    plt.ylabel("Mean Execution Time (seconds)")
    plt.title(f"Execution Time of Algorithms ({x_label})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"execution_time_comparison_{x_axis}.png"))
    plt.show()

# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_algorithms.py <function name>")
        sys.exit(1)

    agg_function = sys.argv[1].upper()  # Aggregate function, e.g., MAX, SUM
    base_folder = "dataset-max-w6"
    my_algo_path = "max-main.py"
    external_algo_path = "Trendline-Outlier-Detection/main.py"
    result_folder = f"results-w6/{agg_function}"
    output_folders = {
        "rows": result_folder + "/rows",
        "groups": result_folder + "/groups",
        "violations": result_folder + "/violations",
    }

    for folder, output_folder in output_folders.items():
        dataset_folder = os.path.join(base_folder, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Starting dataset processing for {folder} with {agg_function}...")
        results = process_datasets_parallel(dataset_folder, my_algo_path, external_algo_path, agg_function)

        # Save results to a CSV file
        results_csv = os.path.join(output_folder, "comparison_results.csv")
        results.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

        # Generate plots
        if folder == "rows":
            plot_results(results, output_folder, "num_rows", "Number of Rows")
        elif folder == "groups":
            plot_results(results, output_folder, "num_groups", "Number of Groups")
        elif folder == "violations":
            plot_results(results, output_folder, "violation_percentage", "Violation Percentage")

        print(f"Plots saved to {output_folder}\n\n")
