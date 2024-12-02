import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt

def run_algorithm(command, is_external=False):
    """
    Run an algorithm as a subprocess and measure its time and rows removed.

    Args:
        command (list): The command to execute the algorithm.
        is_external (bool): Whether the algorithm is external (handles output parsing differently).

    Returns:
        dict: A dictionary with execution time and rows removed.
    """
    start_time = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    end_time = time.time()

    execution_time = end_time - start_time
    rows_removed = 0

    if is_external:
        # For the external algorithm, parse the "The removed tuples are:" section
        output_lines = result.stdout.splitlines()
        in_removed_section = False
        for line in output_lines:
            if "The removed tuples are:" in line:
                in_removed_section = True
                continue
            if in_removed_section:
                if line.strip() == "":  # End of section
                    break
                rows_removed += 1  # Count each row in the removed tuples section
    else:
        # For my algorithm, parse "Total tuple removals"
        for line in result.stdout.splitlines():
            if "Total tuple removals" in line:
                rows_removed = int(float(line.split(":")[-1].strip()))

    return {"time": execution_time, "rows_removed": rows_removed}



def create_line_plots(results, output_folder):
    """
    Create line plots for rows removed and execution time.

    Args:
        results (list): List of dictionaries containing performance metrics for each dataset.
        output_folder (str): Folder to save the line plots.
    """
    # Ensure the results folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    datasets = [result["dataset"] for result in results]
    rows_removed_my_algo = [result["My Algorithm"]["rows_removed"] for result in results]
    rows_removed_ext_algo = [result["External Algorithm"]["rows_removed"] for result in results]
    time_my_algo = [result["My Algorithm"]["time"] for result in results]
    time_ext_algo = [result["External Algorithm"]["time"] for result in results]

    # Line plot for rows removed
    plt.figure(figsize=(10, 6))
    plt.plot(datasets, rows_removed_my_algo, marker="o", label="My Algorithm")
    plt.plot(datasets, rows_removed_ext_algo, marker="o", label="External Algorithm")
    plt.xlabel("Dataset", fontsize=14)
    plt.ylabel("Rows Removed", fontsize=14)
    plt.title("Rows Removed Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "rows_removed_comparison.png"))
    plt.close()

    # Line plot for execution time
    plt.figure(figsize=(10, 6))
    plt.plot(datasets, time_my_algo, marker="o", label="My Algorithm")
    plt.plot(datasets, time_ext_algo, marker="o", label="External Algorithm")
    plt.xlabel("Dataset", fontsize=14)
    plt.ylabel("Execution Time (s)", fontsize=14)
    plt.title("Execution Time Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "execution_time_comparison.png"))
    plt.close()


if __name__ == "__main__":
    # Inputs
    dataset_folder = "dataset-sum"  # Folder containing the datasets
    aggregation_function = "SUM"  # Options: "MAX", "SUM"
    column_to_aggregate = "B"
    grouping_column = "A"
    output_folder = "results-sum"

    # Initialize results
    all_results = []

    # Iterate over all datasets in the folder
    for dataset_file in sorted(os.listdir(dataset_folder)):
        if dataset_file.endswith(".csv"):
            dataset_path = os.path.join(dataset_folder, dataset_file)
            print(f"\nProcessing dataset: {dataset_file}")

            # Commands for both algorithms
            my_algorithm_command = ["python", f"{aggregation_function.lower()}-main.py", dataset_path]
            other_algorithm_command = [
                "python", "Trendline-Outlier-Detection/main.py", aggregation_function, dataset_path,
                column_to_aggregate, grouping_column
            ]

            # Run my algorithm
            print("Running my algorithm...")
            my_algorithm_results = run_algorithm(my_algorithm_command)

            # Run the external algorithm
            print("Running the external algorithm...")
            other_algorithm_results = run_algorithm(other_algorithm_command, is_external=True)

            # Collect results for this dataset
            all_results.append({
                "dataset": dataset_file,
                "My Algorithm": my_algorithm_results,
                "External Algorithm": other_algorithm_results,
            })

    # Print summary of results
    print("\nSummary of Results:")
    for result in all_results:
        print(f"Dataset: {result['dataset']}")
        print(f"  My Algorithm: Rows Removed = {result['My Algorithm']['rows_removed']}, "
              f"Execution Time = {result['My Algorithm']['time']:.4f}s")
        print(f"  External Algorithm: Rows Removed = {result['External Algorithm']['rows_removed']}, "
              f"Execution Time = {result['External Algorithm']['time']:.4f}s")

    # Create comparison line plots
    create_line_plots(all_results, output_folder)
    print(f"\nComparison plots saved in the '{output_folder}' folder.")
