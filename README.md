# Monotonicity Constraint Tuple Deletion

## Overview
This project focuses on grouped datasets, where each group is identified by a categorical variable.
The goal is to make sure that the values of a dependent variable, calculated using a chosen aggregation function, do not decrease as a function of the ordered group ids.
The main challenge is to efficiently decide which rows to remove to fix the violations while preserving as much of the data as possible.

To achieve this, we implement and compare two algorithms:
1. **Dynamic Programming (DP) Algorithm** - Guarantees minimal deletions but is computationally expensive [[LINK](https://github.com/idanco5d/Trendline-Outlier-Detection)].
2. **Greedy Algorithm** - Provides a near-optimal solution much faster, making it more suitable for large datasets.

This project involves:
- Generating synthetic datasets with controlled monotonicity violations.
- Implementing and optimizing the Greedy algorithm.
- Evaluating different aggregation functions: **Sum, Max, Avg, and Median**.

## Project Structure
```
├── dataset_generator.py              # Generates synthetic datasets with controlled monotonicity violations
├── compare_algorithms_single_ds.py   # Compares DP and Greedy algorithms on a single dataset
├── compare_algorithms.py             # Runs experiments on multiple datasets
├── aggr-main.py                      # Implements a greedy algorithm to remove tuples violating monotonicity
├── plots_for_main.py                 # Generates plots for visualization
```

## Installation
Ensure you have Python 3 installed along with the required dependencies:
```sh
pip install pandas numpy matplotlib tqdm sortedcontainers
```

## Explanation of Files and How to Run

### `dataset_generator.py`
#### Running Command:
```sh
python dataset_generator.py
```
#### Explanation:
This script generates synthetic datasets with controlled monotonicity violations. It allows the user to specify parameters such as:
- **Number of groups** (`num_groups`)
- **Number of rows** (`num_rows`)
- **Violation percentage** (`violations_percentage`)
- **Aggregation function** (`sum`, `max`, `avg`, `median`)

It ensures that the dataset initially follows an increasing order in aggregation values while introducing controlled violations for testing.

#### Outputs:
- CSV dataset files in `{output_folder}/datasets/`
- Scatter and bar plots in `{output_folder}/scatter_plots/` and `{output_folder}/bar_plots/`

The generated files a **name suffix**, which encodes key dataset parameters in the following format:
`{file_type}_{agg_func}_g{num_groups}_r{num_rows}_v{violations_percentage}`

### `aggr-main.py`
#### Running Command:
```sh
python aggr-main.py {dataset_path} {agg_function} --grouping_column {grouping_column} --aggregation_column {aggregation_column} --output_folder {output_folder}
```
#### Explanation:
This script implements a **Greedy Algorithm** that iteratively removes tuples to restore monotonicity.
It calculates the impact of each tuple removal and prioritizes the ones that minimize overall monotonicity violations (the one with the highest impact).
- `dataset_path`: Path to the dataset CSV file
- `agg_function`: Aggregation function (`sum`, `max`, `avg`, `median`)
- `grouping_column`: Column to group by (default: `A`)
- `aggregation_column`: Column to aggregate (default: `B`)
- `output_folder`: Folder to store results

#### Outputs:
- Filtered dataset in `{output_folder}/result-{dataset}.csv`
- Removed tuples in `{output_folder}/removed-{dataset}.csv`
- Plots based on `plots_for_main.py` as described below.

### `plots_for_main.py`
#### Running Command:
This script is used internally for generating visualizations and does not require direct execution.
#### Explanation:
Generates plots for:
- Aggregation values before and after tuple removal
- Impact of tuple removal per iteration

#### Outputs - Plots:
  - Execution time comparison
  - Rows removed comparison
  - Impact of removal per iteration

### `compare_algorithms_single_ds.py`
#### Running Command:
```sh
python compare_algorithms_single_ds.py {dataset_path} {agg_function} --output_folder {output_folder}
```
#### Explanation:
This script runs both the **DP Algorithm** and the **Greedy Algorithm** on a single dataset and compares:
- Execution time
- Number of tuples removed
- Overlap between removed tuples

It helps analyze how well the Greedy approach performs compared to the optimal DP method.

#### Outputs:
- CSV results (`comparison_results.csv` in `{output_folder}`)
- Execution time and rows removed comparison plots in addition to an overlap plot.

### `compare_algorithms.py`
#### Running Command:
```sh
python compare_algorithms.py --agg_function {agg_function} --dataset_folder {dataset_folder} --results_folder {results_folder} --timeout_min {timeout} --grouping_column {grouping_column} --aggregation_column {aggregation_column}
```
#### Explanation:
Runs experiments on multiple datasets to evaluate performance across different dataset configurations:
- `agg_function`: Aggregation function (`sum`, `max`, `avg`, `median`)
- `dataset_folder`: Folder containing datasets
- `results_folder`: Folder to store results
- `timeout`: Timeout for each algorithm in minutes
- `grouping_column`: Column to group by (default: `A`)
- `aggregation_column`: Column to aggregate (default: `B`)

The script compares **DP and Greedy algorithms** over various settings, such as the number of rows, groups, and violation percentages.
The results presented in the output files and plots are computed as the mean of multiple runs with the same parameter configuration.
For each experiment, datasets are generated with the same number of groups, rows, and violation percentages. 
The final reported results (such as execution time, number of rows removed, and overlap) represent the average values across these runs to provide a more stable and reliable evaluation of algorithm performance.

#### Outputs:
- Aggregated comparison results (`comparison_results.csv`)
- Execution time comparison plots
- Rows removed comparison plots
