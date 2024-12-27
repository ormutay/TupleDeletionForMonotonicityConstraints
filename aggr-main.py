import pandas as pd
import numpy as np
import time
import argparse
import os
from tqdm import tqdm
tqdm.pandas()
from plots_for_main import plot_aggregation, plot_impact_per_iteration

# --- Preprocessing ---
def preprocess_group_values_with_indices(df, grouping_column, aggregation_column):
    """Preprocess to save values and their indices grouped by group ID."""
    group_data = {}
    for group_id, group_df in df.groupby(grouping_column):
        group_data[group_id] = {}
        for value, indices in group_df.groupby(aggregation_column).groups.items():
            group_data[group_id][value] = list(indices)

    return group_data


def calculate_group_stats(df, agg_func, grouping_column, aggregation_column):
    """Calculate Alpha(A_i) and MVI ( Measure of Violations Index) for adjacent groups."""
    group_agg = df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    alphas = group_agg[aggregation_column].values
    mvis = np.append(alphas[:-1] - alphas[1:], 0)
    mvis = np.maximum(0, mvis)  # Replace negative values with 0

    group_stats = {
        group: {"Alpha(A_i)": alpha, "MVI": mvi, "prev": group - 1, "next": group + 1}
        for group, alpha, mvi in zip(group_agg[grouping_column], alphas, mvis)
    }

    return group_stats


# --- Group Updates ---
def update_group_stats(group_stats, max_impact_group, new_group_alpha, new_group_mvi, new_prev_group_mvi):
    """Update Alpha(A_i) and MVI fields for max_impact_group and its neighbors in group_stats."""
    # Update Alpha(A_i) and MVI for max_impact_group
    group_stats[max_impact_group]["Alpha(A_i)"] = new_group_alpha
    group_stats[max_impact_group]["MVI"] = new_group_mvi

    # Update MVI for the previous group (if it exists)
    prev_group = group_stats[max_impact_group]["prev"]
    if prev_group in group_stats:
        group_stats[prev_group]["MVI"] = new_prev_group_mvi


# --- Impact Calculation ---
def calculate_impact(group_stats, group_id, new_alpha):
    """ Calculate the impact of deleting a tuple (that creates a new alpha value). """
    current_mvi, next_group_id, prev_group_id = group_stats[group_id]["MVI"], group_stats[group_id]["next"], group_stats[group_id]["prev"]

    next_alpha = group_stats[next_group_id]["Alpha(A_i)"] if next_group_id in group_stats else float("inf")
    prev_alpha = group_stats[prev_group_id]["Alpha(A_i)"] if prev_group_id in group_stats else float("-inf")

    prev_mvi = group_stats[prev_group_id]["MVI"] if prev_group_id in group_stats else 0

    new_mvi = max(0, new_alpha - next_alpha)
    new_prev_mvi = max(0, prev_alpha - new_alpha)

    impact = (current_mvi - new_mvi) + (prev_mvi - new_prev_mvi)

    return impact, new_mvi, new_prev_mvi


def calculate_tuple_removal_impact_max(group_data, group_stats, group_id, group_impacts):
    """Calculate impact of removing max tuples in a group for max aggregation."""
    max_value = group_stats[group_id]["Alpha(A_i)"]

    tuples_to_remove = group_data[group_id][max_value]
    remaining_values = [v for v in group_data[group_id] if v != max_value]

    #todo: can I avoid using max here?
    new_alpha = max(remaining_values) if remaining_values else 0
    impact, new_mvi, new_prev_mvi = calculate_impact(group_stats, group_id, new_alpha)

    group_impacts[group_id] = [
        {
            "tuple_index": index,
            "value": max_value,
            "impact": impact,
            "group_id": group_id,
            "new_mvi": new_mvi,
            "new_prev_mvi": new_prev_mvi,
            "new_alpha": new_alpha
        }
        for index in tuples_to_remove
    ]


def calculate_tuple_removal_impact_sum(group_data, group_stats, group_id, group_impacts):
    """Calculate impact of removing each tuple in a group for sum aggregation."""
    current_alpha = group_stats[group_id]["Alpha(A_i)"]
    impacts = []

    for value, indices in group_data[group_id].items():
        new_alpha = current_alpha - value
        impact, new_mvi, new_prev_mvi = calculate_impact(group_stats, group_id, new_alpha)
        for index in indices:
            impacts.append({
                "tuple_index": index,
                "value": value,
                "impact": impact,
                "group_id": group_id,
                "new_mvi": new_mvi,
                "new_prev_mvi": new_prev_mvi,
                "new_alpha": new_alpha
            })

    group_impacts[group_id] = impacts


def calculate_tuple_removal_impact_avg(group_data, group_stats, group_id, group_impacts, group_sums, group_counts):
    """Calculate impact of removing each tuple in a group for avg aggregation."""
    impacts = []

    for value, indices in group_data[group_id].items():
        if group_counts[group_id] > 1:
            new_alpha = (group_sums[group_id] - value) / (group_counts[group_id] - 1)
        else:
            new_alpha = 0

        impact, new_mvi, new_prev_mvi = calculate_impact(group_stats, group_id, new_alpha)
        for index in indices:
            impacts.append({
                "tuple_index": index,
                "value": value,
                "impact": impact,
                "group_id": group_id,
                "new_mvi": new_mvi,
                "new_prev_mvi": new_prev_mvi,
                "new_alpha": new_alpha
            })

    group_impacts[group_id] = impacts

def calculate_tuple_removal_impact_median(group_data, group_stats, group_id, group_impacts):
    impacts = []

    for value, indices in group_data[group_id].items():
        remaining_values = [v for v in group_data[group_id] if v != value]
        new_alpha = np.median(remaining_values) if remaining_values else 0
        impact, new_mvi, new_prev_mvi = calculate_impact(group_stats, group_id, new_alpha)
        for index in indices:
            impacts.append({
                "tuple_index": index,
                "value": value,
                "impact": impact,
                "group_id": group_id,
                "new_mvi": new_mvi,
                "new_prev_mvi": new_prev_mvi,
                "new_alpha": new_alpha
            })

    group_impacts[group_id] = impacts


def calculate_group_impacts(group, group_data, group_stats, agg_func, group_impacts, group_impact_calculated, group_sums=None, group_counts=None):
    if not group_impact_calculated[group]:
        if agg_func == "max":
            calculate_tuple_removal_impact_max(group_data, group_stats, group, group_impacts)
        elif agg_func == "sum":
            calculate_tuple_removal_impact_sum(group_data, group_stats, group, group_impacts)
        elif agg_func in {"mean", "avg"}:
            calculate_tuple_removal_impact_avg(group_data, group_stats, group, group_impacts,
                                               group_sums, group_counts)
        elif agg_func == "median":
            calculate_tuple_removal_impact_median(group_data, group_stats, group, group_impacts)

        group_impact_calculated[group] = True

# --- Greedy Algorithm ---
def initialize_group_data_and_stats(df, grouping_column, aggregation_column, agg_func):
    group_data = preprocess_group_values_with_indices(df, grouping_column, aggregation_column)
    group_stats = calculate_group_stats(df, agg_func, grouping_column, aggregation_column)
    Smvi = sum(stats["MVI"] for stats in group_stats.values())

    # Precompute initial sums and counts for each group (for avg aggregation)
    group_sums, group_counts = None, None
    if agg_func in {"mean", "avg"}:
        group_sums = {group: sum(v * len(i) for v, i in group_data[group].items()) for group in group_data}
        group_counts = {group: sum(len(i) for i in group_data[group].values()) for group in group_data}

    # Initialize group impact tracking
    group_impact_calculated = {group_id: False for group_id in group_stats}
    group_impacts = {group_id: [] for group_id in group_stats}

    return group_data, group_stats, Smvi, group_sums, group_counts, group_impact_calculated, group_impacts

def handle_removal_and_update(group_data, group_stats, max_impact_data, agg_func, group_sums, group_counts,
                              removed_indices, group_impact_calculated):
    max_impact_idx = max_impact_data["tuple_index"]
    max_impact_value = max_impact_data["value"]
    max_impact_group = max_impact_data["group_id"]

    if agg_func == "max":
        tuples_to_remove = group_data[max_impact_group][max_impact_value]
        removed_indices.extend(tuples_to_remove)
        del group_data[max_impact_group][max_impact_value]
        num_tuples_to_remove = len(tuples_to_remove)
    else:
        group_data[max_impact_group][max_impact_value].remove(max_impact_idx)
        if not group_data[max_impact_group][max_impact_value]:
            del group_data[max_impact_group][max_impact_value]
        removed_indices.append(max_impact_idx)
        num_tuples_to_remove = 1

    if agg_func in {"mean", "avg"}:
        group_sums[max_impact_group] -= max_impact_value
        group_counts[max_impact_group] -= 1

    group_impact_calculated[max_impact_group] = False
    if max_impact_group - 1 in group_stats:
        group_impact_calculated[max_impact_group - 1] = False
    if max_impact_group + 1 in group_stats:
        group_impact_calculated[max_impact_group + 1] = False

    return num_tuples_to_remove, max_impact_group, max_impact_value

def log_iteration(iteration_logs, iteration, Smvi, max_impact_group, max_impact_value, num_tuples_to_remove, max_impact_impact, fallback_used):
    iteration_logs.append({
        "Iteration": iteration,
        "Original Smvi": Smvi,
        "Tuple Removed Group Value": max_impact_group,
        "Tuple Removed Aggregation Value": max_impact_value,
        "Number of Tuples Removed": num_tuples_to_remove,
        "Impact": max_impact_impact,
        "Fallback Used": fallback_used
    })

def greedy_algorithm(df, agg_func, grouping_column, aggregation_column, output_csv):
    """Greedy algorithm to minimize Smvi by removing tuples."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    tuple_removals = 0
    start_time = time.time()
    iteration_logs = []
    removed_indices = []

    group_data, group_stats, Smvi, group_sums, group_counts, group_impact_calculated, group_impacts = initialize_group_data_and_stats(df, grouping_column, aggregation_column, agg_func)

    progress_bar = tqdm(desc="Greedy loop", unit=" iteration")
    while Smvi > 0:
        violating_groups = [group_id for group_id, stats in group_stats.items() if stats["MVI"] > 0]

        if agg_func in {"avg", "mean", "median"}: # In the case of these aggregation functions, we can also try to inc the alpha val of group+1
            violating_groups += [group_id + 1 for group_id in violating_groups if (group_id + 1) in group_stats]

        for group in violating_groups:
            calculate_group_impacts(group, group_data, group_stats, agg_func, group_impacts,
                                    group_impact_calculated, group_sums, group_counts)

        # Find the maximum impact
        max_impact_data = max((impact_data
                               for group_id in violating_groups
                               for impact_data in group_impacts[group_id]),
                              key=lambda x: x["impact"],
                              default=None)

        if max_impact_data is None:
            print("No valid impacts found. Stopping.")
            break

        max_impact_impact = max_impact_data["impact"]
        fallback_used = max_impact_impact <= 0

        num_tuples_to_remove, max_impact_group, max_impact_value = handle_removal_and_update(group_data, group_stats,
                                                                                             max_impact_data, agg_func,
                                                                                             group_sums, group_counts,
                                                                                             removed_indices, group_impact_calculated)


        log_iteration(iteration_logs, iteration, Smvi, max_impact_group, max_impact_value,
                      num_tuples_to_remove, max_impact_impact, fallback_used)

        Smvi -= max_impact_impact
        update_group_stats(group_stats, max_impact_group,
                           max_impact_data["new_alpha"], max_impact_data["new_mvi"], max_impact_data["new_prev_mvi"])

        tuple_removals += num_tuples_to_remove
        iteration += 1
        progress_bar.set_postfix({"Current Smvi": Smvi, "Fallback Used": fallback_used})
        progress_bar.update(1)

    print("No violations remain. Algorithm completed.")
    progress_bar.close()
    end_time = time.time()
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration logs saved to {output_csv}")

    result_df = df.drop(index=removed_indices)
    return result_df


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm with a specified aggregation function.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("agg_func", type=str, choices=["sum", "max", "avg", "median"], help="The aggregation function to use.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The column to group by.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The column to aggregate.")
    parser.add_argument("--output_folder", type=str, default="results", help="The folder to save the output CSV files.")
    args = parser.parse_args()

    csv_name = os.path.basename(args.csv_file)  # Extract the file name without the path
    function_map = {"sum": "sum", "max": "max", "avg": "mean", "median": "median"}
    agg_func_name = args.agg_func.upper()
    pandas_agg_func = function_map[args.agg_func]

    print(f"Processing file: {csv_name} with aggregation function: {agg_func_name}")

    df = pd.read_csv(args.csv_file)[[args.grouping_column, args.aggregation_column]].copy()

    # todo: hack because price is << 1:
    if args.csv_file == "may_transactions.csv" or args.csv_file == "may_transactions-reduced.csv":
        df['price'] = df['price']*1000

    output_csv = os.path.join(args.output_folder, f"logs-{csv_name}-{agg_func_name}.csv")
    result_df = greedy_algorithm(df, pandas_agg_func, grouping_column=args.grouping_column, aggregation_column=args.aggregation_column, output_csv=output_csv)
    print("Finished running the greedy algorithm.")

    print("Plotting results...")
    # Plot aggregated values before and after the algorithm
    plot_aggregation(df, args.grouping_column, args.aggregation_column,
                     f"Before Algorithm - {csv_name}({agg_func_name})",
                     f"{args.output_folder}/plots/before_algorithm-{agg_func_name}-{csv_name}.pdf",
                     pandas_agg_func, agg_func_name)
    plot_aggregation(result_df, args.grouping_column, args.aggregation_column,
                     f"After Algorithm - {csv_name}({agg_func_name})",
                     f"{args.output_folder}/plots/after_algorithm.pdf-{agg_func_name}-{csv_name}.pdf",
                     pandas_agg_func, agg_func_name)

    # Plot impact per iteration
    plot_impact_per_iteration(output_csv, f"{args.output_folder}/plots/impact_per_iteration-{agg_func_name}-{csv_name}.pdf")
