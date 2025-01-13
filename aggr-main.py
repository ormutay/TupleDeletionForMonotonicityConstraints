import pandas as pd
import numpy as np
import time
import argparse
import os
from tqdm import tqdm
tqdm.pandas()
from plots_for_main import plot_aggregation, plot_impact_per_iteration
from sortedcontainers import SortedList

# --- Preprocessing ---
def preprocess_group_values_with_indices(df, grouping_column, aggregation_column):
    """Preprocess to save values and their indices grouped by group ID."""
    groups_data = {}
    for group_id, group_df in df.groupby(grouping_column):
        groups_data[group_id] = {}
        for value, indices in group_df.groupby(aggregation_column).groups.items():
            groups_data[group_id][value] = list(indices)

    return groups_data


def calculate_groups_stats(df, agg_func, grouping_column, aggregation_column):
    """Calculate Alpha(A_i) and MVI ( Measure of Violations Index) for adjacent groups."""
    group_agg = df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    alphas = group_agg[aggregation_column].values
    mvis = np.append(alphas[:-1] - alphas[1:], 0)
    mvis = np.maximum(0, mvis)  # Replace negative values with 0

    groups_stats = {
        group: {"Alpha(A_i)": alpha, "MVI": mvi, "prev": group - 1, "next": group + 1}
        for group, alpha, mvi in zip(group_agg[grouping_column], alphas, mvis)
    }

    return groups_stats


# --- Group Updates ---
def update_groups_stats(groups_stats, max_impact_group, new_group_alpha, new_group_mvi, new_prev_group_mvi):
    """Update Alpha(A_i) and MVI fields for max_impact_group and its neighbors in groups_stats."""
    # Update Alpha(A_i) and MVI for max_impact_group
    groups_stats[max_impact_group]["Alpha(A_i)"] = new_group_alpha
    groups_stats[max_impact_group]["MVI"] = new_group_mvi

    # Update MVI for the previous group (if it exists)
    prev_group = groups_stats[max_impact_group]["prev"]
    if prev_group in groups_stats:
        groups_stats[prev_group]["MVI"] = new_prev_group_mvi


# --- Impact Calculation ---
def calculate_impact(groups_stats, group_id, new_alpha):
    """ Calculate the impact of deleting a tuple (that creates a new alpha value). """
    current_mvi, next_group_id, prev_group_id = groups_stats[group_id]["MVI"], groups_stats[group_id]["next"], groups_stats[group_id]["prev"]

    next_alpha = groups_stats[next_group_id]["Alpha(A_i)"] if next_group_id in groups_stats else float("inf")
    prev_alpha = groups_stats[prev_group_id]["Alpha(A_i)"] if prev_group_id in groups_stats else float("-inf")

    prev_mvi = groups_stats[prev_group_id]["MVI"] if prev_group_id in groups_stats else 0

    new_mvi = max(0, new_alpha - next_alpha)
    new_prev_mvi = max(0, prev_alpha - new_alpha)

    impact = (current_mvi - new_mvi) + (prev_mvi - new_prev_mvi)

    return impact, new_mvi, new_prev_mvi


def calculate_tuple_removal_impact_max(groups_data, groups_stats, group_id, groups_impacts):
    """Calculate impact of removing max tuples in a group for max aggregation."""
    max_value = groups_stats[group_id]["Alpha(A_i)"]

    tuples_to_remove = groups_data[group_id][max_value]
    remaining_values = [v for v in groups_data[group_id] if v != max_value]

    #todo: can I avoid using max here?
    new_alpha = max(remaining_values) if remaining_values else 0
    impact, new_mvi, new_prev_mvi = calculate_impact(groups_stats, group_id, new_alpha)

    groups_impacts[group_id] = [
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


def calculate_tuple_removal_impact_sum(groups_data, groups_stats, group_id, groups_impacts):
    """Calculate impact of removing each tuple in a group for sum aggregation."""
    current_alpha = groups_stats[group_id]["Alpha(A_i)"]
    impacts = []

    for value, indices in groups_data[group_id].items():
        new_alpha = current_alpha - value
        impact, new_mvi, new_prev_mvi = calculate_impact(groups_stats, group_id, new_alpha)
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

    groups_impacts[group_id] = impacts


def calculate_tuple_removal_impact_avg(groups_data, groups_stats, group_id, groups_impacts, groups_sum, groups_count):
    """Calculate impact of removing each tuple in a group for avg aggregation."""
    impacts = []

    for value, indices in groups_data[group_id].items():
        if groups_count[group_id] > 1:
            new_alpha = (groups_sum[group_id] - value) / (groups_count[group_id] - 1)
        else:
            new_alpha = 0

        impact, new_mvi, new_prev_mvi = calculate_impact(groups_stats, group_id, new_alpha)
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

    groups_impacts[group_id] = impacts


def v2_calculate_tuple_removal_impact_median(groups_stats, group_id, groups_impacts, groups_sorted_values, groups_count):
    impacts = []

    sorted_values = groups_sorted_values[group_id]
    len_after_removal = groups_count[group_id] - 1
    median_idx = len_after_removal // 2
    odd_number_of_elements = len_after_removal % 2 == 1

    for value, index in sorted_values:
        sorted_values.remove((value, index))

        if odd_number_of_elements:
            new_alpha = sorted_values[median_idx][0]
        else:
            new_alpha = (sorted_values[median_idx - 1][0] + sorted_values[median_idx][0]) / 2

        impact, new_mvi, new_prev_mvi = calculate_impact(groups_stats, group_id, new_alpha)
        impacts.append({
            "tuple_index": index,
            "value": value,
            "impact": impact,
            "group_id": group_id,
            "new_mvi": new_mvi,
            "new_prev_mvi": new_prev_mvi,
            "new_alpha": new_alpha
        })

        sorted_values.add((value, index))

    groups_impacts[group_id] = impacts

def v3_calculate_tuple_removal_impact_median(groups_stats, group_id, groups_impacts, groups_sorted_values, groups_count):
    impacts = []

    sorted_values = groups_sorted_values[group_id]
    group_count = groups_count[group_id]
    current_median_idx = group_count // 2
    odd_number_of_elements = group_count % 2 == 1

    if odd_number_of_elements:
        new_alpha_index_smaller_than_median = (sorted_values[current_median_idx][0] + sorted_values[current_median_idx + 1][0]) / 2
        new_alpha_index_equal_than_median = (sorted_values[current_median_idx - 1][0] + sorted_values[current_median_idx + 1][0]) / 2
        new_alpha_index_bigger_than_median = (sorted_values[current_median_idx - 1][0] + sorted_values[current_median_idx][0]) / 2
    else:
        new_alpha_index_smaller_than_median = sorted_values[current_median_idx][0]
        new_alpha_index_equal_than_median = sorted_values[current_median_idx - 1][0]
        new_alpha_index_bigger_than_median = sorted_values[current_median_idx - 1][0]

    impact_smaller, new_mvi_smaller, new_prev_mvi_smaller = calculate_impact(groups_stats, group_id, new_alpha_index_smaller_than_median)
    impact_equal, new_mvi_equal, new_prev_mvi_equal = calculate_impact(groups_stats, group_id, new_alpha_index_equal_than_median)
    impact_bigger, new_mvi_bigger, new_prev_mvi_bigger = calculate_impact(groups_stats, group_id, new_alpha_index_bigger_than_median)

    for list_index, (value, df_index) in enumerate(sorted_values):
        if list_index < current_median_idx:
            new_alpha = new_alpha_index_smaller_than_median
            impact, new_mvi, new_prev_mvi = impact_smaller, new_mvi_smaller, new_prev_mvi_smaller
        elif list_index == current_median_idx:
            new_alpha = new_alpha_index_equal_than_median
            impact, new_mvi, new_prev_mvi = impact_equal, new_mvi_equal, new_prev_mvi_equal
        else:
            new_alpha = new_alpha_index_bigger_than_median
            impact, new_mvi, new_prev_mvi = impact_bigger, new_mvi_bigger, new_prev_mvi_bigger

        impacts.append({
            "tuple_index": df_index,
            "value": value,
            "impact": impact,
            "group_id": group_id,
            "new_mvi": new_mvi,
            "new_prev_mvi": new_prev_mvi,
            "new_alpha": new_alpha
        })

    groups_impacts[group_id] = impacts


from bisect import bisect_left, bisect_right

def v4_count_ranges(sorted_list, lower_bound, upper_bound):
    # Find indices using binary search
    low_index = bisect_left(sorted_list, lower_bound)  # First index >= lower_bound
    high_index = bisect_right(sorted_list, upper_bound)  # First index > upper_bound

    # Calculate counts
    below_count = low_index  # Values before `low_index`
    between_count = high_index - low_index  # Values in range [lower_bound, upper_bound]
    above_count = len(sorted_list) - high_index  # Values after `high_index`

    return below_count, between_count, above_count


def v4_new_calculate_tuple_removal_impact_median(groups_stats, group_id, groups_impacts, groups_sorted_values, groups_count):
    impacts = []

    sorted_values = groups_sorted_values[group_id]
    lower_bound = groups_stats[group_id-1]["Alpha(A_i)"]
    upper__bound = groups_stats[group_id-1]["Alpha(A_i)"]
    below_count, between_count, above_count = v4_count_ranges(sorted_values, lower_bound, upper__bound)
    num_tuples_to_remove_from_start = 0
    num_tuples_to_remove_from_end = 0

    #todo the problem is that both ifs can happen, and I want to be able to calculate for both and also in the end delete togther the relevent tuples
    if groups_stats[group_id - 1]["MVI"] > 0: #med to low
        num_tuples_to_remove_from_start = below_count - between_count - above_count
        new_sorted_values = sorted_values[num_tuples_to_remove_from_start:]
        removed_tuples = sorted_values[:num_tuples_to_remove_from_start]

    if groups_stats[group_id]["MVI"] > 0: #med to high
        num_tuples_to_remove_from_end = above_count - below_count - between_count
        new_sorted_values = sorted_values[:-num_tuples_to_remove_from_end]
        removed_tuples = sorted_values[-num_tuples_to_remove_from_end:]

    new_len = len(new_sorted_values)
    if new_len % 2 == 1:
        new_alpha =  new_sorted_values[new_len // 2]
    else:
        new_alpha = (new_sorted_values[new_len // 2 - 1] + new_sorted_values[new_len // 2]) / 2

    impact, new_mvi, new_prev_mvi = calculate_impact(groups_stats, group_id, new_alpha)
    for value, index in removed_tuples:
        impacts.append({
            "tuple_index": index,
            "value": value,
            "impact": impact,
            "group_id": group_id,
            "new_mvi": new_mvi,
            "new_prev_mvi": new_prev_mvi,
            "new_alpha": new_alpha
        })
    groups_impacts[group_id] = impacts


def calculate_groups_impacts(group, groups_data, groups_stats, agg_func, groups_impacts, group_impact_calculated, groups_sum, groups_count, groups_sorted_values):
    if not group_impact_calculated[group]:
        if agg_func == "max":
            calculate_tuple_removal_impact_max(groups_data, groups_stats, group, groups_impacts)
        elif agg_func == "sum":
            calculate_tuple_removal_impact_sum(groups_data, groups_stats, group, groups_impacts)
        elif agg_func in {"mean", "avg"}:
            calculate_tuple_removal_impact_avg(groups_data, groups_stats, group, groups_impacts,
                                               groups_sum, groups_count)
        elif agg_func == "median":
            v3_calculate_tuple_removal_impact_median(groups_stats, group, groups_impacts, groups_sorted_values, groups_count)

        group_impact_calculated[group] = True

# --- Greedy Algorithm ---
def initialize_group_data_and_stats(df, grouping_column, aggregation_column, agg_func):
    groups_data = preprocess_group_values_with_indices(df, grouping_column, aggregation_column)
    groups_stats = calculate_groups_stats(df, agg_func, grouping_column, aggregation_column)
    Smvi = sum(stats["MVI"] for stats in groups_stats.values())

    # Precompute initial sums and counts for each group (for avg aggregation)
    groups_sum, groups_count = None, None
    if agg_func in {"mean", "avg"}:
        groups_sum = {group: sum(v * len(i) for v, i in groups_data[group].items()) for group in groups_data}
        groups_count = {group: sum(len(i) for i in groups_data[group].values()) for group in groups_data}
    elif agg_func == "median":
        groups_count = {group: sum(len(i) for i in groups_data[group].values()) for group in groups_data}


    # dict of SortedList for each group (group_id -> SortedList(values*count, index))
    groups_sorted_values = {group: SortedList([(v, idx)
                                               for v, idx_list in groups_data[group].items()
                                               for idx in idx_list], key=lambda x: x[0])
                            for group in groups_data} # Sort by the value

    # Initialize group impact tracking
    group_impact_calculated = {group_id: False for group_id in groups_stats}
    groups_impacts = {group_id: [] for group_id in groups_stats}

    return groups_data, groups_stats, Smvi, groups_sum, groups_count, group_impact_calculated, groups_impacts, groups_sorted_values

def find_max_impact_data(groups_impacts, violating_groups, additional_groups, agg_func):
    max_impact_data = None

    if agg_func in {"avg", "mean", "median"}:
        # min value logic
        max_impact_data_additional = max((impact_data
                                          for group_id in additional_groups
                                          for impact_data in groups_impacts[group_id]),
                                         key=lambda x: (x["impact"], -x["value"]),
                                         default=None)
        # max value logic
        max_impact_data_regular = max((impact_data
                                       for group_id in violating_groups if group_id not in additional_groups
                                       for impact_data in groups_impacts[group_id]),
                                      key=lambda x: (x["impact"], x["value"]),
                                      default=None)

        # overall max impact data (prioritizing "regular" groups)
        if max_impact_data_regular is None and max_impact_data_additional is None:
            return None
        if max_impact_data_additional is None and max_impact_data_regular is not None:
            max_impact_data = max_impact_data_regular
        elif max_impact_data_additional is not None and max_impact_data_regular is None:
            max_impact_data = max_impact_data_additional
        else:
            max_impact_data = max_impact_data_additional \
                              if max_impact_data_additional["impact"] > max_impact_data_regular["impact"] \
                              else max_impact_data_regular
    else:
        max_impact_data = max((impact_data
                               for group_id in violating_groups
                               for impact_data in groups_impacts[group_id]),
                              key=lambda x: (x["impact"], x["value"]),
                              default=None)

    return max_impact_data


def handle_removal_and_update(groups_data, groups_stats, max_impact_data, agg_func, groups_sum, groups_count,
                              removed_indices, group_impact_calculated, groups_sorted_values):
    max_impact_idx = max_impact_data["tuple_index"]
    max_impact_value = max_impact_data["value"]
    max_impact_group = max_impact_data["group_id"]

    if agg_func == "max":
        tuples_to_remove = groups_data[max_impact_group][max_impact_value]
        removed_indices.extend(tuples_to_remove)
        del groups_data[max_impact_group][max_impact_value]
        num_tuples_to_remove = len(tuples_to_remove)
    else:
        groups_data[max_impact_group][max_impact_value].remove(max_impact_idx)
        if not groups_data[max_impact_group][max_impact_value]:
            del groups_data[max_impact_group][max_impact_value]
        removed_indices.append(max_impact_idx)
        num_tuples_to_remove = 1

    if agg_func in {"mean", "avg"}:
        groups_sum[max_impact_group] -= max_impact_value
        groups_count[max_impact_group] -= 1
    elif agg_func == "median":
        groups_count[max_impact_group] -= 1
        groups_sorted_values[max_impact_group].remove((max_impact_value, max_impact_idx))

    group_impact_calculated[max_impact_group] = False
    if max_impact_group - 1 in groups_stats:
        group_impact_calculated[max_impact_group - 1] = False
    if max_impact_group + 1 in groups_stats:
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

    groups_data, groups_stats, Smvi, groups_sum, groups_count, group_impact_calculated, groups_impacts, groups_sorted_values = initialize_group_data_and_stats(df, grouping_column, aggregation_column, agg_func)

    progress_bar = tqdm(desc="Greedy loop", unit=" iteration")
    while Smvi > 0:

        violating_groups = [group_id for group_id, stats in groups_stats.items() if stats["MVI"] > 0]
        additional_groups = None

        if agg_func in {"avg", "mean", "median"}: # In the case of these aggregation functions, we can also try to inc the alpha val of group+1
            additional_groups = [group_id + 1 for group_id in violating_groups if (group_id + 1) in groups_stats]
            violating_groups.extend(additional_groups)
            violating_groups = list(set(violating_groups))

        for group in violating_groups:
            calculate_groups_impacts(group, groups_data, groups_stats, agg_func, groups_impacts,
                                    group_impact_calculated, groups_sum, groups_count, groups_sorted_values)

        # Find the maximum impact
        max_impact_data = find_max_impact_data(groups_impacts, violating_groups, additional_groups, agg_func)


        if max_impact_data is None:
            print("\033[91mNo valid impacts found. Stopping.\033[0m")
            break

        max_impact_impact = max_impact_data["impact"]
        fallback_used = max_impact_impact <= 0

        num_tuples_to_remove, max_impact_group, max_impact_value = handle_removal_and_update(groups_data, groups_stats,
                                                                                             max_impact_data, agg_func,
                                                                                             groups_sum, groups_count,
                                                                                             removed_indices, group_impact_calculated,
                                                                                             groups_sorted_values)

        log_iteration(iteration_logs, iteration, Smvi, max_impact_group, max_impact_value,
                      num_tuples_to_remove, max_impact_impact, fallback_used)

        Smvi -= max_impact_impact
        update_groups_stats(groups_stats, max_impact_group,
                           max_impact_data["new_alpha"], max_impact_data["new_mvi"], max_impact_data["new_prev_mvi"])

        tuple_removals += num_tuples_to_remove
        iteration += 1
        progress_bar.set_postfix({"Current Smvi": Smvi, "Fallback Used": fallback_used,
                                  "violating groups": len(violating_groups), "impact":max_impact_data["impact"],
                                  "group": max_impact_group, "index": max_impact_data["tuple_index"]})
        progress_bar.update(1)

    progress_bar.close()
    end_time = time.time()
    print("\033[92mNo violations remain. Algorithm completed.\033[0m")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration logs saved to {output_csv}")

    removed_df = df.loc[removed_indices]

    result_df = df.drop(index=removed_indices)
    return result_df, removed_df


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

    print(f"\033[1mProcessing file: {csv_name} with aggregation function: {agg_func_name}\033[0m")

    df = pd.read_csv(args.csv_file)[[args.grouping_column, args.aggregation_column]].copy()

    # todo: hack because price is << 1:
    if "may_transactions" in csv_name:
        print("Multiplying 'price' by 1,000,000 to handle numerical instability")
        df['price'] = df['price']*10000

    output_csv = os.path.join(args.output_folder, f"logs-{csv_name}-{agg_func_name}.csv")
    result_df, removed_df = greedy_algorithm(df, pandas_agg_func, grouping_column=args.grouping_column, aggregation_column=args.aggregation_column, output_csv=output_csv)
    print("\033[1mFinished running the greedy algorithm.\033[0m")

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

    print("Saving results to csv...")
    result_df["original_index"] = result_df.index
    removed_df["original_index"] = removed_df.index
    result_df.to_csv(os.path.join(args.output_folder, f"result-{csv_name}-{agg_func_name}.csv"), index=True)
    removed_df.to_csv(os.path.join(args.output_folder, f"removed-{csv_name}-{agg_func_name}.csv"), index=True)

    print("The removed tuples are: \n", removed_df)
