import pandas as pd
import numpy as np
import time
import argparse
import os
from tqdm import tqdm
tqdm.pandas()
from plots_for_main import plot_aggregation, plot_impact_per_iteration

# --- Preprocessing ---
def preprocess_sort(df, grouping_column, aggregation_column):
    """Sort DataFrame by groups and values in descending order."""
    return df.sort_values(by=[grouping_column, aggregation_column], ascending=[True, False]).reset_index(drop=True)

# --- MVI Calculation ---
def calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column):
    """
    Calculate Measure of Violations Index (MVI) for adjacent groups.
    Aggregates group values using the provided function (agg_func).
    """
    grouped_df = sorted_df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    grouped_df.rename(columns={aggregation_column: "Alpha(A_i)"}, inplace=True)
    grouped_df["MVI"] = grouped_df["Alpha(A_i)"].diff(-1).fillna(0).clip(lower=0)
    return grouped_df

def calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column):
    """
    Calculate Alpha(A_i) and MVI ( Measure of Violations Index) for adjacent groups.
    """
    group_agg = sorted_df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    alphas = group_agg[aggregation_column].values
    mvis = np.append(alphas[:-1] - alphas[1:], 0)
    mvis = np.maximum(0, mvis)  # Replace negative values with 0

    group_stats = {
        group: {"Alpha(A_i)": alpha, "MVI": mvi, "prev": group - 1, "next": group + 1}
        for group, alpha, mvi in zip(group_agg[grouping_column], alphas, mvis)
    }

    return group_stats


# # def update_group_stats(grouped_df, max_impact_group, new_group_alpha, new_group_mvi, new_prev_group_mvi, grouping_column):
#     """
#     Update Alpha(A_i) and MVI fields for max_impact_group and its neighbors in grouped_df.
#     """
#     # Update Alpha(A_i) for max_impact_group
#     grouped_df.loc[grouped_df[grouping_column] == max_impact_group, "Alpha(A_i)"] = new_group_alpha
#
#     # Update MVI for max_impact_group
#     grouped_df.loc[grouped_df[grouping_column] == max_impact_group, "MVI"] = new_group_mvi
#
#     # Update MVI for max_impact_group-1 (if it exists)
#     if max_impact_group - 1 in grouped_df[grouping_column].values:
#         grouped_df.loc[grouped_df[grouping_column] == max_impact_group - 1, "MVI"] = new_prev_group_mvi
#
#     return grouped_df


# --- Group Updates ---
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


# --- Impact Calculation ---


def calculate_tuple_removal_impact_max(sorted_df, group_stats, group_id, grouping_column, aggregation_column, group_impacts):
    """Calculate impact of removing max tuples in a group for max aggregation."""
    max_value = group_stats[group_id]["Alpha(A_i)"]
    group_tuples = sorted_df[sorted_df[grouping_column] == group_id]

    tuples_to_remove = group_tuples[group_tuples[aggregation_column] == max_value]
    remaining_tuples = group_tuples[group_tuples[aggregation_column] != max_value]

    #todo: in this feutre I can use the fat the df is sorted to find the new max value
    new_alpha = remaining_tuples[aggregation_column].max() if not remaining_tuples.empty else 0
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
        for index in tuples_to_remove.index
    ]

def calculate_tuple_removal_impact_sum(sorted_df, group_stats, group_id, grouping_column, aggregation_column, group_impacts):
    """Calculate impact of removing each tuple in a group for sum aggregation."""
    group_tuples = sorted_df[sorted_df[grouping_column] == group_id]
    tuple_values = group_tuples[aggregation_column].values

    current_alpha = group_stats[group_id]["Alpha(A_i)"]
    new_alphas = current_alpha - tuple_values

    impacts = [
        calculate_impact(group_stats, group_id, new_alpha)
        for new_alpha in new_alphas
    ]

    group_impacts[group_id] = [
        {
            "tuple_index": group_tuples.index[i],
            "value": tuple_values[i],
            "impact": impact[0],
            "group_id": group_id,
            "new_mvi": impact[1],
            "new_prev_mvi": impact[2],
            "new_alpha": new_alpha,
        }
        for i, (new_alpha, impact) in enumerate(zip(new_alphas, impacts))
    ]


def calculate_tuple_removal_impact_avg(sorted_df, group_stats, group_id, grouping_column, aggregation_column, group_sums, group_counts, group_impacts):
    """Calculate impact of removing each tuple in a group for avg aggregation."""
    group_tuples = sorted_df[sorted_df[grouping_column] == group_id]
    tuple_values = group_tuples[aggregation_column].values

    if group_counts[group_id] > 1:
        new_alphas = (group_sums[group_id] - tuple_values) / (group_counts[group_id] - 1)
    else:
        new_alphas = np.zeros_like(tuple_values)

    impacts = [
        calculate_impact(group_stats, group_id, new_alpha)
        for new_alpha in new_alphas
    ]

    group_impacts[group_id] = [
        {
            "tuple_index": group_tuples.index[i],
            "value": tuple_values[i],
            "impact": impact[0],
            "group_id": group_id,
            "new_mvi": impact[1],
            "new_prev_mvi": impact[2],
            "new_alpha": new_alpha,
        }
        for i, (new_alpha, impact) in enumerate(zip(new_alphas, impacts))
    ]


# --- Greedy Algorithm ---
def greedy_algorithm(df, agg_func, grouping_column, aggregation_column, output_csv):
    """Greedy algorithm to minimize Smvi by removing tuples."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    tuple_removals = 0
    start_time = time.time()
    iteration_logs = []

    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)
    group_stats = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)
    Smvi = sum(stats["MVI"] for stats in group_stats.values())

    # Precompute initial sums and counts for each group
    group_sums = None
    group_counts = None
    if agg_func in {"mean", "avg"}:
        group_sums = sorted_df.groupby(grouping_column)[aggregation_column].sum().to_dict()
        group_counts = sorted_df.groupby(grouping_column)[aggregation_column].count().to_dict()

    # Initialize group impact tracking
    group_impact_calculated = {group_id: False for group_id in group_stats}
    group_impacts = {group_id: [] for group_id in group_stats}

    progress_bar = tqdm(desc="Greedy loop", unit=" iteration")

    while Smvi > 0:
        violating_groups = [group_id for group_id, stats in group_stats.items() if stats["MVI"] > 0]

        for group in violating_groups:
            if not group_impact_calculated[group]:
                if agg_func == "max":
                    calculate_tuple_removal_impact_max(sorted_df, group_stats, group, grouping_column,
                                                        aggregation_column, group_impacts)
                elif agg_func == "sum":
                    calculate_tuple_removal_impact_sum(sorted_df, group_stats, group, grouping_column,
                                                        aggregation_column, group_impacts)
                elif agg_func in {"mean", "avg"}:
                    calculate_tuple_removal_impact_avg(sorted_df, group_stats, group, grouping_column,
                                                        aggregation_column, group_sums, group_counts, group_impacts)
                group_impact_calculated[group] = True

        # Find the maximum impact
        max_impact_data = max((impact_data
                               for group_id in violating_groups
                               for impact_data in group_impacts[group_id]),
                              key=lambda x: x["impact"],
                              default=None)

        if max_impact_data is None:
            print("No valid impacts found. Stopping.")
            break

        max_impact_idx, max_impact_value, max_impact_impact, max_impact_group, new_group_mvi, new_prev_group_mvi, new_group_alpha = (
            max_impact_data["tuple_index"],
            max_impact_data["value"],
            max_impact_data["impact"],
            max_impact_data["group_id"],
            max_impact_data["new_mvi"],
            max_impact_data["new_prev_mvi"],
            max_impact_data["new_alpha"]
        )

        fallback_used = False
        if max_impact_impact <= 0:
            fallback_used = True

        # todo: I think in the case of max the index will be evrything in the impact data and I dont need the if
        if agg_func == "max":
            tuples_to_remove = [impact_data["tuple_index"]
                                for impact_data in group_impacts[max_impact_group]
                                if impact_data["value"] == group_stats[max_impact_group]["Alpha(A_i)"]]
            num_tuples_to_remove = len(tuples_to_remove)
        else:
            tuples_to_remove = max_impact_idx
            num_tuples_to_remove = 1

        if agg_func in {"mean", "avg"}:
            group_sums[max_impact_group] -= max_impact_value
            group_counts[max_impact_group] -= 1

        #todo: do this in the end, keep track of the tuples to remove so I can remove them in the end and still calculate the impact correctly
        #todo do i need to do = here?
        sorted_df = sorted_df.drop(index=tuples_to_remove).reset_index(drop=True)

        group_impact_calculated[max_impact_group] = False
        if max_impact_group - 1 in group_stats:
            group_impact_calculated[max_impact_group - 1] = False
        if max_impact_group + 1 in group_stats:
            group_impact_calculated[max_impact_group + 1] = False


        iteration_logs.append({
            "Iteration": iteration,
            "Original Smvi": Smvi,
            "Tuple Removed Group Value": max_impact_group,
            "Tuple Removed Aggregation Value": max_impact_value,
            "Number of Tuples Removed": num_tuples_to_remove,
            "Impact": max_impact_impact,
            "Fallback Used": fallback_used
        })

        Smvi -= max_impact_impact
        tuple_removals += num_tuples_to_remove
        iteration += 1

        #todo switch back after bug is fixed
        #group_stats = update_group_stats(grouped_df, max_impact_group, new_group_mvi, new_prev_group_mvi, new_group_alpha, grouping_column)
        group_stats = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)

        progress_bar.set_postfix({"Current Smvi": Smvi, "Fallback Used": fallback_used})
        progress_bar.update(1)

    print("No violations remain. Algorithm completed.")
    progress_bar.close()
    end_time = time.time()
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration logs saved to {output_csv}")
    return sorted_df

# --- Main Entry Point ---
#python aggr-main.py <input_csv> <agg_func> --grouping_column <grouping_column> --aggregation_column <aggregation_column> --output_csv <output_csv>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm with a specified aggregation function.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("agg_func", type=str, choices=["sum", "max", "avg"], help="The aggregation function to use.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The column to group by.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The column to aggregate.")
    parser.add_argument("--output_folder", type=str, default="results", help="The folder to save the output CSV files.")
    args = parser.parse_args()

    csv_name = os.path.basename(args.csv_file)  # Extract the file name without the path
    function_map = {"sum": "sum", "max": "max", "avg": "mean"}
    agg_func_name = args.agg_func.upper()
    pandas_agg_func = function_map[args.agg_func]

    print(f"Processing file: {csv_name} with aggregation function: {agg_func_name}")

    df = pd.read_csv(args.csv_file)
    df = df[[args.grouping_column, args.aggregation_column]]
    # todo: terrible hack:
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
