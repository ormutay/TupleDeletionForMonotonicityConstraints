import pandas as pd
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
def calculate_impact(grouped_df, group_index, new_group_alpha):
    """Update MVI for the affected group (i) and its neighbors."""
    original_group_mvi = grouped_df.loc[group_index, "MVI"]
    original_prev_group_mvi = grouped_df.loc[group_index - 1, "MVI"] if group_index - 1 in grouped_df.index else 0

    next_group_alpha = grouped_df.loc[group_index + 1, "Alpha(A_i)"] if group_index + 1 in grouped_df.index else float("inf")
    prev_group_alpha = grouped_df.loc[group_index - 1, "Alpha(A_i)"] if group_index - 1 in grouped_df.index else float("-inf")

    # Update MVI for group i
    new_group_mvi = max(0, new_group_alpha - next_group_alpha)
    # Update MVI for group i-1
    new_prev_group_mvi = max(0, prev_group_alpha - new_group_alpha)

    impact = (original_group_mvi - new_group_mvi) + (original_prev_group_mvi - new_prev_group_mvi)

    return impact, new_group_mvi, new_prev_group_mvi

# --- Impact Calculation ---
def calculate_tuple_impact_helper(grouped_df, group_index, row, aggregation_column, grouping_column):
    """Helper function for the impact calculation."""
    impact, new_group_mvi, new_prev_group_mvi = calculate_impact(grouped_df, group_index, row['new_alpha'])
    return row.name, row[aggregation_column], impact, row[grouping_column], new_group_mvi, new_prev_group_mvi, row['new_alpha']


def calculate_tuple_removal_impact_max(sorted_df, grouped_df, group, grouping_column, aggregation_column, group_impacts):
    """Calculate the impact of removing all max value tuples in a group."""
    #todo I can precalculate the group index so I dont have to do it each time
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]
    orig_alpha = grouped_df.iloc[group_index]["Alpha(A_i)"]
    group_tuples = sorted_df[sorted_df[grouping_column] == group]
    max_tuples = group_tuples[group_tuples[aggregation_column] == orig_alpha]

    #todo check if I can avoid dropping the max tuples
    #remaining_values = group_tuples.loc[~group_tuples.index.isin(max_tuples.index), aggregation_column].values
    #new_group_alpha = np.max(remaining_values) if remaining_values.size > 0 else 0
    remaining_group = group_tuples.drop(index=max_tuples.index)
    new_group_alpha = remaining_group[aggregation_column].max() if not remaining_group.empty else 0
    impact, new_group_mvi, new_prev_group_mvi = calculate_impact(grouped_df, group_index, new_group_alpha)

    group_impacts[group] = [(idx, orig_alpha, impact, group, new_group_mvi, new_prev_group_mvi, new_group_alpha) for idx in max_tuples.index]

#todo: can I not use copy here?
def calculate_tuple_removal_impact_sum(sorted_df, grouped_df, group, grouping_column, aggregation_column, group_impacts):
    """Calculate the impact of removing each tuple in a group individually."""
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]
    group_tuples = sorted_df[sorted_df[grouping_column] == group].copy()

    orig_alpha = grouped_df.loc[group_index, "Alpha(A_i)"]
    group_tuples['new_alpha'] = orig_alpha - group_tuples[aggregation_column]
    group_impacts[group] = group_tuples.apply(lambda row: calculate_tuple_impact_helper(grouped_df, group_index, row, aggregation_column, grouping_column), axis=1).tolist()


def calculate_tuple_removal_impact_avg(sorted_df, grouped_df, group, grouping_column, aggregation_column, group_sums, group_counts, group_impacts):
    """Calculate the impact of removing each tuple in a group individually for AVG."""
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]
    group_tuples = sorted_df[sorted_df[grouping_column] == group].copy()

    if group_counts[group] > 1:
        group_tuples['new_alpha'] = (group_sums[group] - group_tuples[aggregation_column]) / (group_counts[group] - 1)
    else:
        group_tuples['new_alpha'] = 0

    group_impacts[group] = group_tuples.apply(lambda row: calculate_tuple_impact_helper(grouped_df, group_index, row, aggregation_column, grouping_column), axis=1).tolist()

# --- Greedy Algorithm ---
def greedy_algorithm(df, agg_func, grouping_column, aggregation_column, output_csv):
    """Greedy algorithm to minimize Smvi by removing tuples."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)
    tuple_removals = 0
    start_time = time.time()
    iteration_logs = []

    # Precompute initial sums and counts for each group
    group_sums = None
    group_counts = None
    if agg_func in {"mean", "avg"}:
        group_sums = sorted_df.groupby(grouping_column)[aggregation_column].sum().to_dict()
        group_counts = sorted_df.groupby(grouping_column)[aggregation_column].count().to_dict()

    # Initialize group impact tracking
    groups = sorted_df[grouping_column].unique()
    group_impact_calculated = {group: False for group in groups}
    group_impacts = {group: [] for group in groups}

    progress_bar = tqdm(desc="Greedy loop", unit=" iteration")

    grouped_df = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)
    Smvi = grouped_df["MVI"].sum()

    while Smvi > 0:

        violating_groups = grouped_df[grouped_df["MVI"] > 0][grouping_column].unique()

        for group in violating_groups:
            if not group_impact_calculated[group]:
                if agg_func == "max":
                    calculate_tuple_removal_impact_max(sorted_df, grouped_df, group, grouping_column,
                                                              aggregation_column, group_impacts)
                elif agg_func == "sum":
                    calculate_tuple_removal_impact_sum(sorted_df, grouped_df, group, grouping_column,
                                                              aggregation_column, group_impacts)
                elif agg_func in {"mean", "avg"}:
                    calculate_tuple_removal_impact_avg(sorted_df, grouped_df, group, grouping_column,
                                                              aggregation_column, group_sums, group_counts,
                                                              group_impacts)
                else:
                    raise ValueError("Unsupported aggregation function")
                group_impact_calculated[group] = True

        impacts = [impact
                   for group in violating_groups
                   for impact in group_impacts[group]
                   ]

        # Flatten and validate impacts
        impacts = [item for sublist in impacts for item in (sublist if isinstance(sublist, list) else [sublist])]
        impacts = [impact for impact in impacts if isinstance(impact, (list, tuple)) and len(impact) >= 7]

        if not impacts:
            print("No valid impacts found. Stopping.")
            break

        # Find the maximum impact
        max_impact = max(impacts, key=lambda x: x[2])
        max_impact_idx, max_impact_value, max_impact_impact, max_impact_group, new_group_mvi, new_prev_group_mvi, new_group_alpha = max_impact

        fallback_used = False
        if max_impact_impact <= 0:
            fallback_used = True

        if agg_func == "max":
            tuples_to_remove = [idx for idx, max_value, _, group, _, _, _ in impacts if max_value == max_impact_value and group == max_impact_group]
            num_tuples_to_remove = len(tuples_to_remove)
        else:
            tuples_to_remove = max_impact_idx
            num_tuples_to_remove = 1

        # todo: do this in the end, keep track of the tuples to remove so I can remove them in the end and still calculate the impact correctly
        #todo do i need to do = here?
        sorted_df = sorted_df.drop(index=tuples_to_remove).reset_index(drop=True)

        group_impact_calculated[max_impact_group] = False
        if max_impact_group - 1 in groups:
            group_impact_calculated[max_impact_group - 1] = False
        if max_impact_group + 1 in groups:
            group_impact_calculated[max_impact_group + 1] = False

        if agg_func in {"mean", "avg"}:
            group_sums[max_impact_group] -= max_impact_value
            group_counts[max_impact_group] -= 1

        Smvi -= max_impact_impact

        iteration_logs.append({
            "Iteration": iteration,
            "Updated Smvi": Smvi,
            "Tuple Removed Group Value": max_impact_group,
            "Tuple Removed Aggregation Value": max_impact_value,
            "Number of Tuples Removed": num_tuples_to_remove,
            "Impact": max_impact_impact,
            "Fallback Used": fallback_used
        })

        tuple_removals += num_tuples_to_remove
        iteration += 1

        #todo switch back after bug is fixed
        #grouped_df = update_group_stats(grouped_df, max_impact_group, new_group_mvi, new_prev_group_mvi, new_group_alpha, grouping_column)
        grouped_df = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)

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
