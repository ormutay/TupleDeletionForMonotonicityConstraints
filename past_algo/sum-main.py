import pandas as pd
import time
import argparse
import os

# --- Preprocessing ---
def preprocess_sort(df, grouping_column="A", aggregation_column="B"):
    """Sort DataFrame by groups and values in descending order."""
    return df.sort_values(by=[grouping_column, aggregation_column], ascending=[True, False]).reset_index(drop=True)


# --- MVI Calculation ---
def calculate_group_stats(sorted_df, agg_func, grouping_column="A", aggregation_column="B"):
    """
    Calculate Measure of Violations Index (MVI) for adjacent groups.
    Aggregates group values using the provided function (agg_func).
    """
    grouped_df = sorted_df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    grouped_df.rename(columns={aggregation_column: "Alpha(A_i)"}, inplace=True)
    grouped_df["MVI"] = grouped_df["Alpha(A_i)"].diff(-1).fillna(0).clip(lower=0)
    return grouped_df

# --- Group Updates ---

def update_group_mvi(grouped_df, group_index, new_group_alpha):
    """Update MVI for the affected group (i) and its neighbors."""
    grouped_df.loc[group_index, "Alpha(A_i)"] = new_group_alpha

    # Update MVI for group i-1
    if group_index > 0:
        prev_group_index = group_index - 1
        prev_group_alpha = grouped_df.loc[prev_group_index, "Alpha(A_i)"]
        grouped_df.loc[prev_group_index, "MVI"] = max(
            0, prev_group_alpha - new_group_alpha
        )

    # Update MVI for group i
    next_group_alpha = (
        grouped_df.loc[group_index + 1, "Alpha(A_i)"]
        if group_index + 1 in grouped_df.index else float("inf")
    )
    grouped_df.loc[group_index, "MVI"] = max(0, new_group_alpha - next_group_alpha)

    return grouped_df

# --- Impact Calculation ---
def find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func, grouping_column="A", aggregation_column="B", group_sums=None):
    """Simulate removing each tuple in a group and calculate its impact."""
    impacts = []
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]
    group_tuples = sorted_df[sorted_df[grouping_column] == group]

    for _, row in group_tuples.iterrows():
        temp_df = sorted_df.drop(index=row.name)
        remaining_group = temp_df[temp_df[grouping_column] == group]

        if group_sums is not None:
            group_sums[group] -= row[aggregation_column]
            new_group_alpha = group_sums[group]
        else:
            new_group_alpha = remaining_group[aggregation_column].apply(agg_func) if not remaining_group.empty else 0

        updated_grouped_df = update_group_mvi(grouped_df.copy(), group_index, new_group_alpha)

        impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
        impacts.append((row.name, row[aggregation_column], impact))

        if group_sums is not None:
            group_sums[group] += row[aggregation_column]  # Restore the sum for subsequent calculations

    return impacts

# --- Main Algorithm ---
def greedy_algorithm(df, agg_func, output_csv="results/iteration_log.csv", grouping_column="A", aggregation_column="B"):
    """Greedy algorithm to minimize Smvi by removing tuples only and log iterations to CSV."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)

    tuple_removals = 0
    start_time = time.time()

    # Precompute initial sums for each group
    group_sums = sorted_df.groupby(grouping_column)[aggregation_column].sum().to_dict()

    # Collect iteration details for CSV
    iteration_logs = []

    while True:
        grouped_df = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)
        Smvi = grouped_df["MVI"].sum()

        if Smvi == 0:
            print("No violations remain. Algorithm completed.")
            break

        if Smvi < 0:
            print("!!!! Smvi < 0 !!!!, Algorithm completed.")
            break

        max_impact = None
        for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
            group = row[grouping_column]
            impacts = find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func, grouping_column, aggregation_column, group_sums)
            best_impact = max(impacts, key=lambda x: x[2])  # Highest impact
            if max_impact is None or best_impact[2] > max_impact[2]:
                max_impact = best_impact

        # Fallback to the largest non-positive impact if no positive impacts are available
        fallback_used = False
        if max_impact is None or max_impact[2] <= 0:
            fallback_used = True
            max_impact = None
            for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
                group = row[grouping_column]
                impacts = find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func, grouping_column, aggregation_column, group_sums)
                fallback_impact = max(impacts, key=lambda x: x[2])  # Largest impact (may be ≤ 0)
                if max_impact is None or fallback_impact[2] > max_impact[2]:
                    max_impact = fallback_impact

            # If still no valid impacts, terminate the loop
            if max_impact is None:
                print("No valid impacts found. Stopping.")
                break

        # Retrieve grouping and aggregation column values
        removed_tuple = sorted_df.loc[max_impact[0]]
        removed_group_value = removed_tuple[grouping_column]
        removed_aggregation_value = removed_tuple[aggregation_column]

        # Update group sum
        group_sums[removed_group_value] -= removed_aggregation_value

        # Remove the tuple with the largest impact (positive or fallback)
        sorted_df = sorted_df.drop(index=max_impact[0]).reset_index(drop=True)

        # Log this iteration
        iteration_logs.append({
            "Iteration": iteration,
            "Current Smvi": Smvi,
            "Tuple Removed Group Value": removed_group_value,
            "Tuple Removed Aggregation Value": removed_aggregation_value,
            "Impact": max_impact[2],
            "Fallback Used": fallback_used
        })

        iteration += 1
        tuple_removals += 1
        print("finished iteration", iteration)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nAlgorithm Summary:")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {total_time:.4f} seconds")

    # Write iteration logs to CSV
    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration log saved to {output_csv}")

    return sorted_df


# python/py -3.13 sum-main.py <path_to_csv_file> --grouping_column <group_col> --aggregation_column <agg_col> --output_csv <path_to_output_csv>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm on a specified CSV file.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The name of the grouping column.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The name of the aggregation column.")
    parser.add_argument("--output_csv", type=str, default="results_csv(single_sum)/iteration_log.csv", help="Path for the output CSV file.")
    args = parser.parse_args()

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        exit(1)

    print(f"Processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file, usecols=[args.grouping_column, args.aggregation_column])
    except ValueError as e:
        print(f"Error: {e}. Ensure the specified columns exist in the CSV file.")
        exit(1)

    grouping_column = args.grouping_column
    aggregation_column = args.aggregation_column

    result_df, output_csv = greedy_algorithm(df, agg_func="sum", output_csv=args.output_csv,
                                 grouping_column=grouping_column, aggregation_column=aggregation_column)
