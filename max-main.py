import pandas as pd
import time
import argparse
import os

# --- Preprocessing ---
def preprocess_sort(df, grouping_column="A", aggregation_column="B"):
    """Sort DataFrame by grouping and aggregation columns in descending order."""
    return df.sort_values(by=[grouping_column, aggregation_column], ascending=[True, False]).reset_index(drop=True)


# --- MVI Calculation ---
def calculate_group_stats(sorted_df, grouping_column="A", aggregation_column="B"):
    """
    Calculate Measure of Violations Index (MVI) for adjacent groups.
    Only considers positive violations where Alpha(A_i) > Alpha(A_{i+1}).
    """
    grouped_df = sorted_df.groupby(grouping_column).first().reset_index()
    grouped_df.rename(columns={aggregation_column: "Alpha(A_i)"}, inplace=True)
    grouped_df["MVI"] = grouped_df["Alpha(A_i)"].diff(-1).fillna(0).clip(lower=0)  # Use clip for clarity
    return grouped_df


# --- Group Updates ---
def update_group_mvi(grouped_df, group_index, new_group_i_alpha_val):
    """Update MVI for the affected group (i) and (i-1)."""

    grouped_df.loc[group_index, "Alpha(A_i)"] = new_group_i_alpha_val

    # Update MVI for group i-1
    if group_index > 0:
        prev_group_index = group_index - 1
        grouped_df.loc[prev_group_index, "MVI"] = max(
            0, grouped_df.loc[prev_group_index, "Alpha(A_i)"] - new_group_i_alpha_val
        )

    # Update MVI for group i
    next_group_alpha_val = (
        grouped_df.loc[group_index + 1, "Alpha(A_i)"]
        if group_index + 1 in grouped_df.index else float("inf")
    )
    grouped_df.loc[group_index, "MVI"] = max(0, new_group_i_alpha_val - next_group_alpha_val)

    return grouped_df


# --- Impact Calculation ---
def calculate_tuple_removal_impact(sorted_df, grouped_df, group, grouping_column="A", aggregation_column="B"):
    """Calculate the impact of removing the max value tuples in a group."""
    max_value = sorted_df[sorted_df[grouping_column] == group][aggregation_column].max()  # Find the max value in the group
    max_tuples = sorted_df[(sorted_df[grouping_column] == group) & (sorted_df[aggregation_column] == max_value)]  # All tuples with max value
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]

    # Simulate removal of the max value tuples
    temp_df = sorted_df.drop(index=max_tuples.index)  # Remove all tuples with max value
    remaining_group = temp_df[temp_df[grouping_column] == group]
    new_group_i_alpha_val = remaining_group[aggregation_column].max() if not remaining_group.empty else float("-inf")
    updated_grouped_df = update_group_mvi(grouped_df.copy(), group_index, new_group_i_alpha_val)

    impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
    return {
        "Index": max_tuples.index.tolist(),
        "Group": group,
        "MaxValue": max_value,
        "Impact": impact,
        "RowsRemoved": len(max_tuples)
    }


def calculate_impact(sorted_df, grouped_df, grouping_column="A", aggregation_column="B"):
    """Calculate impacts of tuple removals."""
    impacts = []
    for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
        group = row[grouping_column]
        impact = calculate_tuple_removal_impact(sorted_df, grouped_df, group, grouping_column, aggregation_column)
        impacts.append(impact)
    return (
        pd.DataFrame(impacts, columns=["Index", "Group", "MaxValue", "Impact", "RowsRemoved"])
        .astype({"RowsRemoved": "int"})
        .sort_values(by=["Impact", "RowsRemoved"], ascending=[False, True])
        .reset_index(drop=True)
    )


# --- Main Algorithm ---
def greedy_algorithm(df, grouping_column="A", aggregation_column="B", output_csv="results/iteration_log.csv"):
    """Greedy algorithm to minimize Smvi by removing tuples."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)

    tuple_removals = 0
    start_time = time.time()

    # Collect iteration details for CSV
    iteration_logs = []

    while True:
        grouped_df = calculate_group_stats(sorted_df, grouping_column, aggregation_column)
        Smvi = grouped_df["MVI"].sum()

        if Smvi == 0:
            print("No violations remain. Algorithm completed.")
            break

        if Smvi < 0:
            print("!!!! Smvi < 0 !!!!, Algorithm completed.")
            break

        impact_df = calculate_impact(sorted_df, grouped_df, grouping_column, aggregation_column)

        if impact_df.empty:
            print("No impacts available. Stopping.")
            break

        # Find the tuple with the highest impact
        max_impact = impact_df.iloc[0]
        fallback_used = False

        # Check if impact is non-positive
        if max_impact["Impact"] <= 0:
            fallback_used = True
            print(f"Fallback used at iteration {iteration} due to non-positive impact.")

        # Remove the tuple(s) with the highest impact
        removed_group_value = max_impact["Group"]
        removed_aggregation_value = max_impact["MaxValue"]
        num_removed = len(max_impact["Index"])
        sorted_df = sorted_df.drop(index=max_impact["Index"]).reset_index(drop=True)

        # Log this iteration
        iteration_logs.append({
            "Iteration": iteration,
            "Current Smvi": Smvi,
            "Tuple Removed Group Value": removed_group_value,
            "Tuple Removed Aggregation Value": removed_aggregation_value,
            "Number of Tuples Removed": num_removed,
            "Impact": max_impact["Impact"],
            "Fallback Used": fallback_used
        })

        tuple_removals += num_removed
        iteration += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("\nAlgorithm Summary:")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {total_time:.4f} seconds")

    # Write iteration logs to CSV
    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration log saved to {output_csv}")

    return sorted_df, output_csv

# python/py -3.13 max-main.py <path_to_csv_file> --grouping_column <group_col> --aggregation_column <agg_col> --output_csv <path_to_output_csv>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm on a specified CSV file.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The name of the grouping column.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The name of the aggregation column.")
    parser.add_argument("--output_csv", type=str, default="processed_output.csv", help="Path for the output CSV file.")
    parser.add_argument("--output_folder", type=str, default="results", help="Folder to save the output files.")
    args = parser.parse_args()

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        exit(1)

    print(f"Processing file: {csv_file}")

    # Load only the necessary columns
    try:
        df = pd.read_csv(csv_file, usecols=[args.grouping_column, args.aggregation_column])
    except ValueError as e:
        print(f"Error: {e}. Ensure the specified columns exist in the CSV file.")
        exit(1)

    # Plot the initial state
    grouping_column = args.grouping_column
    aggregation_column = args.aggregation_column
    output_folder = args.output_folder

    # Run the algorithm
    result_df, output_csv = greedy_algorithm(df, grouping_column=grouping_column,
                                 aggregation_column=aggregation_column, output_csv=args.output_csv)
