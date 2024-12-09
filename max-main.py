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
def handle_empty_group(grouped_df, group_index):
    """Update MVI for group i-1 when group i becomes empty."""
    if group_index > 0:
        prev_group_index = group_index - 1
        next_group_alpha = (
            grouped_df.loc[group_index + 1, "Alpha(A_i)"]
            if group_index + 1 in grouped_df.index else float("-inf")
        )
        grouped_df.loc[prev_group_index, "MVI"] = max(
            0, grouped_df.loc[prev_group_index, "Alpha(A_i)"] - next_group_alpha
        )
    return grouped_df.drop(index=group_index).reset_index(drop=True)


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
def calculate_next_group_removal_impact(sorted_df, grouped_df, group_index, grouping_column="A"):
    """Calculate the impact of removing all rows in group i+1."""
    next_group_index = group_index + 1
    if next_group_index not in grouped_df.index:
        return None  # No group i+1 exists (so I can't try to remove it)

    next_group = grouped_df.loc[next_group_index, grouping_column]

    updated_grouped_df = grouped_df.drop(index=next_group_index).reset_index(drop=True)
    group_alpha_val = updated_grouped_df.loc[group_index, "Alpha(A_i)"]

    if next_group_index + 1 in updated_grouped_df.index:  # Update MVI if i+2 exists
        new_next_group_alpha_val = updated_grouped_df.loc[group_index + 1, "Alpha(A_i)"]
        updated_grouped_df.loc[group_index, "MVI"] = max(0, group_alpha_val - new_next_group_alpha_val)
    else:
        updated_grouped_df.loc[group_index, "MVI"] = 0  # No next group

    impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
    rows_removed = int(len(sorted_df[sorted_df[grouping_column] == next_group]))
    return impact, rows_removed


def find_options(sorted_df, grouped_df, group, grouping_column="A", aggregation_column="B"):
    """Calculate impact of removing the max tuple(s) or group i+1."""
    max_value = sorted_df[sorted_df[grouping_column] == group][aggregation_column].max()  # Find the max value in the group
    max_tuples = sorted_df[(sorted_df[grouping_column] == group) & (sorted_df[aggregation_column] == max_value)]  # All tuples with max value
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]

    # Option 1 - Max tuples removal from group
    temp_df = sorted_df.drop(index=max_tuples.index)  # Remove all tuples with max value
    remaining_group = temp_df[temp_df[grouping_column] == group]
    new_group_i_alpha_val = remaining_group[aggregation_column].max() if not remaining_group.empty else float("-inf")
    updated_grouped_df = (
        handle_empty_group(grouped_df.copy(), group_index)
        if new_group_i_alpha_val == float("-inf")
        else update_group_mvi(grouped_df.copy(), group_index, new_group_i_alpha_val)
    )
    impact_max_tuples = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()

    # Option 2 - Next group removal
    impact_group_removal, rows_removed = calculate_next_group_removal_impact(sorted_df, grouped_df, group_index)

    return {
        "tuple_removal": (max_tuples.index.tolist(), group, max_value, impact_max_tuples, len(max_tuples)),
        "group_removal": (
            None,
            grouped_df.loc[group_index + 1, grouping_column],
            None,
            impact_group_removal,
            rows_removed,
        )
        if impact_group_removal is not None
        else None,
    }


def calculate_impact(sorted_df, grouped_df, grouping_column="A", aggregation_column="B"):
    """Calculate impacts of tuple and group removals."""
    impacts = []
    for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
        group = row[grouping_column]
        options = find_options(sorted_df, grouped_df, group)
        impacts.append(options["tuple_removal"])  # Tuple removal includes multiple tuples now
        if options["group_removal"]:
            impacts.append(options["group_removal"])
    return (
        pd.DataFrame(impacts, columns=["Index", grouping_column, aggregation_column, "Impact", "RowsRemoved"])
        .astype({"RowsRemoved": "int"})
        .sort_values(by=["Impact", "RowsRemoved"], ascending=[False, True])
        .reset_index(drop=True)
    )


# --- Main Algorithm ---
def greedy_algorithm(df, grouping_column="A", aggregation_column="B", output_csv="results/iteration_log.csv"):
    """Greedy algorithm to minimize Smvi by removing tuples or groups."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)

    tuple_removals = 0
    group_removals = 0
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

        if impact_df.empty or impact_df.iloc[0]["Impact"] <= 0: #todo: this was relavent only in the case of tuple removal without group removal
            print("No positive impact available. Exiting loop.")
            break

        max_impact = impact_df.iloc[0]
        deleted_group = False
        if isinstance(max_impact["Index"], list):  # Tuple removal for multiple tuples
            removed_group_value = max_impact[grouping_column]
            removed_aggregation_value = max_impact[aggregation_column]
            num_removed = len(max_impact["Index"])
            sorted_df = sorted_df.drop(index=max_impact["Index"]).reset_index(drop=True)

        else:  # Group removal
            removed_group_value = max_impact[grouping_column]
            removed_aggregation_value = None
            num_removed = int(max_impact["RowsRemoved"])
            sorted_df = sorted_df[sorted_df[grouping_column] != removed_group_value].reset_index(drop=True)
            deleted_group = True
            group_removals += 1

        iteration_logs.append({
            "Iteration": iteration,
            "Current Smvi": Smvi,
            "Tuple Removed Group Value": removed_group_value,
            "Tuple Removed Aggregation Value": removed_aggregation_value,
            "Number of Tuples Removed": num_removed,
            "Impact": max_impact["Impact"],
            "Group Deleted": deleted_group
        })

        tuple_removals += num_removed
        iteration += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("\nAlgorithm Summary:")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total group removals: {group_removals}")
    print(f"Total execution time: {total_time:.4f} seconds")

    # Write iteration logs to CSV
    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    print(f"Iteration log saved to {output_csv}")

    return sorted_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm on a specified CSV file.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The name of the grouping column.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The name of the aggregation column.")
    parser.add_argument("--output_csv", type=str, default="processed_output.csv", help="Path for the output CSV file.")
    args = parser.parse_args()

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        exit(1)

    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)
    result_df = greedy_algorithm(df, args.grouping_column, args.aggregation_column, args.output_csv)
