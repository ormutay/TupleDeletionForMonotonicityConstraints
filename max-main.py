
""" for me:
.reset_index(drop=True):
                       resets the index after sorting, so the row indices are sequential (0, 1, 2, ...).
                       drop=True avoids adding the old index as a new column. 

grouped_df["MVI"] = grouped_df["Alpha(A_i)"].diff(-1).fillna(0).apply(lambda x: max(0, x))
                        diff(-1): Computes the difference between the current group’s Alpha and the next group’s Alpha.
                        fillna(0): Replaces NaN values with 0.
                        apply(lambda x: max(0, x)): Replaces negative values with 0.
"""


import pandas as pd
import time
import argparse
import os

# --- Preprocessing ---
def preprocess_sort(df):
    """Sort DataFrame by groups (A) and values (B) in descending order."""
    return df.sort_values(by=["A", "B"], ascending=[True, False]).reset_index(drop=True)

# --- MVI Calculation ---
def calculate_group_stats(sorted_df):
    """
    Calculate Measure of Violations Index (MVI) for adjacent groups.
    Only considers positive violations where Alpha(A_i) > Alpha(A_{i+1}).
    """
    grouped_df = sorted_df.groupby("A").first().reset_index()
    grouped_df.rename(columns={"B": "Alpha(A_i)"}, inplace=True)
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
def calculate_next_group_removal_impact(sorted_df, grouped_df, group_index):
    """Calculate the impact of removing all rows in group i+1."""
    next_group_index = group_index + 1
    if next_group_index not in grouped_df.index:
        return None  # No group i+1 exists (so I can't try to remove it)

    next_group = grouped_df.loc[next_group_index, "A"]

    updated_grouped_df = grouped_df.drop(index=next_group_index).reset_index(drop=True)
    group_alpha_val = updated_grouped_df.loc[group_index, "Alpha(A_i)"]

    if next_group_index + 1 in updated_grouped_df.index:  # Update MVI if i+2 exists
        new_next_group_alpha_val = updated_grouped_df.loc[group_index + 1, "Alpha(A_i)"]
        updated_grouped_df.loc[group_index, "MVI"] = max(0, group_alpha_val - new_next_group_alpha_val)
    else:
        updated_grouped_df.loc[group_index, "MVI"] = 0  # No next group

    impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
    rows_removed = int(len(sorted_df[sorted_df["A"] == next_group]))
    return impact, rows_removed

def find_options(sorted_df, grouped_df, group):
    """Calculate impact of removing the max tuple or group i+1."""
    max_tuple = sorted_df[sorted_df["A"] == group].iloc[0]
    group_index = grouped_df[grouped_df["A"] == group].index[0]

    # Option 1 - Max tuple removal from group
    temp_df = sorted_df.drop(index=max_tuple.name)
    remaining_group = temp_df[temp_df["A"] == group]
    new_group_i_alpha_val = remaining_group["B"].max() if not remaining_group.empty else float("-inf")
    updated_grouped_df = handle_empty_group(grouped_df.copy(), group_index) if new_group_i_alpha_val == float("-inf") else update_group_mvi(grouped_df.copy(), group_index, new_group_i_alpha_val)
    impact_max_tuple = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()

    # Option 2- Next group removal
    impact_group_removal, rows_removed = calculate_next_group_removal_impact(sorted_df, grouped_df, group_index)

    return {
        "tuple_removal": (max_tuple.name, group, max_tuple["B"], impact_max_tuple, 1),
        "group_removal": (None, grouped_df.loc[group_index + 1, "A"], None, impact_group_removal, rows_removed) if impact_group_removal is not None else None,
    }

def calculate_impact(sorted_df, grouped_df):
    """Calculate impacts of tuple and group removals."""
    impacts = []
    for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
        group = row["A"]
        options = find_options(sorted_df, grouped_df, group)
        impacts.append(options["tuple_removal"]) #todo tuple removal can also have an impact of 0
        if options["group_removal"]:
            impacts.append(options["group_removal"])
    return pd.DataFrame(impacts, columns=["Index", "A", "B", "Impact", "RowsRemoved"]).astype({"RowsRemoved": "int"}).sort_values(by=["Impact", "RowsRemoved"], ascending=[False, True]).reset_index(drop=True)

# --- Main Algorithm ---
def greedy_algorithm(df):
    """Greedy algorithm to minimize Smvi by removing tuples or groups."""
    iteration = 0
    sorted_df = preprocess_sort(df)

    tuple_removals = 0
    group_removals = 0
    start_time = time.time()

    while True:
        grouped_df = calculate_group_stats(sorted_df)
        Smvi = grouped_df["MVI"].sum()
        print(f"\nIteration {iteration}: Current Smvi = {Smvi}")
        if Smvi == 0:
            print("No violations remain. Algorithm completed.")
            break

        impact_df = calculate_impact(sorted_df, grouped_df)
        print("Impact DataFrame:")
        print(impact_df)

        if impact_df.empty or impact_df.iloc[0]["Impact"] <= 0: #todo: this was relavent only in the case of tuple removal without group removal
            print("No positive impact available. Exiting loop.")
            break

        max_impact = impact_df.iloc[0]
        if pd.notna(max_impact["Index"]):  # Tuple removal
            sorted_df = sorted_df.drop(index=int(max_impact["Index"])).reset_index(drop=True)
            print(f"Removed tuple at index {int(max_impact['Index'])} with impact = {max_impact['Impact']}")
            tuple_removals += 1
        else:  # Group removal
            sorted_df = sorted_df[sorted_df["A"] != max_impact["A"]].reset_index(drop=True)
            print(f"Removed group {max_impact['A']} with impact = {max_impact['Impact']} and rows removed = {max_impact['RowsRemoved']}")
            tuple_removals += int(max_impact["RowsRemoved"])
            group_removals += 1
        iteration += 1

    end_time = time.time()
    total_time = end_time - start_time
    print("\nAlgorithm Summary:")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total group removals: {group_removals}")
    print(f"Total execution time: {total_time:.4f} seconds")
    return sorted_df

# if __name__ == "__main__":
#     cases = [
#         {
#             "description": "Example from presntation (Normal Case)",
#             "data": {"A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], "B": [1, 2, 6, 3, 4, 4, 2, 3, 4, 1, 1, 2]},
#             "ideal_deleted_rows": 4,
#         },
#         {
#             "description": "Random Values (Normal Case)",
#             "data": {"A": [1, 1, 2, 2, 3, 3], "B": [2, 4, 3, 6, 5, 1]},
#             "ideal_deleted_rows": 2,
#         },
#         {
#             "description": "delete 1 vs 2 rows (Normal Case)",
#             "data": {"A": [1, 1, 2, 2, 3, 3], "B": [1, 2, 3, 4, 3, 3]},
#             "ideal_deleted_rows": 1,
#
#         },
#         {
#             "description": "Middle Group Removal (Normal Case)",
#             "data": {"A": [1, 1, 2, 2, 3, 3], "B": [4, 5, 3, 2, 1, 6]},
#             "ideal_deleted_rows": 2,
#
#         },
#         {
#             "description": "Single Row per Group (Normal Case)",
#             "data": {"A": [1, 2, 3, 4], "B": [5, 3, 6, 4]},
#             "ideal_deleted_rows": 2,
#
#         },
#         {
#             "description": "Single Group (Edge Case)",
#             "data": {"A": [1, 1, 1], "B": [5, 3, 2]},
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "Reverse Monotonic (Edge Case)",
#             "data": {"A": [1, 1, 1, 2, 2, 2], "B": [6, 5, 4, 3, 2, 1]},
#             "ideal_deleted_rows": 3,
#         },
#         {
#             "description": "Strictly Monotonic (Edge Case)",
#             "data": {"A": [1, 1, 1, 2, 2, 2, 3, 3, 3], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "Last Group Removal (Edge Case)",
#             "data": {"A": [1, 1, 2, 2, 3, 3], "B": [1, 2, 4, 4, 3, 3]},
#             "ideal_deleted_rows": 2,
#         },
#     ]
#
#     for case in cases:
#         print("\033[1;94m" + f"Running Case: {case['description']}" + "\033[0m")  # Bold and Blue for description
#         df = pd.DataFrame(case["data"])
#         print("Input DataFrame:")
#         print(df)
#         print("ideal_deleted_rows:", case.get("ideal_deleted_rows", "N/A"))
#         print("\033[1m\n running greedy algorithm..\033[0m")
#         result = greedy_algorithm(df)
#         print("Final DataFrame After Greedy Algorithm:")
#         print(result)
#        print("\033[92m" + "=" * 50 + "\033[0m")  # Green separator line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm on a specified CSV file.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    args = parser.parse_args()

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        exit(1)

    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)
    print("Input DataFrame:")
    print(df)

    result_df = greedy_algorithm(df)

    # # Save the processed output
    # output_file = f"processed_{os.path.basename(csv_file)}"
    # result_df.to_csv(output_file, index=False)
    # print(f"Processed file saved to {output_file}")