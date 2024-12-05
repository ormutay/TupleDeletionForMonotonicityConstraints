import pandas as pd
import time
import argparse
import os

# --- Preprocessing ---
def preprocess_sort(df):
    """Sort DataFrame by groups (A) and values (B) in descending order."""
    return df.sort_values(by=["A", "B"], ascending=[True, False]).reset_index(drop=True)

# --- MVI Calculation ---
def calculate_group_stats(sorted_df, agg_func):
    """
    Calculate Measure of Violations Index (MVI) for adjacent groups.
    Aggregates group values using the provided function (agg_func).
    """
    grouped_df = sorted_df.groupby("A")["B"].apply(agg_func).reset_index()
    grouped_df.rename(columns={"B": "Alpha(A_i)"}, inplace=True)
    grouped_df["MVI"] = grouped_df["Alpha(A_i)"].diff(-1).fillna(0).clip(lower=0)
    return grouped_df

# --- Group Updates ---

def update_group_mvi(grouped_df, group_index, new_group_alpha, agg_func):
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
def find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func):
    """Simulate removing each tuple in a group and calculate its impact."""
    impacts = []
    group_index = grouped_df[grouped_df["A"] == group].index[0]
    group_tuples = sorted_df[sorted_df["A"] == group]

    for _, row in group_tuples.iterrows():
        temp_df = sorted_df.drop(index=row.name)
        remaining_group = temp_df[temp_df["A"] == group]
        new_group_alpha = remaining_group["B"].apply(agg_func) if not remaining_group.empty else 0
        updated_grouped_df = update_group_mvi(grouped_df.copy(), group_index, new_group_alpha, agg_func)

        impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
        impacts.append((row.name, row["B"], impact))

    return impacts

# --- Main Algorithm ---
def greedy_algorithm(df, agg_func):
    """Greedy algorithm to minimize Smvi by removing tuples only."""
    iteration = 0
    sorted_df = preprocess_sort(df)

    tuple_removals = 0
    start_time = time.time()

    while True:
        grouped_df = calculate_group_stats(sorted_df, agg_func)
        Smvi = grouped_df["MVI"].sum()
        print(f"\nIteration {iteration}: Current Smvi = {Smvi}")

        if Smvi == 0:
            print("No violations remain. Algorithm completed.")
            break

        max_impact = None
        for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
            group = row["A"]
            impacts = find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func)
            best_impact = max(impacts, key=lambda x: x[2])  # Highest impact
            if max_impact is None or best_impact[2] > max_impact[2]:
                max_impact = best_impact

        if max_impact is None or max_impact[2] <= 0:
            # No positive impacts available: fallback to largest non-positive impact
            print("No positive impact available. Continuing with largest non-positive impact.")
            max_impact = None
            for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
                group = row["A"]
                impacts = find_tuple_removal_impact(sorted_df, grouped_df, group, agg_func)
                fallback_impact = max(impacts, key=lambda x: x[2])  # Largest impact (may be ≤ 0)
                if max_impact is None or fallback_impact[2] > max_impact[2]:
                    max_impact = fallback_impact

            # If still no valid impacts, terminate the loop
            if max_impact is None:
                print("No valid impacts found. Stopping.")
                break

        # Remove the tuple with the largest impact (positive or fallback)
        sorted_df = sorted_df.drop(index=max_impact[0]).reset_index(drop=True)
        print(f"Removed tuple ({sorted_df.loc[max_impact[0], 'A']}, {max_impact[1]}) with impact {max_impact[2]}")
        iteration += 1
        tuple_removals += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("\nAlgorithm Summary:")
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {total_time:.4f} seconds")

    return sorted_df

# if __name__ == "__main__":
#     cases = [
#         {
#             "description": "Random Values (Normal Case)",
#             "data": {
#             "A": [1, 1, 1, 2, 2, 2, 3, 3, 3],
#             "B": [6, 5, 4, 3, 2, 1, 7, 8, 9],
#             },
#             "ideal_deleted_rows": 2,
#         },
#         {
#             "description": "Example from presentation (Normal Case)",
#             "data": {
#                 "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
#                 "B": [1, 2, 6, 3, 4, 4, 2, 3, 4, 1, 1, 2],
#             },
#             "ideal_deleted_rows": 4,
#         },
#         {
#             "description": "Random Values (Normal Case)",
#             "data": {
#                 "A": [1, 1, 1, 2, 2, 2, 3, 3, 3],
#                 "B": [6, 5, 4, 3, 2, 1, 7, 8, 9],
#             },
#             "ideal_deleted_rows": 2,
#         },
#         {
#             "description": "Strictly Increasing Groups (No Deletions Needed)",
#             "data": {
#                 "A": [1, 1, 1, 2, 2, 2, 3, 3, 3],
#                 "B": [1, 2, 3, 4, 5, 6, 7, 8, 9],
#             },
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "Reverse Order Groups (Edge Case)", #todo: need to talk about this
#             "data": {
#                 "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#                 "B": [9, 8, 7, 6, 5, 4, 3, 3, 2, 1],
#             },
#             "ideal_deleted_rows": 3,
#         },
#         {
#             "description": "Single Row per Group (Normal Case)",
#             "data": {
#                 "A": [1, 2, 3, 4],
#                 "B": [4, 3, 2, 1],
#             },
#             "ideal_deleted_rows": 3,
#         },
#         {
#             "description": "Empty Dataset (Edge Case)",
#             "data": {
#                 "A": [],
#                 "B": [],
#             },
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "Single Group with Multiple Rows (No Deletions Needed)",
#             "data": {
#                 "A": [1, 1, 1, 1],
#                 "B": [10, 9, 8, 7],
#             },
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "All Groups Have Equal Sum (Edge Case)",
#             "data": {
#                 "A": [1, 1, 2, 2, 3, 3],
#                 "B": [5, 5, 5, 5, 5, 5],
#             },
#             "ideal_deleted_rows": 0,
#         },
#         {
#             "description": "Mixed Violations (Complex Case)",
#             "data": {
#                 "A": [1, 1, 2, 2, 3, 3, 4, 4],
#                 "B": [10, 2, 8, 3, 6, 1, 5, 7],
#             },
#             "ideal_deleted_rows": 3,
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
#         result = greedy_algorithm(df, agg_func="sum")
#         print("Final DataFrame After Greedy Algorithm:")
#         print(result)
#         print("\033[92m" + "=" * 50 + "\033[0m")  # Green separator line


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

    result_df = greedy_algorithm(df, agg_func="sum")

    # # Save the processed output
    # output_file = f"processed_{os.path.basename(csv_file)}"
    # result_df.to_csv(output_file, index=False)
    # print(f"Processed file saved to {output_file}")