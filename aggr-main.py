import pandas as pd
import time
import argparse
import os
import matplotlib.pyplot as plt

# --- Plotting ---
def plot_aggregation(df, grouping_column, aggregation_column, title, output_file, agg_func):
    """Plot the aggregated values of the DataFrame."""
    aggregated_df = df.groupby(grouping_column)[aggregation_column].apply(agg_func).reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(aggregated_df[grouping_column], aggregated_df[aggregation_column], color='skyblue')
    plt.xlabel(grouping_column)
    plt.ylabel(f"{agg_func}({aggregation_column})")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file, format='pdf')
    plt.close()
    print(f"Plot saved to {output_file}")


def plot_impact_per_iteration(log_file, output_file):
    """Plot the impact per iteration based on the iteration log."""
    log_data = pd.read_csv(log_file)

    if "Impact" not in log_data.columns or "Iteration" not in log_data.columns:
        print("Error: Missing required columns in the log file.")
        return

    colors = log_data["Impact"].apply(lambda x: 'green' if x > 0 else ('yellow' if x == 0 else 'red')).tolist()

    plt.figure(figsize=(10, 6))
    for i, impact in enumerate(log_data["Impact"]):
        plt.plot([log_data["Iteration"].iloc[i]], [impact], marker='o', markersize=8,
                 color='black', markerfacecolor=colors[i], markeredgecolor='black')

    plt.plot(log_data["Iteration"], log_data["Impact"], color='black', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("Impact")
    plt.title("Impact per Iteration")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_file, format="pdf")
    plt.close()
    print(f"Impact per iteration plot saved to {output_file}")


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
def calculate_tuple_removal_impact_max(sorted_df, grouped_df, group, grouping_column="A", aggregation_column="B"):
    """Calculate the impact of removing all max value tuples in a group."""
    max_value = sorted_df[sorted_df[grouping_column] == group][aggregation_column].max()
    max_tuples = sorted_df[(sorted_df[grouping_column] == group) & (sorted_df[aggregation_column] == max_value)]
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]

    temp_df = sorted_df.drop(index=max_tuples.index)
    remaining_group = temp_df[temp_df[grouping_column] == group]
    new_group_alpha = remaining_group[aggregation_column].max() if not remaining_group.empty else 0
    updated_grouped_df = update_group_mvi(grouped_df.copy(), group_index, new_group_alpha)

    impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
    return [(idx, max_value, impact) for idx in max_tuples.index]

def calculate_tuple_removal_impact_sum(sorted_df, grouped_df, group, grouping_column="A", aggregation_column="B"):
    """Calculate the impact of removing each tuple in a group individually."""
    group_index = grouped_df[grouped_df[grouping_column] == group].index[0]
    group_tuples = sorted_df[sorted_df[grouping_column] == group]

    impacts = []
    for _, row in group_tuples.iterrows():
        temp_df = sorted_df.drop(index=row.name)
        remaining_group = temp_df[temp_df[grouping_column] == group]
        new_group_alpha = remaining_group[aggregation_column].sum() if not remaining_group.empty else 0
        updated_grouped_df = update_group_mvi(grouped_df.copy(), group_index, new_group_alpha)

        impact = grouped_df["MVI"].sum() - updated_grouped_df["MVI"].sum()
        impacts.append((row.name, row[aggregation_column], impact))

    return impacts

# --- Greedy Algorithm ---
def greedy_algorithm(df, agg_func, grouping_column="A", aggregation_column="B", output_csv="results/iteration_log.csv"):
    """Greedy algorithm to minimize Smvi by removing tuples."""
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iteration = 0
    sorted_df = preprocess_sort(df, grouping_column, aggregation_column)
    tuple_removals = 0
    start_time = time.time()
    iteration_logs = []

    while True:
        grouped_df = calculate_group_stats(sorted_df, agg_func, grouping_column, aggregation_column)
        Smvi = grouped_df["MVI"].sum()

        if Smvi == 0:
            print("No violations remain. Algorithm completed.")
            break

        impacts = []
        tuples_to_remove = []

        for _, row in grouped_df[grouped_df["MVI"] > 0].iterrows():
            group = row[grouping_column]
            if agg_func == "MAX":
                impacts.extend(calculate_tuple_removal_impact_max(sorted_df, grouped_df, group, grouping_column, aggregation_column))
            elif agg_func == "SUM":
                impacts.extend(calculate_tuple_removal_impact_sum(sorted_df, grouped_df, group, grouping_column, aggregation_column))
            else:
                raise ValueError("Unsupported aggregation function")

        if not impacts:
            print("No valid impacts found. Stopping.")
            break

        max_impact = max(impacts, key=lambda x: x[2])

        fallback_used = False
        if max_impact[2] <= 0:
            print("No positive impacts found. Using fallback strategy.")
            fallback_used = True

        tuples_to_remove = [idx for idx, _, _ in impacts if _ == max_impact[1]] if agg_func == max else [max_impact[0]]
        sorted_df = sorted_df.drop(index=tuples_to_remove).reset_index(drop=True)

        iteration_logs.append({
            "Iteration": iteration,
            "Current Smvi": Smvi,
            "Tuple Removed Group Value": sorted_df.loc[tuples_to_remove[0], grouping_column] if tuples_to_remove else None,
            "Tuple Removed Aggregation Value": max_impact[1],
            "Number of Tuples Removed": len(tuples_to_remove),
            "Impact": max_impact[2],
            "Fallback Used": fallback_used
        })

        tuple_removals += len(tuples_to_remove)
        iteration += 1

    end_time = time.time()
    print(f"Total tuple removals: {tuple_removals}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

    pd.DataFrame(iteration_logs).to_csv(output_csv, index=False)
    return sorted_df

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the greedy algorithm with a specified aggregation function.")
    parser.add_argument("csv_file", type=str, help="The path to the input CSV file.")
    parser.add_argument("agg_func", type=str, choices=["sum", "max"], help="The aggregation function to use.")
    parser.add_argument("--grouping_column", type=str, default="A", help="The column to group by.")
    parser.add_argument("--aggregation_column", type=str, default="B", help="The column to aggregate.")
    parser.add_argument("--output_csv", type=str, default="results/iteration_log.csv", help="The path for the output CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    agg_func = sum if args.agg_func == "sum" else max

    greedy_algorithm(df, agg_func, grouping_column=args.grouping_column, aggregation_column=args.aggregation_column, output_csv=args.output_csv)
