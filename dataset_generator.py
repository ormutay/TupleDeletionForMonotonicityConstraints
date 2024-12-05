import pandas as pd
import random
import matplotlib.pyplot as plt
import os

def create_dataset(
    num_groups, num_rows, agg_func_name, output_folder="dataset",
    index=0, violations_percentage=10, name_suffix="default"
):
    """
    Create a dataset with columns A and B that is almost monotonic with respect to the aggregation function.

    Args:
        num_groups (int): Number of unique groups (values of A).
        num_rows (int): Total number of rows in the dataset.
        agg_func_name (str): Aggregation function name ('sum' or 'max').
        output_folder (str): Name of the output folder for saving files.
        index (int): Index of the dataset (for unique naming).
        violations_percentage (int): Percentage of groups to introduce violations.
        name_suffix (str): Custom suffix for naming output files.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    if agg_func_name not in ["sum", "max"]:
        raise ValueError("Aggregation function must be 'sum' or 'max'.")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Assign groups to rows
    groups = [i % num_groups + 1 for i in range(num_rows)]
    random.shuffle(groups)  # Shuffle groups to distribute randomly

    # Generate random values for column B
    B_values = [random.randint(1, 100) for _ in range(num_rows)]

    # Create DataFrame
    df = pd.DataFrame({"A": groups, "B": B_values})

    # Ensure groups are almost monotonic
    grouped = df.groupby("A")
    group_agg = grouped["B"].agg(agg_func_name).reset_index()
    group_agg.sort_values(by="A", inplace=True)

    # Introduce controlled violations based on violations_percentage
    num_violations = max(1, int((violations_percentage / 100) * num_groups))  # At least 1 violation
    for _ in range(num_violations):
        group_idx = random.randint(0, num_groups - 2)  # Pick a random group (except the last one)
        if agg_func_name == "sum":
            group_agg.loc[group_idx + 1, "B"] += random.randint(10, 50)  # Increase the next group's sum
        elif agg_func_name == "max":
            group_agg.loc[group_idx + 1, "B"] = max(
                group_agg.loc[group_idx, "B"] + random.randint(1, 10),
                group_agg.loc[group_idx + 1, "B"],
            )

    # Reassign B values to the original DataFrame
    monotonic_B = group_agg.set_index("A")["B"].to_dict()
    df["B"] = df["A"].map(lambda a: random.randint(1, monotonic_B[a]))

    # Save the dataset with a custom suffix
    dataset_filename = os.path.join(output_folder, f"dataset_{name_suffix}_{index}.csv")
    df.to_csv(dataset_filename, index=False)
    print(f"Dataset saved to {dataset_filename}")

    # Plot and save the dataset
    plot_dataset(df, group_agg, agg_func_name, output_folder, index, name_suffix)

    return df


def plot_dataset(df, group_agg, agg_func_name, output_folder, index, name_suffix):
    """
    Generate and save bar and scatter plots of the dataset.

    Args:
        df (pd.DataFrame): The dataset to visualize.
        group_agg (pd.DataFrame): Aggregated group data.
        agg_func_name (str): Aggregation function name ('sum' or 'max').
        output_folder (str): Name of the output folder for saving plots.
        index (int): Index of the dataset (for unique naming).
        name_suffix (str): Custom suffix for naming output files.
    """
    # Scatter plot
    scatter_plot_file = os.path.join(output_folder, f"scatter_plot_{name_suffix}_{index}.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["A"], df["B"], alpha=0.6, c="blue", label="Data Points")
    plt.xlabel("A (Groups)")
    plt.ylabel("B (Values)")
    plt.title("Scatter Plot of Dataset")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(scatter_plot_file)
    print(f"Scatter plot saved to {scatter_plot_file}")
    plt.close()

    # Bar plot
    bar_plot_file = os.path.join(output_folder, f"bar_plot_{name_suffix}_{index}.png")
    plt.figure(figsize=(8, 6))
    plt.bar(group_agg["A"], group_agg["B"], alpha=0.7, color="green", label=f"Group {agg_func_name.upper()}")
    plt.xlabel("A (Groups)")
    plt.ylabel(f"B ({agg_func_name.upper()})")
    plt.title(f"Bar Plot of Group Aggregation ({agg_func_name.upper()})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(bar_plot_file)
    print(f"Bar plot saved to {bar_plot_file}")
    plt.close()

if __name__ == "__main__":
    num_datasets_per_setting = 3  # Generate 3 datasets for each setting

    # Case 1: 5 groups and rows from 100 to 1000 in increments of 100
    for num_rows in range(100, 600, 100):
        for i in range(num_datasets_per_setting):
            create_dataset(num_groups=5, num_rows=num_rows, agg_func_name="sum", output_folder="dataset-sum-w6/rows",
                           index=i, violations_percentage=10,
                           name_suffix=f"sum_g5_r{num_rows}_v10")
            create_dataset(num_groups=5, num_rows=num_rows, agg_func_name="max", output_folder="dataset-max-w6/rows",
                           index=i, violations_percentage=10,
                           name_suffix=f"max_g5_r{num_rows}_v10")

    # Case 2: 500 rows and groups from 5 to 50 in increments of 5
    for num_groups in range(5, 30, 5):
        for i in range(num_datasets_per_setting):
            create_dataset(num_groups=num_groups, num_rows=500, agg_func_name="sum", output_folder="dataset-sum-w6/groups",
                           index=i, violations_percentage=10,
                           name_suffix=f"sum_g{num_groups}_r500_v10")
            create_dataset(num_groups=num_groups, num_rows=500, agg_func_name="max", output_folder="dataset-max-w6/groups",
                           index=i, violations_percentage=10,
                           name_suffix=f"max_g{num_groups}_r500_v10")

    Case 3: 5 groups, 300 rows, violations_percentage from 5 to 50 in increments of 5
    for violations_percentage in range(5, 30, 5):
        for i in range(num_datasets_per_setting):
            create_dataset(num_groups=5, num_rows=300, agg_func_name="sum", output_folder="dataset-sum-w6/violations",
                           index=i, violations_percentage=violations_percentage,
                           name_suffix=f"sum_g5_r300_v{violations_percentage}")
            create_dataset(num_groups=5, num_rows=300, agg_func_name="max", output_folder="dataset-max-w6/violations",
                           index=i, violations_percentage=violations_percentage,
                           name_suffix=f"max_g5_r300_v{violations_percentage}")
