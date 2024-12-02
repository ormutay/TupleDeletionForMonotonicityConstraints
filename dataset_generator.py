import pandas as pd
import random
import matplotlib.pyplot as plt
import os

def create_dataset(num_groups, num_rows, agg_func_name, output_folder="dataset", index=0):
    """
    Create a dataset with columns A and B that is almost monotonic (90%) with respect to the aggregation function.

    Args:
        num_groups (int): Number of unique groups (values of A).
        num_rows (int): Total number of rows in the dataset.
        agg_func_name (str): Aggregation function name ('sum' or 'max').
        output_folder (str): Name of the output folder for saving files.
        index (int): Index of the dataset (for unique naming).

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

    # Introduce controlled violations (10% of groups)
    num_violations = max(1, int(0.1 * num_groups))  # At least 1 violation
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

    # Save the dataset with unique name
    output_csv = os.path.join(output_folder, f"dataset_{index}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

    # Plot and save the dataset
    plot_dataset(df, group_agg, agg_func_name, output_folder, index)

    return df


def plot_dataset(df, grouped_df, agg_func_name, output_folder, index):
    """
    Plot the dataset to visualize the raw data and the grouped aggregation, and save the plots.

    Args:
        df (pd.DataFrame): The dataset to plot.
        grouped_df (pd.DataFrame): Grouped aggregation data.
        agg_func_name (str): The aggregation function used.
        output_folder (str): Folder to save the plots.
        index (int): Index of the dataset (for unique naming).
    """
    # Plot original dataset
    plt.figure(figsize=(12, 6))
    plt.scatter(df["A"], df["B"], color="blue", alpha=0.7, label="Raw Data (B values)")
    plt.xlabel("Groups (A)", fontsize=14)
    plt.ylabel("Values (B)", fontsize=14)
    plt.title(f"Original Dataset {index}", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_folder, f"scatter_plot_{index}.png")
    plt.savefig(scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    plt.close()

    # Plot aggregated values
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df["A"], grouped_df["B"], color="skyblue", alpha=0.8, edgecolor="black",
            label=f"Aggregated B ({agg_func_name})")
    plt.xlabel("Groups (A)", fontsize=14)
    plt.ylabel(f"Aggregated B ({agg_func_name})", fontsize=14)
    plt.title(f"Grouped Aggregation {index}", fontsize=16)
    plt.xticks(grouped_df["A"], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    bar_plot_path = os.path.join(output_folder, f"bar_plot_{index}.png")
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.close()


if __name__ == "__main__":
    # Example Usage: Generate multiple datasets
    num_datasets = 3
    for i in range(num_datasets):
        create_dataset(num_groups=5, num_rows=100, agg_func_name="sum", output_folder="dataset-sum", index=i)
        create_dataset(num_groups=5, num_rows=100, agg_func_name="max", output_folder="dataset-max", index=i)

    for i in range(num_datasets):
        create_dataset(num_groups=10, num_rows=100, agg_func_name="sum", output_folder="dataset-sum", index=i+3)
        create_dataset(num_groups=10, num_rows=100, agg_func_name="max", output_folder="dataset-max", index=i+3)

    for i in range(num_datasets):
        create_dataset(num_groups=5, num_rows=200, agg_func_name="sum", output_folder="dataset-sum", index=i+6)
        create_dataset(num_groups=5, num_rows=200, agg_func_name="max", output_folder="dataset-max", index=i+6)

    for i in range(num_datasets):
        create_dataset(num_groups=10, num_rows=200, agg_func_name="sum", output_folder="dataset-sum", index=i+9)
        create_dataset(num_groups=10, num_rows=200, agg_func_name="max", output_folder="dataset-max", index=i+9)