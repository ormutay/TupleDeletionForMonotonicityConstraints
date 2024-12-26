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
    """
    # Map aggregation function name to the actual function
    function_map = {"sum": "sum", "max": "max", "avg": "mean", "median": "median"}
    if agg_func_name not in function_map:
        raise ValueError("Aggregation function must be 'sum', 'max', 'median' or 'avg'.")

    agg_func = function_map[agg_func_name]

    # Ensure the output folder exists
    datasets_output_folder = os.path.join(output_folder, "datasets")
    if not os.path.exists(datasets_output_folder):
        os.makedirs(datasets_output_folder)

    # Assign groups to rows
    groups = [i % num_groups + 1 for i in range(num_rows)]
    random.shuffle(groups)  # Shuffle groups to distribute randomly

    # Generate random values for column B
    B_values = [random.randint(1, 100) for _ in range(num_rows)]

    # Create DataFrame
    df = pd.DataFrame({"A": groups, "B": B_values})

    # Ensure groups are almost monotonic
    grouped = df.groupby("A")
    group_agg = grouped["B"].agg(agg_func).reset_index()
    group_agg.sort_values(by="A", inplace=True)

    # Introduce controlled violations
    num_violations = max(1, int((violations_percentage / 100) * num_groups))
    for _ in range(num_violations):
        group_idx = random.randint(0, num_groups - 2)  # Pick a random group (except the last one)
        if agg_func_name == "sum":
            group_agg.loc[group_idx + 1, "B"] += random.randint(10, 50)  # Increase the next group's sum
        elif agg_func_name == "max":
            group_agg.loc[group_idx + 1, "B"] = max(
                group_agg.loc[group_idx, "B"] + random.randint(1, 10),
                group_agg.loc[group_idx + 1, "B"],
            )
        elif agg_func_name == "avg":
            # Ensure group_idx + 2 is within bounds
            if group_idx + 2 < len(group_agg):
                current_group_count = len(df[df["A"] == group_idx + 1])
                next_group_count = len(df[df["A"] == group_idx + 2])
                if next_group_count > 0:
                    avg_adjustment = random.randint(5, 20)  # Introduce a meaningful violation
                    group_agg.loc[group_idx + 2, "B"] = (
                            (group_agg.loc[group_idx + 2, "B"] * next_group_count + avg_adjustment)
                            / next_group_count
                    )
        elif agg_func_name == "median":
            for _ in range(num_violations):
                group_idx = random.randint(0, num_groups - 2)  # Pick a random group (except the last one)
                current_group_values = df[df["A"] == group_idx + 1]["B"].tolist()
                next_group_values = df[df["A"] == group_idx + 2]["B"].tolist()

                if next_group_values:
                    violation_value = random.randint(min(next_group_values), max(next_group_values))
                    df.loc[df["A"] == group_idx + 2, "B"] = violation_value

    # Reassign B values to the original DataFrame
    monotonic_B = group_agg.set_index("A")["B"].to_dict()
    df["B"] = df["A"].map(lambda a: random.randint(1, int(monotonic_B[a])))

    # Save the dataset
    dataset_filename = os.path.join(datasets_output_folder, f"dataset_{name_suffix}_{index}.csv")
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
        agg_func_name (str): Aggregation function name
        output_folder (str): Name of the output folder for saving plots.
        index (int): Index of the dataset (for unique naming).
        name_suffix (str): Custom suffix for naming output files.
    """
    # Ensure subfolders exist
    scatter_plot_folder = os.path.join(output_folder, "scatter_plots")
    bar_plot_folder = os.path.join(output_folder, "bar_plots")
    os.makedirs(scatter_plot_folder, exist_ok=True)
    os.makedirs(bar_plot_folder, exist_ok=True)

    # Scatter plot
    scatter_plot_file = os.path.join(scatter_plot_folder, f"scatter_plot_{name_suffix}_{index}.png")
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
    bar_plot_file = os.path.join(bar_plot_folder, f"bar_plot_{name_suffix}_{index}.png")
    plt.figure(figsize=(8, 6))
    plt.bar(group_agg["A"], group_agg["B"], alpha=0.7, color="green", label=f"Group {agg_func_name.upper()}")
    plt.xlabel("A (Groups)")
    plt.ylabel(f"B ({agg_func_name.upper()})")
    plt.title(f"Bar Plot of Group Aggregation ({agg_func_name.upper()})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(bar_plot_file)
    print(f"Bar plot saved to {bar_plot_file}")
    plt.close()


def run_dataset_generation(configurations, num_datasets_per_setting):
    for config in configurations:
        for param_value in config['range']:
            for i in range(num_datasets_per_setting):
                for agg_func in config['agg_funcs']:
                    create_dataset(
                        num_groups=config.get('num_groups', param_value),
                        num_rows=config.get('num_rows', param_value),
                        agg_func_name=agg_func,
                        output_folder=f"w8/dataset-{agg_func}-w8/{config['folder']}",
                        index=i,
                        violations_percentage=config.get('violations_percentage', param_value),
                        name_suffix=f"{agg_func}_g{config.get('num_groups', param_value)}_r{config.get('num_rows', param_value)}_v{config.get('violations_percentage', param_value)}"
                    )

if __name__ == "__main__":
    num_datasets_per_setting = 3  # Number of datasets per setting
    agg_funcs = ["sum", "max", "avg", "median"]
    configurations = [
        {
            'range': range(100, 300, 100),
            'folder': 'rows',
            'agg_funcs': agg_funcs,
            'num_groups': 10,
            'violations_percentage': 10,
        },
        {
            'range': range(5, 15, 5),
            'folder': 'groups',
            'agg_funcs': agg_funcs,
            'num_rows': 500,
            'violations_percentage': 10,
        },
        {
            'range': range(5, 15, 5),
            'folder': 'violations',
            'agg_funcs': agg_funcs,
            'num_groups': 10,
            'num_rows': 500,
        }
    ]

    run_dataset_generation(configurations, num_datasets_per_setting)
