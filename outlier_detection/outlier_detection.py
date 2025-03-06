import pandas as pd
import numpy as np
import argparse
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import time
import os

# ANSI color codes
BLUE_BOLD = "\033[94m\033[1m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_dataset(dataset_path):
    """Load dataset from CSV file."""
    return pd.read_csv(dataset_path)


def is_monotonic(df, group_col, agg_col, agg_func):
    """Check if B is non-decreasing across groups of A based on the specified aggregation function."""
    agg_funcs = {
        "max": lambda x: x.max(),
        "sum": lambda x: x.sum(),
        "avg": lambda x: x.mean(),
        "median": lambda x: x.median(),
    }
    if agg_func not in agg_funcs:
        raise ValueError("Unsupported aggregation function. Choose from 'max', 'sum', 'avg', 'median'.")

    grouped = df.groupby(group_col)[agg_col].apply(agg_funcs[agg_func])
    return (grouped.diff().dropna() >= 0).all()


# def remove_until_monotonic(df, outliers, max_removal_pct, agg_func, group_col, agg_col):
#     """Gradually remove outliers until monotonicity is restored or max_removal_pct is reached."""
#     df_after_outlier_removal = df.copy()
#     total_rows = len(df)
#
#     max_removals = int(total_rows * max_removal_pct / 100)
#     print(f"Total rows in DataFrame: {total_rows}, max removals: {max_removals}")
#
#     removed = []
#     print(f"Removing outliers until monotonicity is restored or {max_removal_pct}%(={max_removals}) of rows are removed...")
#
#     for index, row in outliers.iterrows():
#         if len(removed) >= max_removals:
#             break
#         df_after_outlier_removal = df_after_outlier_removal.drop(index)
#         removed.append(index)
#         if is_monotonic(df_after_outlier_removal,group_col=group_col, agg_col=agg_col, agg_func=agg_func):
#             print(f"Removed {len(removed)} outliers, monotonic.")
#             break
#         print(f"Removed {len(removed)} outliers, still not monotonic.")
#     print(f"Removed {len(removed)} outliers, still not monotonic.")
#     return df_after_outlier_removal, removed

def remove_outliers(df, outliers, max_removal_pct, agg_func, group_col, agg_col):
    df_after_outlier_removal = df.copy()
    total_rows = len(df)

    # Compute max removals allowed
    max_removals = int(total_rows * max_removal_pct / 100)
    max_possible_removals = min(max_removals, len(outliers))
    print(f"Total rows in DataFrame: {total_rows}, max removals allowed: {max_removals}, actual removals: {max_possible_removals}")

    # Take first N outliers and remove all at once
    outliers_to_remove = outliers.iloc[:max_possible_removals]
    df_after_outlier_removal = df_after_outlier_removal.drop(outliers_to_remove.index)

    # Check if monotonicity is restored
    if is_monotonic(df_after_outlier_removal, group_col=group_col, agg_col=agg_col, agg_func=agg_func):
        print(f"Removed {len(outliers_to_remove)} outliers, monotonic.")
    else:
        print(f"Removed {len(outliers_to_remove)} outliers, still not monotonic.")

    return df_after_outlier_removal, outliers_to_remove.index.tolist()


def save_results(df, method_name, output_folder, dataset_name, agg_func):
    """Save the outlier detection results to a CSV file."""
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"removed_{dataset_name}_{method_name}_{agg_func}.csv")
    df.to_csv(output_file, index=False)
    print(f"Outlier removal results saved to {output_file}")

# Any value too far from the mean (in terms of standard deviations) is considered an outlier.
# Adjust threshold: Higher value = fewer outliers, Lower value = more outliers.
def z_score_outliers(df, column, threshold):
    """Detect outliers using Z-score method with configurable threshold."""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores > threshold]

# Detects points that are far from their neighbors.
# Adjust neighbors: Increase → More robust but may miss some outliers, Decrease → More local and sensitive to small change.
def knn_outliers(df, column, neighbors):
    """Detect outliers using k-Nearest Neighbors (kNN) based method with configurable neighbors."""
    lof = LocalOutlierFactor(n_neighbors=neighbors)
    outliers = lof.fit_predict(df[[column]])
    return df[outliers == -1]

# Isolates outliers using random partitions (random decision trees).
# Adjust contamination: Increase → Looser, More outliers detected, Decrease → Stricter, Fewer outliers detected.
def isolation_forest_outliers(df, column, contamination):
    """Detect outliers using Isolation Forest with configurable contamination parameter."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(df[[column]])
    return df[predictions == -1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier Detection Methods")
    parser.add_argument("dataset_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--agg_func", type=str, choices=["max", "sum", "avg", "median"],
                        help="Aggregation function to check monotonicity")
    parser.add_argument("--group_col", type=str, help="Column to group by")
    parser.add_argument("--agg_col", type=str, help="Column to aggregate")
    parser.add_argument("--max_removal_pct", type=float, default=100, help="Maximum percentage of rows to remove")
    parser.add_argument("--output_folder", type=str, default="outlier_results", help="Output folder")
    parser.add_argument("--method", type=str, help="Outlier detection method", choices=["z_score", "knn", "isolation_forest"])
    parser.add_argument("--param", type=float, help="Method parameter")

    args = parser.parse_args()

    method_param = args.param
    method_name = args.method
    if method_name == "knn":
        method_param = int(method_param)

    print(f"{BLUE_BOLD}Running {method_name} outlier detection method...{RESET}\n"
          f"{BLUE_BOLD}method parameter:{RESET} {BOLD}{method_param}{RESET}")

    df = load_dataset(args.dataset_path)
    df = df.dropna() #does not modify the original dataframe
    #todo delete this is just for testing
    #df = df.sample(n=500000, random_state=42)
    dataset_name = os.path.basename(args.dataset_path).replace(".csv", "")

    methods = {
        "z_score": lambda df, col: z_score_outliers(df, col, method_param),
        "knn": lambda df, col: knn_outliers(df, col, method_param),
        "isolation_forest": lambda df, col: isolation_forest_outliers(df, col, method_param),
    }

    start_time = time.time()
    method = methods[method_name]
    outliers = method(df, args.agg_col)
    print(f"Detected {len(outliers)} outliers")
    df_after_outlier_removal, removed_rows = remove_outliers(df, outliers, args.max_removal_pct,
                                                             args.agg_func, args.group_col, args.agg_col)
    save_results(df_after_outlier_removal, method_name, args.output_folder, dataset_name, args.agg_func)
    #todo: save in to a log/file
    print(f"Method: {method_name}, Removed rows: {len(removed_rows)}, Time: {time.time() - start_time:.4f} sec")
