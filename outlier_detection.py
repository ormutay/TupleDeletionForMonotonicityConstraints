import pandas as pd
import numpy as np
import os
import argparse
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import os

# ANSI color codes
BLUE_BOLD = "\033[94m\033[1m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_dataset(dataset_path):
    """Load dataset from CSV file."""
    return pd.read_csv(dataset_path)


def is_monotonic(df, group_col="A", value_col="B", agg_func="max"):
    """Check if B is non-decreasing across groups of A based on the specified aggregation function."""
    agg_funcs = {
        "max": lambda x: x.max(),
        "sum": lambda x: x.sum(),
        "avg": lambda x: x.mean(),
        "median": lambda x: x.median(),
    }
    if agg_func not in agg_funcs:
        raise ValueError("Unsupported aggregation function. Choose from 'max', 'sum', 'avg', 'median'.")

    grouped = df.groupby(group_col)[value_col].apply(agg_funcs[agg_func])
    return (grouped.diff().dropna() >= 0).all()


def remove_until_monotonic(df, outliers, max_removal_pct, agg_func):
    """Gradually remove outliers until monotonicity is restored or max_removal_pct is reached."""
    total_rows = len(df)
    max_removals = int(total_rows * max_removal_pct / 100)
    removed = []

    for index, row in outliers.iterrows():
        if len(removed) >= max_removals:
            break
        df = df.drop(index)
        removed.append(index)
        if is_monotonic(df, agg_func=agg_func):
            print(f"Monotonicity restored after removing {len(removed)} outliers.")
            break
    print(f"Removed {len(removed)} outliers, still not monotonic.")
    return df, removed


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

# Identifies dense regions in data and classifies points far from these regions as outliers.
# Adjust eps: Increase → More points included in a cluster (More sensitive to outliers), Decrease → More clusters formed (Less sensitive).
# Adjust min_samples: Increase → More points required to form a cluster (Fewer outliers) , Decrease → More outliers detected.
def dbscan_outliers(df, column, eps, min_samples):
    """Detect outliers using DBSCAN clustering with configurable eps and min_samples."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df[[column]])
    return df[labels == -1]

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
    parser.add_argument("--column", type=str, default="B", help="Column to analyze")
    parser.add_argument("--max_removal_pct", type=float, default=100, help="Maximum percentage of rows to remove")
    parser.add_argument("--output_folder", type=str, default="outlier_results", help="Output folder")

    parser.add_argument("--z_score_threshold", type=float, default=1.5, help="Threshold for Z-score method")
    parser.add_argument("--knn_neighbors", type=int, default=10, help="Number of neighbors for kNN method")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="Epsilon for DBSCAN method")
    parser.add_argument("--dbscan_min_samples", type=int, default=10, help="Minimum samples for DBSCAN method")
    parser.add_argument("--isolation_contamination", type=float, default=0.1,
                        help="Contamination level for Isolation Forest method")

    args = parser.parse_args()

    z_score_threshold = args.z_score_threshold
    knn_neighbors = args.knn_neighbors
    dbscan_eps = args.dbscan_eps
    dbscan_min_samples = args.dbscan_min_samples
    isolation_contamination = args.isolation_contamination

    # Parameters for each method
    # z_score_threshold = 1.5
    # knn_neighbors = 10
    # dbscan_eps = 0.4
    # dbscan_min_samples = 12
    # isolation_contamination = 0.1
    print(f"{BLUE_BOLD}The parameters for each method are:{RESET}\n"
          f"{BLUE_BOLD}Z-score threshold:{RESET} {BOLD}{z_score_threshold}{RESET}\n"
          f"{BLUE_BOLD}kNN neighbors:{RESET} {BOLD}{knn_neighbors}{RESET}\n"
          f"{BLUE_BOLD}DBSCAN eps:{RESET} {BOLD}{dbscan_eps}{RESET}, "
          f"{BLUE_BOLD}min_samples:{RESET} {BOLD}{dbscan_min_samples}{RESET}\n"
          f"{BLUE_BOLD}Isolation Forest contamination:{RESET} {BOLD}{isolation_contamination}{RESET}")

    df = load_dataset(args.dataset_path)
    dataset_name = os.path.basename(args.dataset_path).replace(".csv", "")

    methods = {
        "z_score": lambda df, col: z_score_outliers(df, col, z_score_threshold),
        "knn": lambda df, col: knn_outliers(df, col, knn_neighbors),
        "dbscan": lambda df, col: dbscan_outliers(df, col, dbscan_eps, dbscan_min_samples),
        "isolation_forest": lambda df, col: isolation_forest_outliers(df, col, isolation_contamination),
    }

    for method_name, method in methods.items():
        print(f"{BLUE_BOLD}Running {method_name} outlier detection...{RESET}")
        outliers = method(df, args.column)
        df_filtered, removed_rows = remove_until_monotonic(df, outliers, args.max_removal_pct, args.agg_func)
        save_results(df_filtered, method_name, args.output_folder, dataset_name, args.agg_func)