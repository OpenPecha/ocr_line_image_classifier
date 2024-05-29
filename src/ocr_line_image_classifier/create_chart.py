import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_threshold_from_filename(filename):
    """Extract the threshold from the filename."""
    return float(filename.split("_")[-1].split(".csv")[0])


def read_csv_files(directory):
    """Read CSV files from the specified directory and extract the threshold from the filename."""

    data: Dict[str, List[float]]

    data = {
        "thresholds": [],
        "true_positives": [],
        "false_positives": [],
        "true_negatives": [],
        "false_negatives": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Extract the threshold from the filename
            threshold = extract_threshold_from_filename(filename)
            data["thresholds"].append(threshold)

            # Read the CSV file
            df = pd.read_csv(os.path.join(directory, filename))

            # Extract the stats
            data["true_positives"].append(df["true_positives"][0])
            data["false_positives"].append(df["false_positives"][0])
            data["true_negatives"].append(df["true_negatives"][0])
            data["false_negatives"].append(df["false_negatives"][0])
            data["precision"].append(df["precision"][0])
            data["recall"].append(df["recall"][0])
            data["f1_score"].append(df["f1_score"][0])

    return data


def sort_data_by_threshold(data):
    """Sort the data by thresholds."""
    sorted_indices = np.argsort(data["thresholds"])
    for key in data:
        data[key] = np.array(data[key])[sorted_indices]
    return data


def plot_counts(
    ax, x, true_positives, false_positives, true_negatives, false_negatives
):
    """Plot the counts as a bar chart."""
    width = 0.15
    bars1 = ax.bar(x - 1.5 * width, true_positives, width, label="True Positives")
    bars2 = ax.bar(x - 0.5 * width, false_positives, width, label="False Positives")
    bars3 = ax.bar(x + 0.5 * width, true_negatives, width, label="True Negatives")
    bars4 = ax.bar(x + 1.5 * width, false_negatives, width, label="False Negatives")

    ax.set_ylabel("Count")
    ax.set_title(
        "Counts of True Positives, False Positives, True Negatives, False Negatives"
    )
    ax.legend(loc="upper right")

    # Add value labels on top of the bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


def plot_metrics(ax, x, precision, recall, f1_score):
    """Plot precision, recall, and F1 score as a line chart."""
    ax.plot(x, precision, color="tab:red", marker="o", label="Precision")
    ax.plot(x, recall, color="tab:green", marker="o", label="Recall")
    ax.plot(x, f1_score, color="tab:blue", marker="o", label="F1 Score")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1 Score")
    ax.legend(loc="upper right")


def main(directory):
    data = read_csv_files(directory)
    data = sort_data_by_threshold(data)

    x = np.arange(len(data["thresholds"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    plot_counts(
        ax1,
        x,
        data["true_positives"],
        data["false_positives"],
        data["true_negatives"],
        data["false_negatives"],
    )
    plot_metrics(ax2, x, data["precision"], data["recall"], data["f1_score"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(data["thresholds"])

    plt.tight_layout()
    plt.show()
