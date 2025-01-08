import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from collections import Counter
from clustering import clusterConcepts

TRAIN_CONCEPT_PATH = "data/celeba/output/concepts_train.csv"
PLOTS_PATH = "experiments/plots/CelebA/"
CLUSTERS_PATH = "experiments/clusters/CelebA/"
VISUALIZATION = True

# load concepts
concepts = pd.read_csv(TRAIN_CONCEPT_PATH, index_col="file_name")
concepts = np.array(concepts)

# read header, it contains the labels

with open(TRAIN_CONCEPT_PATH, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    labels = next(reader)  # Reads the first row as the header
    labels = [label for label in labels if label != "file_name"]  # remove id column


no_clusters = 2

table_data = clusterConcepts(concepts, no_clusters=no_clusters)
table_data.to_csv(CLUSTERS_PATH + "CelebA_clusters_idx.csv", index=False)

table_data = clusterConcepts(concepts, no_clusters=no_clusters, str_labels=labels)
table_data.to_csv(CLUSTERS_PATH + "CelebA_clusters_str.csv", index=False)

# We will append random noise to our concepts to check clustering stability:
no_concepts = len(labels)
no_observations = len(concepts)
no_noisy = int(no_concepts * 0.1)

# Set seed
np.random.seed(1)

concepts = np.column_stack(
    [
        concepts,
        np.random.binomial(n=1, p=0.5, size=(no_observations, no_noisy)),
    ]
)

labels = labels + [f"Noise {i}" for i in range(no_noisy)]

no_clusters = 2

table_data = clusterConcepts(concepts, no_clusters=no_clusters)
table_data.to_csv(CLUSTERS_PATH + "CelebA_clusters_noisy_idx.csv", index=False)

table_data = clusterConcepts(concepts, no_clusters=no_clusters, str_labels=labels)
table_data.to_csv(CLUSTERS_PATH + "CelebA_clusters_noisy_str.csv", index=False)

if VISUALIZATION:
    sns.set_theme(context="paper", style="white")

    # Increase font sizes globally
    plt.rcParams.update(
        {
            "font.size": 18,  # General font size
            "axes.titlesize": 24,  # Title font size
            "axes.labelsize": 20,  # Axis label font size
            "xtick.labelsize": 18,  # X-tick label font size
            "ytick.labelsize": 18,  # Y-tick label font size
            "legend.fontsize": 16,  # Legend font size
            "legend.title_fontsize": 18,  # Legend title font size
        }
    )

    # Extract Suffixes
    suffix_dict = {}
    cluster_shift = len("cluster_")
    for cluster in table_data.columns:
        capitalized = f"Cluster ${cluster[cluster_shift :]}$"
        suffixes = []

        for entry in table_data[cluster]:
            if "Noise" in entry:
                suffixes.append("Noise")
            else:
                suffixes.append("Original")

        suffix_dict[capitalized] = sorted(suffixes)

    # Count the frequency of each suffix across all lists
    suffix_counts = Counter(
        item for sublist in suffix_dict.values() for item in sublist
    )

    # Flatten and get unique strings
    unique_items = sorted(
        set(item for sublist in suffix_dict.values() for item in sublist)
    )

    color_shape_map = {"Original": ("black", "s"), "Noise": ("red", "s")}

    # Track which labels have already been added to avoid duplicates in the legend
    already_labeled = set()

    # Create the plot
    plt.figure(figsize=(15, len(suffix_dict) * 2))
    y_offset = 0

    for key, lst in suffix_dict.items():
        x_vals = (
            np.linspace(0, len(lst) - 1, len(lst)) * 1
        )  # Increase spacing between symbols
        y_vals = [y_offset] * len(lst)

        # Plot each string with its assigned color and shape
        for i, item in enumerate(lst):
            color, shape = color_shape_map[item]
            label = item if item not in already_labeled else None  # Add label only once
            if label is not None:
                label = label.capitalize()

            plt.scatter(
                x_vals[i],
                y_vals[i],
                color=color,
                marker=shape,
                s=250,
                label=label,
            )
            already_labeled.add(item)  # Mark this item as labeled

        y_offset += 1  # Move to the next line

    # Beautify the plot
    plt.yticks(range(len(suffix_dict)), suffix_dict.keys())
    plt.xticks([])  # No x-axis labels
    plt.legend(bbox_to_anchor=(0, -0.2), loc="lower left", ncol=2)
    plt.savefig(PLOTS_PATH + "CelebA_clusters_noisy.pdf", bbox_inches="tight")
