import pandas as pd
import numpy as np
import csv

from clustering import clusterConcepts

TRAIN_CONCEPT_PATH = "data/celeba/output/concepts_train.csv"
CLUSTERS_PATH = "experiments/clusters/CelebA/"

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
