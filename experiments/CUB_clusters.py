import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import radialtree as rt  # https://github.com/koonimaru/radialtree
from clustpy.hierarchical import Diana

TRAIN_CONCEPT_PATH = "data/CUB_200_2011_concepts_train.csv"
concepts = pd.read_csv(TRAIN_CONCEPT_PATH, index_col="image_id")
concepts = np.array(concepts)

ATTRIBUTE_PATH = "data/CUB_200_2011/attributes/attributes.txt"

attributes = pd.read_csv(
    ATTRIBUTE_PATH,
    sep=r"\s+",
    names=["concept_id", "description"],
    index_col="concept_id",
)
label_dict = attributes["description"].to_dict()
labels = [label_dict[i + 1][4:] for i in range(len(label_dict))]

PLOTS_PATH = "plots/CUB/"

metric = "jaccard"
distance_matrix = pdist(concepts.T, metric=metric)

# Apply agglomerative clustering using complete linkage
linkage_matrix = linkage(distance_matrix, method="complete")

"""
no_concepts = concepts.shape[1]
trivial = list(range(1, no_concepts + 1))

smallest_t = 0
for t in np.linspace(0, 1, int(1e4)):
    if len(set(fcluster(Z=linkage_matrix, t=t, criterion="distance"))) < 5:
        break
    else:
        smallest_t = t

no_clusters = len(set(fcluster(Z=linkage_matrix, t=smallest_t, criterion="distance")))

# Visualize dendrogram
out = dendrogram(
    Z=linkage_matrix,
    p=no_clusters,
    labels=labels,
    orientation="left",
    color_threshold=smallest_t,
    no_plot=True,
)
rt.plot(out, figsize=(15, 15), pallete="Set1")
plt.savefig(PLOTS_PATH + "labeled.pdf", dpi=400)

out = dendrogram(
    Z=linkage_matrix,
    p=no_clusters,
    labels=labels,
    truncate_mode="lastp",
    orientation="left",
    show_leaf_counts=True,
    no_plot=True,
)
rt.plot(out)
plt.savefig(PLOTS_PATH + "collapsed.pdf", dpi=300)
"""

# Apply divisive clustering using same metric

diana = Diana(metric=metric)
diana.fit(concepts.T)
no_clusters = 4
clusters = diana.flat_clustering(n_leaf_nodes_to_keep=no_clusters)

clust_str_labels = {label: [] for label in range(no_clusters)}
clust_labels = {label: [] for label in range(no_clusters)}

for i in range(len(clusters)):
    clust_str_labels[clusters[i]].append(label_dict[i + 1])
    clust_labels[clusters[i]].append(i)

data = clust_labels
max_len = max(len(v) for v in data.values())  # Find the longest list
table_data = {
    key: value + [""] * (max_len - len(value)) for key, value in data.items()
}  # Pad shorter lists
table_data = pd.DataFrame.from_dict(table_data, orient="columns")
table_data.to_csv("clusters_idx.csv", index=False)

data = clust_str_labels
max_len = max(len(v) for v in data.values())  # Find the longest list
table_data = {
    key: value + [""] * (max_len - len(value)) for key, value in data.items()
}  # Pad shorter lists
table_data = pd.DataFrame.from_dict(table_data, orient="columns")
table_data.to_csv("clusters_str.csv", index=False)
