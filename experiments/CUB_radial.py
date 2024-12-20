import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import radialtree as rt  # https://github.com/koonimaru/radialtree


CONCEPT_PATH = "data/CUB_200_2011_concepts.csv"
concepts = pd.read_csv(CONCEPT_PATH, index_col="image_id")
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
