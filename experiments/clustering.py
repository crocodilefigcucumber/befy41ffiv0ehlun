from pandas import DataFrame
from clustpy.hierarchical import Diana


def clusterConcepts(concepts, no_clusters, label_dict=None, metric="jaccard"):
    """Perform DIANA clustering on concepts, export as DataFrame

    Keyword arguments:
    concepts -- np.array of shape (N,n_concepts)
    label_dict -- only pass if you want string outputs
    """

    diana = Diana(metric=metric)
    diana.fit(concepts.T)
    clusters = diana.flat_clustering(n_leaf_nodes_to_keep=no_clusters)

    if label_dict is not None:
        clust_str_labels = {label: [] for label in range(no_clusters)}

        for i in range(len(clusters)):
            clust_str_labels[clusters[i]].append(label_dict[i])

        data = clust_str_labels
    else:
        clust_labels = {label: [] for label in range(no_clusters)}
        for i in range(len(clusters)):
            clust_labels[clusters[i]].append(i)

        data = clust_labels

    max_len = max(len(v) for v in data.values())  # Find the longest list
    table_data = {
        key: value + [""] * (max_len - len(value)) for key, value in data.items()
    }  # Pad shorter lists
    return DataFrame.from_dict(table_data, orient="columns")
