import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


from config import config

def load_data():
    """
    Load data based on the dataset specified in the config.

    Args:
    --

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
            predicted_concepts, groundtruth_concepts, cluster_assignments, labels.
    """
    # Define dataset-specific file paths
    
    dataset_paths = {
        'CUB': {
            'predicted_file': 'data/cub/output/cub_prediction_matrices.npz',
            'cluster_file': 'experiments/clusters/CUB/CUB_clusters_idx.csv',
            'groundtruth_file': 'data/cub/output/concepts_train.csv',
            'n_concept' : 312, 
        },
        'Awa2': {
            'predicted_file': 'data/awa2/output/awa2_prediction_matrices.npz',
            'cluster_file': 'experiments/clusters/AwA2/AwA2_clusters_idx.csv',
            'groundtruth_file' : 'data/awa2/output/concepts_train.csv',
            'n_concept' : 312, 

        },
        'CelebA': {
            'predicted_file': 'data/celeba/output/celeba_prediction_matrices.npz',
            'cluster_file': 'experiments/clusters/CelebA/CelebA_clusters_idx.csv',
            'groundtruth_file' : 'data/celeba/output/concepts_train.csv',
            'n_concept' : 312, 
        },
    }

    dataset = config['dataset']

    if dataset not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    paths = dataset_paths[dataset]

    predicted_file_path = paths['predicted_file']
    cluster_file_path = paths['cluster_file']
    groundtruth_file = paths['groundtruth_file']

    # Validate that files exist
    if not os.path.exists(groundtruth_file):
        raise FileNotFoundError(f"GT concepts file not found: {groundtruth_file}")

    if not os.path.exists(predicted_file_path):
        raise FileNotFoundError(f"Predicted concepts file not found: {predicted_file_path}")
    
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file_path}")
    
    # Load predicted concepts
    predicted_data = np.load(predicted_file_path)
    
    predicted_concepts = torch.tensor(predicted_data['first'], dtype=torch.float32)
    print(predicted_concepts.size())
    # Load GT concepts
    groundtruth_data = pd.read_csv(groundtruth_file)
    groundtruth_concepts = torch.tensor(groundtruth_data.values,dtype=torch.float32)

    # Load cluster assignments
    concept_dict = pd.read_csv(cluster_file_path)
    concept_dict_melted = concept_dict.T.melt(ignore_index=False, var_name="concept").dropna(subset=['concept'])
    cluster_assingments = pd.crosstab(concept_dict_melted.index, concept_dict_melted['value'])
    cluster_assingments = torch.tensor(cluster_assingments.values, dtype=torch.float32)

    return predicted_concepts, groundtruth_concepts, cluster_assingments