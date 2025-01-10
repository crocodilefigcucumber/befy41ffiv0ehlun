import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_data_from_config(config):
    """
    Generate data based on the dataset specified in the config.

    Args:
        config (dict): Configuration dictionary containing dataset info.

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
        },
        'Awa2': {
            'predicted_file': 'data/awa2/output/awa2_prediction_matrices.npz',
            'cluster_file': 'experiments/clusters/AwA2/AwA2_clusters_idx.csv',
            'groundtruth_file' : 'data/awa2/output/concepts_train.csv',
        },
        'CelebA': {
            'predicted_file': 'data/celeba/output/celeba_prediction_matrices.npz',
            'cluster_file': 'experiments/clusters/CelebA/CelebA_clusters_idx.csv',
            'groundtruth_file' : 'data/celeba/output/concepts_train.csv'
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
    if not os.path.exists(cluster_file_path):
    if not os.path.exists(predicted_file_path):
        raise FileNotFoundError(f"Predicted concepts file not found: {predicted_file_path}")
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file_path}")
    
    # Load predicted concepts
    predicted_data = np.load(predicted_file_path)
    predicted_concepts = torch.tensor(predicted_data['first'], dtype=torch.float32)
    # Load GT concepts
    groundtruth_data = pd.read_csv(groundtruth_file)
    groundtruth_concepts = torch.tensor(groundtruth_data,dtype=torch.float32)
    # Load cluster assignments
    cluster_data = pd.read_csv(cluster_file_path)
    cluster_assignments =.....
    return predicted_concepts, groundtruth_concepts, cluster_assignments