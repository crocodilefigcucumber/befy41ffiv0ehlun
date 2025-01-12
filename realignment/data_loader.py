import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

from config import config


def load_data(config):
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
            'groundtruth_file': 'data/cub/output/concepts_test.csv',
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
    # Load GT concepts
    groundtruth_data = pd.read_csv(groundtruth_file)
    # Drop id column
    groundtruth_data = groundtruth_data.drop(groundtruth_data.columns[0], axis=1) 
    groundtruth_concepts = torch.tensor(groundtruth_data.values,dtype=torch.float32)

    # Load cluster assignments
    concept_dict = pd.read_csv(cluster_file_path)
    concept_dict_melted = concept_dict.T.melt(ignore_index=False, var_name="concept").dropna(subset=['concept'])
    cluster_assignments = pd.crosstab(concept_dict_melted.index, concept_dict_melted['value'])
    cluster_assignments.index = range(len(cluster_assignments))
    # Initialize the result list
    result = [-1] * len(cluster_assignments.columns)  # Default to -1 for columns without a `1`

    # Iterate over columns to find the 0-indexed row number of `1`s
    for col_idx in range(len(cluster_assignments.columns)):
        for row_idx in range(len(cluster_assignments)):
            if cluster_assignments.iloc[row_idx, col_idx] == 1:
                result[col_idx] = row_idx  # Assign the 0-indexed row number
                break  # Stop after finding the first `1`
    input_size = paths['n_concept']
    output_size = paths['n_concept']
    number_clusters = len(cluster_assignments)
    return predicted_concepts, groundtruth_concepts, cluster_assignments, input_size, output_size, number_clusters

class CustomDataset(Dataset):
    def __init__(self, predicted_concepts, groundtruth_concepts):
        self.predicted_concepts = predicted_concepts
        self.groundtruth_concepts = groundtruth_concepts

    def __len__(self):
        return self.predicted_concepts.size(0)


    def __getitem__(self, idx):
        return (
            self.predicted_concepts[idx],
            self.groundtruth_concepts[idx],
        )

def create_dataloaders(predicted_concepts, groundtruth_concepts, config):
    """
    Creates DataLoaders for training and validation using an 80/20 split.

    Parameters:
        predicted_concepts (Tensor): Tensor of predicted concepts.
        groundtruth_concepts (Tensor): Tensor of groundtruth concepts.
        batch_size (int): Batch size for the DataLoader.
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create the dataset
    dataset = CustomDataset(predicted_concepts, groundtruth_concepts)
    
    # Split indices for 80/20 split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=config['seed'], 
        shuffle=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        Subset(dataset, train_indices), 
        batch_size=config['batch_size'], 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), 
        batch_size=config['batch_size'], 
        shuffle=False, 
        pin_memory=True
    )

    return train_loader, val_loader