import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset


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
    # Load GT concepts
    groundtruth_data = pd.read_csv(groundtruth_file)
    print(predicted_concepts)
    groundtruth_concepts = torch.tensor(groundtruth_data.values,dtype=torch.float32)

    # Load cluster assignments
    concept_dict = pd.read_csv(cluster_file_path)
    concept_dict_melted = concept_dict.T.melt(ignore_index=False, var_name="concept").dropna(subset=['concept'])
    cluster_assingments = pd.crosstab(concept_dict_melted.index, concept_dict_melted['value'])
    cluster_assingments = torch.tensor(cluster_assingments.values, dtype=torch.float32)
    print(f"Predicted Concepts Size: {predicted_concepts.size(0)}")
    print(f"Groundtruth Concepts Size: {groundtruth_concepts.size(0)}")
    print(f"Cluster Assignments Size: {cluster_assingments.size(2)}")

    return predicted_concepts, groundtruth_concepts, cluster_assingments
load_data(config)

def create_splits(file_path):
    """
    Splits the train data from train_test_split.txt into train and validation sets (80/20).
    Args:
        file_path (str): Path to the train_test_split.txt file.
    Returns:
        train_split (list): List of indices for the training set.
        val_split (list): List of indices for the validation set.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["id", "split"])
    train_ids = data[data['split'] == 1]['id'].tolist()
    
    # Shuffle train IDs
    shuffled_train_ids = np.random.permutation(train_ids)
    
    # Calculate the split point for 80/20 split
    n_train = int(len(train_ids) * 0.8)
    
    # Assign 80% to 'train' and 20% to 'val'
    train_split = shuffled_train_ids[:n_train].tolist()
    val_split = shuffled_train_ids[n_train:].tolist()
    
    return train_split, val_split


class CustomDataset(Dataset):
    def __init__(self, predicted_concepts, groundtruth_concepts, cluster_assignments):
        self.predicted_concepts = predicted_concepts
        self.groundtruth_concepts = groundtruth_concepts
        self.cluster_assignments = cluster_assignments

    def __len__(self):
        return self.predicted_concepts.size(0)

    def __getitem__(self, idx):
        return (
            self.predicted_concepts[idx],
            self.groundtruth_concepts[idx],
            self.cluster_assignments[idx],
        )

def create_dataloaders(predicted_concepts, groundtruth_concepts, cluster_assignments, train_split, val_split, batch_size=32):
    dataset = CustomDataset(predicted_concepts, groundtruth_concepts, cluster_assignments)
    train_loader = DataLoader(Subset(dataset, train_split), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_split), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Main Test Code
if __name__ == "__main__":
    # Configuration for the dataset
    config = {'dataset': 'CUB'}

    # Load data
    predicted_concepts, groundtruth_concepts, cluster_assignments = load_data(config)

    # Create splits
    file_path = 'data/cub/CUB_200_2011/train_test_split.txt'
    train_split, val_split = create_splits(file_path)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        predicted_concepts, groundtruth_concepts, cluster_assignments, train_split, val_split, batch_size=32
    )