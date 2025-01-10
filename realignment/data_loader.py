import torch
from torch.utils.data import TensorDataset, DataLoader

#  Set input_size and output_size based on number of concepts (k)
config['input_size'] = config['data_generation']['k']
config['output_size'] = config['data_generation']['k']
# =========================
# READ IN DATA
# =========================
def generate_data_from_config(config):
    """
    Generate data based on the dataset specified in the config.

    Args:
        config (dict): Configuration dictionary containing dataset info.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
            predicted_concepts, groundtruth_concepts, cluster_assignments, labels.
    """
    dataset = config['dataset']
    labels_count = config['labels_count']
    seed = config['seed']

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

    if dataset not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    paths = dataset_paths[dataset]
    predicted_file_path = paths['predicted_file']
    cluster_file_path = paths['cluster_file']

        # Validate that files exist
    if not os.path.exists(predicted_file_path):
        raise FileNotFoundError(f"Predicted concepts file not found: {predicted_file_path}")
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file_path}")
    
        # Load predicted concepts
    predicted_data = np.load(predicted_file_path)
    predicted_concepts = torch.tensor(predicted_data['first'], dtype=torch.float32)
    cluster_data = pd.read_csv(cluster_file_path)

    return predicted_concepts, groundtruth_concepts, cluster_assignments, labels

def generate_synthetic_data(k: int, n: int, J: int, m: int, seed: int):
    torch.manual_seed(seed)
    predicted_concepts = torch.rand(n, k)
    groundtruth_concepts = (torch.rand(n, k) > 0.5).float()
    cluster_assignments = {cid: [] for cid in range(m)}
    for concept_idx in range(k):
        assigned_cluster = torch.randint(low=0, high=m, size=(1,)).item()
        cluster_assignments[assigned_cluster].append(concept_idx)
    labels = torch.randint(low=0, high=J, size=(n,))
    return predicted_concepts, groundtruth_concepts, cluster_assignments, labels
print()