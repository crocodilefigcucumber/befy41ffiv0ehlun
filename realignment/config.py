import torch

# =========================
# Hyperparameters Configuration
# =========================
config = {
    # Model Parameters
    'model': 'MultiLSTM',                # Model type: 'LSTM', 'MultiLSTM', or 'Baseline'
    'hidden_size': 256,                 # Number of hidden units in LSTM
    'num_layers': 5,                    # Number of LSTM layers
    'input_format': 'original_and_intervened_inplace',  # Input format for the model

    # Dataset
    'dataset': 'CUB',  # Options: 'CUB', 'Awa2', 'CelebA'
    'seed' : 42,

    # Training Parameters
    'learning_rate': 0.0001,            # Learning rate for optimizer
    'weight_decay': 1e-5,               # Weight decay (L2 regularization)
    'batch_size': 64,                   # Batch size for training
    'epochs': 100,                        # Number of training epochs
    'max_interventions': 10,            # Maximum number of interventions per trajectory

    # Intervention Policy
    'intervention_policy_train': 'ucp',     # Policy for training interventions
    'intervention_policy_validate': 'ucp',  # Policy for validation interventions

    # Verbosity
    'verbose': True,                    # If True, prints detailed intervention information

    # Device Configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Automatically select 'cuda' if available

    # Adapter Configuration
    'adapter_path': None,                # Path to adapter model (set to None if not used)

    # Early Stopping Parameters
    'early_stop_patience': 10,           # Number of epochs with no improvement after which training will be stopped
}