import torch

# =========================
# Hyperparameters Configuration
# =========================
config = {
    # Model Parameters
    'model': 'LSTM',                # Model type: 'LSTM', 'MultiLSTM', or 'Baseline'
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
    'epochs': 50,                        # Number of training epochs
    'max_interventions': 10,            # Maximum number of interventions per trajectory

    # Intervention Policy
    'intervention_policy_train': 'ucp',     # Policy for training interventions
    'intervention_policy_validate': 'ucp',  # Policy for validation interventions

    # Verbosity
    'verbose': True,                    # If True, prints detailed intervention information

    # Device Configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Automatically select 'cuda' if available


    # REMOVE LATER

    # # Data Generation Parameters
    # 'data_generation': {
    #     'k': 50,             # Number of concepts
    #     'n_total': 12000,    # Total number of observations (training + validation)
    #     'n_train': 10000,    # Number of training observations
    #     'n_val': 2000,       # Number of validation observations
    #     'J': 5,              # Number of target classes
    #     'm': 10,             # Number of concept clusters (used only for MultiLSTM)
    #     'seed': 42,          # Random seed
    # },

    # Adapter Configuration
    'adapter_path': None,                # Path to adapter model (set to None if not used)

    # Early Stopping Parameters
    'early_stop_patience': 10,           # Number of epochs with no improvement after which training will be stopped
}