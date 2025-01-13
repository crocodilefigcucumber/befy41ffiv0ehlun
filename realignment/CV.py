import torch
from torch import nn
import os
import json
from config import config as default_config
from concept_corrector_models import (
    BaselineConceptCorrector,
    LSTMConceptCorrector,
    MultiLSTMConceptCorrector,
    GRUConceptCorrector,
    RNNConceptCorrector,
    MultiGRUConceptCorrector,
    MultiRNNConceptCorrector,
)
from train import train_model
from eval import evaluate_baseline
from data_loader import load_data, create_dataloaders, CustomDataset

 
# =========================
# Main Function
# =========================
def main():
    config = default_config
    device = config['device']
    print(f"Using device: {device}")
    predicted_concepts, groundtruth_concepts, concept_to_cluster, input_size, output_size, number_clusters = load_data(config)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        predicted_concepts, groundtruth_concepts, config
    )
    model_type = config['model']
    print(f"Selected model type: {model_type}")
    
    # Initialize the appropriate Concept Corrector model
    if model_type == 'MultiLSTM':
        ConceptCorrectorClass = MultiLSTMConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            m_clusters=number_clusters,
            concept_to_cluster=concept_to_cluster,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'LSTM':
        ConceptCorrectorClass = LSTMConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'Baseline':
        ConceptCorrectorClass = BaselineConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            output_size=output_size,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'GRU':
        ConceptCorrectorClass = GRUConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'RNN':
        ConceptCorrectorClass = RNNConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'MultiGRU':
        ConceptCorrectorClass = MultiGRUConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            m_clusters=number_clusters,
            concept_to_cluster=concept_to_cluster,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'MultiRNN':
        ConceptCorrectorClass = MultiRNNConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            m_clusters=number_clusters,
            concept_to_cluster=concept_to_cluster,
            input_format=config['input_format']
        ).to(device)


    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"{model_type} model initialized.")
    
    # Initialize model weights
    for name, param in concept_corrector.named_parameters():
        if 'weight' in name and param.ndimension() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    print("Model weights initialized.")
    
    # Load adapter if provided
    adapter = None
    if config['adapter_path'] is not None:
        if os.path.exists(config['adapter_path']):
            adapter = torch.load(config['adapter_path']).to(device)
            print(f"Using adapter from: {config['adapter_path']}")
        else:
            raise FileNotFoundError(f"Adapter path {config['adapter_path']} does not exist.")
    else:
        print("No adapter path provided. Skipping adapter loading.")
    
    # Train the model if not Baseline
    if config['model'] != 'Baseline':
        train_model(concept_corrector, train_loader, val_loader, device, config, concept_to_cluster, adapter)
    else:

        # For Baseline model, save the config dictionary
        # Create directory structure for Baseline model
        baseline_dir = os.path.join('trained_models', config['dataset'], 'Baseline')
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Save the config dictionary as config.json in Baseline directory
        config_save_path = os.path.join(baseline_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_save_path}")
        
        # Perform intervention and evaluation for Baseline model
        print("\nBaseline Model Evaluation with Interventions:")
        verbose = config['verbose']
        # Evaluate on Training Data
        print("Intervening on Training Data:")
        evaluate_baseline(concept_corrector, train_loader, device, config, concept_to_cluster, adapter, phase='Training', verbose=verbose)
        # Evaluate on Validation Data
        print("Intervening on Validation Data:")
        evaluate_baseline(concept_corrector, val_loader, device, config, concept_to_cluster, adapter, phase='Validation', verbose=verbose)
        print("Baseline model evaluation completed.")


if __name__ == '__main__':
    main()
