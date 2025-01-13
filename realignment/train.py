import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

from eval import evaluate_model
from intervention_utils import ucp
from train_utils import compute_loss


# =========================
# Training Function
# =========================
def train_model(
    concept_corrector: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    concept_to_cluster: list,
    adapter: nn.Module=None, 
    run_idx=0
):
    optimizer = optim.Adam(
        concept_corrector.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("Optimizer and scheduler initialized.")
    
    criterion = nn.BCELoss()
    print("Loss function initialized.")
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = config['early_stop_patience']
    train_losses = []
    val_losses = []

    # model saving
    trained_models_dir = os.path.join('trained_models', config['dataset'], config['model'])
    os.makedirs(trained_models_dir, exist_ok=True)
    final_model_filename = f"run_{run_idx}_best_model.pth"
    final_model_path = os.path.join(trained_models_dir, final_model_filename)
    
    # Save the config dictionary as config.json
    config_save_path = os.path.join(trained_models_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize a variable to store the best state_dict
    best_model_state = None

    # Initial Evaluation Before Training
    print("\nInitial Evaluation Before Training:")
    initial_train_loss = evaluate_model(
        concept_corrector, train_loader, device, config, concept_to_cluster,
        adapter, phase='Initial Training'
    )
    initial_val_loss = evaluate_model(
        concept_corrector, val_loader, device, config, concept_to_cluster,
        adapter, phase='Initial Validation'
    )
    
    # Main Training Loop
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        concept_corrector.train()
        train_loss = 0.0
        
        for batch in train_loader:
            predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            # CHANGED: Extended the if-condition for new model types
            if config['model'] in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
            elif config['model'] in ['LSTM', 'GRU', 'RNN']:
                initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
            elif config['model'] == 'Baseline':
                initial_hidden = None
            else:
                raise ValueError(f"Unsupported model type: {config['model']}")

            if adapter is not None:
                # If your adapter is similarly single- or multi-cluster, handle it here
                if config['model'] in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                    predicted_concepts, _ = adapter.forward_single_timestep(
                        predicted_concepts,
                        torch.zeros_like(predicted_concepts),
                        predicted_concepts,
                        initial_hidden
                    )
                elif config['model'] in ['LSTM', 'GRU', 'RNN']:
                    predicted_concepts, _ = adapter.forward_single_timestep(
                        predicted_concepts,
                        torch.zeros_like(predicted_concepts),
                        predicted_concepts,
                        initial_hidden
                    )
                # Baseline remains unchanged

            loss = compute_loss(
                concept_corrector,
                predicted_concepts,
                groundtruth_concepts,
                initial_hidden,
                ucp,
                config['max_interventions'],
                criterion,
                concept_to_cluster,
                config['model'],
                verbose=False
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        average_train_loss = train_loss / len(train_loader)
        train_losses.append(average_train_loss)
        
        # Validation Phase
        concept_corrector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]
                
                # CHANGED: Same extension for new models
                if config['model'] in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                    initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                elif config['model'] in ['LSTM', 'GRU', 'RNN']:
                    initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                elif config['model'] == 'Baseline':
                    initial_hidden = None
                else:
                    raise ValueError(f"Unsupported model type: {config['model']}")

                if adapter is not None:
                    if config['model'] in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                        predicted_concepts, _ = adapter.forward_single_timestep(
                            predicted_concepts,
                            torch.zeros_like(predicted_concepts),
                            predicted_concepts,
                            initial_hidden
                        )
                    elif config['model'] in ['LSTM', 'GRU', 'RNN']:
                        predicted_concepts, _ = adapter.forward_single_timestep(
                            predicted_concepts,
                            torch.zeros_like(predicted_concepts),
                            predicted_concepts,
                            initial_hidden
                        )
                
                loss = compute_loss(
                    concept_corrector,
                    predicted_concepts,
                    groundtruth_concepts,
                    initial_hidden,
                    ucp,
                    config['max_interventions'],
                    criterion,
                    concept_to_cluster,
                    config['model'],
                    verbose=False
                )
                val_loss += loss.item()
        
        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        
        # Early Stopping Check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            early_stop_counter = 0
            print(f"Epoch {epoch}: Improved validation loss to {best_val_loss:.4f}.")
            # Store the best model state_dict
            best_model_state = concept_corrector.state_dict()
        else:
            early_stop_counter += 1
            print(f"Epoch {epoch}: Validation loss did not improve ({average_val_loss:.4f}).")
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{config['epochs']}], "
              f"Train Loss: {average_train_loss:.4f}, "
              f"Val Loss: {average_val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    # Save the best model state dictionary if available
    if best_model_state is not None:
        torch.save(best_model_state, final_model_path)
        print(f"Best model saved to {final_model_path}")
    else:
        print("No improvement observed during training. Model not saved.")
