import torch
from torch import nn
from torch.utils.data import DataLoader

from intervention_utils import ucp
from train_utils import compute_loss


# =========================
# Evaluation Function for Non-Baseline Models
# =========================

def evaluate_model(
    concept_corrector: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    concept_to_cluster: list,
    adapter: nn.Module=None,
    phase: str='Validation'
):
    concept_corrector.eval()
    total_loss = 0.0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batch in loader:
            predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]

            if config['model'] in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
            elif config['model'] in ['LSTM', 'GRU', 'RNN']:
                initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
            elif config['model'] == 'Baseline':
                initial_hidden = None
            else:
                raise ValueError(f"Unsupported model type: {config['model']}")

            # If adapter is used
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

            # Finally, compute the loss
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
                verbose=False  # Set verbose to False for non-Baseline models
            )
            total_loss += loss.item()
    
    average_loss = total_loss / len(loader)
    print(f"{phase} Loss: {average_loss:.4f}")
    return average_loss


# =========================
# Evaluation Function for Baseline Model
# =========================
def evaluate_baseline(concept_corrector: nn.Module, loader: DataLoader, device: torch.device, config: dict, concept_to_cluster: list, adapter: nn.Module=None, phase: str='Validation', verbose: bool=False):
    concept_corrector.eval()
    total_loss = 0.0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batch in loader:
            predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]
            # No hidden states for Baseline
            loss = compute_loss(
                concept_corrector,
                predicted_concepts,
                groundtruth_concepts,
                None,
                ucp,
                config['max_interventions'],
                criterion,
                concept_to_cluster,
                config['model'],
                verbose=verbose  # Pass the verbose flag here
            )
            total_loss += loss.item()
    average_loss = total_loss / len(loader)
    print(f"{phase} Loss for Baseline model: {average_loss:.4f}")
    return average_loss
