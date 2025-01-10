import torch
from torch import nn

from intervention_utils import intervene

# =========================
# Trajectory Sampling
# =========================
def sample_trajectory(concept_corrector: nn.Module, concepts: torch.Tensor, groundtruth_concepts: torch.Tensor, initial_hidden, intervention_policy, max_interventions: int, concept_to_cluster: list, model_type: str, verbose: bool=False):
    with torch.no_grad():
        all_inputs = []
        all_masks = []
        all_original_predictions = []
        all_groundtruths = []
        if model_type == 'MultiLSTM':
            hidden_states = initial_hidden
        elif model_type == 'LSTM':
            hidden = initial_hidden
        elif model_type == 'Baseline':
            pass
        already_intervened_concepts = torch.zeros_like(concepts)
        original_predictions = concepts.detach().clone()
        all_inputs.append(concepts.detach().clone())
        all_masks.append(already_intervened_concepts.detach().clone())
        all_original_predictions.append(original_predictions)
        all_groundtruths.append(groundtruth_concepts.detach().clone())
        
        if model_type == 'MultiLSTM':
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden_states
            )
        elif model_type == 'LSTM':
            out, hidden = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden
            )
        elif model_type == "Baseline":
            out = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions
            )
        concepts = out
        
        for intervention_step in range(1, max_interventions + 1):
            # Only print replacements for Baseline model
            print_replacements = (model_type == 'Baseline') and verbose
            concepts, already_intervened_concepts = intervene(
                concepts, already_intervened_concepts, groundtruth_concepts, intervention_policy, verbose=print_replacements
            )
            all_inputs.append(concepts.detach().clone())
            all_masks.append(already_intervened_concepts.detach().clone())
            all_original_predictions.append(original_predictions)
            all_groundtruths.append(groundtruth_concepts.detach().clone())
            if model_type == 'MultiLSTM':
                intervened = (concepts != original_predictions).float()
                concepts_to_intervene = torch.argmax(intervened, dim=1)
                selected_clusters = torch.tensor([concept_to_cluster[c.item()] for c in concepts_to_intervene]).to(concepts.device)
                unique_clusters = torch.unique(selected_clusters)
                out, hidden_states = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions, hidden_states, selected_clusters, unique_clusters.tolist()
                )
                concepts = out
            elif model_type == 'LSTM':
                out, hidden = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions, hidden
                )
                concepts = out
            elif model_type == 'Baseline':
                out = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions
                )
        all_inputs = torch.stack(all_inputs, dim=1)
        all_masks = torch.stack(all_masks, dim=1)
        all_original_predictions = torch.stack(all_original_predictions, dim=1)
        all_groundtruths = torch.stack(all_groundtruths, dim=1)
        if torch.min(all_inputs) < 0 or torch.max(all_inputs) > 1:
            print("Warning: All inputs have values outside the [0, 1] range.")
        return all_inputs, all_masks, all_original_predictions, all_groundtruths
    
# =========================
# Loss Computation
# =========================
def compute_loss(concept_corrector: nn.Module, concepts: torch.Tensor, groundtruth_concepts: torch.Tensor, initial_hidden, intervention_policy, max_interventions: int, criterion, concept_to_cluster: list, model_type: str, verbose: bool=False):
    all_inputs, all_masks, all_original_predictions, all_groundtruths = sample_trajectory(
        concept_corrector, concepts, groundtruth_concepts, initial_hidden, intervention_policy, max_interventions, concept_to_cluster, model_type, verbose=verbose
    )
    if model_type == 'MultiLSTM':
        hidden_states = concept_corrector.prepare_initial_hidden(all_inputs.size(0), all_inputs.device)
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, hidden_states)
    elif model_type == 'LSTM':
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, None)
    elif model_type == 'Baseline':
        out = concept_corrector.forward(all_inputs, all_masks, all_original_predictions)
    return criterion(out, all_groundtruths)