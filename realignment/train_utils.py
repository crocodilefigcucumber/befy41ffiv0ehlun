import torch
from torch import nn

from intervention_utils import intervene


# =========================
# Trajectory Sampling
# =========================

def sample_trajectory(
    concept_corrector: nn.Module,
    concepts: torch.Tensor,
    groundtruth_concepts: torch.Tensor,
    initial_hidden,
    intervention_policy,
    max_interventions: int,
    concept_to_cluster: list,
    model_type: str,
    verbose: bool=False
):
    with torch.no_grad():
        all_inputs = []
        all_masks = []
        all_original_predictions = []
        all_groundtruths = []

        # Initialize hidden states depending on the model type
        if model_type in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
            hidden_states = initial_hidden
        elif model_type in ['LSTM', 'GRU', 'RNN']:
            hidden = initial_hidden
        elif model_type == 'Baseline':
            pass  # Baseline has no hidden states
        
        # We'll track which concepts were intervened on
        already_intervened_concepts = torch.zeros_like(concepts)
        original_predictions = concepts.detach().clone()

        # Save the initial states
        all_inputs.append(concepts.detach().clone())
        all_masks.append(already_intervened_concepts.detach().clone())
        all_original_predictions.append(original_predictions)
        all_groundtruths.append(groundtruth_concepts.detach().clone())

        # First forward pass
        if model_type == 'MultiLSTM':
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden_states
            )
        elif model_type == 'LSTM':
            out, hidden = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden
            )
        elif model_type == 'MultiGRU':
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden_states
            )
        elif model_type == 'GRU':
            out, hidden = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden
            )
        elif model_type == 'MultiRNN':
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden_states
            )
        elif model_type == 'RNN':
            out, hidden = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden
            )
        elif model_type == 'Baseline':
            out = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        concepts = out

        # Perform up to 'max_interventions' sequential interventions
        for intervention_step in range(1, max_interventions + 1):
            # For Baseline, optionally print each replaced concept
            print_replacements = (model_type == 'Baseline') and verbose

            # Perform the actual intervention
            concepts, already_intervened_concepts = intervene(
                concepts,
                already_intervened_concepts,
                groundtruth_concepts,
                intervention_policy,
                verbose=print_replacements
            )

            # Log the updated states
            all_inputs.append(concepts.detach().clone())
            all_masks.append(already_intervened_concepts.detach().clone())
            all_original_predictions.append(original_predictions)
            all_groundtruths.append(groundtruth_concepts.detach().clone())

            # For multi-cluster models, we typically intervene on the cluster(s)
            # that changed, then pass only those to forward_single_timestep
            if model_type in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
                # Identify which cluster(s) changed
                intervened = (concepts != original_predictions).float()
                concepts_to_intervene = torch.argmax(intervened, dim=1)
                selected_clusters = torch.tensor(
                    [concept_to_cluster[c.item()] for c in concepts_to_intervene]
                ).to(concepts.device)
                unique_clusters = torch.unique(selected_clusters)

                if model_type == 'MultiLSTM':
                    out, hidden_states = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions,
                        hidden_states, selected_clusters, unique_clusters.tolist()
                    )
                elif model_type == 'MultiGRU':
                    out, hidden_states = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions,
                        hidden_states, selected_clusters, unique_clusters.tolist()
                    )
                elif model_type == 'MultiRNN':
                    out, hidden_states = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions,
                        hidden_states, selected_clusters, unique_clusters.tolist()
                    )
                else:
                    raise ValueError(f"Unexpected multi-cluster model type: {model_type}")

            elif model_type in ['LSTM', 'GRU', 'RNN']:
                if model_type == 'LSTM':
                    out, hidden = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions, hidden
                    )
                elif model_type == 'GRU':
                    out, hidden = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions, hidden
                    )
                elif model_type == 'RNN':
                    out, hidden = concept_corrector.forward_single_timestep(
                        concepts, already_intervened_concepts, original_predictions, hidden
                    )
                else:
                    raise ValueError(f"Unexpected single-cluster RNN type: {model_type}")

            elif model_type == 'Baseline':
                out = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Update concepts
            concepts = out

        # Stack everything for final return
        all_inputs = torch.stack(all_inputs, dim=1)
        all_masks = torch.stack(all_masks, dim=1)
        all_original_predictions = torch.stack(all_original_predictions, dim=1)
        all_groundtruths = torch.stack(all_groundtruths, dim=1)

        # Optionally warn if values drift outside [0, 1]
        if torch.min(all_inputs) < 0 or torch.max(all_inputs) > 1:
            print("Warning: Some concept values are outside the [0, 1] range.")

        return all_inputs, all_masks, all_original_predictions, all_groundtruths

    

# =========================
# Loss Computation
# =========================

def compute_loss(
    concept_corrector: nn.Module,
    concepts: torch.Tensor,
    groundtruth_concepts: torch.Tensor,
    initial_hidden,
    intervention_policy,
    max_interventions: int,
    criterion,
    concept_to_cluster: list,
    model_type: str,
    verbose: bool=False
):
    all_inputs, all_masks, all_original_predictions, all_groundtruths = sample_trajectory(
        concept_corrector, concepts, groundtruth_concepts, initial_hidden,
        intervention_policy, max_interventions, concept_to_cluster, model_type,
        verbose=verbose
    )

    #  Handle each model type explicitly
    if model_type == 'MultiLSTM':
        hidden_states = concept_corrector.prepare_initial_hidden(all_inputs.size(0), all_inputs.device)
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, hidden_states)

    elif model_type == 'LSTM':
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, None)

    elif model_type == 'Baseline':
        out = concept_corrector.forward(all_inputs, all_masks, all_original_predictions)

    elif model_type == 'GRU':
        # For single-cluster GRU, you don’t have a list of hidden states,
        # just a single hidden. So pass 'None' or an initial hidden if needed.
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, None)

    elif model_type == 'RNN':
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, None)

    elif model_type == 'MultiGRU':
        hidden_states = concept_corrector.prepare_initial_hidden(all_inputs.size(0), all_inputs.device)
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, hidden_states)

    elif model_type == 'MultiRNN':
        hidden_states = concept_corrector.prepare_initial_hidden(all_inputs.size(0), all_inputs.device)
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, hidden_states)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Finally compute the loss
    return criterion(out, all_groundtruths)
