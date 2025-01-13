import torch
from torch import nn

from intervention_utils import intervene

def realign_concepts(
    concept_corrector: nn.Module,
    concept_vector: torch.Tensor,
    groundtruth_concepts: torch.Tensor,
    device: torch.device,
    config: dict,
    concept_to_cluster: list,
    intervention_policy,  # e.g. ucp(...)
    verbose: bool = False
) -> torch.Tensor:
    """
    Perform multi-step realignment on 'concept_vector' at inference time,
    replicating the same iterative "forward -> intervene -> forward -> intervene"
    logic used in training (sample_trajectory).

    Args:
        concept_corrector (nn.Module):
            - BaselineConceptCorrector
            - LSTMConceptCorrector / MultiLSTMConceptCorrector
            - GRUConceptCorrector / MultiGRUConceptCorrector
            - RNNConceptCorrector / MultiRNNConceptCorrector
          Must be already trained or loaded from a checkpoint.

        concept_vector (torch.Tensor): Initial predicted concept vector(s).
            Shape: (batch_size, num_concepts).

        groundtruth_concepts (torch.Tensor): Ground-truth concepts for these samples.
            Shape: (batch_size, num_concepts).

        device (torch.device): CPU or CUDA device.

        config (dict): Should contain:
            - 'model' (str): which model type we have
            - 'max_interventions' (int): how many forward->intervene steps to run
            - possibly other keys like 'verbose' or 'input_format'.

        concept_to_cluster (list): For multi-cluster models, cluster assignment for each concept.
            Single-cluster or Baseline models can ignore this.

        intervention_policy (Callable): a function like ucp(...) or random_intervention_policy(...)
            that chooses which concept(s) to intervene on at each step.

        verbose (bool): if True, we print out which concepts get replaced each step
            (just as your code does in 'intervene(..., verbose=True)').

    Returns:
        torch.Tensor: final realigned concept vectors, shape (batch_size, num_concepts),
                      after up to 'max_interventions' steps of partial updates.
    """
    model_type = config['model']
    max_interventions = config['max_interventions']

    # Move input Tensors to device
    concept_vector = concept_vector.to(device)
    groundtruth_concepts = groundtruth_concepts.to(device)

    # Prepare hidden states if needed
    batch_size = concept_vector.size(0)

    if model_type in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
        hidden_states = concept_corrector.prepare_initial_hidden(batch_size, device)
        hidden = None
    elif model_type in ['LSTM', 'GRU', 'RNN']:
        hidden = concept_corrector.prepare_initial_hidden(batch_size, device)
        hidden_states = None
    elif model_type == 'Baseline':
        hidden = None
        hidden_states = None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # track which concepts were intervened on so far
    already_intervened_concepts = torch.zeros_like(concept_vector).to(device)

    # original_predictions is unmodified CBM output
    # that the corrector can overwrite partially each step
    original_predictions = concept_vector.clone().detach()

    # current concept vector we keep updating
    concepts = concept_vector.clone().detach().to(device)

    # Perform up to max_interventions iteration
    for step in range(max_interventions):

        # 1) Forward single timestep
        if model_type in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts,
                already_intervened_concepts,
                original_predictions,
                hidden_states
            )
        elif model_type in ['LSTM', 'GRU', 'RNN']:
            out, hidden = concept_corrector.forward_single_timestep(
                concepts,
                already_intervened_concepts,
                original_predictions,
                hidden
            )
        else:  # Baseline
            out = concept_corrector.forward_single_timestep(
                concepts,
                already_intervened_concepts,
                original_predictions
            )

        # Update concepts with corrector's realigned output
        concepts = out

        # 2) Intervene on whichever concept(s) policy says to fix (just like in sample_trajectory)
        concepts, already_intervened_concepts = intervene(
            concepts,                        # current concept predictions
            already_intervened_concepts,     # which are already replaced
            groundtruth_concepts,            # ground truth concepts
            intervention_policy,             # e.g. ucp
            verbose=verbose
        )

    # After done up to max_interventions forward -> intervene cycles,
    # concepts is the final realigned concept vector for each sample.
    return concepts

