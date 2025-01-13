import torch
from torch import nn


def realign_concepts(
    concept_corrector: nn.Module,
    concept_vector: torch.Tensor,
    device: torch.device,
    config: dict,
    concept_to_cluster: list = None
) -> torch.Tensor:
    """
    Given a trained concept corrector model (LSTM, GRU, RNN, or multi-cluster variant)
    OR the BaselineConceptCorrector (not actually trained),
    and an initial predicted concept vector, return the realigned concept vector.
    
    Args:
        concept_corrector (nn.Module): One of the following:
            - BaselineConceptCorrector
            - LSTMConceptCorrector, MultiLSTMConceptCorrector
            - GRUConceptCorrector, MultiGRUConceptCorrector
            - RNNConceptCorrector, MultiRNNConceptCorrector
        concept_vector (torch.Tensor): The initial predicted concept vector(s).
            Shape can be (batch_size, num_concepts).
        device (torch.device): The device on which computations will be done (CPU or CUDA).
        config (dict): Contains 'model' and possibly 'input_format' or other config options.
        concept_to_cluster (list, optional): Only needed for multi-cluster models.

    Returns:
        realigned_vector (torch.Tensor): The realigned concept vector(s),
            of the same shape as `concept_vector`.
    """
    model_type = config['model']

    # Make sure concept_vector is on the correct device
    concept_vector = concept_vector.to(device)

    # 1) Determine hidden states if needed
    batch_size = concept_vector.size(0)  # could be 1 or more

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

    # 2) Prepare placeholder for 'already_intervened_concepts' and 'original_predictions'
    #    We can assume no interventions yet: everything is 0 in 'already_intervened_concepts'.
    already_intervened = torch.zeros_like(concept_vector).to(device)    # shape: (batch_size, num_concepts)
    original_predictions = concept_vector.clone().detach().to(device)  # same shape

    # 3) Call forward_single_timestep or forward to get the realigned concepts
    if model_type in ['MultiLSTM', 'MultiGRU', 'MultiRNN']:
        # multi-cluster variants
        out, hidden_states = concept_corrector.forward_single_timestep(
            concept_vector,                  # inputs
            already_intervened,             # already_intervened_concepts
            original_predictions,           # original_predictions
            hidden_states,                  # hidden states list
            selected_clusters=None,         # not doing any selective intervention
            selected_cluster_ids=None
        )
    elif model_type in ['LSTM', 'GRU', 'RNN']:
        # single-cluster variants
        out, hidden = concept_corrector.forward_single_timestep(
            concept_vector,
            already_intervened,
            original_predictions,
            hidden
        )
    elif model_type == 'Baseline':
        # Baseline doesn't do realignment; it only merges concept_vector with "original_predictions"
        # using the mask. Because there's no realignment, the model's forward just returns:
        #   output = already_intervened * concept_vector + (1 - already_intervened) * original_predictions
        out = concept_corrector.forward_single_timestep(
            concept_vector,
            already_intervened,
            original_predictions
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return out
