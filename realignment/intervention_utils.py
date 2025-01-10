import torch


# =========================
# Intervention Policy
# =========================
def ucp(concepts: torch.Tensor, already_intervened_concepts: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    importances = 1.0 / (torch.abs(concepts - 0.5) + eps)
    importances[already_intervened_concepts == 1] = -1e10
    return importances

# =========================
# Intervention Function
# =========================
def intervene(concepts: torch.Tensor, already_intervened_concepts: torch.Tensor, groundtruth_concepts: torch.Tensor, intervention_policy, return_selected_concepts: bool=False, verbose: bool=False):
    importances = intervention_policy(concepts, already_intervened_concepts)
    concepts_to_intervene = torch.argmax(importances, dim=1)
    
    if verbose:
        # Display which concepts are being replaced for each sample in the batch
        batch_size = concepts.size(0)
        for i in range(batch_size):
            concept_idx = concepts_to_intervene[i].item()
            original_value = concepts[i, concept_idx].item()
            groundtruth_value = groundtruth_concepts[i, concept_idx].item()
            print(f"Sample {i}: Replacing concept {concept_idx} value {original_value:.4f} with ground truth {groundtruth_value:.4f}")

    concepts[range(concepts.size(0)), concepts_to_intervene] = groundtruth_concepts[range(concepts.size(0)), concepts_to_intervene]
    already_intervened_concepts[range(concepts.size(0)), concepts_to_intervene] = 1
    if not return_selected_concepts:
        return concepts, already_intervened_concepts
    else:
        return concepts, already_intervened_concepts, concepts_to_intervene