import numpy as np
from scipy.special import softmax


"""
CODE TAKEN FROM arxiv:2410.07858 APPENDIX A
Copyright (authors, by listing in paper):
Emanuele Palumbo
Moritz Vandenhirtz
Alain Ryser
Imant Daunhawer
Julia E. Vogt
"""


def L2H(logits):
    """
    L2H Algorithm.
    Args:
    logits: Logits from model (N x K) where N number of datapoints in the dataset
    and K is the number of clusters
    Returns:
    steps: Merging steps characterizing the hierarchy
    """
    # Number of clusters is equal to size of last dimension in the logits
    K = logits.shape[-1]
    # Initialize groups of clusters to single clusters
    groups = [(c,) for c in range(K)]
    # Initialize list of steps that characterize hierarchy
    steps = []
    # Given the logits for the whole dataset, compute assignments and predicted probabilities
    softmaxed_logits = softmax(logits, axis=-1)
    assignments = np.argmax(softmaxed_logits, axis=-1)
    pred_probs = np.max(softmaxed_logits, axis=-1)
    for step in range(1, K):
        # Compute score for for each group (which chosen aggregation function)
        score_per_gr = {}
        for group in groups:
            score_per_gr[group] = sum(
                [np.mean(pred_probs[assignments == c]) for c in group]
            )
        # Get the group with the lowest score (lsg), will be merged at this iteration
        lsg = min(score_per_gr, key=score_per_gr.get)
        # Get the logits for datapoints assigned to the lowest score group
        logits_lsg = logits[np.where(np.isin(assignments, lsg))[0]]
        # Reassign datapoints in lsg to only clusters not in lsg,
        # and re-compute predicted probabilities
        msm_logits_lsg = np.zeros_like(logits_lsg)
        cls_not_in_lsg = [c for c in range(K) if c not in lsg]
        cls_in_lsg = [c for c in range(K) if c in lsg]
        msm_logits_lsg[:, cls_not_in_lsg] = softmax(
            logits_lsg[:, cls_not_in_lsg], axis=-1
        )
        msm_logits_lsg[:, cls_in_lsg] = 0.0
        reassignments = np.argmax(msm_logits_lsg, axis=-1)
        re_pred_probs = np.max(msm_logits_lsg, axis=-1)
        # Compute the total reassigned predicted probability per cluster and average across
        # clusters in each group.Then select the group with the highest average.
        re_pp_per_group = {
            group: np.mean([np.sum(re_pred_probs[reassignments == c]) for c in group])
            for group in groups
            if group != lsg
        }
        mtg = max(re_pp_per_group, key=re_pp_per_group.get)
        # Merge `lsg` with `mtg` and update `groups`.
        groups = [gr for gr in groups if gr not in [lsg, mtg]] + [lsg + mtg]
        # Add merging in current iteration to steps
        steps.append((lsg, mtg))
    return steps
