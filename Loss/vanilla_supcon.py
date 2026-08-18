"""Vanilla Supervised Contrastive Loss (Khosla et al., 2020).

A clean, standalone implementation of SupCon with **no** SIGReg regularizer and
**no** similarity-weighted (soft) positives (unlike ``SupConSoftPosLoss``). This
is the supervised, single-view form used by the training pipeline: a batch of
L2-normalized embeddings with integer labels, where every same-label sample is a
positive.

SupCon uses the *coupled* InfoNCE denominator (all non-self samples, i.e. both
positives and negatives) and averages the positive log-probabilities uniformly
over each anchor's positive set (the ``L_out`` formulation). For an anchor ``i``
with positive set ``P(i)`` and all non-self samples ``A(i)``::

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(s_ip / T) / sum_{a in A(i)} exp(s_ia / T) )
"""

from typing import Dict

import torch
from torch import Tensor

from .utils import batch_knn_accuracy


def vanilla_supcon_loss(embeddings: Tensor, labels: Tensor,
                        temperature: float = 0.1, **kwargs) -> Dict[str, Tensor]:
    """Supervised, single-view vanilla SupCon (``L_out`` formulation).

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        labels: (N,) integer labels.
        temperature: softmax temperature.

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    temperature = kwargs.get("temperature", temperature)
    device = embeddings.device
    n = embeddings.size(0)
    labels = labels.view(-1, 1)

    self_mask = torch.eye(n, device=device)
    pos_mask = torch.eq(labels, labels.t()).float()
    pos_mask = (pos_mask - self_mask).clamp(min=0.0)

    # Cosine-similarity logits (embeddings are already normalized).
    logits = embeddings @ embeddings.t() / temperature
    # Numerical stability: subtract per-row max before exponentiating.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Coupled denominator: all non-self samples (positives and negatives).
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0
    # Mean log-likelihood over each anchor's positives (L_out).
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1)

    if valid.any():
        loss = -mean_log_prob_pos[valid].mean()
    else:
        loss = torch.zeros((), device=device, requires_grad=True)

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(logits, labels, self_mask)
        metrics = {
            "SupCon": loss.detach(),
            "pos_fraction": valid.float().mean(),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
