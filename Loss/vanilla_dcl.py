"""Vanilla Decoupled Contrastive Loss (Yeh et al., 2022).

A clean, standalone implementation of DCL with **no** SIGReg regularizer and
**no** false-negative suspicion re-weighting (unlike ``DCLSIGRegLoss``). This is
the supervised, single-view form used by the training pipeline: a batch of
L2-normalized embeddings with integer labels, where every same-label sample is a
positive.

DCL keeps the InfoNCE numerator/denominator structure but **decouples** the
positive from the denominator: the negative log-sum-exp is taken over the
negatives only (positives and self excluded). For an anchor ``i`` with positive
set ``P(i)`` and negatives ``N(i)``::

    L_i = -mean_{p in P(i)} s_ip / T  +  logsumexp_{j in N(i)} s_ij / T

which reduces to the original two-view DCL when each anchor has a single
positive.
"""

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from .utils import batch_knn_accuracy

# log(0) surrogate used to mask out excluded logits before log-sum-exp.
_SMALL_NUM = np.log(1e-45)


def vanilla_dcl_loss(embeddings: Tensor, labels: Tensor,
                     temperature: float = 0.1, **kwargs) -> Dict[str, Tensor]:
    """Supervised, single-view vanilla DCL.

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
    # Decoupling: the denominator excludes positives *and* the self-pair.
    non_neg_mask = pos_mask + self_mask

    # Cosine-similarity logits (embeddings are already normalized).
    sim = embeddings @ embeddings.t() / temperature

    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0

    # Positive term: -mean similarity to same-label samples.
    pos_term = -(pos_mask * sim).sum(dim=1) / pos_per_anchor.clamp(min=1)
    # Decoupled negative term: log-sum-exp over negatives only.
    neg_term = torch.logsumexp(sim + non_neg_mask * _SMALL_NUM, dim=1)

    per_anchor = pos_term + neg_term
    if valid.any():
        loss = per_anchor[valid].mean()
    else:
        loss = torch.zeros((), device=device, requires_grad=True)

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(sim, labels, self_mask)
        metrics = {
            "DCL": loss.detach(),
            "pos_loss": pos_term[valid].mean().detach() if valid.any()
            else torch.zeros((), device=device),
            "neg_loss": neg_term.mean().detach(),
            "pos_fraction": valid.float().mean(),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
