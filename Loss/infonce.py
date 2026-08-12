from typing import Dict

import torch
from torch import Tensor

from .utils import batch_knn_accuracy


def infonce_loss(embeddings: Tensor, labels: Tensor,
                 temperature: float = 0.1, **kwargs) -> Dict[str, Tensor]:
    """Supervised contrastive (SupCon / InfoNCE) loss.

    Pulls embeddings sharing a label together and pushes embeddings from
    different labels apart. All same-label samples in the batch act as
    positives for a given anchor.

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

    # Positive mask: same label, excluding the diagonal (self-pairs).
    pos_mask = torch.eq(labels, labels.t()).float().to(device)
    self_mask = torch.eye(n, device=device)
    pos_mask = pos_mask - self_mask
    pos_mask = pos_mask.clamp(min=0.0)

    # Cosine-similarity logits (embeddings are already normalized).
    logits = embeddings @ embeddings.t() / temperature
    # Numerical stability: subtract per-row max before exponentiating.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Denominator excludes the self-pair.
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0
    # Mean log-likelihood over each anchor's positives.
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1)

    if valid.any():
        loss = -mean_log_prob_pos[valid].mean()
    else:
        loss = torch.zeros((), device=device, requires_grad=True)

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(logits, labels, self_mask)
        metrics = {
            "InfoNCE": loss.detach(),
            "pos_fraction": valid.float().mean(),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
