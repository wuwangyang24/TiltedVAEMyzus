from typing import Dict

import numpy as np
import torch
from torch import Tensor

from .utils import sigreg_loss, batch_knn_accuracy, gaussianity_metrics

_SMALL_NUM = np.log(1e-45)


def dcl_sigreg_loss(
    embeddings: Tensor, labels: Tensor,
    sigreg_weight: float = 0.1,
    temperature: float = 0.1,
    sigreg_slices: int = 512,
    sigreg_num_freqs: int = 33,
    sigreg_t_max: float = 8.0,
    **kwargs,
) -> Dict[str, Tensor]:
    """Decoupled Contrastive Loss (Yeh et al., 2022) + SIGReg.

    Decouples the positive and negative gradient contributions so they
    don't suppress each other.  A hyperparameter ``sigreg_weight`` (lambda)
    interpolates between the DCL negative term and SIGReg:

    loss = pos + lambda * neg + (1 - lambda) * SIGReg
    """
    temperature = kwargs.get("temperature", temperature)
    device = embeddings.device
    n, d = embeddings.shape
    labels_col = labels.view(-1, 1)

    # Masks.
    self_mask = torch.eye(n, device=device)
    pos_mask = torch.eq(labels_col, labels_col.t()).float().to(device)
    pos_mask = (pos_mask - self_mask).clamp(min=0.0)
    neg_mask = 1.0 - torch.eq(labels_col, labels_col.t()).float().to(device)
    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0

    # Cosine similarity (embeddings are L2-normalized).
    sim = embeddings @ embeddings.t() / temperature

    # Positive term: mean similarity to same-label samples.
    pos_sim = (pos_mask * sim).sum(dim=1) / pos_per_anchor.clamp(min=1)
    if valid.any():
        pos_loss = -pos_sim[valid].mean()
    else:
        pos_loss = torch.zeros((), device=device, requires_grad=True)

    # Negative term (DCL): logsumexp over negatives only.
    non_neg_mask = (1.0 - neg_mask) + self_mask  # mask out positives and self
    neg_logits = sim + non_neg_mask.clamp(max=1.0) * _SMALL_NUM
    neg_loss = torch.logsumexp(neg_logits, dim=1).mean()

    # SIGReg on un-normalized embeddings for collapse prevention.
    sr_loss = sigreg_loss(embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

    lam = sigreg_weight
    loss = pos_loss + lam * neg_loss + (1.0 - lam) * sr_loss

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(sim, labels_col, self_mask)
        metrics = {
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "sigreg_loss": sr_loss.detach(),
            "pos_fraction": valid.float().mean(),
            "emb_std": embeddings.std(dim=0).mean(),
            **gaussianity_metrics(embeddings),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
