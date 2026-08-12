from typing import Dict

import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import sigreg_loss, batch_knn_accuracy, gaussianity_metrics


def contrastive_sigreg_loss(
    embeddings: Tensor, labels: Tensor,
    sigreg_weight: float = 0.1,
    sigreg_slices: int = 512,
    sigreg_num_freqs: int = 33,
    sigreg_t_max: float = 8.0,
    **kwargs,
) -> Dict[str, Tensor]:
    """Supervised contrastive loss with SIGReg replacing negatives.

    Positive attraction: for each anchor, minimize mean squared distance to
    same-label embeddings. Collapse prevention: SIGReg pushes the embedding
    distribution toward an isotropic Gaussian instead of using negatives.

    loss = (1 - lambda) * pos_attraction + lambda * SIGReg
    """
    device = embeddings.device
    n, d = embeddings.shape
    labels_col = labels.view(-1, 1)

    # Positive mask: same label, excluding self.
    pos_mask = torch.eq(labels_col, labels_col.t()).float().to(device)
    pos_mask = (pos_mask - torch.eye(n, device=device)).clamp(min=0.0)
    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0

    # Positive attraction via cosine similarity on normalized embeddings.
    normed = F.normalize(embeddings, dim=1)
    sim = normed @ normed.t()
    mean_pos_sim = (pos_mask * sim).sum(dim=1) / pos_per_anchor.clamp(min=1)
    if valid.any():
        pos_loss = 1.0 - mean_pos_sim[valid].mean()
    else:
        pos_loss = torch.zeros((), device=device, requires_grad=True)

    # SIGReg on un-normalized embeddings for collapse prevention.
    sr_loss = sigreg_loss(embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

    lam = sigreg_weight
    loss = (1.0 - lam) * pos_loss + lam * sr_loss

    with torch.no_grad():
        self_mask = torch.eye(n, device=device)
        knn_accs = batch_knn_accuracy(sim, labels_col, self_mask)
        metrics = {
            "pos_loss": pos_loss.detach(),
            "sigreg_loss": sr_loss.detach(),
            "pos_fraction": valid.float().mean(),
            "emb_std": embeddings.std(dim=0).mean(),
            **gaussianity_metrics(embeddings),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
