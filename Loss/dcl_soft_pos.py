from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_knn_accuracy, gaussianity_metrics

_SMALL_NUM = np.log(1e-45)


def sinkhorn_normalize(M: Tensor, n_iters: int = 5) -> Tensor:
    """Sinkhorn-Knopp iterations to produce a doubly-stochastic matrix."""
    M = M.clamp(min=1e-12)
    for _ in range(n_iters):
        M = M / M.sum(dim=1, keepdim=True).clamp(min=1e-12)
        M = M / M.sum(dim=0, keepdim=True).clamp(min=1e-12)
    return M


class DCLSoftPosLoss(nn.Module):
    """DCL with similarity-weighted positives.

    Standard DCL averages the positive-pair similarities uniformly.  This
    variant re-weights positive pairs via a softmax over their cosine
    similarities: pairs that are already close receive larger weight, while
    dissimilar positives are down-weighted.  The effect is to encourage
    tighter, better-separated sub-clusters within each class.

    The negative term is plain DCL (all negatives weighted equally).

    Args:
        pos_weight_tau: temperature for the softmax that turns positive-pair
            similarities into weights.  Lower values sharpen the distribution
            (more weight on the closest positives).
        sinkhorn: if True, replace per-row softmax with Sinkhorn-Knopp
            iterations to produce a doubly-stochastic weight matrix.
        sinkhorn_iters: number of Sinkhorn-Knopp iterations.
    """

    def __init__(self, pos_weight_tau: float = 0.1,
                 sinkhorn: bool = False,
                 sinkhorn_iters: int = 5) -> None:
        super().__init__()
        self.pos_weight_tau = pos_weight_tau
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, embeddings: Tensor, labels: Tensor,
                temperature: float = 0.1,
                test_labels: Optional[Tensor] = None,
                **kwargs) -> Dict[str, Tensor]:
        temperature = kwargs.get("temperature", temperature)
        device = embeddings.device
        n, d = embeddings.shape
        labels_flat = labels.view(-1).long()

        normed = F.normalize(embeddings, dim=1)

        # Masks.
        labels_col = labels_flat.view(-1, 1)
        self_mask = torch.eye(n, device=device)
        pos_mask = torch.eq(labels_col, labels_col.t()).float()
        pos_mask = (pos_mask - self_mask).clamp(min=0.0)
        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        # Cosine similarity scaled by temperature.
        sim = normed @ normed.t() / temperature

        # --- Soft-weighted positive term ---
        raw_cos = normed @ normed.t()
        pos_weight_logits = raw_cos / self.pos_weight_tau
        if self.sinkhorn:
            pos_weights = (pos_weight_logits.exp()) * pos_mask
            pos_weights = sinkhorn_normalize(pos_weights, self.sinkhorn_iters)
            pos_weights = pos_weights * pos_mask
        else:
            pos_weight_logits = pos_weight_logits + (1.0 - pos_mask) * _SMALL_NUM
            pos_weights = F.softmax(pos_weight_logits, dim=1) * pos_mask

        weighted_pos_sim = (pos_weights * sim).sum(dim=1)
        if valid.any():
            pos_loss = -weighted_pos_sim[valid].mean()
        else:
            pos_loss = torch.zeros((), device=device, requires_grad=True)

        # --- Plain DCL negative term ---
        non_neg_mask = pos_mask + self_mask
        neg_logits = sim + non_neg_mask * _SMALL_NUM
        neg_loss = torch.logsumexp(neg_logits, dim=1).mean()

        loss = pos_loss + neg_loss

        with torch.no_grad():
            knn_accs = batch_knn_accuracy(sim, labels_col, self_mask)
            pw_ent = -(pos_weights * (pos_weights + 1e-12).log()).sum(dim=1)
            pw_ent = pw_ent[valid].mean() if valid.any() else torch.zeros((), device=device)
            metrics = {
                "pos_loss": pos_loss.detach(),
                "neg_loss": neg_loss.detach(),
                "pos_weight_entropy": pw_ent,
                "pos_fraction": valid.float().mean(),
                "emb_std": embeddings.std(dim=0).mean(),
                **gaussianity_metrics(embeddings),
                **knn_accs,
            }

        return {"loss": loss, **metrics}
