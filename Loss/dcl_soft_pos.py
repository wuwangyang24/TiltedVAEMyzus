from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .utils import sigreg_loss, batch_knn_accuracy, gaussianity_metrics

_SMALL_NUM = np.log(1e-45)


class DCLSoftPosLoss(nn.Module):
    """DCL + SIGReg with similarity-weighted positives.

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
    """

    def __init__(self, pos_weight_tau: float = 0.1) -> None:
        super().__init__()
        self.pos_weight_tau = pos_weight_tau

    def forward(self, embeddings: Tensor, labels: Tensor,
                sigreg_weight: float = 0.1,
                temperature: float = 0.1,
                sigreg_slices: int = 512,
                sigreg_num_freqs: int = 33,
                sigreg_t_max: float = 8.0,
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
        neg_mask = 1.0 - torch.eq(labels_col, labels_col.t()).float()
        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        # Cosine similarity scaled by temperature.
        sim = normed @ normed.t() / temperature

        # --- Soft-weighted positive term ---
        raw_cos = normed @ normed.t()
        pos_weight_logits = raw_cos / self.pos_weight_tau
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

        # SIGReg on batch embeddings.
        sr_loss = sigreg_loss(
            embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

        loss = pos_loss + neg_loss + sigreg_weight * sr_loss

        with torch.no_grad():
            knn_accs = batch_knn_accuracy(sim, labels_col, self_mask)
            pw_ent = -(pos_weights * (pos_weights + 1e-12).log()).sum(dim=1)
            pw_ent = pw_ent[valid].mean() if valid.any() else torch.zeros((), device=device)
            metrics = {
                "pos_loss": pos_loss.detach(),
                "neg_loss": neg_loss.detach(),
                "sigreg_loss": sr_loss.detach(),
                "pos_weight_entropy": pw_ent,
                "pos_fraction": valid.float().mean(),
                "emb_std": embeddings.std(dim=0).mean(),
                **gaussianity_metrics(embeddings),
                **knn_accs,
            }

        return {"loss": loss, **metrics}
