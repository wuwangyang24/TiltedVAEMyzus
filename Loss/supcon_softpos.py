from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .utils import sigreg_loss, batch_knn_accuracy, gaussianity_metrics
from .dcl_soft_pos import sinkhorn_normalize

_SMALL_NUM = np.log(1e-45)


class SupConSoftPosLoss(nn.Module):
    """Supervised contrastive (SupCon) loss with similarity-weighted positives.

    Standard SupCon averages each anchor's positive-pair log-probabilities
    uniformly.  This variant re-weights positive pairs via a softmax over their
    cosine similarities (the exact scheme used by :class:`DCLSoftPosLoss`):
    pairs that are already close receive larger weight, while dissimilar
    positives are down-weighted, encouraging tighter within-class sub-clusters.

    Unlike DCL, the SupCon denominator stays *coupled* — it includes both
    positives and negatives (all non-self samples).  As in
    :class:`DCLSoftPosLoss`, a SIGReg regularizer on the batch embeddings is
    added to prevent collapse.

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
        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        # Cosine-similarity logits scaled by temperature.
        logits = normed @ normed.t() / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # --- Soft positive weights (same scheme as DCLSoftPos) ---
        raw_cos = normed @ normed.t()
        pos_weight_logits = raw_cos / self.pos_weight_tau
        if self.sinkhorn:
            pos_weights = pos_weight_logits.exp() * pos_mask
            pos_weights = sinkhorn_normalize(pos_weights, self.sinkhorn_iters)
            pos_weights = pos_weights * pos_mask
        else:
            pos_weight_logits = pos_weight_logits + (1.0 - pos_mask) * _SMALL_NUM
            pos_weights = F.softmax(pos_weight_logits, dim=1) * pos_mask

        # SupCon (coupled) denominator: negatives at weight 1, positive terms
        # re-weighted by the same soft positive weights.
        neg_mask = (1.0 - self_mask - pos_mask).clamp(min=0.0)
        denom_coeff = neg_mask + pos_weights
        exp_logits = torch.exp(logits) * denom_coeff
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted log-likelihood over each anchor's positives.
        weighted_log_prob_pos = (pos_weights * log_prob).sum(dim=1)
        if valid.any():
            supcon_loss = -weighted_log_prob_pos[valid].mean()
        else:
            supcon_loss = torch.zeros((), device=device, requires_grad=True)

        # SIGReg on batch embeddings.
        sr_loss = sigreg_loss(
            embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

        loss = supcon_loss + sigreg_weight * sr_loss

        with torch.no_grad():
            knn_accs = batch_knn_accuracy(logits, labels_col, self_mask)
            pw_ent = -(pos_weights * (pos_weights + 1e-12).log()).sum(dim=1)
            pw_ent = pw_ent[valid].mean() if valid.any() else torch.zeros((), device=device)
            metrics = {
                "supcon_loss": supcon_loss.detach(),
                "sigreg_loss": sr_loss.detach(),
                "pos_weight_entropy": pw_ent,
                "pos_fraction": valid.float().mean(),
                "emb_std": embeddings.std(dim=0).mean(),
                **gaussianity_metrics(embeddings),
                **knn_accs,
            }

            # Alignment of positive-pair weights with the test_cat taxonomy:
            # do same-test_cat positives receive more weight than diff-test_cat ones?
            if test_labels is not None:
                tl = test_labels.view(-1, 1)
                same_test_pos = pos_mask * torch.eq(tl, tl.t()).float()
                diff_test_pos = pos_mask * torch.ne(tl, tl.t()).float()
                w_same = (pos_weights * same_test_pos).sum() / same_test_pos.sum().clamp(min=1)
                w_diff = (pos_weights * diff_test_pos).sum() / diff_test_pos.sum().clamp(min=1)
                metrics["pos_weight_same_testcat"] = w_same
                metrics["pos_weight_diff_testcat"] = w_diff
                metrics["pos_weight_testcat_ratio"] = w_same / w_diff.clamp(min=1e-12)

        return {"loss": loss, **metrics}
