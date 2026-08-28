"""InfoNCE (SupCon) with similarity-weighted positives.

Same supervised InfoNCE / SupCon objective as :func:`infonce_loss`, but instead
of averaging each anchor's positives uniformly, the positive pairs are
re-weighted by a softmax over their cosine similarity (the exact scheme used by
:class:`DCLSoftPosLoss`): already-close positives get more weight, dissimilar
positives less, which encourages tighter within-class sub-clusters.

Unlike DCL, the InfoNCE denominator stays *coupled* — it includes both
positives and negatives (all non-self samples).
"""

from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_knn_accuracy
from .dcl_soft_pos import sinkhorn_normalize

_SMALL_NUM = np.log(1e-45)


def infonce_softpos_loss(embeddings: Tensor, labels: Tensor,
                         temperature: float = 0.1,
                         pos_weight_tau: float = 0.1,
                         use_pos_weighting: bool = True,
                         denom_pos_weight: bool = False,
                         sinkhorn: bool = False,
                         sinkhorn_iters: int = 5,
                         pos_weight_sim: Tensor = None,
                         test_labels: Tensor = None,
                         **kwargs) -> Dict[str, Tensor]:
    """Supervised InfoNCE / SupCon with soft (similarity-weighted) positives.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        labels: (N,) integer labels.
        temperature: softmax temperature of the contrastive logits.
        pos_weight_tau: temperature of the softmax that turns positive-pair
            cosine similarities into weights (lower = sharper, more weight on
            the closest positives).
        use_pos_weighting: if False, weight all positives uniformly.
        denom_pos_weight: if True, also re-weight the positive terms inside the
            denominator by the soft positive weights (negatives stay at 1).
        sinkhorn: if True, use Sinkhorn-Knopp iterations for a doubly-stochastic
            positive-weight matrix instead of per-row softmax.
        sinkhorn_iters: number of Sinkhorn-Knopp iterations.
        test_labels: optional (N,) evaluation (``test_cat``) labels, used only
            for monitoring whether positive-pair weights align with test_cat.

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    temperature = kwargs.get("temperature", temperature)
    device = embeddings.device
    n = embeddings.size(0)
    labels = labels.view(-1, 1)

    self_mask = torch.eye(n, device=device)
    pos_mask = torch.eq(labels, labels.t()).float()
    pos_mask = (pos_mask - self_mask).clamp(min=0.0)
    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0

    # Cosine-similarity logits (embeddings are already normalized).
    logits = embeddings @ embeddings.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Soft positive weights from cosine similarity (same scheme as DCLSoftPos).
    # When ``pos_weight_sim`` is supplied (e.g. cosine similarities from an EMA
    # "teacher" copy of the model), it drives the positive weights instead of
    # the online embeddings' own similarities.
    raw_cos = embeddings @ embeddings.t() if pos_weight_sim is None else pos_weight_sim
    if not use_pos_weighting:
        pos_weights = pos_mask / pos_per_anchor.clamp(min=1).unsqueeze(1)
    else:
        pos_weight_logits = raw_cos / pos_weight_tau
        if sinkhorn:
            pos_weights = pos_weight_logits.exp() * pos_mask
            pos_weights = sinkhorn_normalize(pos_weights, sinkhorn_iters)
            pos_weights = pos_weights * pos_mask
        else:
            pos_weight_logits = pos_weight_logits + (1.0 - pos_mask) * _SMALL_NUM
            pos_weights = F.softmax(pos_weight_logits, dim=1) * pos_mask

    # SupCon denominator over all non-self samples.  With ``denom_pos_weight``
    # the positive terms inside the denominator are re-weighted by the same soft
    # positive weights (negatives stay at weight 1).
    if denom_pos_weight:
        neg_mask = (1.0 - self_mask - pos_mask).clamp(min=0.0)
        denom_coeff = neg_mask + pos_weights
    else:
        denom_coeff = 1.0 - self_mask
    exp_logits = torch.exp(logits) * denom_coeff
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Weighted log-likelihood over each anchor's positives.
    weighted_log_prob_pos = (pos_weights * log_prob).sum(dim=1)
    if valid.any():
        loss = -weighted_log_prob_pos[valid].mean()
    else:
        loss = torch.zeros((), device=device, requires_grad=True)

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(logits, labels, self_mask)
        pw_ent = -(pos_weights * (pos_weights + 1e-12).log()).sum(dim=1)
        pw_ent = pw_ent[valid].mean() if valid.any() else torch.zeros((), device=device)
        metrics = {
            "InfoNCE_SoftPos": loss.detach(),
            "pos_weight_entropy": pw_ent,
            "pos_fraction": valid.float().mean(),
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
