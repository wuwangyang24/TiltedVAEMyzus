"""Multi-Similarity (MS) loss (Wang et al., CVPR 2019).

Port of the official implementation
(https://github.com/msight-tech/research-ms-loss), vectorized over the batch.

For every anchor ``i`` the pairs are first *mined*: a negative is kept when its
similarity exceeds the anchor's hardest positive minus ``margin``, and a
positive is kept when its similarity falls below the anchor's hardest negative
plus ``margin``. The kept pairs are then weighted through the soft-max style
terms::

    L_i = 1/alpha * log(1 + sum_{p} exp(-alpha * (s_ip - lambda)))
        + 1/beta  * log(1 + sum_{n} exp( beta  * (s_in - lambda)))

with ``alpha = scale_pos``, ``beta = scale_neg`` and ``lambda = thresh``. As in
the reference code the per-anchor losses are summed and divided by the batch
size (anchors without mined pairs contribute zero).
"""

from typing import Dict

import torch
from torch import Tensor

from .utils import batch_knn_accuracy


def multi_similarity_loss(embeddings: Tensor, labels: Tensor,
                          thresh: float = 0.5, margin: float = 0.1,
                          scale_pos: float = 2.0, scale_neg: float = 40.0,
                          **kwargs) -> Dict[str, Tensor]:
    """Multi-Similarity loss on L2-normalized embeddings.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        labels: (N,) integer labels.
        thresh: similarity offset ``lambda``.
        margin: mining margin ``epsilon``.
        scale_pos: positive scale ``alpha``.
        scale_neg: negative scale ``beta``.

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    thresh = kwargs.get("thresh", thresh)
    margin = kwargs.get("margin", margin)
    scale_pos = kwargs.get("scale_pos", scale_pos)
    scale_neg = kwargs.get("scale_neg", scale_neg)

    device = embeddings.device
    n = embeddings.size(0)
    labels = labels.view(-1, 1)

    sim_mat = embeddings @ embeddings.t()
    self_mask = torch.eye(n, device=device, dtype=torch.bool)
    same_label = torch.eq(labels, labels.t())

    # The reference code drops pairs with similarity ~1 (this removes the self
    # pair and exact duplicates).
    pos_mask = same_label & ~self_mask & (sim_mat < 1.0 - 1e-5)
    neg_mask = ~same_label

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)

    # Hardest positive / negative per anchor (used as the mining thresholds).
    min_pos = sim_mat.masked_fill(~pos_mask, float("inf")).min(dim=1).values
    max_neg = sim_mat.masked_fill(~neg_mask, float("-inf")).max(dim=1).values

    mined_neg = neg_mask & (sim_mat + margin > min_pos.unsqueeze(1))
    mined_pos = pos_mask & (sim_mat - margin < max_neg.unsqueeze(1))

    valid = has_pos & has_neg & mined_pos.any(dim=1) & mined_neg.any(dim=1)

    pos_terms = torch.exp(-scale_pos * (sim_mat - thresh)) * mined_pos
    neg_terms = torch.exp(scale_neg * (sim_mat - thresh)) * mined_neg

    pos_loss = torch.log1p(pos_terms.sum(dim=1)) / scale_pos
    neg_loss = torch.log1p(neg_terms.sum(dim=1)) / scale_neg

    per_anchor = (pos_loss + neg_loss) * valid
    loss = per_anchor.sum() / n

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(sim_mat, labels, self_mask.float())
        metrics = {
            "MS": loss.detach(),
            "pos_loss": (pos_loss * valid).sum().detach() / max(int(valid.sum()), 1),
            "neg_loss": (neg_loss * valid).sum().detach() / max(int(valid.sum()), 1),
            "pos_fraction": valid.float().mean(),
            "mined_pos_per_anchor": mined_pos.float().sum(dim=1).mean(),
            "mined_neg_per_anchor": mined_neg.float().sum(dim=1).mean(),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
