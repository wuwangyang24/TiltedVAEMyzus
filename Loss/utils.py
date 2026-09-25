from typing import Dict

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def gaussianity_metrics(z: Tensor) -> Dict[str, Tensor]:
    """Diagnostics of how close the embedding batch is to an isotropic
    standard Gaussian N(0, I). All are monitoring-only (no gradient).

    Ideal values for a true N(0, I) sample:
      * ``emb_mean_abs``   -> 0   (zero-centered)
      * ``emb_std``        -> 1   (unit per-dimension variance; see caller)
      * ``emb_norm_ratio`` -> 1   (E[||z||^2] / D equals 1)
    """
    m, d = z.shape
    emb_mean_abs = z.mean(dim=0).abs().mean()
    emb_norm_ratio = (z.pow(2).sum(dim=1).mean() / d)

    return {
        "emb_mean_abs": emb_mean_abs,
        "emb_norm_ratio": emb_norm_ratio,
    }


def sinkhorn_normalize(M: Tensor, n_iters: int = 5) -> Tensor:
    """Sinkhorn-Knopp iterations to produce a doubly-stochastic matrix."""
    M = M.clamp(min=1e-12)
    for _ in range(n_iters):
        M = M / M.sum(dim=1, keepdim=True).clamp(min=1e-12)
        M = M / M.sum(dim=0, keepdim=True).clamp(min=1e-12)
    return M


@torch.no_grad()
def batch_knn_accuracy(logits: Tensor, labels: Tensor,
                       self_mask: Tensor) -> dict:
    """Top-1/3/5 fraction of anchors whose nearest neighbours share their label."""
    masked = logits.masked_fill(self_mask.bool(), float("-inf"))
    labels = labels.view(-1)
    n = labels.size(0)
    result = {}
    for k, suffix in ((1, "batch_knn_acc"), (3, "batch_knn_top3_acc"), (5, "batch_knn_top5_acc")):
        if k >= n:
            result[suffix] = torch.tensor(1.0, device=logits.device)
            continue
        topk_idx = masked.topk(k, dim=1).indices
        hits = (labels[topk_idx] == labels.unsqueeze(1)).any(dim=1)
        result[suffix] = hits.float().mean()
    return result
