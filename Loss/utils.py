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


def sigreg_loss(z: Tensor, num_slices: int = 512, num_freqs: int = 33,
                t_max: float = 8.0) -> Tensor:
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Projects the embeddings onto ``num_slices`` random directions drawn
    uniformly on the unit sphere and, for each 1-D projection, measures its
    deviation from a standard normal N(0, 1) with the Epps-Pulley
    empirical-characteristic-function goodness-of-fit statistic. Averaged
    over slices this is a differentiable, unbiased estimate of the distance
    between the embedding distribution and an isotropic Gaussian.
    """
    m, d = z.shape
    device, dtype = z.device, z.dtype

    # Random projection directions, uniform on the unit sphere.
    dirs = torch.randn(d, num_slices, device=device, dtype=dtype)
    dirs = F.normalize(dirs, dim=0)
    proj = z @ dirs                                   # (M, num_slices)

    # Frequency grid and Gaussian weighting w(t) = exp(-t^2 / 2).
    t = torch.linspace(-t_max, t_max, num_freqs, device=device, dtype=dtype)
    weight = torch.exp(-0.5 * t ** 2)                 # (F,)

    # Empirical characteristic function per slice: E_j[exp(i t x_j)].
    tp = t.view(1, -1, 1) * proj.t().unsqueeze(1)     # (num_slices, F, M)
    emp_re = torch.cos(tp).mean(dim=2)                # (num_slices, F)
    emp_im = torch.sin(tp).mean(dim=2)
    tgt_re = torch.exp(-0.5 * t ** 2)                 # N(0,1) CF (imag = 0)

    diff2 = (emp_re - tgt_re) ** 2 + emp_im ** 2      # (num_slices, F)
    dt = t[1] - t[0]
    # Epps-Pulley statistic includes the sample-size factor N (= M here),
    # which sets its magnitude relative to the prediction term.
    stat = m * (diff2 * weight).sum(dim=1) * dt       # (num_slices,)
    return stat.mean()


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
