from typing import Dict

import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import sigreg_loss, gaussianity_metrics


def lejepa_loss(view_embeddings: Tensor,
                sigreg_weight: float = 0.05,
                sigreg_slices: int = 512,
                sigreg_num_freqs: int = 33,
                sigreg_t_max: float = 8.0,
                **kwargs) -> Dict[str, Tensor]:
    """LeJEPA self-supervised loss (Balestriero & LeCun, 2025).

    Combines two terms and needs no labels, via the convex weighting
    ``loss = (1 - lambda) * prediction + lambda * SIGReg``:
      * Prediction (invariance): embeddings of different augmented views of
        the same image are pulled together (each view towards the per-image
        mean over views).
      * SIGReg: the aggregate embedding distribution is regularized towards
        an isotropic standard Gaussian via the sketched Epps-Pulley
        characteristic-function test, which provably prevents collapse.

    Args:
        view_embeddings: (V, N, D) raw (un-normalized) embeddings, where V
            is the number of augmented views and N the images per batch.
        sigreg_weight: lambda in [0, 1] balancing SIGReg vs prediction
            (paper default 0.05).
        sigreg_slices: number of random 1-D projections for SIGReg.
        sigreg_num_freqs: quadrature points for the Epps-Pulley integral.
        sigreg_t_max: half-width of the frequency integration grid.

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    if view_embeddings.dim() != 3:
        raise ValueError(
            "lejepa_loss expects (V, N, D) view embeddings, got "
            f"shape {tuple(view_embeddings.shape)}."
        )
    v, n, d = view_embeddings.shape

    # Prediction / invariance: pull each view towards the per-image mean.
    mean_emb = view_embeddings.mean(dim=0, keepdim=True)      # (1, N, D)
    pred_loss = ((view_embeddings - mean_emb) ** 2).sum(dim=-1).mean()

    # SIGReg over all views/images stacked together.
    z = view_embeddings.reshape(v * n, d)
    sr_loss = sigreg_loss(z, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

    # Convex LeJEPA weighting (Balestriero & LeCun, 2025): lambda balances
    # the isotropic-Gaussian (SIGReg) and invariance (prediction) terms.
    lam = sigreg_weight
    loss = (1.0 - lam) * pred_loss + lam * sr_loss

    with torch.no_grad():
        # Cross-view alignment: mean cosine similarity between the two most
        # separated views (view 0 vs view 1) as a collapse/quality monitor.
        z0 = F.normalize(view_embeddings[0], dim=1)
        z1 = F.normalize(view_embeddings[min(1, v - 1)], dim=1)
        view_cos = (z0 * z1).sum(dim=1).mean()
        metrics = {
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sr_loss.detach(),
            "view_cos_sim": view_cos,
            "emb_std": z.std(dim=0).mean(),
            **gaussianity_metrics(z),
        }
    return {"loss": loss, **metrics}
