"""Grafit loss (Touvron et al., 2020).

*Grafit: Learning fine-grained image representations with coarse labels*
combines two similarity-based cross-entropy terms that differ only by their
positive set::

    L(P) = -log( sum_{p in P(i)} exp(s_ip / T) / sum_{a in A(i)} exp(s_ia / T) )

where ``A(i)`` is every non-self sample of the batch and ``s_ij`` is the cosine
similarity between L2-normalized embeddings. The *instance* term uses the other
views of the same image as positives, keeping fine-grained (intra-class)
information that the coarse labels cannot express; the *coarse* term uses all
samples sharing the anchor's coarse label. The two are mixed by ``lam``::

    L = lam * L_inst + (1 - lam) * L_coarse

The reference implementation draws the negatives from a momentum memory bank;
here the batch itself plays that role (as for the other losses in this folder).
"""

from typing import Dict, Optional

import torch
from torch import Tensor

from .utils import batch_knn_accuracy


def _softmax_term(logits: Tensor, pos_mask: Tensor,
                  denom_mask: Tensor) -> tuple:
    """``-log(sum_pos exp / sum_all exp)`` averaged over anchors with positives."""
    exp_logits = torch.exp(logits)
    pos_sum = (exp_logits * pos_mask).sum(dim=1)
    all_sum = (exp_logits * denom_mask).sum(dim=1)
    valid = pos_mask.sum(dim=1) > 0
    per_anchor = -torch.log((pos_sum + 1e-12) / (all_sum + 1e-12))
    if valid.any():
        loss = per_anchor[valid].mean()
    else:
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return loss, valid


def grafit_loss(embeddings: Tensor, labels: Tensor,
                lam: float = 0.5, temperature: float = 0.1,
                instance_ids: Optional[Tensor] = None,
                **kwargs) -> Dict[str, Tensor]:
    """Grafit joint instance / coarse-label loss on L2-normalized embeddings.

    Args:
        embeddings: ``(N, D)`` embeddings, or ``(V, N, D)`` when several views
            of the same ``N`` images are available (views are then flattened
            and used as the instance-level positives).
        labels: ``(N,)`` coarse integer labels (shared across views).
        lam: weight of the instance term; ``1 - lam`` weights the coarse term.
        temperature: softmax temperature ``T``.
        instance_ids: optional ``(N,)`` ids identifying the crops that come from
            the same image. Required to get a non-degenerate instance term when
            ``embeddings`` is 2-D.

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    lam = kwargs.get("lam", lam)
    temperature = kwargs.get("temperature", temperature)

    if embeddings.ndim == 3:
        v, n, d = embeddings.shape
        if instance_ids is None:
            instance_ids = torch.arange(n, device=embeddings.device).repeat(v)
        else:
            instance_ids = instance_ids.view(-1).repeat(v)
        labels = labels.view(-1).repeat(v)
        embeddings = embeddings.reshape(v * n, d)
    elif instance_ids is None:
        # Single view: every sample is its own instance, so the instance term
        # has no positive and contributes nothing.
        instance_ids = torch.arange(embeddings.size(0), device=embeddings.device)

    device = embeddings.device
    n = embeddings.size(0)
    labels = labels.view(-1, 1)
    instance_ids = instance_ids.view(-1, 1)

    self_mask = torch.eye(n, device=device)
    not_self = 1.0 - self_mask
    inst_mask = torch.eq(instance_ids, instance_ids.t()).float() * not_self
    coarse_mask = torch.eq(labels, labels.t()).float() * not_self

    logits = embeddings @ embeddings.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    inst_loss, inst_valid = _softmax_term(logits, inst_mask, not_self)
    coarse_loss, coarse_valid = _softmax_term(logits, coarse_mask, not_self)

    loss = lam * inst_loss + (1.0 - lam) * coarse_loss

    with torch.no_grad():
        knn_accs = batch_knn_accuracy(logits, labels, self_mask)
        metrics = {
            "Grafit": loss.detach(),
            "grafit_instance": inst_loss.detach(),
            "grafit_coarse": coarse_loss.detach(),
            "instance_pos_fraction": inst_valid.float().mean(),
            "pos_fraction": coarse_valid.float().mean(),
            **knn_accs,
        }
    return {"loss": loss, **metrics}
