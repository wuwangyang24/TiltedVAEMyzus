"""MaskCon loss (Feng & Patras, CVPR 2023).

*MaskCon: Masked Contrastive Learning for Coarse-Labelled Dataset* builds, for
every query, a soft target over ``[k+, queue]`` that interpolates between the
purely self-supervised target (only the other view ``k+`` is positive) and
inter-sample relations that are **masked by the coarse labels**: candidates with
a different coarse label get target 0, candidates sharing it get a weight given
by their similarity to the key branch ``k+``::

    p_i  = softmax_{j: same coarse}(k . m_j / t0)_i        (masked soft labels)
    z_i  = p_i / max_j p_j                                 (rescaled to [0, 1])
    y    = normalize([1, z])                               (MaskCon target)
    y_hat = w * y + (1 - w) * onehot(k+)
    L    = - sum_i y_hat_i * log_softmax([q.k+, q.M] / t)_i

``w = 1`` recovers pure MaskCon, ``w = 0`` recovers MoCo/SimCLR-style
self-supervision, and ``t0 -> inf`` recovers SupCon over the coarse labels.

As in the official implementation (MrChenFeng/MaskCon_CVPR2023) the queried
candidates ``m_j`` come from a MoCo-style FIFO memory queue filled by a momentum
encoder, and the soft labels are produced by the (gradient-free) key branch.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_NEG_INF = -1e4


class MaskConQueue(nn.Module):
    """MoCo-style FIFO queue of momentum-encoder keys and their coarse labels."""

    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.size = size
        self.register_buffer("queue", torch.zeros(size, dim))
        self.register_buffer("queue_labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(size, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: Tensor, labels: Tensor) -> None:
        keys = keys.detach().to(self.queue.dtype).to(self.queue.device)
        labels = labels.view(-1).to(self.queue_labels.device)
        n = min(keys.size(0), self.size)
        keys, labels = keys[:n], labels[:n]
        ptr = int(self.ptr)
        # Write in (at most) two chunks so any batch / queue size combination works.
        first = min(n, self.size - ptr)
        self.queue[ptr:ptr + first] = keys[:first]
        self.queue_labels[ptr:ptr + first] = labels[:first]
        self.filled[ptr:ptr + first] = True
        rest = n - first
        if rest > 0:
            self.queue[:rest] = keys[first:]
            self.queue_labels[:rest] = labels[first:]
            self.filled[:rest] = True
        self.ptr[0] = (ptr + n) % self.size


@torch.no_grad()
def _knn_accuracy(logits: Tensor, labels: Tensor, cand_labels: Tensor,
                  cand_mask: Tensor) -> Dict[str, Tensor]:
    """Top-1/3/5 coarse-label agreement of each query with its nearest candidates."""
    masked = logits.masked_fill(~cand_mask, float("-inf"))
    labels = labels.view(-1, 1)
    n_cand = int(cand_mask.sum(dim=1).min().item())
    result = {}
    for k, suffix in ((1, "batch_knn_acc"), (3, "batch_knn_top3_acc"),
                      (5, "batch_knn_top5_acc")):
        if k > n_cand:
            result[suffix] = torch.ones((), device=logits.device)
            continue
        topk_idx = masked.topk(k, dim=1).indices
        hits = (cand_labels[topk_idx] == labels).any(dim=1)
        result[suffix] = hits.float().mean()
    return result


def maskcon_loss(embeddings: Tensor, labels: Tensor, keys: Optional[Tensor] = None,
                 temperature: float = 0.1, soft_temperature: float = 0.1,
                 w: float = 1.0, queue: Optional[MaskConQueue] = None,
                 update_queue: bool = True, **kwargs) -> Dict[str, Tensor]:
    """MaskCon loss on L2-normalized embeddings.

    Args:
        embeddings: ``(N, D)`` online (query) embeddings ``q``.
        labels: ``(N,)`` coarse integer labels.
        keys: ``(N, D)`` momentum-encoder embeddings ``k+`` of a second view of
            the same images. When omitted the queries stand in for the keys, so
            the explicit positive degenerates to the query itself; pass a second
            view whenever available.
        temperature: contrastive temperature ``t`` of the main objective.
        soft_temperature: temperature ``t0`` used to generate the soft labels.
        w: mixing weight between the MaskCon target and the purely
            self-supervised one (``1.0`` = pure MaskCon).
        queue: optional :class:`MaskConQueue` holding the candidates ``M``. With
            no queue the other keys in the batch play that role.
        update_queue: enqueue this batch's keys after the loss (train only).

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    temperature = kwargs.get("temperature", temperature)
    soft_temperature = kwargs.get("soft_temperature", soft_temperature)
    w = kwargs.get("w", w)

    device, dtype = embeddings.device, embeddings.dtype
    n = embeddings.size(0)
    labels = labels.view(-1)
    q = embeddings
    k = q.detach() if keys is None else keys.detach()

    if queue is not None:
        cand = queue.queue.to(device=device, dtype=dtype)
        cand_labels = queue.queue_labels.to(device)
        cand_mask = queue.filled.to(device).unsqueeze(0).expand(n, -1)
    else:
        # In-batch fallback: every other key of the batch is a candidate.
        cand = k
        cand_labels = labels
        cand_mask = ~torch.eye(n, device=device, dtype=torch.bool)

    # Masked soft labels from the (gradient-free) key branch.
    with torch.no_grad():
        coarse_mask = (labels.view(-1, 1) == cand_labels.view(1, -1)) & cand_mask
        soft_logits = (k @ cand.t()) / soft_temperature
        soft_logits = soft_logits.masked_fill(~coarse_mask, 0.0)
        soft_logits = soft_logits - soft_logits.max(dim=1, keepdim=True).values
        soft = soft_logits.exp() * coarse_mask
        soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-12)
        # Rescale so the closest same-coarse candidate is as positive as k+.
        soft = soft / soft.max(dim=1, keepdim=True).values.clamp_min(1e-12)

        target = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), soft], dim=1)
        target = target / target.sum(dim=1, keepdim=True)
        self_target = torch.zeros_like(target)
        self_target[:, 0] = 1.0
        target = w * target + (1.0 - w) * self_target

    l_pos = (q * k).sum(dim=1, keepdim=True)
    l_neg = q @ cand.t()
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    logits_mask = torch.cat(
        [torch.ones(n, 1, device=device, dtype=torch.bool), cand_mask], dim=1)
    logits = logits.masked_fill(~logits_mask, _NEG_INF)

    loss = -(F.log_softmax(logits, dim=1) * target).sum(dim=1).mean()

    with torch.no_grad():
        n_same = coarse_mask.sum(dim=1)
        metrics = {
            "MaskCon": loss.detach(),
            "maskcon_soft_mass": (target[:, 1:].sum(dim=1)).mean(),
            "maskcon_soft_candidates": n_same.float().mean(),
            "pos_fraction": (n_same > 0).float().mean(),
            **_knn_accuracy(l_neg, labels, cand_labels, cand_mask),
        }
        if queue is not None:
            metrics["maskcon_queue_fill"] = queue.filled.float().mean()

    if queue is not None and update_queue:
        queue.enqueue(k, labels)
    return {"loss": loss, **metrics}
