from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .utils import sigreg_loss, batch_knn_accuracy, gaussianity_metrics

_SMALL_NUM = np.log(1e-45)


class DCLSIGRegLoss(nn.Module):
    """Decoupled Contrastive Loss (Yeh et al., 2022) + SIGReg with a
    false-negative-aware memory bank.

    Vanilla DCL treats every cross-label pair as an equally-weighted negative,
    which penalizes *false negatives*: pairs that carry different labels but
    actually depict semantically close classes. This variant estimates, for
    each pair of classes, a *suspicion* value ``p_ij`` and down-weights the
    corresponding negatives by ``1 - p_ij``.

    Mechanism:
      1. A no-gradient EMA of the per-class mean embedding is kept in a memory
         bank (one row per class label).
      2. Cross-class cosine similarities of the bank means are squashed with a
         sigmoid into a suspicion ``p_ij`` in (0, 1): close class means -> high
         suspicion (likely false negative), distant means -> low suspicion.
      3. Each negative pair is weighted by ``1 - p_ij`` instead of 1.
      4. SIGReg is applied to the class-mean embeddings (with gradient re-
         injected for the classes present in the batch) to keep the class
         centroids spread like an isotropic Gaussian.

    Final loss: ``pos_loss + suspicion_weighted_neg_loss + lambda * sigreg``.

    Args:
        ema_momentum: EMA momentum ``m`` for the memory bank update
            ``mean <- m * mean + (1 - m) * batch_mean``.
        suspicion_tau: temperature of the suspicion sigmoid.
        suspicion_bias: similarity offset subtracted before the sigmoid; a pair
            is "suspicious" only once its mean-similarity exceeds this value.
    """

    def __init__(self, ema_momentum: float = 0.9,
                 suspicion_tau: float = 0.1,
                 suspicion_bias: float = 0.5,
                 normal_dcl: bool = False) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.suspicion_tau = suspicion_tau
        self.suspicion_bias = suspicion_bias
        # When True, fall back to plain DCL+SIGReg: every negative is weighted
        # equally (no suspicion memory bank) and SIGReg acts on the batch
        # embeddings directly.
        self.normal_dcl = normal_dcl
        # Lazily-sized memory-bank buffers (allocated on first forward).
        self.register_buffer("class_means", torch.empty(0), persistent=True)
        self.register_buffer("initialized", torch.empty(0, dtype=torch.bool),
                             persistent=True)

    def _ensure_capacity(self, num_classes: int, dim: int,
                         device, dtype) -> None:
        """Grow the memory bank to hold at least ``num_classes`` rows."""
        cur = self.class_means.size(0)
        if cur >= num_classes and self.class_means.numel() > 0:
            return
        new_means = torch.zeros(num_classes, dim, device=device, dtype=dtype)
        new_init = torch.zeros(num_classes, dtype=torch.bool, device=device)
        if cur > 0:
            new_means[:cur] = self.class_means.to(device=device, dtype=dtype)
            new_init[:cur] = self.initialized.to(device=device)
        self.class_means = new_means
        self.initialized = new_init

    @torch.no_grad()
    def _update_memory(self, batch_means: Tensor, present_mask: Tensor) -> None:
        """EMA-update the memory bank from detached per-class batch means."""
        m = self.ema_momentum
        new_init = present_mask & (~self.initialized)
        upd = present_mask & self.initialized
        self.class_means[new_init] = batch_means[new_init]
        self.class_means[upd] = (
            m * self.class_means[upd] + (1.0 - m) * batch_means[upd])
        self.initialized |= present_mask

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
        num_classes = int(labels_flat.max().item()) + 1

        self._ensure_capacity(num_classes, d, device, embeddings.dtype)

        # SIGReg and the class-mean bank operate in the raw (un-normalized)
        # Euclidean space, where the isotropic-Gaussian target is well defined;
        # the contrastive/suspicion terms use the L2-normalized (cosine) space.
        normed = F.normalize(embeddings, dim=1)

        # Per-class batch means (gradient-carrying) via one-hot scatter, in the
        # raw embedding space for SIGReg / bank storage.
        onehot = F.one_hot(labels_flat, self.class_means.size(0)).to(embeddings.dtype)
        counts = onehot.sum(0)                              # (C,)
        sums = onehot.t() @ embeddings                      # (C, D)
        present_mask = counts > 0
        safe_counts = counts.clamp(min=1).unsqueeze(1)
        class_means_batch = sums / safe_counts              # (C, D), grad

        # Masks.
        labels_col = labels_flat.view(-1, 1)
        self_mask = torch.eye(n, device=device)
        pos_mask = torch.eq(labels_col, labels_col.t()).float()
        pos_mask = (pos_mask - self_mask).clamp(min=0.0)
        neg_mask = 1.0 - torch.eq(labels_col, labels_col.t()).float()
        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        # Cosine similarity on the L2-normalized embeddings.
        sim = normed @ normed.t() / temperature

        # Positive term: mean similarity to same-label samples.
        pos_sim = (pos_mask * sim).sum(dim=1) / pos_per_anchor.clamp(min=1)
        if valid.any():
            pos_loss = -pos_sim[valid].mean()
        else:
            pos_loss = torch.zeros((), device=device, requires_grad=True)

        non_neg_mask = pos_mask + self_mask                 # mask out positives and self

        if self.normal_dcl:
            # Plain DCL: every negative weighted equally; SIGReg on the batch
            # embeddings directly (no suspicion memory bank).
            neg_logits = sim + non_neg_mask * _SMALL_NUM
            neg_loss = torch.logsumexp(neg_logits, dim=1).mean()
            sr_loss = sigreg_loss(
                embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)
        else:
            # Suspicion weights from the detached memory-bank means (pre-update).
            # Close class means -> high p_ij -> small (1 - p_ij) weight on that pair.
            with torch.no_grad():
                bank_normed = F.normalize(self.class_means, dim=1)
                sample_means = bank_normed[labels_flat]         # (N, D)
                mean_sim = sample_means @ sample_means.t()      # (N, N)
                p = torch.sigmoid((mean_sim - self.suspicion_bias) / self.suspicion_tau)
                neg_weight = (1.0 - p).clamp(min=1e-6)

            # Weighted decoupled negative term: log( sum_j w_ij * exp(sim_ij) ).
            log_w = torch.log(neg_weight)
            neg_logits = sim + log_w + non_neg_mask * _SMALL_NUM
            neg_loss = torch.logsumexp(neg_logits, dim=1).mean()

            # SIGReg over a pool of class means: live batch means (gradient) for
            # the classes in this batch, plus the last stored EMA means
            # (detached) for the bank classes absent from the batch.
            absent_mask = self.initialized & (~present_mask)
            pool = [class_means_batch[present_mask]]
            if absent_mask.any():
                pool.append(self.class_means[absent_mask].detach())
            pool_means = torch.cat(pool, dim=0)
            if pool_means.size(0) >= 2:
                sr_loss = sigreg_loss(
                    pool_means, sigreg_slices, sigreg_num_freqs, sigreg_t_max)
            else:
                sr_loss = torch.zeros((), device=device)

        lam = sigreg_weight
        loss = pos_loss + neg_loss + lam * sr_loss

        with torch.no_grad():
            knn_accs = batch_knn_accuracy(sim, labels_col, self_mask)
            metrics = {
                "pos_loss": pos_loss.detach(),
                "neg_loss": neg_loss.detach(),
                "sigreg_loss": sr_loss.detach(),
                "pos_fraction": valid.float().mean(),
                "emb_std": embeddings.std(dim=0).mean(),
                **gaussianity_metrics(embeddings),
                **knn_accs,
            }

            # Suspicion diagnostics (only when the memory bank is active).
            if not self.normal_dcl:
                metrics["suspicion_mean"] = (
                    (p * neg_mask).sum() / neg_mask.sum().clamp(min=1))
                # Break down suspicion over negatives by --test_cat relationship:
                # negatives sharing a test_cat are the likely false negatives.
                if test_labels is not None:
                    tl = test_labels.view(-1, 1)
                    same_test = torch.eq(tl, tl.t()).float()
                    same_test_neg = neg_mask * same_test
                    diff_test_neg = neg_mask * (1.0 - same_test)
                    metrics["suspicion_same_testcat"] = (
                        (p * same_test_neg).sum() / same_test_neg.sum().clamp(min=1))
                    metrics["suspicion_diff_testcat"] = (
                        (p * diff_test_neg).sum() / diff_test_neg.sum().clamp(min=1))

        # Update the EMA bank for this batch's classes (detached for storage).
        if not self.normal_dcl:
            self._update_memory(class_means_batch.detach(), present_mask)

        return {"loss": loss, **metrics}
