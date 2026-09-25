"""BuCSFR loss (Shi et al., ICCV 2025).

*Learning Separable Fine-Grained Representation via Dendrogram Construction from
Coarse Labels for Fine-grained Visual Recognition* trains a MoCo-style
contrastive model whose positives and negatives are selected by a **dendrogram**
built bottom-up inside every coarse class:

1. Every few epochs the (momentum) embeddings of the whole training set are
   clustered per coarse class with k-means, giving an over-segmentation of the
   unknown fine-grained classes.
2. The two clusters of a coarse class whose *coding-rate* entropies are the most
   redundant are merged, one merge per class per round, so the leaf partition
   walks up the dendrogram as training progresses
   (:class:`BuCSFRDendrogram`).
3. The resulting cluster assignment drives instance selection against the
   momentum queue: candidates sharing the anchor's coarse label are sampled as
   *positives* with probability proportional to their soft assignment to the
   anchor's cluster, and as *negatives* with the complementary probability.
   Candidates of a different coarse class are always negatives.
4. The selected positives are softly weighted by their key-branch similarity and
   the resulting distribution is the target of a cross-entropy over the MoCo
   logits; a coarse-label classification loss is mixed in with weight
   ``1 - alpha``.

Differences from the reference implementation
(BeCarefulOfYournaoke/BuCSFR, ``moco/builder.py`` + ``utils/run_hkmeans``):
faiss/GPU k-means is replaced by a torch k-means, and unselected negatives are
removed from the denominator (``-inf``) instead of having their logit zeroed.
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .maskcon import MaskConQueue

_NEG_INF = -1e4
# Fixed prototype temperature of the reference implementation (`div(0.1)`).
_PROTO_TEMPERATURE = 0.1


@torch.no_grad()
def _kmeans(x: Tensor, k: int, iters: int = 20,
            generator: Optional[torch.Generator] = None) -> Tensor:
    """Plain Lloyd k-means on ``(n, d)`` features. Returns ``(n,)`` assignments."""
    n = x.size(0)
    k = min(k, n)
    perm = torch.randperm(n, device=x.device, generator=generator)[:k]
    centroids = x[perm].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        for c in range(k):
            members = x[assign == c]
            if members.numel():
                centroids[c] = members.mean(dim=0)
    return assign


@torch.no_grad()
def _coding_rate(z: Tensor) -> Tensor:
    """Gaussian coding rate ``(n + d)/2 * log2 det(I + z z^T / (n * eps))``.

    This is the entropy ``L`` of the reference ``new_new_entropy_counter`` (with
    its ``exiu / 128`` constant rewritten as ``eps = 1 / d``).
    """
    n, d = z.shape
    lam = d / max(n, 1)
    gram = z @ z.t() if n <= d else z.t() @ z
    gram = gram * lam
    gram = gram + torch.eye(gram.size(0), device=z.device, dtype=z.dtype)
    logdet = torch.linalg.slogdet(gram).logabsdet / torch.log(
        torch.tensor(2.0, device=z.device, dtype=z.dtype))
    return logdet * ((n + d) / 2.0)


class BuCSFRDendrogram:
    """Bottom-up dendrogram over the fine-grained structure of each coarse class.

    The first call k-means-partitions every coarse class into
    ``clusters_per_class`` leaves; each subsequent call merges (at most) the one
    most redundant cluster pair per coarse class, so the number of clusters
    decreases monotonically towards the true fine-grained granularity.

    Args:
        clusters_per_class: initial number of leaves per coarse class (the paper
            recommends 3-4x the expected number of fine-grained classes).
        threshold: hyper-parameter ``T`` of the paper. A pair is merged when
            ``min(L_j, L_k) / I_jk < T``; larger values merge more eagerly.
        kmeans_iters: Lloyd iterations of the leaf k-means.
        seed: RNG seed of the k-means initialization.
    """

    def __init__(self, clusters_per_class: int = 20, threshold: float = 1.1,
                 kmeans_iters: int = 20, seed: int = 0) -> None:
        self.init_clusters_per_class = clusters_per_class
        self.threshold = threshold
        self.kmeans_iters = kmeans_iters
        self.seed = seed
        # coarse label -> current number of clusters (decreases with merges).
        self.clusters_per_class: Dict[int, int] = {}

    @torch.no_grad()
    def build(self, features: Tensor, coarse_labels: Tensor) -> Dict[str, Tensor]:
        """Cluster ``features`` (L2-normalized, ``(N, D)``) per coarse class.

        Returns ``{"im2cluster": (N,), "centroids": (C, D), "density": (C,)}``
        with globally unique cluster ids.
        """
        device = features.device
        generator = torch.Generator(device=device).manual_seed(self.seed)
        coarse_labels = coarse_labels.view(-1)

        im2cluster = torch.full((features.size(0),), -1, dtype=torch.long,
                                device=device)
        centroids: List[Tensor] = []
        offset = 0
        for coarse in coarse_labels.unique().tolist():
            idx = (coarse_labels == coarse).nonzero(as_tuple=True)[0]
            z = features[idx]
            k = self.clusters_per_class.get(coarse, self.init_clusters_per_class)
            k = max(min(k, z.size(0)), 1)

            assign = _kmeans(z, k, self.kmeans_iters, generator)
            assign, k = self._merge_once(z, assign, k)

            class_centroids = torch.stack([
                z[assign == c].mean(dim=0) if (assign == c).any()
                else torch.zeros(z.size(1), device=device, dtype=z.dtype)
                for c in range(k)
            ])
            im2cluster[idx] = assign + offset
            centroids.append(class_centroids)
            self.clusters_per_class[coarse] = k
            offset += k

        centroid_matrix = F.normalize(torch.cat(centroids, dim=0), dim=1)
        # The reference keeps a constant prototype concentration of 0.1.
        density = torch.full((centroid_matrix.size(0),), _PROTO_TEMPERATURE,
                             device=device, dtype=centroid_matrix.dtype)
        return {"im2cluster": im2cluster, "centroids": centroid_matrix,
                "density": density}

    @torch.no_grad()
    def _merge_once(self, z: Tensor, assign: Tensor, k: int):
        """Merge the most redundant cluster pair of one coarse class, if any.

        The merge score of a pair is ``min(L_j, L_k) / (L_j + L_k - L_jk)``:
        the smaller cluster's coding rate over the coding rate the merge saves.
        Pairs are tried in increasing order and the first admissible one
        (``0 < score < threshold``) is merged, mirroring the reference's single
        merge per clustering round.
        """
        if k < 2:
            return assign, k

        rates = [_coding_rate(z[assign == c]) if (assign == c).sum() > 1 else None
                 for c in range(k)]

        scores = []
        for j in range(k):
            for m in range(j + 1, k):
                if rates[j] is None or rates[m] is None:
                    continue
                n_j = int((assign == j).sum())
                n_m = int((assign == m).sum())
                combined = _coding_rate(torch.cat([z[assign == j], z[assign == m]]))
                mutual = rates[j] + rates[m] - combined
                if mutual <= 0:
                    continue
                smaller = rates[m] if n_j >= n_m else rates[j]
                score = float(smaller / mutual)
                if score > 0:
                    scores.append((score, j, m))

        if not scores:
            return assign, k
        score, j, m = min(scores)
        if score >= self.threshold:
            return assign, k

        assign = assign.clone()
        assign[assign == m] = j
        # Compact the ids so they stay contiguous in [0, k - 1).
        assign[assign > m] -= 1
        return assign, k - 1


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


def bucsfr_loss(embeddings: Tensor, labels: Tensor, keys: Optional[Tensor] = None,
                cluster_labels: Optional[Tensor] = None,
                centroids: Optional[Tensor] = None,
                density: Optional[Tensor] = None,
                class_logits: Optional[Tensor] = None,
                temperature: float = 0.2, alpha: float = 0.5,
                queue: Optional[MaskConQueue] = None,
                update_queue: bool = True, **kwargs) -> Dict[str, Tensor]:
    """BuCSFR loss on L2-normalized embeddings.

    Args:
        embeddings: ``(N, D)`` online (query) embeddings ``q``.
        labels: ``(N,)`` coarse integer labels.
        keys: ``(N, D)`` momentum-encoder embeddings of a second view. Falls
            back to the (detached) queries when omitted.
        cluster_labels: ``(N,)`` dendrogram cluster id of each anchor, from
            :meth:`BuCSFRDendrogram.build`. ``None`` (warmup, before the first
            dendrogram) degenerates to plain MoCo over the coarse queue.
        centroids: ``(C, D)`` normalized cluster centroids.
        density: ``(C,)`` per-cluster prototype concentration.
        class_logits: ``(N, num_coarse)`` logits of the coarse classifier; adds
            the supervised term weighted by ``1 - alpha``.
        temperature: contrastive temperature of the MoCo objective.
        alpha: weight of the contrastive term in
            ``alpha * L_con + (1 - alpha) * L_ce``.
        queue: momentum-key queue holding the selection candidates. Without one
            the other keys of the batch play that role.
        update_queue: enqueue this batch's keys after the loss (train only).

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    device, dtype = embeddings.device, embeddings.dtype
    n = embeddings.size(0)
    labels = labels.view(-1)
    q = embeddings
    k = q.detach() if keys is None else keys.detach()

    if queue is not None:
        # Clone: the candidates stay in the graph of l_neg while enqueue()
        # mutates the queue buffer in place at the end of this call.
        cand = queue.queue.to(device=device, dtype=dtype).clone()
        cand_labels = queue.queue_labels.to(device)
        cand_mask = queue.filled.to(device).unsqueeze(0).expand(n, -1)
    else:
        cand = k
        cand_labels = labels
        cand_mask = ~torch.eye(n, device=device, dtype=torch.bool)

    same_coarse = (labels.view(-1, 1) == cand_labels.view(1, -1)) & cand_mask
    l_pos = (q * k).sum(dim=1, keepdim=True)
    l_neg = q @ cand.t()

    has_dendrogram = (cluster_labels is not None and centroids is not None
                      and cand_mask.any())
    if has_dendrogram:
        with torch.no_grad():
            centroids = centroids.to(device=device, dtype=dtype)
            proto_logits = cand @ centroids.t()
            if density is not None:
                proto_logits = proto_logits / density.to(device=device, dtype=dtype).clamp_min(1e-3)
            else:
                proto_logits = proto_logits / _PROTO_TEMPERATURE
            # P(candidate belongs to the anchor's dendrogram cluster), (N, M).
            p_cluster = proto_logits.softmax(dim=-1)[:, cluster_labels.view(-1)].t()

            pos_p = p_cluster * same_coarse
            pos_p = pos_p / pos_p.max(dim=1, keepdim=True).values.clamp_min(1e-12)
            pos_mask = torch.bernoulli(pos_p.clamp(0.0, 0.999)).bool() & same_coarse

            neg_p = (1.0 - p_cluster) * same_coarse
            neg_p = neg_p / neg_p.max(dim=1, keepdim=True).values.clamp_min(1e-12)
            neg_mask = torch.bernoulli(neg_p.clamp(0.0, 0.999)).bool() & same_coarse
            # Candidates of another coarse class are unconditional negatives.
            keep_mask = neg_mask | pos_mask | (cand_mask & ~same_coarse)

            # Soft weights of the selected positives, from the key branch.
            pos_sim = (k @ cand.t()) / _PROTO_TEMPERATURE
            pos_sim = pos_sim.masked_fill(~pos_mask, float("-inf"))
            pos_weight = pos_sim.softmax(dim=1)
            pos_weight = torch.where(pos_mask, pos_weight,
                                     torch.zeros_like(pos_weight))
            # Rescale so the best selected positive matches the explicit key.
            pos_weight = pos_weight / pos_weight.max(dim=1, keepdim=True).values.clamp_min(1e-12)
            target = torch.cat(
                [torch.ones(n, 1, device=device, dtype=dtype), pos_weight], dim=1)
            target = target / target.sum(dim=1, keepdim=True)
    else:
        keep_mask = cand_mask
        target = torch.zeros(n, cand.size(0) + 1, device=device, dtype=dtype)
        target[:, 0] = 1.0
        pos_mask = torch.zeros_like(same_coarse)

    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    logits_mask = torch.cat(
        [torch.ones(n, 1, device=device, dtype=torch.bool), keep_mask], dim=1)
    logits = logits.masked_fill(~logits_mask, _NEG_INF)

    con_loss = -(F.log_softmax(logits, dim=1) * target).sum(dim=1).mean()
    if class_logits is not None:
        ce_loss = F.cross_entropy(class_logits, labels.long())
        loss = alpha * con_loss + (1.0 - alpha) * ce_loss
    else:
        ce_loss = torch.zeros((), device=device)
        loss = con_loss

    with torch.no_grad():
        metrics = {
            "BuCSFR": loss.detach(),
            "bucsfr_con_loss": con_loss.detach(),
            "bucsfr_ce_loss": ce_loss.detach(),
            "bucsfr_selected_pos": pos_mask.sum(dim=1).float().mean(),
            "bucsfr_selected_neg": keep_mask.sum(dim=1).float().mean(),
            "bucsfr_dendrogram_active": torch.tensor(
                float(has_dendrogram), device=device),
            "pos_fraction": (same_coarse.sum(dim=1) > 0).float().mean(),
            **_knn_accuracy(l_neg, labels, cand_labels, cand_mask),
        }
        if class_logits is not None:
            metrics["ce_top1"] = (class_logits.argmax(dim=1)
                                  == labels.long()).float().mean()
        if queue is not None:
            metrics["bucsfr_queue_fill"] = queue.filled.float().mean()

    if queue is not None and update_queue:
        queue.enqueue(k, labels)
    return {"loss": loss, **metrics}
