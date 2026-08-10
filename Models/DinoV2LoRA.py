import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover - timm is a declared dependency
    _HAS_TIMM = False


# Default backbone name -> timm model id. Uses the LVD-142M DINOv2 weights.
_BACKBONE_TO_TIMM = {
    "vit_small_patch14_dinov2": "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2": "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2": "vit_large_patch14_dinov2.lvd142m",
}


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (Hu et al., 2021) wrapper around a frozen ``nn.Linear``.

    The original linear weights are frozen; only the low-rank update
    ``B @ A`` (scaled by ``alpha / rank``) is trainable. ``B`` is zero-initialized
    so the adapted layer starts identical to the pretrained one.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A ~ Kaiming, B = 0  ->  initial LoRA update is zero.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _inject_lora(model: nn.Module, targets: List[str], rank: int,
                 alpha: int, dropout: float) -> int:
    """Replace every ``nn.Linear`` whose leaf name matches one of ``targets``
    with a :class:`LoRALinear`. Returns the number of layers adapted."""
    to_replace: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            to_replace.append(name)

    for name in to_replace:
        *parents, attr = name.split(".")
        parent = model
        for p in parents:
            parent = getattr(parent, p)
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, rank, alpha, dropout))

    return len(to_replace)


class DinoV2LoRA(nn.Module):
    """LoRA-adapted DINOv2 backbone with a projection head for contrastive
    (supervised InfoNCE / SupCon) representation learning.

    The pretrained DINOv2 backbone is frozen; only the injected LoRA adapters
    and the projection head are trained. ``forward`` returns L2-normalized
    embeddings suitable for a cosine-similarity contrastive objective.

    Args:
        backbone: DINOv2 variant name (see ``_BACKBONE_TO_TIMM``).
        img_size: square input size fed to the backbone (must be a multiple of
            the patch size, 14).
        embedding_dim: dimension of the output (projected) embedding.
        proj_hidden_dim: hidden width of the 2-layer projection MLP.
        lora_rank: rank of the LoRA update.
        lora_alpha: LoRA scaling numerator (effective scale = alpha / rank).
        lora_dropout: dropout applied to the LoRA input.
        lora_targets: leaf module names to adapt (e.g. ["qkv"], ["qkv", "proj"]).
        temperature: softmax temperature for the InfoNCE / SupCon loss.
        use_proj_head: if True, add a 2-layer MLP projection head on top of
            the backbone features; otherwise output the L2-normalized backbone
            features directly.
        pretrained: load pretrained DINOv2 weights.
    """

    # The pipeline uses this flag to skip image reconstruction/sampling logging.
    supports_image_generation = False

    def __init__(self,
                 backbone: str = "vit_small_patch14_dinov2",
                 img_size: int = 224,
                 embedding_dim: int = 256,
                 proj_hidden_dim: int = 2048,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 lora_targets: Optional[List[str]] = None,
                 temperature: float = 0.1,
                 use_proj_head: bool = True,
                 pretrained: bool = True) -> None:
        super().__init__()

        if not _HAS_TIMM:
            raise ImportError(
                "timm is required for DinoV2LoRA. Install it with `pip install timm`."
            )

        if backbone not in _BACKBONE_TO_TIMM:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONE_TO_TIMM)}."
            )

        self.backbone_name = backbone
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_proj_head = use_proj_head
        lora_targets = lora_targets or ["qkv"]

        # Feature-extractor backbone (num_classes=0 -> pooled features, no head).
        self.backbone = timm.create_model(
            _BACKBONE_TO_TIMM[backbone],
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        feat_dim = self.backbone.num_features

        # Freeze the full backbone, then inject trainable LoRA adapters.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n_adapted = _inject_lora(
            self.backbone, lora_targets, lora_rank, lora_alpha, lora_dropout)
        if n_adapted == 0:
            raise ValueError(
                f"No LoRA targets matched {lora_targets} in backbone '{backbone}'."
            )
        self._n_lora_layers = n_adapted

        # Trainable projection head mapping backbone features -> embedding space.
        if self.use_proj_head:
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_hidden_dim),
                nn.GELU(),
                nn.Linear(proj_hidden_dim, embedding_dim),
            )
        else:
            self.projection = None

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable (LoRA + projection head) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, x: Tensor, normalize: bool = True) -> Tensor:
        """Return embeddings for a batch of images [N, 3, H, W].

        By default the embeddings are L2-normalized (for the cosine-similarity
        contrastive objective). Pass ``normalize=False`` to obtain the raw
        projected features, e.g. for the LeJEPA / SIGReg objective whose target
        is an isotropic Gaussian in unbounded Euclidean space.
        """
        feats = self.backbone(x)          # (N, feat_dim)
        if self.projection is not None:
            feats = self.projection(feats)  # (N, embedding_dim)
        return F.normalize(feats, dim=1) if normalize else feats

    def forward(self, x: Tensor, normalize: bool = True, **kwargs) -> Tensor:
        return self.encode(x, normalize=normalize)

    def loss_function(self, embeddings: Tensor, labels: Tensor,
                      **kwargs) -> Dict[str, Tensor]:
        """Supervised contrastive (SupCon / InfoNCE) loss.

        Pulls embeddings sharing a synthesis-program label together and pushes
        embeddings from different programs apart. All same-label samples in the
        batch act as positives for a given anchor.

        Args:
            embeddings: (N, D) L2-normalized embeddings.
            labels: (N,) integer synthesis-program labels.

        Returns a dict with the scalar ``loss`` and monitoring metrics.
        """
        temperature = kwargs.get("temperature", self.temperature)
        device = embeddings.device
        n = embeddings.size(0)
        labels = labels.view(-1, 1)

        # Positive mask: same label, excluding the diagonal (self-pairs).
        pos_mask = torch.eq(labels, labels.t()).float().to(device)
        self_mask = torch.eye(n, device=device)
        pos_mask = pos_mask - self_mask
        pos_mask = pos_mask.clamp(min=0.0)

        # Cosine-similarity logits (embeddings are already normalized).
        logits = embeddings @ embeddings.t() / temperature
        # Numerical stability: subtract per-row max before exponentiating.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Denominator excludes the self-pair.
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0
        # Mean log-likelihood over each anchor's positives.
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1)

        if valid.any():
            loss = -mean_log_prob_pos[valid].mean()
        else:
            # No positive pairs in the batch (e.g. all-distinct labels): no signal.
            loss = torch.zeros((), device=device, requires_grad=True)

        with torch.no_grad():
            knn_accs = self._batch_knn_accuracy(logits, labels, self_mask)
            metrics = {
                "InfoNCE": loss.detach(),
                "pos_fraction": valid.float().mean(),
                **knn_accs,
            }
        return {"loss": loss, **metrics}

    def contrastive_sigreg_loss_function(
        self, embeddings: Tensor, labels: Tensor,
        sigreg_weight: float = 0.1,
        sigreg_slices: int = 512,
        sigreg_num_freqs: int = 33,
        sigreg_t_max: float = 8.0,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Supervised contrastive loss with SIGReg replacing negatives.

        Positive attraction: for each anchor, minimize mean squared distance to
        same-label embeddings. Collapse prevention: SIGReg pushes the embedding
        distribution toward an isotropic Gaussian instead of using negatives.

        loss = (1 - lambda) * pos_attraction + lambda * SIGReg
        """
        temperature = kwargs.get("temperature", self.temperature)
        device = embeddings.device
        n, d = embeddings.shape
        labels_col = labels.view(-1, 1)

        # Positive mask: same label, excluding self.
        pos_mask = torch.eq(labels_col, labels_col.t()).float().to(device)
        pos_mask = (pos_mask - torch.eye(n, device=device)).clamp(min=0.0)
        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        # Positive attraction via cosine similarity on normalized embeddings.
        normed = F.normalize(embeddings, dim=1)
        sim = normed @ normed.t() / temperature
        mean_pos_sim = (pos_mask * sim).sum(dim=1) / pos_per_anchor.clamp(min=1)
        if valid.any():
            pos_loss = -mean_pos_sim[valid].mean()
        else:
            pos_loss = torch.zeros((), device=device, requires_grad=True)

        # SIGReg on un-normalized embeddings for collapse prevention.
        sigreg_loss = self._sigreg_loss(
            embeddings, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

        lam = sigreg_weight
        loss = (1.0 - lam) * pos_loss + lam * sigreg_loss

        with torch.no_grad():
            self_mask = torch.eye(n, device=device)
            knn_accs = self._batch_knn_accuracy(sim, labels_col, self_mask)
            metrics = {
                "pos_loss": pos_loss.detach(),
                "sigreg_loss": sigreg_loss.detach(),
                "pos_fraction": valid.float().mean(),
                "emb_std": embeddings.std(dim=0).mean(),
                **self._gaussianity_metrics(embeddings),
                **knn_accs,
            }
        return {"loss": loss, **metrics}

    def lejepa_loss_function(self, view_embeddings: Tensor,
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
                "lejepa_loss_function expects (V, N, D) view embeddings, got "
                f"shape {tuple(view_embeddings.shape)}."
            )
        v, n, d = view_embeddings.shape

        # Prediction / invariance: pull each view towards the per-image mean.
        mean_emb = view_embeddings.mean(dim=0, keepdim=True)      # (1, N, D)
        pred_loss = ((view_embeddings - mean_emb) ** 2).sum(dim=-1).mean()

        # SIGReg over all views/images stacked together.
        z = view_embeddings.reshape(v * n, d)
        sigreg_loss = self._sigreg_loss(
            z, sigreg_slices, sigreg_num_freqs, sigreg_t_max)

        # Convex LeJEPA weighting (Balestriero & LeCun, 2025): lambda balances
        # the isotropic-Gaussian (SIGReg) and invariance (prediction) terms.
        lam = sigreg_weight
        loss = (1.0 - lam) * pred_loss + lam * sigreg_loss

        with torch.no_grad():
            # Cross-view alignment: mean cosine similarity between the two most
            # separated views (view 0 vs view 1) as a collapse/quality monitor.
            z0 = F.normalize(view_embeddings[0], dim=1)
            z1 = F.normalize(view_embeddings[min(1, v - 1)], dim=1)
            view_cos = (z0 * z1).sum(dim=1).mean()
            metrics = {
                "pred_loss": pred_loss.detach(),
                "sigreg_loss": sigreg_loss.detach(),
                "view_cos_sim": view_cos,
                "emb_std": z.std(dim=0).mean(),
                **self._gaussianity_metrics(z),
            }
        return {"loss": loss, **metrics}

    @staticmethod
    @torch.no_grad()
    def _gaussianity_metrics(z: Tensor) -> Dict[str, Tensor]:
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

    @staticmethod
    def _sigreg_loss(z: Tensor, num_slices: int = 512, num_freqs: int = 33,
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

    @staticmethod
    @torch.no_grad()
    def _batch_knn_accuracy(logits: Tensor, labels: Tensor,
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
