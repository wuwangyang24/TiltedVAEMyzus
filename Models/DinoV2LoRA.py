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

    def encode(self, x: Tensor) -> Tensor:
        """Return L2-normalized embeddings for a batch of images [N, 3, H, W]."""
        feats = self.backbone(x)          # (N, feat_dim)
        if self.projection is not None:
            feats = self.projection(feats)  # (N, embedding_dim)
        return F.normalize(feats, dim=1)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.encode(x)

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
