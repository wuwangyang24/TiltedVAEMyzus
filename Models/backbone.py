from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from Loss import (
    infonce_loss, contrastive_sigreg_loss, DCLSIGRegLoss, DCLSoftPosLoss,
    sigreg_loss, batch_knn_accuracy, gaussianity_metrics,
    vanilla_dcl_loss, infonce_softpos_loss, SupConSoftPosLoss,
    vanilla_supcon_loss,
)

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover - timm is a declared dependency
    _HAS_TIMM = False


# Supported fully fine-tuned backbones (timm model ids).
_SUPPORTED_BACKBONES = (
    "resnet18", "resnet50", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
    "convnext_tiny",
)


class Backbone(nn.Module):
    """Fully fine-tuned convolutional / ViT backbone with an optional projection
    head for supervised contrastive (InfoNCE / SupCon) representation learning.

    Unlike :class:`DinoV2LoRA`, the entire backbone is trainable (full
    fine-tuning, no LoRA). ``forward`` returns L2-normalized embeddings suitable
    for a cosine-similarity contrastive objective, so this model is a drop-in
    replacement inside :class:`ContrastiveExperiment`.

    Despite the class name, the ``backbone`` argument selects which timm model to
    fine-tune (``resnet18``, ``resnet50``, ``vit_small_patch16_224``,
    ``swin_tiny_patch4_window7_224`` or ``convnext_tiny``).

    Args:
        backbone: timm model id to fully fine-tune (see ``_SUPPORTED_BACKBONES``).
        img_size: square input size fed to the backbone (any size >= 32).
        embedding_dim: dimension of the output (projected) embedding.
        proj_hidden_dim: hidden width of the 2-layer projection MLP.
        temperature: softmax temperature for the InfoNCE / SupCon loss.
        use_proj_head: if True, add a 2-layer MLP projection head on top of the
            backbone features; otherwise output the L2-normalized backbone
            features directly.
        pretrained: load ImageNet-pretrained backbone weights.
    """

    # The pipeline uses this flag to skip image reconstruction/sampling logging.
    supports_image_generation = False

    def __init__(self,
                 backbone: str = "resnet18",
                 img_size: int = 224,
                 embedding_dim: int = 256,
                 proj_hidden_dim: int = 2048,
                 temperature: float = 0.1,
                 use_proj_head: bool = True,
                 dcl_ema_momentum: float = 0.9,
                 dcl_suspicion_tau: float = 0.1,
                 dcl_suspicion_bias: float = 0.5,
                 dcl_suspicion_standardize: bool = False,
                 dcl_normal: bool = False,
                 dcl_soft_pos: bool = False,
                 dcl_soft_pos_tau: float = 0.1,
                 supcon_soft_pos: bool = False,
                 supcon_soft_pos_tau: float = 0.1,
                 supcon_denom_pos_weight: bool = False,
                 sinkhorn: bool = False,
                 sinkhorn_iters: int = 5,
                 grad_checkpointing: bool = False,
                 pretrained: bool = True) -> None:
        super().__init__()

        if not _HAS_TIMM:
            raise ImportError(
                "timm is required for Backbone. Install it with `pip install timm`."
            )

        if backbone not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose from {list(_SUPPORTED_BACKBONES)}."
            )

        self.backbone_name = backbone
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_proj_head = use_proj_head

        # Feature-extractor backbone (num_classes=0 -> pooled features, no head).
        # The whole backbone is trainable (full fine-tuning). Transformer
        # backbones need img_size to build position embeddings for non-default
        # input sizes.
        create_kwargs = dict(pretrained=pretrained, num_classes=0)
        if backbone.startswith(("vit", "swin")):
            create_kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **create_kwargs)
        feat_dim = self.backbone.num_features

        # Trade compute for memory: recompute backbone activations in the
        # backward pass instead of storing them (allows larger batches).
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)

        # Trainable projection head mapping backbone features -> embedding space.
        if self.use_proj_head:
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_hidden_dim),
                nn.GELU(),
                nn.Linear(proj_hidden_dim, embedding_dim),
            )
        else:
            self.projection = None

        # Stateful DCL+SIGReg loss with its EMA class-mean memory bank.
        self.dcl_sigreg_loss = DCLSIGRegLoss(
            ema_momentum=dcl_ema_momentum,
            suspicion_tau=dcl_suspicion_tau,
            suspicion_bias=dcl_suspicion_bias,
            suspicion_standardize=dcl_suspicion_standardize,
            normal_dcl=dcl_normal,
        )

        self.dcl_soft_pos_loss = DCLSoftPosLoss(
            pos_weight_tau=dcl_soft_pos_tau,
            sinkhorn=sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
        ) if dcl_soft_pos else None

        self.supcon_soft_pos_loss = SupConSoftPosLoss(
            pos_weight_tau=supcon_soft_pos_tau,
            sinkhorn=sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            denom_pos_weight=supcon_denom_pos_weight,
        ) if supcon_soft_pos else None

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable (backbone + projection head) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, x: Tensor, normalize: bool = True) -> Tensor:
        """Return embeddings for a batch of images [N, 3, H, W].

        By default the embeddings are L2-normalized (for the cosine-similarity
        contrastive objective). Pass ``normalize=False`` to obtain the raw
        projected features (e.g. for the SIGReg objective whose target is an
        isotropic Gaussian in unbounded Euclidean space).
        """
        feats = self.backbone(x)          # (N, feat_dim)
        if self.projection is not None:
            feats = self.projection(feats)  # (N, embedding_dim)
        return F.normalize(feats, dim=1) if normalize else feats

    def forward(self, x: Tensor, normalize: bool = True, **kwargs) -> Tensor:
        return self.encode(x, normalize=normalize)

    def loss_function(self, embeddings: Tensor, labels: Tensor,
                      **kwargs) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return infonce_loss(embeddings, labels, **kwargs)

    def contrastive_sigreg_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        return contrastive_sigreg_loss(embeddings, labels, **kwargs)

    def dcl_sigreg_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        return self.dcl_sigreg_loss(embeddings, labels, **kwargs)

    def dcl_soft_pos_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        return self.dcl_soft_pos_loss(embeddings, labels, **kwargs)

    def supcon_soft_pos_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return self.supcon_soft_pos_loss(embeddings, labels, **kwargs)

    def vanilla_dcl_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return vanilla_dcl_loss(embeddings, labels, **kwargs)

    def vanilla_supcon_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return vanilla_supcon_loss(embeddings, labels, **kwargs)

    def infonce_softpos_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return infonce_softpos_loss(embeddings, labels, **kwargs)

    @staticmethod
    @torch.no_grad()
    def _gaussianity_metrics(z: Tensor) -> Dict[str, Tensor]:
        return gaussianity_metrics(z)

    @staticmethod
    def _sigreg_loss(z: Tensor, num_slices: int = 512, num_freqs: int = 33,
                     t_max: float = 8.0) -> Tensor:
        return sigreg_loss(z, num_slices, num_freqs, t_max)

    @staticmethod
    @torch.no_grad()
    def _batch_knn_accuracy(logits: Tensor, labels: Tensor,
                            self_mask: Tensor) -> dict:
        return batch_knn_accuracy(logits, labels, self_mask)
