from typing import Any, Dict

import torch
import pytorch_lightning as pl

from Models import DinoV2LoRA


class ContrastiveExperiment(pl.LightningModule):
    """LightningModule wrapping the DINOv2+LoRA model for supervised contrastive
    (InfoNCE / SupCon) training on synthesis-program labels.

    Only the LoRA adapters and the projection head are optimized; the DINOv2
    backbone stays frozen.

    Args:
        model: the :class:`DinoV2LoRA` model to train.
        lr: learning rate for the AdamW optimizer.
        weight_decay: L2 weight decay for the optimizer.
        temperature: softmax temperature for the contrastive loss.
        scheduler_gamma: multiplicative LR decay per epoch (None to disable).
    """

    def __init__(self,
                 model: DinoV2LoRA,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 temperature: float = 0.1,
                 scheduler_gamma: float = 0.95,
                 scheduler: str = "exponential",
                 warmup_epochs: int = 0,
                 max_epochs: int = 100) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any) -> Dict[str, torch.Tensor]:
        images, labels = batch
        embeddings = self.model(images)
        return self.model.loss_function(
            embeddings, labels, temperature=self.temperature)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss_dict["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_gamma is None and self.scheduler_type == "exponential":
            return optimizer

        if self.scheduler_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-7
            )
        else:
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.scheduler_gamma
            )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, main_scheduler],
                milestones=[self.warmup_epochs]
            )
        else:
            scheduler = main_scheduler

        return [optimizer], [scheduler]
