from typing import Any, Dict, List

import torch
import pytorch_lightning as pl
import torchvision.utils as vutils

from Models import VAE


class VAEExperiment(pl.LightningModule):
    """
    LightningModule that wraps the VAE model for training and validation.

    Handles the optimization step, loss logging, and periodic sampling /
    reconstruction of images for visual inspection (logged to W&B).

    Args:
        model: the VAE model to train
        lr: learning rate for the Adam optimizer
        weight_decay: L2 weight decay for the optimizer
        kld_weight: weight applied to the KL divergence term (M_N).
            Typically set to batch_size / dataset_size.
        scheduler_gamma: multiplicative LR decay per epoch (None to disable)
        num_samples: number of images to sample/reconstruct for logging
    """

    def __init__(self,
                 model: VAE,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 kld_weight: float = 0.005,
                 scheduler_gamma: float = 0.95,
                 num_samples: int = 16) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.kld_weight = kld_weight
        self.scheduler_gamma = scheduler_gamma
        self.num_samples = num_samples
        # Save hyperparameters (excluding the model object) for reproducibility.
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return self.model(x, **kwargs)

    def _step(self, batch: Any) -> Dict[str, torch.Tensor]:
        images, _ = batch
        results = self.model(images)
        loss_dict = self.model.loss_function(*results, M_N=self.kld_weight)
        return loss_dict

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss_dict["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss_dict["loss"]

    def on_validation_epoch_end(self) -> None:
        self._log_images()

    @torch.no_grad()
    def _log_images(self) -> None:
        # Only log images when using the W&B logger.
        if not hasattr(self.logger, "experiment"):
            return

        val_loader = self.trainer.datamodule.val_dataloader()
        images, _ = next(iter(val_loader))
        images = images[: self.num_samples].to(self.device)

        recons = self.model.generate(images)
        samples = self.model.sample(self.num_samples, self.device)

        recon_grid = vutils.make_grid(recons, nrow=4, normalize=True, value_range=(0, 1))
        input_grid = vutils.make_grid(images, nrow=4, normalize=True, value_range=(0, 1))
        sample_grid = vutils.make_grid(samples, nrow=4, normalize=True, value_range=(0, 1))

        try:
            import wandb
            self.logger.experiment.log({
                "inputs": wandb.Image(input_grid),
                "reconstructions": wandb.Image(recon_grid),
                "samples": wandb.Image(sample_grid),
                "global_step": self.global_step,
            })
        except ImportError:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_gamma is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]
