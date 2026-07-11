import math
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
        anneal_kld: enable sigmoid annealing of the KL weight over training
            steps (ramps the effective weight from ~0 up to ``kld_weight``).
        anneal_k: steepness of the sigmoid annealing schedule.
        anneal_x0: global step at which the sigmoid schedule reaches its
            midpoint (half of the target KL weight).
        anneal_end: final KL weight the sigmoid schedule saturates to when
            annealing is enabled (e.g. 1.0).
        au_threshold: variance threshold on the aggregated posterior mean
            used to count active units (AU).
    """

    def __init__(self,
                 model: VAE,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 kld_weight: float = 0.005,
                 scheduler_gamma: float = 0.95,
                 num_samples: int = 16,
                 anneal_kld: bool = False,
                 anneal_k: float = 0.0025,
                 anneal_x0: int = 2500,
                 anneal_end: float = 1.0,
                 au_threshold: float = 0.01) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.kld_weight = kld_weight
        self.scheduler_gamma = scheduler_gamma
        self.num_samples = num_samples
        self.anneal_kld = anneal_kld
        self.anneal_k = anneal_k
        self.anneal_x0 = anneal_x0
        self.anneal_end = anneal_end
        self.au_threshold = au_threshold
        # Running sufficient statistics for epoch-level latent metrics
        # (KL per dim, AU), aggregated across DDP ranks at epoch end.
        self._val_mu_sum: torch.Tensor = None
        self._val_mu_sqsum: torch.Tensor = None
        self._val_kld_sum: torch.Tensor = None
        self._val_count: int = 0
        # Fixed batch (on CPU) reused for image logging to avoid rebuilding the
        # val DataLoader every epoch.
        self._log_images_batch: torch.Tensor = None
        # Save hyperparameters (excluding the model object) for reproducibility.
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return self.model(x, **kwargs)

    def _kld_weight(self) -> float:
        """Current KL weight. When annealing is disabled, returns the fixed
        ``kld_weight``. When enabled, follows a sigmoid schedule that ramps
        from ~0 up to ``anneal_end`` (e.g. 1.0).

        Note: no world-size scaling is applied. The loss uses mean reductions
        and DDP averages gradients, so the recon:KL ratio (= kld_weight) is
        invariant to the number of GPUs.
        """
        if not self.anneal_kld:
            return self.kld_weight
        factor = 1.0 / (1.0 + math.exp(-self.anneal_k * (self.global_step - self.anneal_x0)))
        return self.anneal_end * factor

    def _step(self, batch: Any, kld_weight: float):
        images, _ = batch
        results = self.model(images)  # [recons, input, mu, log_var]
        loss_dict = self.model.loss_function(*results, M_N=kld_weight)
        mu = results[2]
        return loss_dict, mu

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        kld_weight = self._kld_weight()
        loss_dict, _ = self._step(batch, kld_weight)
        kld_per_dim = loss_dict.pop("KLD_per_dim")
        # Per-step train metrics: skip sync_dist to avoid an all-reduce every
        # step under DDP (these are monitoring-only).
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=True,
        )
        # Mean KL contribution per latent dimension.
        self.log("train_KLD_per_dim_mean", kld_per_dim.mean(),
                 on_step=True, on_epoch=True)
        self.log("kld_weight", kld_weight, on_step=True, on_epoch=False)
        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        self._val_mu_sum = None
        self._val_mu_sqsum = None
        self._val_kld_sum = None
        self._val_count = 0

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Validate against the full target KL weight for a comparable metric.
        loss_dict, mu = self._step(batch, self.kld_weight)
        kld_per_dim = loss_dict.pop("KLD_per_dim")
        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        # Accumulate fixed-size running statistics (robust to uneven shard
        # sizes and cheap to all-gather across DDP ranks).
        bs = mu.size(0)
        mu_sum = mu.sum(dim=0)
        mu_sqsum = (mu * mu).sum(dim=0)
        kld_batch_sum = kld_per_dim * bs  # kld_per_dim is a batch mean
        if self._val_mu_sum is None:
            self._val_mu_sum = mu_sum
            self._val_mu_sqsum = mu_sqsum
            self._val_kld_sum = kld_batch_sum
        else:
            self._val_mu_sum += mu_sum
            self._val_mu_sqsum += mu_sqsum
            self._val_kld_sum += kld_batch_sum
        self._val_count += bs
        return loss_dict["loss"]

    def on_validation_epoch_end(self) -> None:
        self._log_latent_metrics()
        self._log_images()

    @torch.no_grad()
    def _log_latent_metrics(self) -> None:
        """Log total KL, KL per latent dimension, and the number of active
        units (AU) aggregated over the full validation set (across DDP ranks).

        Active units follow Burda et al. (2016): a latent dimension is
        considered active if the variance of its posterior mean across the
        dataset exceeds ``au_threshold`` (default 0.01).
        """
        if self._val_mu_sum is None:
            return

        count = torch.tensor(float(self._val_count), device=self.device)
        mu_sum = self._val_mu_sum
        mu_sqsum = self._val_mu_sqsum
        kld_sum = self._val_kld_sum

        # Aggregate running statistics across DDP ranks (no-op on single GPU).
        count = self.all_gather(count).sum()
        mu_sum = self.all_gather(mu_sum).sum(dim=0)
        mu_sqsum = self.all_gather(mu_sqsum).sum(dim=0)
        kld_sum = self.all_gather(kld_sum).sum(dim=0)

        mean = mu_sum / count
        au_variance = mu_sqsum / count - mean ** 2      # population variance [D]
        active_units = int((au_variance > self.au_threshold).sum().item())
        kld_per_dim = kld_sum / count                   # [D]

        # Values are identical on all ranks after all_gather -> no sync needed.
        self.log("val_active_units", float(active_units))
        self.log("val_total_KLD", kld_per_dim.sum())
        self.log("val_KLD_per_dim_mean", kld_per_dim.mean())

        if self.trainer.is_global_zero and hasattr(self.logger, "experiment"):
            try:
                import wandb
                self.logger.experiment.log({
                    "val_KLD_per_dim_hist": wandb.Histogram(kld_per_dim.cpu().numpy()),
                    "val_AU_variance_hist": wandb.Histogram(au_variance.cpu().numpy()),
                    "global_step": self.global_step,
                })
            except ImportError:
                pass

        self._val_mu_sum = None
        self._val_mu_sqsum = None
        self._val_kld_sum = None
        self._val_count = 0

    @torch.no_grad()
    def _log_images(self) -> None:
        # Only log images when using the W&B logger.
        if not hasattr(self.logger, "experiment"):
            return

        # Fetch a fixed batch once and reuse it, so we don't rebuild the val
        # DataLoader (and re-spawn workers) on every validation epoch.
        if self._log_images_batch is None:
            val_loader = self.trainer.datamodule.val_dataloader()
            images, _ = next(iter(val_loader))
            self._log_images_batch = images[: self.num_samples].clone()
        images = self._log_images_batch.to(self.device)

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
