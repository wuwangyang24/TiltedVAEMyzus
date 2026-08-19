from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
                 max_epochs: int = 100,
                 contrastive_sigreg_loss: bool = False,
                 dcl_sigreg_loss: bool = False,
                 dcl_soft_pos_loss: bool = False,
                 vanilla_dcl: bool = False,
                 infonce_softpos: bool = False,
                 supcon_softpos: bool = False,
                 vanilla_supcon: bool = False,
                 pos_weight_tau: float = 0.1,
                 supcon_soft_pos_tau: float = 0.1,
                 denom_pos_weight: bool = False,
                 tau_annealing: bool = False,
                 supcon_tau_start: float = 0.1,
                 supcon_tau_end: float = 0.1,
                 no_pos_weight_epoch: int = 0,
                 sinkhorn: bool = False,
                 sinkhorn_iters: int = 5,
                 sigreg_weight: float = 0.1,
                 sigreg_slices: int = 512) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.contrastive_sigreg_loss = contrastive_sigreg_loss
        self.dcl_sigreg_loss = dcl_sigreg_loss
        self.dcl_soft_pos_loss = dcl_soft_pos_loss
        self.vanilla_dcl = vanilla_dcl
        self.infonce_softpos = infonce_softpos
        self.supcon_softpos = supcon_softpos
        self.vanilla_supcon = vanilla_supcon
        self.pos_weight_tau = pos_weight_tau
        self.supcon_soft_pos_tau = supcon_soft_pos_tau
        self.denom_pos_weight = denom_pos_weight
        self.tau_annealing = tau_annealing
        self.supcon_tau_start = supcon_tau_start
        self.supcon_tau_end = supcon_tau_end
        if no_pos_weight_epoch < 0:
            raise ValueError("no_pos_weight_epoch must be non-negative")
        self.no_pos_weight_epoch = no_pos_weight_epoch
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _current_supcon_tau(self) -> float:
        if not self.tau_annealing:
            return self.supcon_soft_pos_tau
        weighted_epoch = max(self.current_epoch - self.no_pos_weight_epoch, 0)
        weighted_epochs = max(self.max_epochs - self.no_pos_weight_epoch, 1)
        progress = min(weighted_epoch / max(weighted_epochs - 1, 1), 1.0)
        return self.supcon_tau_start + (
            self.supcon_tau_end - self.supcon_tau_start
        ) * progress

    def _use_pos_weighting(self) -> bool:
        return self.current_epoch >= self.no_pos_weight_epoch

    def _step(self, batch: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Support both (images, labels) and (images, train_labels, test_labels)
        if len(batch) == 3:
            images, labels, test_labels = batch
        else:
            images, labels = batch
            test_labels = None

        if self.dcl_soft_pos_loss:
            embeddings = self.model(images, normalize=False)
            loss_dict = self.model.dcl_soft_pos_loss_function(
                embeddings, labels, temperature=self.temperature,
                sigreg_weight=self.sigreg_weight,
                sigreg_slices=self.sigreg_slices,
                test_labels=test_labels)
        elif self.dcl_sigreg_loss:
            embeddings = self.model(images, normalize=False)
            loss_dict = self.model.dcl_sigreg_loss_function(
                embeddings, labels, temperature=self.temperature,
                sigreg_weight=self.sigreg_weight,
                sigreg_slices=self.sigreg_slices,
                test_labels=test_labels)
        elif self.contrastive_sigreg_loss:
            embeddings = self.model(images, normalize=False)
            loss_dict = self.model.contrastive_sigreg_loss_function(
                embeddings, labels, temperature=self.temperature,
                sigreg_weight=self.sigreg_weight,
                sigreg_slices=self.sigreg_slices)
        elif self.vanilla_dcl:
            embeddings = self.model(images)
            loss_dict = self.model.vanilla_dcl_loss_function(
                embeddings, labels, temperature=self.temperature)
        elif self.vanilla_supcon:
            embeddings = self.model(images)
            loss_dict = self.model.vanilla_supcon_loss_function(
                embeddings, labels, temperature=self.temperature)
        elif self.infonce_softpos:
            embeddings = self.model(images)
            loss_dict = self.model.infonce_softpos_loss_function(
                embeddings, labels, temperature=self.temperature,
                pos_weight_tau=self.pos_weight_tau,
                use_pos_weighting=self._use_pos_weighting(),
                denom_pos_weight=self.denom_pos_weight,
                sinkhorn=self.sinkhorn, sinkhorn_iters=self.sinkhorn_iters,
                test_labels=test_labels)
        elif self.supcon_softpos:
            embeddings = self.model(images, normalize=False)
            supcon_tau = self._current_supcon_tau()
            loss_dict = self.model.supcon_soft_pos_loss_function(
                embeddings, labels, temperature=self.temperature,
                pos_weight_tau=supcon_tau,
                use_pos_weighting=self._use_pos_weighting(),
                sigreg_weight=self.sigreg_weight,
                sigreg_slices=self.sigreg_slices,
                test_labels=test_labels)
        else:
            embeddings = self.model(images)
            loss_dict = self.model.loss_function(
                embeddings, labels, temperature=self.temperature)

        return loss_dict, embeddings, labels, test_labels

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict, _, _, _ = self._step(batch)
        if self.supcon_softpos:
            self.log("train_supcon_tau", self._current_supcon_tau(),
                     on_step=False, on_epoch=True)
        if self.supcon_softpos or self.infonce_softpos:
            self.log("train_pos_weight_active", float(self._use_pos_weighting()),
                     on_step=False, on_epoch=True)
        self.log(
            "train_loss", loss_dict["loss"],
            on_step=True, on_epoch=True, prog_bar=True,
        )
        for key in ("suspicion_mean", "suspicion_same_testcat", "suspicion_diff_testcat",
                    "pos_weight_same_testcat", "pos_weight_diff_testcat",
                    "pos_weight_testcat_ratio"):
            if key in loss_dict:
                self.log(f"train_{key}", loss_dict[key], on_step=True, on_epoch=True)
        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        self._val_embeddings = []
        self._val_train_labels = []
        self._val_test_labels = []

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict, embeddings, train_labels, test_labels = self._step(batch)
        self.log(
            "val_loss", loss_dict["loss"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        for key in ("pos_weight_same_testcat", "pos_weight_diff_testcat",
                    "pos_weight_testcat_ratio"):
            if key in loss_dict:
                self.log(f"val_{key}", loss_dict[key], on_step=False, on_epoch=True,
                         sync_dist=True)
        if test_labels is not None:
            self._val_embeddings.append(embeddings.detach().cpu())
            self._val_train_labels.append(train_labels.detach().view(-1).cpu())
            self._val_test_labels.append(test_labels.detach().view(-1).cpu())
        return loss_dict["loss"]

    def on_validation_epoch_end(self) -> None:
        if not getattr(self, "_val_embeddings", None):
            return

        val_embeddings = torch.cat(self._val_embeddings, dim=0)
        val_train_labels = torch.cat(self._val_train_labels, dim=0)
        val_test_labels = torch.cat(self._val_test_labels, dim=0)
        self._val_embeddings = []
        self._val_train_labels = []
        self._val_test_labels = []

        val_embeddings = self._gather_across_ranks(val_embeddings)
        val_train_labels = self._gather_across_ranks(val_train_labels)
        val_test_labels = self._gather_across_ranks(val_test_labels)

        device = self.device
        val_embeddings = val_embeddings.to(device)
        val_train_labels = val_train_labels.to(device)
        val_test_labels = val_test_labels.to(device)

        # KNN and linear probe on test_cat labels.
        knn = self._full_set_knn_accuracy(val_embeddings, val_test_labels)
        self.log_dict(
            {
                "val_knn_acc": knn[1],
                "val_knn_top3_acc": knn[3],
                "val_knn_top5_acc": knn[5],
            },
            prog_bar=True, sync_dist=False,
        )

        probe_test = self._linear_probe(val_embeddings, val_test_labels)
        self.log_dict(
            {
                "val_linprobe_top1": probe_test["top1_acc"],
                "val_linprobe_top5": probe_test["top5_acc"],
            },
            prog_bar=False, sync_dist=False,
        )

        # Linear probe on train_cat labels.
        probe_train = self._linear_probe(val_embeddings, val_train_labels)
        self.log_dict(
            {
                "val_linprobe_traincat_top1": probe_train["top1_acc"],
                "val_linprobe_traincat_top5": probe_train["top5_acc"],
            },
            prog_bar=False, sync_dist=False,
        )

    @staticmethod
    def _gather_across_ranks(tensor: torch.Tensor) -> torch.Tensor:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor
        gathered: list = [None] * world_size
        torch.distributed.all_gather_object(gathered, tensor.cpu())
        return torch.cat([t.to(tensor.device) for t in gathered], dim=0)

    @staticmethod
    @torch.no_grad()
    def _linear_probe(
        embeddings: torch.Tensor, labels: torch.Tensor,
        train_fraction: float = 0.8, seed: int = 42,
        lr: float = 0.1, epochs: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Train/test linear probe on the val embeddings."""
        n = embeddings.size(0)
        num_classes = int(labels.max().item()) + 1
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split = int(n * train_fraction)
        train_idx, test_idx = perm[:split], perm[split:]

        normed = F.normalize(embeddings, dim=1)
        train_e, train_l = normed[train_idx], labels[train_idx]
        test_e, test_l = normed[test_idx], labels[test_idx]

        dim = train_e.size(1)
        classifier = torch.nn.Linear(dim, num_classes, device=train_e.device)

        with torch.enable_grad():
            optimizer = torch.optim.LBFGS(classifier.parameters(), lr=lr, max_iter=20)
            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(classifier(train_e), train_l)
                loss.backward()
                return loss.detach()
            for _ in range(epochs):
                optimizer.step(closure)

        classifier.eval()
        logits = classifier(test_e)
        top1 = (logits.argmax(dim=1) == test_l).float().mean()
        k = min(5, num_classes)
        top5 = (logits.topk(k, dim=1).indices == test_l.unsqueeze(1)).any(dim=1).float().mean()
        return {"top1_acc": top1, "top5_acc": top5}

    @staticmethod
    @torch.no_grad()
    def _full_set_knn_accuracy(
        embeddings: torch.Tensor, labels: torch.Tensor,
        ks: Tuple[int, ...] = (1, 3, 5), chunk_size: int = 1024,
    ) -> Dict[int, torch.Tensor]:
        """Top-k KNN accuracy over the entire val set (cosine, leave-one-out).

        A sample is a top-k hit if any of its k nearest neighbours across the
        full set shares its label. Computed in row chunks to bound memory.
        """
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        n = embeddings.size(0)
        max_k = min(max(ks), n - 1)
        if max_k < 1:
            return {k: torch.tensor(1.0, device=embeddings.device) for k in ks}

        hits = {k: torch.zeros(n, dtype=torch.bool, device=embeddings.device) for k in ks}
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sim = embeddings[start:end] @ embeddings.t()       # (chunk, n)
            rows = torch.arange(end - start, device=embeddings.device)
            sim[rows, torch.arange(start, end, device=embeddings.device)] = float("-inf")
            topk_idx = sim.topk(max_k, dim=1).indices           # (chunk, max_k)
            match = labels[topk_idx] == labels[start:end].unsqueeze(1)
            for k in ks:
                hits[k][start:end] = match[:, :min(k, max_k)].any(dim=1)
        return {k: hits[k].float().mean() for k in ks}

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


class LeJEPAExperiment(pl.LightningModule):
    """LightningModule training the DINOv2+LoRA model with the LeJEPA
    self-supervised objective (Balestriero & LeCun, 2025).

    Instead of supervised contrastive learning, this optimizes a label-free
    loss: a prediction/invariance term over multiple augmented views plus the
    SIGReg isotropic-Gaussian regularizer that prevents representation collapse.
    Only the LoRA adapters and projection head are trained.

    Args:
        model: the :class:`DinoV2LoRA` model to train.
        lr: learning rate for the AdamW optimizer.
        weight_decay: L2 weight decay for the optimizer.
        sigreg_weight: weight of the SIGReg term relative to the prediction term.
        sigreg_slices: number of random 1-D projections used by SIGReg.
        sigreg_num_freqs: quadrature points for the Epps-Pulley integral.
        scheduler_gamma: multiplicative LR decay per epoch (None to disable).
        scheduler: 'exponential' or 'cosine'.
        warmup_epochs: linear LR warmup epochs before the main schedule.
        max_epochs: total training epochs (for the cosine schedule).
    """

    def __init__(self,
                 model: DinoV2LoRA,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 sigreg_weight: float = 0.05,
                 sigreg_slices: int = 512,
                 sigreg_num_freqs: int = 33,
                 scheduler_gamma: float = 0.95,
                 scheduler: str = "cosine",
                 warmup_epochs: int = 0,
                 max_epochs: int = 100) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.sigreg_num_freqs = sigreg_num_freqs
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any) -> Dict[str, torch.Tensor]:
        # views: (B, V, C, H, W); labels are unused by the LeJEPA objective.
        views, _ = batch
        b, v = views.shape[0], views.shape[1]
        flat = views.reshape(b * v, *views.shape[2:])
        # Raw (un-normalized) embeddings: SIGReg targets an isotropic Gaussian.
        emb = self.model(flat, normalize=False)             # (B*V, D)
        view_emb = emb.reshape(b, v, -1).permute(1, 0, 2)   # (V, B, D)
        return self.model.lejepa_loss_function(
            view_emb,
            sigreg_weight=self.sigreg_weight,
            sigreg_slices=self.sigreg_slices,
            sigreg_num_freqs=self.sigreg_num_freqs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log(
            "train_loss", loss_dict["loss"],
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss_dict["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict = self._step(batch)
        self.log(
            "val_loss", loss_dict["loss"],
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
