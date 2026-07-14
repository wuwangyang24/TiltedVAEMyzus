"""
classifier_callback.py

PyTorch Lightning Callback that evaluates the chemical-class classification
accuracy of the VAE latent space every N epochs during training.

At the end of a validation epoch (every ``eval_every_n_epochs``), the callback:
  1. Encodes all compound images from the metadata JSON using the current model.
  2. Builds per-compound mean latent features (optionally control-subtracted).
  3. Trains a CatBoost classifier on a train split and evaluates on a test split.
  4. Logs top-1 accuracy, balanced accuracy, and macro F1 to the trainer's logger.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

import pytorch_lightning as pl

# Ensure sibling modules are importable when the callback is used from the
# repo root (train.py).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from classifier_utils import (
    build_mean_latent_features,
    filter_rare_classes_array,
    build_label_encoder,
)

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helper
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _encode_paths(
    rel_paths: List[str],
    root_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    mode: ImageReadMode,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a list of image paths to a (N, D) float32 CPU tensor of latent means."""
    latents: List[torch.Tensor] = []
    for start in range(0, len(rel_paths), batch_size):
        batch_paths = rel_paths[start:start + batch_size]
        imgs = []
        for rel in batch_paths:
            full_path = root_dir / rel
            if not full_path.exists():
                continue
            img = read_image(str(full_path), mode=mode)
            imgs.append(transform(img))
        if not imgs:
            continue
        batch = torch.stack(imgs, dim=0).to(device)
        mu, _ = model.encode(batch)
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0) if latents else torch.empty(0)


def _encode_all_compounds(
    metadata: List[Dict],
    root_dir: Path,
    model: torch.nn.Module,
    img_size: int,
    in_channels: int,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """Encode all compounds from metadata JSON into the embeddings dict format."""
    transform = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ])
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB

    embeddings = {}
    for entry in tqdm(metadata, desc="[ClassifierCallback] Encoding compounds", miniters=5000, dynamic_miniters=False):
        compound_id = str(entry["Compound"])
        plate_dict = {}
        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            treated_paths = plate_data.get("treated", [])
            control_paths = plate_data.get("control", [])

            plate_entry = {}
            if treated_paths:
                plate_entry["treated"] = _encode_paths(
                    treated_paths, root_dir, model, transform, mode,
                    batch_size, device,
                )
            if control_paths:
                control_latents = _encode_paths(
                    control_paths, root_dir, model, transform, mode,
                    batch_size, device,
                )
                if control_latents.numel() > 0:
                    plate_entry["control"] = control_latents.mean(dim=0)

            if plate_entry:
                plate_dict[str(plate_id)] = plate_entry

        if plate_dict:
            embeddings[compound_id] = plate_dict

    return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Callback
# ═══════════════════════════════════════════════════════════════════════════════

class ChemicalClassClassifierCallback(pl.Callback):
    """Evaluate latent-space chemical-class separability during training.

    Args:
        image_metadata_json: path to the JSON file mapping compounds to
            plate/image paths (same format as encode_embeddings.py).
        label_metadata_csv: path to the CSV/Excel with compound labels.
        root_dir: base directory prepended to image paths in the JSON.
        eval_every_n_epochs: run the classifier every N validation epochs.
        compound_col: column name for compound IDs in the label CSV.
        label_col: column name for the class label in the label CSV.
        subtract_control: subtract per-plate averaged control embedding.
        normalize_before_subtract: L2-normalize before subtraction.
        min_compounds_per_class: drop classes with fewer compounds.
        test_split: fraction held out for evaluation.
        filter_by_efficacy: keep only compounds with Efficacy >= this value.
        img_size: image resize target (must match training).
        in_channels: number of image channels.
        batch_size: encoding batch size.
        cb_iterations: CatBoost boosting iterations.
        cb_depth: CatBoost tree depth.
        seed: random seed.
    """

    def __init__(
        self,
        image_metadata_json: str,
        label_metadata_csv: str,
        root_dir: str,
        eval_every_n_epochs: int = 5,
        compound_col: str = "compound",
        label_col: str = "synthesis_program",
        subtract_control: bool = False,
        normalize_before_subtract: bool = False,
        min_compounds_per_class: int = 2,
        test_split: float = 0.2,
        filter_by_efficacy: Optional[float] = 30.0,
        img_size: int = 96,
        in_channels: int = 3,
        batch_size: int = 64,
        cb_iterations: int = 300,
        cb_depth: int = 5,
        seed: int = 42,
        output_dir: str = "results",
    ):
        super().__init__()
        self.image_metadata_json = Path(image_metadata_json)
        self.label_metadata_csv = Path(label_metadata_csv)
        self.root_dir = Path(root_dir)
        self.eval_every_n_epochs = eval_every_n_epochs
        self.compound_col = compound_col
        self.label_col = label_col
        self.subtract_control = subtract_control
        self.normalize_before_subtract = normalize_before_subtract
        self.min_compounds_per_class = min_compounds_per_class
        self.test_split = test_split
        self.filter_by_efficacy = filter_by_efficacy
        self.img_size = img_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.cb_iterations = cb_iterations
        self.cb_depth = cb_depth
        self.seed = seed
        self.output_dir = Path(output_dir)

        # Pre-load static data once
        self._metadata: Optional[List[Dict]] = None
        self._df: Optional[pd.DataFrame] = None
        self._best_balanced_acc: float = 0.0

    def _load_data(self) -> None:
        """Load image metadata and label dataframe once."""
        if self._metadata is not None:
            return

        with open(self.image_metadata_json) as f:
            self._metadata = json.load(f)

        suffix = self.label_metadata_csv.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            self._df = pd.read_excel(self.label_metadata_csv)
        else:
            self._df = pd.read_csv(self.label_metadata_csv)

        if self.filter_by_efficacy is not None and "Efficacy" in self._df.columns:
            self._df = self._df[self._df["Efficacy"] >= self.filter_by_efficacy]

        self._df = self._df[[self.compound_col, self.label_col]].dropna()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        current_epoch = trainer.current_epoch
        print(
            f"  [ClassifierCallback] on_validation_epoch_end called "
            f"(epoch={current_epoch}, is_global_zero={trainer.is_global_zero}, "
            f"has_catboost={_HAS_CATBOOST})",
            flush=True,
        )

        # Only run on the main process
        if not trainer.is_global_zero:
            return

        # Only run every N epochs (epoch 0 is skipped to avoid noise)
        if current_epoch == 0:
            return
        if current_epoch % self.eval_every_n_epochs != 0:
            return

        if not _HAS_CATBOOST:
            return

        self._load_data()
        if self._metadata is None or self._df is None or self._df.empty:
            return

        # ── Encode all compounds with current model ──────────────────────────
        model = pl_module.model
        model.eval()
        device = pl_module.device

        embeddings = _encode_all_compounds(
            metadata=self._metadata,
            root_dir=self.root_dir,
            model=model,
            img_size=self.img_size,
            in_channels=self.in_channels,
            batch_size=self.batch_size,
            device=device,
        )

        if not embeddings:
            return

        # ── Build features and train classifier ──────────────────────────────
        str2idx, classes = build_label_encoder(self._df[self.label_col])

        X, y, cids = build_mean_latent_features(
            embeddings=embeddings,
            compound_col=self._df[self.compound_col],
            label_col=self._df[self.label_col],
            label2idx=str2idx,
            subtract_control=self.subtract_control,
            normalize_before_subtract=self.normalize_before_subtract,
        )

        if X.shape[0] < 10:
            return

        X, y, cids, classes, num_classes = filter_rare_classes_array(
            X, y, cids, classes, self.min_compounds_per_class,
        )

        if num_classes < 2 or X.shape[0] < 10:
            return

        # ── Train/test split ─────────────────────────────────────────────────
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            balanced_accuracy_score, f1_score, accuracy_score,
        )

        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_split,
            random_state=self.seed,
            stratify=strat,
        )

        # ── Train CatBoost ───────────────────────────────────────────────────
        clf = CatBoostClassifier(
            iterations=self.cb_iterations,
            depth=self.cb_depth,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            random_seed=self.seed,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────────
        preds = clf.predict(X_test).astype(int).ravel()
        probs = clf.predict_proba(X_test)

        top1_acc = accuracy_score(y_test, preds)
        balanced_acc = balanced_accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        from sklearn.metrics import top_k_accuracy_score as topk_acc

        metrics = {
            "cls_test/top1_accuracy": top1_acc,
            "cls_test/balanced_accuracy": balanced_acc,
            "cls_test/macro_f1": macro_f1,
            "cls_test/weighted_f1": weighted_f1,
            "cls_test/num_classes": float(num_classes),
            "cls_test/num_compounds": float(X.shape[0]),
        }

        # Top-k accuracy (only meaningful when k < num_classes)
        for k in (3, 5):
            if k < num_classes:
                topk = topk_acc(y_test, probs, k=k, labels=np.arange(num_classes))
                metrics[f"cls_test/top{k}_accuracy"] = topk

        # ── Log metrics ──────────────────────────────────────────────────────
        pl_module.log_dict(metrics, prog_bar=False, logger=True)

        # ── Save confusion matrix ────────────────────────────────────────────
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(y_test, preds, labels=np.arange(num_classes))
        fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(8, num_classes * 0.5)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=90)
        ax.set_title(f"Confusion Matrix — Epoch {current_epoch}")
        fig.tight_layout()

        cm_dir = self.output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        cm_path = cm_dir / f"confusion_matrix_epoch{current_epoch:04d}.png"
        fig.savefig(cm_path, dpi=150)

        # Log to W&B
        try:
            import wandb
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log({
                    "cls_test/confusion_matrix": wandb.Image(fig),
                    "global_step": trainer.global_step,
                })
        except ImportError:
            pass

        plt.close(fig)

        print(
            f"\n  [ClassifierCallback] Epoch {current_epoch}: "
            f"top1_acc={top1_acc:.3f}  balanced_acc={balanced_acc:.3f}  "
            f"weighted_f1={weighted_f1:.3f}  "
            f"({num_classes} classes, {X.shape[0]} compounds)"
            f"  | confusion matrix -> {cm_path}\n",
            flush=True,
        )

        # ── Save best checkpoint by balanced accuracy ────────────────────────
        if balanced_acc > self._best_balanced_acc:
            self._best_balanced_acc = balanced_acc
            ckpt_dir = self.output_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best_balanced_acc.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            print(
                f"  [ClassifierCallback] New best balanced_acc={balanced_acc:.3f} "
                f"— saved checkpoint to {ckpt_path}\n",
                flush=True,
            )
