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
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

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

# ImageNet normalization stats for DINOv2-based models (embedding models that
# expect normalized inputs rather than raw [0, 1] pixels).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helper
# ═══════════════════════════════════════════════════════════════════════════════

class _ImagePathDataset(Dataset):
    """Dataset that loads images by relative path for use with DataLoader workers."""

    def __init__(self, rel_paths: List[str], root_dir: Path, mode: ImageReadMode, transform: T.Compose):
        self.rel_paths = rel_paths
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        full_path = self.root_dir / self.rel_paths[idx]
        if not full_path.exists():
            return None
        img = read_image(str(full_path), mode=self.mode)
        return self.transform(img)


def _collate_skip_none(batch):
    """Collate that drops None entries (missing files)."""
    imgs = [x for x in batch if x is not None]
    if not imgs:
        return None
    return torch.stack(imgs, dim=0)


_NUM_WORKERS = min(8, os.cpu_count() or 1)


@torch.no_grad()
def _encode_paths(
    rel_paths: List[str],
    root_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    mode: ImageReadMode,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> torch.Tensor:
    """Encode a list of image paths to a (N, D) float32 CPU tensor of latent means."""
    ds = _ImagePathDataset(rel_paths, root_dir, mode, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_skip_none,
        pin_memory=num_workers > 0,
        persistent_workers=False,
    )
    latents: List[torch.Tensor] = []
    for batch in tqdm(loader, desc="Encoding batches"):
        if batch is None:
            continue
        batch = batch.to(device, non_blocking=True)
        # VAE encoders return (mu, log_var); embedding models (e.g. DINOv2+LoRA)
        # return a single feature tensor. Support both.
        out = model.encode(batch)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        latents.append(feats.cpu())
    return torch.cat(latents, dim=0) if latents else torch.empty(0)


def _encode_all_compounds(
    metadata: List[Dict],
    root_dir: Path,
    model: torch.nn.Module,
    img_size: int,
    in_channels: int,
    batch_size: int,
    device: torch.device,
    normalize_imagenet: bool = False,
) -> Dict:
    """Encode all compounds from metadata JSON into the embeddings dict format.

    All images are collected and encoded in a single DataLoader pass to avoid
    the overhead of spawning workers per compound.
    """
    tfm = [
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ]
    if normalize_imagenet:
        tfm.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = T.Compose(tfm)
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB

    # Collect all image paths with their (compound, plate, subset) keys.
    all_paths: List[str] = []
    # Each entry: (compound_id, plate_id, subset, start_idx, count)
    index_map: List[tuple] = []

    for entry in metadata:
        compound_id = str(entry["Compound"])
        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            for subset in ("treated", "control"):
                paths = plate_data.get(subset, [])
                if paths:
                    start = len(all_paths)
                    all_paths.extend(paths)
                    index_map.append((compound_id, str(plate_id), subset, start, len(paths)))

    if not all_paths:
        return {}

    # Filter to only existing paths so we can track exact positions.
    valid_indices: List[int] = []
    valid_paths: List[str] = []
    for i, p in enumerate(all_paths):
        if (root_dir / p).exists():
            valid_indices.append(i)
            valid_paths.append(p)

    print(
        f"  [ClassifierCallback] Encoding {len(valid_paths)}/{len(all_paths)} images "
        f"in a single pass ...", flush=True,
    )

    # Encode all valid images in one DataLoader pass with workers.
    latent_tensors = _encode_paths(
        valid_paths, root_dir, model, transform, mode, batch_size, device,
        num_workers=_NUM_WORKERS,
    )

    # Map encoded positions back to original flat indices.
    latent_by_idx: Dict[int, torch.Tensor] = {}
    for j, orig_idx in enumerate(valid_indices):
        latent_by_idx[orig_idx] = latent_tensors[j]

    # Scatter results back into the compound/plate/subset structure.
    embeddings: Dict = {}
    for compound_id, plate_id, subset, start, count in index_map:
        feats_list = [latent_by_idx[i] for i in range(start, start + count)
                      if i in latent_by_idx]
        if not feats_list:
            continue
        feats = torch.stack(feats_list)
        plate_dict = embeddings.setdefault(compound_id, {})
        plate_entry = plate_dict.setdefault(plate_id, {})
        if subset == "treated":
            plate_entry["treated"] = feats
        else:
            plate_entry["control"] = feats.mean(dim=0)

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
        normalize_after_subtract: L2-normalize after subtraction.
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
        normalize_after_subtract: bool = False,
        min_compounds_per_class: int = 30,
        test_split: float = 0.2,
        filter_by_efficacy: Optional[float] = 0,
        img_size: int = 96,
        in_channels: int = 3,
        batch_size: int = 64,
        cb_iterations: int = 300,
        cb_depth: int = 5,
        seed: int = 42,
        output_dir: str = "results",
        ckpt_subdir: str = "",
        normalize_imagenet: bool = False,
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
        self.normalize_after_subtract = normalize_after_subtract
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
        self.ckpt_subdir = ckpt_subdir
        self.normalize_imagenet = normalize_imagenet

        # Pre-load static data once
        self._metadata: Optional[List[Dict]] = None
        self._df: Optional[pd.DataFrame] = None
        self._logging_verified: bool = False

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

    def _get_valid_compound_ids(self) -> Optional[set]:
        """Return the set of compound IDs belonging to classes with enough members.

        If min_compounds_per_class <= 1, returns None (no pre-filtering).
        """
        if self._df is None or self._df.empty:
            return None
        min_cpc = max(self.min_compounds_per_class, 2)
        # Count unique compounds per class (not rows, which may have duplicates)
        class_counts = self._df.groupby(self.label_col)[self.compound_col].nunique()
        valid_classes = set(class_counts[class_counts >= min_cpc].index)
        if not valid_classes:
            return None
        valid_df = self._df[self._df[self.label_col].isin(valid_classes)]
        return set(valid_df[self.compound_col].astype(str).unique())

    def _filter_metadata(self, metadata: List[Dict]) -> List[Dict]:
        """Filter metadata to only include compounds from valid classes."""
        valid_ids = self._get_valid_compound_ids()
        if valid_ids is None:
            return metadata
        filtered = [e for e in metadata if str(e["Compound"]) in valid_ids]
        n_removed = len(metadata) - len(filtered)
        if n_removed > 0:
            print(
                f"  [ClassifierCallback] Pre-filtered metadata: kept {len(filtered)}/{len(metadata)} "
                f"compounds (removed {n_removed} from rare classes with <{self.min_compounds_per_class} members)",
                flush=True,
            )
        return filtered

    def _run_logging_smoke_test(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run a quick classifier on 1% of data at epoch 0 to verify logging works."""
        self._load_data()
        if self._metadata is None or self._df is None or self._df.empty:
            return

        # Pre-filter to only compounds in valid classes, then take 1% for speed
        filtered_metadata = self._filter_metadata(self._metadata)
        subset_size = max(1, len(filtered_metadata) // 10)
        metadata_subset = filtered_metadata[:subset_size]

        model = pl_module.model
        model.eval()
        device = pl_module.device

        embeddings = _encode_all_compounds(
            metadata=metadata_subset,
            root_dir=self.root_dir,
            model=model,
            img_size=self.img_size,
            in_channels=self.in_channels,
            batch_size=self.batch_size,
            device=device,
            normalize_imagenet=self.normalize_imagenet,
        )

        if not embeddings:
            print("  [ClassifierCallback] Smoke test: no embeddings produced, skipping.", flush=True)
            return

        str2idx, classes = build_label_encoder(self._df[self.label_col])
        X, y, cids = build_mean_latent_features(
            embeddings=embeddings,
            compound_col=self._df[self.compound_col],
            label_col=self._df[self.label_col],
            label2idx=str2idx,
            subtract_control=self.subtract_control,
            normalize_before_subtract=self.normalize_before_subtract,
            normalize_after_subtract=self.normalize_after_subtract,
        )

        if X.shape[0] < 4:
            print(
                f"  [ClassifierCallback] Smoke test: only {X.shape[0]} compounds "
                f"from 1% subset, need >=4. Skipping.", flush=True,
            )
            return

        X, y, cids, classes, num_classes = filter_rare_classes_array(
            X, y, cids, classes, self.min_compounds_per_class,
        )

        if num_classes < 2:
            print("  [ClassifierCallback] Smoke test: fewer than 2 classes, skipping.", flush=True)
            return

        # Quick train/test split
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=strat,
        )

        clf = CatBoostClassifier(
            iterations=50, depth=3, learning_rate=0.1,
            auto_class_weights="Balanced",
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            random_seed=self.seed, verbose=0,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test).astype(int).ravel()

        smoke_metrics = {
            "cls_test/smoke_top1_accuracy": accuracy_score(y_test, preds),
            "cls_test/smoke_balanced_accuracy": balanced_accuracy_score(y_test, preds),
        }

        # Attempt to log — this validates that the logging pipeline works
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            try:
                experiment = trainer.logger.experiment
                if experiment is not None:
                    experiment.log(smoke_metrics, commit=True)
                self._logging_verified = True
                print(
                    f"  [ClassifierCallback] Smoke test PASSED — logging works. "
                    f"(acc={smoke_metrics['cls_test/smoke_top1_accuracy']:.3f}, "
                    f"bal_acc={smoke_metrics['cls_test/smoke_balanced_accuracy']:.3f}, "
                    f"{num_classes} classes, {X.shape[0]} compounds from 1% subset)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"  [ClassifierCallback] Smoke test FAILED — logging error: {e}",
                    flush=True,
                )
        else:
            print("  [ClassifierCallback] Smoke test: no logger attached to trainer.", flush=True)

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

        if not _HAS_CATBOOST:
            return

        # On epoch 0, run a quick smoke test on 10% of data to verify logging
        if current_epoch == 0:
            if not self._logging_verified:
                self._run_logging_smoke_test(trainer, pl_module)
            return

        # Only run every N epochs
        if current_epoch % self.eval_every_n_epochs != 0:
            return

        self._load_data()
        if self._metadata is None or self._df is None or self._df.empty:
            return

        # ── Encode only compounds from valid classes ─────────────────────────
        model = pl_module.model
        model.eval()
        device = pl_module.device

        filtered_metadata = self._filter_metadata(self._metadata)
        if not filtered_metadata:
            return

        embeddings = _encode_all_compounds(
            metadata=filtered_metadata,
            root_dir=self.root_dir,
            model=model,
            img_size=self.img_size,
            in_channels=self.in_channels,
            batch_size=self.batch_size,
            device=device,
            normalize_imagenet=self.normalize_imagenet,
        )

        if not embeddings:
            return

        # ── Build features and train classifier ──────────────────────────────
        # Evaluate FOUR variants simultaneously:
        #   1. Raw treated means (no subtraction, no normalization)
        #   2. Control-subtracted treated means
        #   3. L2-normalized treated means
        #   4. Control-subtracted then L2-normalized treated means
        str2idx, classes = build_label_encoder(self._df[self.label_col])

        # (subtract_control, normalize_output, prefix, description)
        variants = [
            (False, False, "cls_test/nosub_", "no control subtraction"),
            (True, False, "cls_test/ctrl_", "control subtracted"),
            (False, True, "cls_test/norm_", "normalized embeddings"),
            (True, True, "cls_test/norm_ctrl_", "normalized control-subtracted"),
        ]

        all_metrics: Dict[str, float] = {}
        wandb_images: Dict[str, object] = {}
        results_by_key: Dict[str, Dict] = {}

        for subtract_flag, normalize_flag, prefix, desc in variants:
            variant_tag = prefix.split("/")[-1].rstrip("_")
            res = self._evaluate_classifier_variant(
                embeddings=embeddings,
                str2idx=str2idx,
                classes=classes,
                subtract_control=subtract_flag,
                normalize_output=normalize_flag,
                current_epoch=current_epoch,
                variant_desc=desc,
                variant_tag=variant_tag,
            )
            if res is None:
                continue
            results_by_key[variant_tag] = res
            # Log only balanced_accuracy per variant (4 total)
            all_metrics[prefix + "balanced_accuracy"] = res["bare_metrics"]["balanced_accuracy"]
            if res["fig"] is not None:
                wandb_images[prefix + "confusion_matrix"] = res["fig"]

        if not results_by_key:
            return

        # ── Log metrics + confusion matrices to W&B ──────────────────────────
        # Use wandb.log() directly — trainer.logger.log_metrics() buffers
        # internally and may never flush when called from a callback in
        # Lightning 2.x.
        try:
            import wandb
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                experiment = trainer.logger.experiment
                if experiment is not None:
                    log_payload = dict(all_metrics)
                    for key, fig in wandb_images.items():
                        log_payload[key] = wandb.Image(fig)
                    experiment.log(log_payload, commit=True)
        except (ImportError, AttributeError) as e:
            print(f"  [ClassifierCallback] W&B logging failed: {e}", flush=True)

        for res in results_by_key.values():
            if res["fig"] is not None:
                plt.close(res["fig"])

        for subtract_flag, normalize_flag, prefix, desc in variants:
            variant_tag = prefix.split("/")[-1].rstrip("_")
            res = results_by_key.get(variant_tag)
            if res is None:
                continue
            m = res["bare_metrics"]
            print(
                f"\n  [ClassifierCallback] Epoch {current_epoch} ({desc}): "
                f"balanced_acc={m['balanced_accuracy']:.3f}  "
                f"({int(m['num_classes'])} classes, {int(m['num_compounds'])} compounds)"
                f"  | confusion matrix -> {res['cm_path']}",
                flush=True,
            )
        print("", flush=True)

    def _evaluate_classifier_variant(
        self,
        embeddings: Dict,
        str2idx: Dict[str, int],
        classes: List[str],
        subtract_control: bool,
        normalize_output: bool,
        current_epoch: int,
        variant_desc: str,
        variant_tag: str,
    ) -> Optional[Dict]:
        """Build per-compound features (optionally control-subtracted and/or
        L2-normalized), train a CatBoost classifier on a train split, evaluate
        on a held-out test split, and render a confusion matrix.

        Returns a dict with ``bare_metrics`` (unprefixed metric-name -> value),
        the matplotlib ``fig`` for the confusion matrix, and ``cm_path``. Returns
        ``None`` when there are too few compounds/classes to evaluate.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            balanced_accuracy_score,
            confusion_matrix, ConfusionMatrixDisplay,
        )

        X, y, cids = build_mean_latent_features(
            embeddings=embeddings,
            compound_col=self._df[self.compound_col],
            label_col=self._df[self.label_col],
            label2idx=str2idx,
            subtract_control=subtract_control,
            normalize_before_subtract=self.normalize_before_subtract,
            normalize_after_subtract=self.normalize_after_subtract,
        )

        # Optionally L2-normalize the per-compound feature vectors.
        if normalize_output and X.shape[0] > 0:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(norms, 1e-8, None)

        if X.shape[0] < 10:
            return None

        X, y, cids, variant_classes, num_classes = filter_rare_classes_array(
            X, y, cids, list(classes), self.min_compounds_per_class,
        )

        if num_classes < 2 or X.shape[0] < 10:
            return None

        # ── Train/test split ─────────────────────────────────────────────────
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

        bare_metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "num_classes": float(num_classes),
            "num_compounds": float(X.shape[0]),
        }

        # ── Save confusion matrix ────────────────────────────────────────────
        cm = confusion_matrix(y_test, preds, labels=np.arange(num_classes))
        fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(8, num_classes * 0.5)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variant_classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=90)
        ax.set_title(f"Confusion Matrix — Epoch {current_epoch} ({variant_desc})")
        fig.tight_layout()

        cm_dir = self.output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        cm_path = cm_dir / f"confusion_matrix_{variant_tag}_epoch{current_epoch:04d}.png"
        fig.savefig(cm_path, dpi=150)

        return {"bare_metrics": bare_metrics, "fig": fig, "cm_path": cm_path}
