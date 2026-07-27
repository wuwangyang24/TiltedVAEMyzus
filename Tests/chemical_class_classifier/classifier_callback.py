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
    """Encode all compounds from metadata JSON into the embeddings dict format."""
    tfm = [
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ]
    # DINOv2-based models were trained on ImageNet-normalized inputs; match that
    # here so callback embeddings are consistent with training.
    if normalize_imagenet:
        tfm.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = T.Compose(tfm)
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB

    embeddings = {}
    for entry in tqdm(metadata, desc="  [ClassifierCallback] Encoding compounds", leave=False):
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
        # Best balanced accuracy tracked independently per variant tag
        # (e.g. "nosub", "ctrl") so each gets its own best checkpoint/embeddings.
        self._best_balanced_acc: Dict[str, float] = {}
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
        subset_size = max(1, len(filtered_metadata) // 100)
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
        # Evaluate BOTH variants simultaneously: without control subtraction
        # (raw treated means) and with per-plate control subtraction. This lets
        # us compare how much the control-subtraction step helps class
        # separability at every evaluation epoch.
        str2idx, classes = build_label_encoder(self._df[self.label_col])

        variants = [
            (False, "cls_test/nosub_", "no control subtraction"),
            (True, "cls_test/ctrl_", "control subtracted"),
        ]

        all_metrics: Dict[str, float] = {}
        wandb_images: Dict[str, object] = {}
        results_by_flag: Dict[bool, Dict] = {}

        for subtract_flag, prefix, desc in variants:
            res = self._evaluate_classifier_variant(
                embeddings=embeddings,
                str2idx=str2idx,
                classes=classes,
                subtract_control=subtract_flag,
                current_epoch=current_epoch,
                variant_desc=desc,
                variant_tag=prefix.split("/")[-1].rstrip("_"),
            )
            if res is None:
                continue
            results_by_flag[subtract_flag] = res
            for name, val in res["bare_metrics"].items():
                all_metrics[prefix + name] = val
            if res["fig"] is not None:
                wandb_images[prefix + "confusion_matrix"] = res["fig"]

        if not results_by_flag:
            return

        # ── Backward-compatible `cls_test/` metrics mirror the configured
        #    self.subtract_control variant (falls back to any available one). ──
        primary = results_by_flag.get(
            self.subtract_control, next(iter(results_by_flag.values()))
        )
        for name, val in primary["bare_metrics"].items():
            all_metrics["cls_test/" + name] = val

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

        for res in results_by_flag.values():
            if res["fig"] is not None:
                plt.close(res["fig"])

        for subtract_flag, prefix, desc in variants:
            res = results_by_flag.get(subtract_flag)
            if res is None:
                continue
            m = res["bare_metrics"]
            print(
                f"\n  [ClassifierCallback] Epoch {current_epoch} ({desc}): "
                f"top1_acc={m['top1_accuracy']:.3f}  "
                f"balanced_acc={m['balanced_accuracy']:.3f}  "
                f"weighted_f1={m['weighted_f1']:.3f}  "
                f"({int(m['num_classes'])} classes, {int(m['num_compounds'])} compounds)"
                f"  | confusion matrix -> {res['cm_path']}",
                flush=True,
            )
        print("", flush=True)

        # ── Save best checkpoint + embeddings independently per variant ──────
        # Each variant may peak at a different epoch, so track its own best
        # balanced accuracy and write to a variant-specific sub-directory.
        for subtract_flag, prefix, desc in variants:
            res = results_by_flag.get(subtract_flag)
            if res is None:
                continue
            variant_tag = prefix.split("/")[-1].rstrip("_")  # "nosub" / "ctrl"
            balanced_acc = res["bare_metrics"]["balanced_accuracy"]
            if balanced_acc <= self._best_balanced_acc.get(variant_tag, 0.0):
                continue
            self._best_balanced_acc[variant_tag] = balanced_acc
            ckpt_dir = self.output_dir / "checkpoints" / self.ckpt_subdir / variant_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best_balanced_acc.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            emb_path = ckpt_dir / "embeddings_best_balanced_acc.pt"
            torch.save(embeddings, emb_path)
            print(
                f"  [ClassifierCallback] New best balanced_acc={balanced_acc:.3f} "
                f"({desc}) — saved checkpoint to {ckpt_path}\n"
                f"  [ClassifierCallback] Saved embeddings to {emb_path}\n",
                flush=True,
            )

    def _evaluate_classifier_variant(
        self,
        embeddings: Dict,
        str2idx: Dict[str, int],
        classes: List[str],
        subtract_control: bool,
        current_epoch: int,
        variant_desc: str,
        variant_tag: str,
    ) -> Optional[Dict]:
        """Build per-compound features (optionally control-subtracted), train a
        CatBoost classifier on a train split, evaluate on a held-out test split,
        and render a confusion matrix.

        Returns a dict with ``bare_metrics`` (unprefixed metric-name -> value),
        the matplotlib ``fig`` for the confusion matrix, and ``cm_path``. Returns
        ``None`` when there are too few compounds/classes to evaluate.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            balanced_accuracy_score, f1_score, accuracy_score,
            top_k_accuracy_score as topk_acc,
            confusion_matrix, ConfusionMatrixDisplay,
        )

        X, y, cids = build_mean_latent_features(
            embeddings=embeddings,
            compound_col=self._df[self.compound_col],
            label_col=self._df[self.label_col],
            label2idx=str2idx,
            subtract_control=subtract_control,
            normalize_before_subtract=self.normalize_before_subtract,
        )

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
        probs = clf.predict_proba(X_test)

        bare_metrics = {
            "top1_accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, preds, average="weighted", zero_division=0),
            "num_classes": float(num_classes),
            "num_compounds": float(X.shape[0]),
        }

        # Top-k accuracy (only meaningful when k < num_classes)
        for k in (3, 5):
            if k < num_classes:
                bare_metrics[f"top{k}_accuracy"] = topk_acc(
                    y_test, probs, k=k, labels=np.arange(num_classes),
                )

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
