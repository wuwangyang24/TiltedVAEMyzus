"""classifier_utils.py

Shared utilities for the CatBoost chemical-class classifier (ported from the
MyzusDINOAdapt synthesis-program classifier, trimmed to the CatBoost path):
  - Per-compound mean-latent feature builder
  - Rare-class filtering & label encoding
  - Result saving (top-1 / top-k reports, confusion matrices, predictions CSV)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report, confusion_matrix,
    top_k_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Feature builder
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along *dim*."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def build_mean_latent_features(
    embeddings: Dict,
    compound_col: pd.Series,
    label_col: pd.Series,
    label2idx: Dict[str, int],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a (num_compounds, D) feature matrix where each row is the mean
    of all treated latents for a compound (optionally control-subtracted).

    Returns
    -------
    X         : (N, D) float32 array
    y         : (N,) int array
    cids      : list of compound ID strings
    """
    comp2label: Dict[str, int] = {}
    for comp, prog in zip(compound_col, label_col):
        comp2label[str(comp)] = label2idx[str(prog)]

    X_rows, y_rows, cids = [], [], []

    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in comp2label:
            continue

        plate_latents: List[torch.Tensor] = []
        for plate_data in plates.values():
            treated = plate_data.get("treated")
            if treated is None or treated.numel() == 0:
                continue
            if subtract_control and "control" in plate_data:
                control = plate_data["control"]
                if normalize_before_subtract:
                    treated = _l2_normalize(treated)
                    control = _l2_normalize(control)
                treated = treated - control.unsqueeze(0)
            plate_latents.append(treated.float())

        if not plate_latents:
            continue

        all_latents = torch.cat(plate_latents, dim=0)       # (M, D)
        mean_latent = all_latents.mean(dim=0).numpy()       # (D,)
        X_rows.append(mean_latent)
        y_rows.append(comp2label[cid])
        cids.append(cid)

    return np.stack(X_rows), np.array(y_rows), cids


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. Class filtering & label remapping
# ═══════════════════════════════════════════════════════════════════════════════

def filter_rare_classes_array(
    X: np.ndarray,
    y: np.ndarray,
    cids: List[str],
    classes: List[str],
    min_compounds_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], int]:
    """Drop classes with fewer than *min_compounds_per_class* compounds
    and remap remaining labels to contiguous 0..K-1.

    Returns (X, y, cids, classes, num_classes).
    """
    min_cpc = max(min_compounds_per_class, 2)
    class_counts = np.bincount(y)
    valid_classes = set(np.where(class_counts >= min_cpc)[0])
    keep_mask = np.array([yi in valid_classes for yi in y])
    n_removed = len(y) - keep_mask.sum()
    if n_removed > 0:
        removed_names = sorted({classes[yi] for yi in y if yi not in valid_classes})
        print(f"  Dropped {n_removed} compound(s) from {len(removed_names)} "
              f"class(es) with <{min_cpc} compounds: {removed_names}")
        X, y, cids = X[keep_mask], y[keep_mask], [c for c, k in zip(cids, keep_mask) if k]

    remaining = sorted(set(y.tolist()))
    old2new = {old: new for new, old in enumerate(remaining)}
    y = np.array([old2new[yi] for yi in y])
    classes = [classes[old] for old in remaining]
    num_classes = len(classes)
    print(f"  {num_classes} classes after filtering, {len(y)} compounds remaining.")
    return X, y, cids, classes, num_classes


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Label encoding
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_encoder(series: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(series.astype(str).unique().tolist())
    str2idx = {c: i for i, c in enumerate(classes)}
    return str2idx, classes


def save_label_encoder(classes: List[str], str2idx: Dict[str, int], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "str2idx": str2idx}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Result saving
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    num_classes: int,
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(7, num_classes * 0.45)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=40)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(num_classes),
        yticks=range(num_classes),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _topk_predictions(val_probs: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k predicted class indices for each sample, shape (N, k)."""
    return np.argsort(val_probs, axis=1)[:, -k:][:, ::-1]


def _topk_confusion_matrix(
    val_true: np.ndarray,
    val_probs: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    """Build a confusion matrix where a prediction counts as correct if the
    true label is within the top-k predictions.  Off-diagonal entries show
    which class was predicted as #1 when the true label was NOT in top-k."""
    topk_idx = _topk_predictions(val_probs, k)  # (N, k)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, (true, top_classes) in enumerate(zip(val_true, topk_idx)):
        if true in top_classes:
            cm[true, true] += 1       # correct under top-k
        else:
            cm[true, top_classes[0]] += 1  # wrong: attribute to top-1 pred
    return cm


def save_results(
    val_true: np.ndarray,
    val_preds: np.ndarray,
    val_probs: np.ndarray,
    val_cids: List[str],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
    cm_title: str,
    file_suffix: str,
    report_header: str,
    save_predictions: bool,
    topk: Tuple[int, ...] = (1, 3, 5),
) -> None:
    """Save classification report, confusion matrix, top-k accuracies, and (optionally) predictions CSV."""
    val_acc = balanced_accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average="weighted", zero_division=0)

    # ── Top-k accuracies (always include top-1) ─────────────────────────────
    all_k = sorted(set((1,) + tuple(topk)))
    topk_results = {}
    for k in all_k:
        if k > num_classes:
            continue
        if k == 1:
            topk_results[k] = float((val_preds == val_true).mean())
        else:
            topk_results[k] = float(top_k_accuracy_score(
                val_true, val_probs, k=k, labels=list(range(num_classes)),
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # Top-1 report & confusion matrix
    # ══════════════════════════════════════════════════════════════════════════
    report_str = classification_report(
        val_true, val_preds,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
    print("\n-- Top-1 Classification Report --")
    print(report_str)
    print(f"Balanced accuracy : {val_acc:.4f}")
    print(f"Weighted F1       : {val_f1:.4f}")
    print(f"Top-1 accuracy    : {topk_results[1]:.4f}")

    report_path = output_dir / f"classification_report_top1{file_suffix}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_header)
        f.write("-- Top-1 Classification Report --\n\n")
        f.write(report_str)
        f.write(f"\nBalanced accuracy : {val_acc:.4f}\n")
        f.write(f"Weighted F1       : {val_f1:.4f}\n")
        f.write(f"Top-1 accuracy    : {topk_results[1]:.4f}\n")
    print(f"Report saved to    : {report_path}")

    cm = confusion_matrix(val_true, val_preds, labels=list(range(num_classes)))
    cm_path = output_dir / f"confusion_matrix_top1{file_suffix}.png"
    _plot_confusion_matrix(cm, classes, num_classes, f"{cm_title} (Top-1)", cm_path)
    print(f"Confusion matrix   : {cm_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Top-k reports & confusion matrices (k > 1)
    # ══════════════════════════════════════════════════════════════════════════
    for k, k_acc in sorted(topk_results.items()):
        if k == 1:
            continue

        # Build top-k adjusted predictions: if true label is in top-k,
        # count as correct (pred = true); otherwise use top-1 prediction.
        topk_idx = _topk_predictions(val_probs, k)
        topk_preds = np.array([
            true if true in row else row[0]
            for true, row in zip(val_true, topk_idx)
        ])

        topk_acc = balanced_accuracy_score(val_true, topk_preds)
        topk_f1 = f1_score(val_true, topk_preds, average="weighted", zero_division=0)

        report_k_str = classification_report(
            val_true, topk_preds,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0,
        )

        print(f"\n-- Top-{k} Classification Report --")
        print(report_k_str)
        print(f"Balanced accuracy : {topk_acc:.4f}")
        print(f"Weighted F1       : {topk_f1:.4f}")
        print(f"Top-{k} accuracy    : {k_acc:.4f}")

        report_k_path = output_dir / f"classification_report_top{k}{file_suffix}.txt"
        with open(report_k_path, "w", encoding="utf-8") as f:
            f.write(report_header)
            f.write(f"-- Top-{k} Classification Report --\n\n")
            f.write(report_k_str)
            f.write(f"\nBalanced accuracy : {topk_acc:.4f}\n")
            f.write(f"Weighted F1       : {topk_f1:.4f}\n")
            f.write(f"Top-{k} accuracy    : {k_acc:.4f}\n")
        print(f"Report saved to    : {report_k_path}")

        # Top-k confusion matrix
        cm_k = _topk_confusion_matrix(val_true, val_probs, k, num_classes)
        cm_k_path = output_dir / f"confusion_matrix_top{k}{file_suffix}.png"
        _plot_confusion_matrix(cm_k, classes, num_classes, f"{cm_title} (Top-{k})", cm_k_path)
        print(f"Confusion matrix   : {cm_k_path}")

    # ── Summary of all top-k accuracies ───────────────────────────────────────
    summary_path = output_dir / f"topk_summary{file_suffix}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_header)
        f.write("-- Top-k Accuracy Summary --\n\n")
        for k, acc in sorted(topk_results.items()):
            f.write(f"Top-{k} accuracy : {acc:.4f}\n")
    print(f"\nTop-k summary      : {summary_path}")

    # ── Predictions CSV ──────────────────────────────────────────────────────
    if save_predictions:
        pred_rows = {
            "compound_id":     val_cids,
            "true_label":      [classes[i] for i in val_true],
            "predicted_label": [classes[i] for i in val_preds],
            "correct":         [t == p for t, p in zip(val_true, val_preds)],
        }
        for cls_idx, cls_name in enumerate(classes):
            pred_rows[f"prob_{cls_name}"] = val_probs[:, cls_idx].tolist()
        # Add top-k correctness columns
        for k in sorted(topk_results):
            if k == 1:
                continue
            topk_idx = _topk_predictions(val_probs, k)
            pred_rows[f"correct_top{k}"] = [t in row for t, row in zip(val_true, topk_idx)]

        pred_df = pd.DataFrame(pred_rows)
        pred_path = output_dir / f"predictions{file_suffix}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

    print(f"Outputs saved to   : {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Dimension-reduction visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cluster_visualization(
    X: np.ndarray,
    y: np.ndarray,
    classes: List[str],
    output_dir: Path,
    file_suffix: str = "",
    split_labels: Optional[np.ndarray] = None,
    seed: int = 42,
    methods: Tuple[str, ...] = ("umap", "tsne", "pca"),
) -> None:
    """Create 2-D scatter plots of *X* coloured by chemical class using
    UMAP, t-SNE and/or PCA.

    Parameters
    ----------
    X : (N, D) feature matrix (already per-compound mean latents).
    y : (N,) integer class labels.
    classes : list of class name strings (indexed by *y*).
    output_dir : directory for saving plots.
    file_suffix : extra suffix appended to output file names.
    split_labels : optional (N,) array of "train" / "test" strings.
        When provided an extra plot per method is generated with marker
        style distinguishing train vs. test points.
    seed : random state.
    methods : which reductions to run (subset of "umap", "tsne", "pca").
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    try:
        from umap import UMAP
        _has_umap = True
    except ImportError:
        _has_umap = False

    num_classes = len(classes)

    # ── Colour palette (use tab20 for many classes, tab10 for few) ────────────
    cmap = plt.cm.tab20 if num_classes > 10 else plt.cm.tab10
    colours = [cmap(i / max(num_classes - 1, 1)) for i in range(num_classes)]

    reducers: Dict[str, np.ndarray] = {}

    if "pca" in methods:
        pca = PCA(n_components=2, random_state=seed)
        reducers["PCA"] = pca.fit_transform(X)

    if "tsne" in methods:
        perplexity = min(30, max(5, X.shape[0] // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                     init="pca", learning_rate="auto")
        reducers["t-SNE"] = tsne.fit_transform(X)

    if "umap" in methods:
        if _has_umap:
            n_neighbors = min(15, max(2, X.shape[0] // 5))
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                           min_dist=0.3, random_state=seed)
            reducers["UMAP"] = reducer.fit_transform(X)
        else:
            print("  [warn] umap-learn not installed — skipping UMAP visualisation.")

    for method_name, coords in reducers.items():
        _save_scatter(coords, y, classes, colours, num_classes,
                      method_name, output_dir, file_suffix)
        if split_labels is not None:
            _save_scatter_split(coords, y, split_labels, classes, colours,
                                num_classes, method_name, output_dir, file_suffix)

    print(f"  Cluster plots saved to : {output_dir}")


def _save_scatter(
    coords: np.ndarray,
    y: np.ndarray,
    classes: List[str],
    colours: list,
    num_classes: int,
    method_name: str,
    output_dir: Path,
    file_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for ci in range(num_classes):
        mask = y == ci
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colours[ci]], label=classes[ci],
                   s=25, alpha=0.75, edgecolors="none")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"Chemical Class Clustering — {method_name}")
    ax.legend(fontsize=7, markerscale=1.5, loc="best",
              ncol=max(1, num_classes // 15), framealpha=0.7)
    fig.tight_layout()
    tag = method_name.lower().replace("-", "")
    fig.savefig(output_dir / f"cluster_{tag}{file_suffix}.png", dpi=150)
    plt.close(fig)


def _save_scatter_split(
    coords: np.ndarray,
    y: np.ndarray,
    split_labels: np.ndarray,
    classes: List[str],
    colours: list,
    num_classes: int,
    method_name: str,
    output_dir: Path,
    file_suffix: str,
) -> None:
    """Like ``_save_scatter`` but uses circle / cross markers to show
    train vs. test membership."""
    fig, ax = plt.subplots(figsize=(10, 8))
    marker_map = {"train": "o", "val": "s", "test": "x"}
    for ci in range(num_classes):
        for split_name, marker in marker_map.items():
            mask = (y == ci) & (split_labels == split_name)
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colours[ci]], label=f"{classes[ci]} ({split_name})",
                       s=25, alpha=0.75, marker=marker, edgecolors="none")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"Chemical Class Clustering — {method_name} (train/test)")
    ax.legend(fontsize=6, markerscale=1.5, loc="best",
              ncol=max(1, num_classes // 8), framealpha=0.7)
    fig.tight_layout()
    tag = method_name.lower().replace("-", "")
    fig.savefig(output_dir / f"cluster_{tag}_split{file_suffix}.png", dpi=150)
    plt.close(fig)
