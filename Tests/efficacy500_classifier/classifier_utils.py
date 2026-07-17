"""
classifier_utils.py

Shared utilities for efficacy-500ppm binary classifiers:
  - Data loading (efficacy.pt and CSV) & binarisation
  - Feature builders (mean-pooled for XGBoost, MIL bags for ABMIL)
  - GatedABMIL model + train / infer helpers
  - LogSumExpMIL model + train / infer helpers
  - Evaluation reporting (classification report, confusion matrix, AUROC, CSV)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = ["inactive", "active"]  # 0 = < threshold, 1 = >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading & binarisation
# ═══════════════════════════════════════════════════════════════════════════════


def load_efficacy(path: str) -> Dict[str, float]:
    """
    Load efficacy.pt → {compound_id: efficacy_value}.

    Expected format: [{'Compound': '...', 'Efficacy': float}, ...]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(entry["Compound"]): float(entry["Efficacy"]) for entry in data}


def binarize_efficacy(
    efficacy: Dict[str, float],
    threshold: float = 70.0,
) -> Dict[str, int]:
    """Return {compound_id: 0 or 1} where 1 means efficacy >= threshold."""
    return {cid: int(val >= threshold) for cid, val in efficacy.items()}


def load_inference_labels(csv_path: str) -> Dict[str, int]:
    """Load inference CSV with 'Compound No' and 'Active' columns."""
    df = pd.read_csv(csv_path)
    return {str(row["Compound No"]).strip(): int(row["Active"]) for _, row in df.iterrows()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Feature builders
# ═══════════════════════════════════════════════════════════════════════════════


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along *dim*."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _collect_plate_latents(
    plates: Dict,
    subtract_control: bool,
    normalize_before_subtract: bool,
) -> List[torch.Tensor]:
    """Collect latent tensors from all plates for a single compound."""
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
    return plate_latents


def build_mil_bags(
    embeddings: Dict,
    cid2label: Dict[str, int],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Build variable-length bags of instance embeddings per compound."""
    bags, labels, cids = [], [], []
    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in cid2label:
            continue
        plate_latents = _collect_plate_latents(plates, subtract_control, normalize_before_subtract)
        if not plate_latents:
            continue
        bags.append(torch.cat(plate_latents, dim=0))
        labels.append(cid2label[cid])
        cids.append(cid)
    return bags, labels, cids


def build_mean_latent_features(
    embeddings: Dict,
    cid2label: Dict[str, int],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (N, D) feature matrix from per-compound mean latents.

    Returns
    -------
    X    : (N, D)
    y    : (N,) int labels (0 or 1)
    cids : list of compound IDs
    """
    X_rows, y_rows, cids = [], [], []

    for compound_id, plates in embeddings.items():
        cid = str(compound_id)
        if cid not in cid2label:
            continue
        plate_latents = _collect_plate_latents(plates, subtract_control, normalize_before_subtract)
        if not plate_latents:
            continue
        all_latents = torch.cat(plate_latents, dim=0)
        mean_latent = all_latents.mean(dim=0).numpy()
        X_rows.append(mean_latent)
        y_rows.append(cid2label[cid])
        cids.append(cid)

    return np.stack(X_rows), np.array(y_rows, dtype=int), cids


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Gated Attention-Based MIL (ABMIL)
# ═══════════════════════════════════════════════════════════════════════════════


class GatedABMIL(nn.Module):
    """Gated Attention-Based Multiple Instance Learning classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.instance_norm = nn.LayerNorm(input_dim)
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        bag : (N_instances, D)

        Returns
        -------
        logit  : (1,) raw logit
        att    : (N_instances,) attention weights
        """
        bag = self.instance_norm(bag)
        V = self.attention_V(bag)  # (N, H)
        U = self.attention_U(bag)  # (N, H)
        att_logits = self.attention_w(V * U)  # (N, 1)
        att = torch.softmax(att_logits, dim=0)  # (N, 1)
        bag_repr = (att * bag).sum(dim=0, keepdim=True)  # (1, D)
        logit = self.classifier(bag_repr).squeeze()  # scalar
        return logit, att.squeeze()


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. LogSumExp MIL
# ═══════════════════════════════════════════════════════════════════════════════


class LogSumExpMIL(nn.Module):
    """LogSumExp-based Multiple Instance Learning classifier.

    Uses LogSumExp pooling as a smooth interpolation between
    mean-pooling (r -> inf) and max-pooling (r -> 0+) over instances:
        pool(H) = r * log( (1/N) * sum(exp(h_i / r)) )
    where r is a learnable temperature parameter.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25,
                 init_r: float = 1.0):
        super().__init__()
        self.instance_norm = nn.LayerNorm(input_dim)
        # Learnable temperature (log-space for positivity)
        self.log_r = nn.Parameter(torch.tensor(float(np.log(init_r))))
        # Instance-level projection before pooling
        self.instance_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        bag : (N_instances, D)

        Returns
        -------
        logit        : (1,) raw logit
        contributions : (N_instances,) per-instance contribution weights
        """
        bag = self.instance_norm(bag)
        h = self.instance_proj(bag)  # (N, D)
        r = torch.exp(self.log_r).clamp(min=1e-6)  # positive temperature
        # LogSumExp pooling: r * log( (1/N) * sum(exp(h/r)) )
        # Using torch.logsumexp for numerical stability
        N = h.shape[0]
        bag_repr = r * (torch.logsumexp(h / r, dim=0) - np.log(N))  # (D,)
        bag_repr = bag_repr.unsqueeze(0)  # (1, D)
        logit = self.classifier(bag_repr).squeeze()  # scalar
        # Instance contributions: softmax of scaled instance norms
        with torch.no_grad():
            contributions = torch.softmax((h / r).sum(dim=-1), dim=0)  # (N,)
        return logit, contributions


class MILBagDataset:
    """Dataset that yields one bag (variable-length tensor) per compound."""

    def __init__(self, bags: List[torch.Tensor], labels: List[int]):
        self.bags = bags
        self.labels = labels

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        return self.bags[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ABMIL train / infer
# ═══════════════════════════════════════════════════════════════════════════════


def train_abmil(
    bags: List[torch.Tensor],
    labels: List[int],
    args: argparse.Namespace,
    device: torch.device,
    eval_bags: List[torch.Tensor] | None = None,
    eval_labels: np.ndarray | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> GatedABMIL:
    """Train Gated ABMIL, return model trained on all data."""
    input_dim = bags[0].shape[1]
    all_labels = np.array(labels)
    has_eval = eval_bags is not None and eval_labels is not None

    # ── Compute pos_weight for class balancing ───────────────────────────
    pos_weight = None
    if args.balance:
        n_pos = int(all_labels.sum())
        n_neg = len(all_labels) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor(n_neg / n_pos, device=device)
            if verbose:
                print(f"  ABMIL pos_weight={pos_weight.item():.3f} (neg={n_neg}, pos={n_pos})")

    # ── Train model on all data ──────────────────────────────────────────
    if verbose:
        print(f"\nTraining ABMIL on all {len(bags)} training compounds ...")
    torch.manual_seed(args.seed)
    model = GatedABMIL(input_dim, args.abmil_hidden, args.abmil_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.abmil_lr, weight_decay=args.abmil_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.abmil_epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = -1.0
    patience_counter = 0
    best_state = None
    training_log = []
    ckpt_dir = output_dir / "checkpoints" if output_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.abmil_epochs), desc="ABMIL Training", disable=not verbose):
        model.train()
        indices = np.random.permutation(len(bags))
        epoch_loss = 0.0
        for i in indices:
            bag = bags[i].to(device)
            # Instance-level dropout augmentation during training
            if bag.shape[0] > 4 and args.abmil_instance_dropout > 0:
                keep_mask = torch.rand(bag.shape[0], device=device) > args.abmil_instance_dropout
                if keep_mask.sum() > 1:
                    bag = bag[keep_mask]
            label = torch.tensor(float(labels[i]), device=device)
            optimizer.zero_grad()
            logit, _ = model(bag)
            loss = criterion(logit, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(bags)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # ── Periodic evaluation on test set ──────────────────────────────
        eval_auroc = None
        eval_f1 = None
        if has_eval and (epoch + 1) % args.abmil_eval_every == 0:
            preds, probas = infer_abmil(model, eval_bags, device)
            eval_auroc = roc_auc_score(eval_labels, probas)
            eval_f1 = f1_score(eval_labels, preds, average="weighted", zero_division=0)
            if verbose:
                print(f"  Epoch {epoch+1}/{args.abmil_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  eval_AUROC={eval_auroc:.4f}  eval_F1={eval_f1:.4f}")

            # Save checkpoint if best
            if eval_auroc > best_auroc:
                best_auroc = eval_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_dir is not None:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": best_state,
                        "auroc": eval_auroc,
                        "f1": eval_f1,
                        "loss": avg_loss,
                    }, ckpt_dir / "best_model.pt")
                    if verbose:
                        print(f"    Saved best checkpoint (AUROC={eval_auroc:.4f})")
            else:
                patience_counter += 1
        elif (epoch + 1) % 50 == 0 or epoch == 0:
            if verbose:
                print(f"  Epoch {epoch+1}/{args.abmil_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}")

        # Fallback: if no eval data, track training loss
        if not has_eval:
            if avg_loss < (best_auroc if best_auroc > 0 else float("inf")) - 1e-4:
                best_auroc = avg_loss  # reuse variable as best_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

        # ── Log this epoch ───────────────────────────────────────────
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "lr": cur_lr,
            "eval_auroc": eval_auroc,
            "eval_f1": eval_f1,
        })

        if patience_counter >= args.abmil_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (patience={args.abmil_patience})")
            break

    # ── Save training log ────────────────────────────────────────────────
    if output_dir is not None and training_log:
        log_df = pd.DataFrame(training_log)
        log_path = output_dir / "training_log.csv"
        log_df.to_csv(log_path, index=False)
        if verbose:
            print(f"Training log saved : {log_path}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if has_eval and verbose:
            print(f"Restored best model (eval AUROC={best_auroc:.4f})")
    if verbose:
        print("Training done.\n")
    return model


def infer_abmil(
    model: GatedABMIL,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, probabilities)."""
    model.eval()
    probas, preds = [], []
    with torch.no_grad():
        for bag in bags:
            logit, _ = model(bag.to(device))
            p = torch.sigmoid(logit).cpu().item()
            probas.append(p)
            preds.append(int(p >= 0.5))
    return np.array(preds), np.array(probas)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LogSumExp MIL — train / infer
# ═══════════════════════════════════════════════════════════════════════════════


def train_logsumexp(
    bags: List[torch.Tensor],
    labels: List[int],
    args: argparse.Namespace,
    device: torch.device,
    eval_bags: List[torch.Tensor] | None = None,
    eval_labels: np.ndarray | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> LogSumExpMIL:
    """Train LogSumExp MIL, return trained model."""
    input_dim = bags[0].shape[1]
    all_labels = np.array(labels)
    has_eval = eval_bags is not None and eval_labels is not None

    # ── Compute pos_weight for class balancing ───────────────────────────
    pos_weight = None
    if args.balance:
        n_pos = int(all_labels.sum())
        n_neg = len(all_labels) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor(n_neg / n_pos, device=device)
            if verbose:
                print(f"  LogSumExp pos_weight={pos_weight.item():.3f} (neg={n_neg}, pos={n_pos})")

    if verbose:
        print(f"\nTraining LogSumExp MIL on all {len(bags)} training compounds ...")
    torch.manual_seed(args.seed)
    model = LogSumExpMIL(
        input_dim, args.lse_hidden, args.lse_dropout, args.lse_init_r,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lse_lr, weight_decay=args.lse_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lse_epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = -1.0
    patience_counter = 0
    best_state = None
    training_log = []
    ckpt_dir = output_dir / "checkpoints" if output_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.lse_epochs), desc="LogSumExp Training", disable=not verbose):
        model.train()
        indices = np.random.permutation(len(bags))
        epoch_loss = 0.0
        for i in indices:
            bag = bags[i].to(device)
            if bag.shape[0] > 4 and args.lse_instance_dropout > 0:
                keep_mask = torch.rand(bag.shape[0], device=device) > args.lse_instance_dropout
                if keep_mask.sum() > 1:
                    bag = bag[keep_mask]
            label = torch.tensor(float(labels[i]), device=device)
            optimizer.zero_grad()
            logit, _ = model(bag)
            loss = criterion(logit, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(bags)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        eval_auroc = None
        eval_f1 = None
        if has_eval and (epoch + 1) % args.lse_eval_every == 0:
            preds, probas = infer_logsumexp(model, eval_bags, device)
            eval_auroc = roc_auc_score(eval_labels, probas)
            eval_f1 = f1_score(eval_labels, preds, average="weighted", zero_division=0)
            if verbose:
                r_val = torch.exp(model.log_r).item()
                print(f"  Epoch {epoch+1}/{args.lse_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  "
                      f"r={r_val:.4f}  eval_AUROC={eval_auroc:.4f}  eval_F1={eval_f1:.4f}")

            if eval_auroc > best_auroc:
                best_auroc = eval_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_dir is not None:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": best_state,
                        "auroc": eval_auroc,
                        "f1": eval_f1,
                        "loss": avg_loss,
                    }, ckpt_dir / "best_model.pt")
                    if verbose:
                        print(f"    Saved best checkpoint (AUROC={eval_auroc:.4f})")
            else:
                patience_counter += 1
        elif (epoch + 1) % 50 == 0 or epoch == 0:
            if verbose:
                r_val = torch.exp(model.log_r).item()
                print(f"  Epoch {epoch+1}/{args.lse_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  r={r_val:.4f}")

        if not has_eval:
            if avg_loss < (best_auroc if best_auroc > 0 else float("inf")) - 1e-4:
                best_auroc = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "lr": cur_lr,
            "r": torch.exp(model.log_r).item(),
            "eval_auroc": eval_auroc,
            "eval_f1": eval_f1,
        })

        if patience_counter >= args.lse_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (patience={args.lse_patience})")
            break

    if output_dir is not None and training_log:
        log_df = pd.DataFrame(training_log)
        log_path = output_dir / "training_log.csv"
        log_df.to_csv(log_path, index=False)
        if verbose:
            print(f"Training log saved : {log_path}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if has_eval and verbose:
            print(f"Restored best model (eval AUROC={best_auroc:.4f})")
    if verbose:
        print("Training done.\n")
    return model


def infer_logsumexp(
    model: LogSumExpMIL,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, probabilities)."""
    model.eval()
    probas, preds = [], []
    with torch.no_grad():
        for bag in bags:
            logit, _ = model(bag.to(device))
            p = torch.sigmoid(logit).cpu().item()
            probas.append(p)
            preds.append(int(p >= 0.5))
    return np.array(preds), np.array(probas)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluation & reporting
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_and_report(
    y_inf: np.ndarray,
    inf_preds: np.ndarray,
    inf_proba: np.ndarray,
    cids_inf: List[str],
    classifier_label: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Compute metrics, print report, and save outputs."""
    inf_acc = balanced_accuracy_score(y_inf, inf_preds)
    inf_f1 = f1_score(y_inf, inf_preds, average="weighted", zero_division=0)
    inf_auroc = roc_auc_score(y_inf, inf_proba)

    report_str = classification_report(
        y_inf, inf_preds, labels=[0, 1],
        target_names=CLASS_NAMES, zero_division=0,
    )
    print("Classification Report (inference):")
    print(report_str)
    print(f"Inference accuracy : {inf_acc:.4f}")
    print(f"Inference F1       : {inf_f1:.4f}")
    print(f"Inference AUROC    : {inf_auroc:.4f}")

    # Save report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classifier           : XGBoost\n")
        f.write(f"Model                : {args.model_name}\n")
        f.write(f"Train embeddings     : {args.embeddings}\n")
        f.write(f"Train efficacy       : {args.efficacy}\n")
        f.write(f"Inference embeddings : {args.inference_embeddings}\n")
        f.write(f"Inference efficacy   : {args.inference_efficacy}\n")
        f.write(f"Threshold            : {args.threshold}\n")
        f.write(f"Classes              : {CLASS_NAMES}\n\n")
        f.write(report_str)
        f.write(f"\nInference accuracy : {inf_acc:.4f}")
        f.write(f"\nInference F1       : {inf_f1:.4f}")
        f.write(f"\nInference AUROC    : {inf_auroc:.4f}\n")
    print(f"Report saved       : {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_inf, inf_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=40)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ylabel="True", xlabel="Predicted",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # AUROC curve
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_inf, inf_proba, name=classifier_label, ax=ax_roc,
    )
    ax_roc.set_title(f"ROC Curve (AUROC = {inf_auroc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "auroc_curve.png", dpi=150)
    plt.close(fig_roc)

    # Predictions CSV
    pred_df = pd.DataFrame({
        "compound_id": cids_inf,
        "true_label": [CLASS_NAMES[i] for i in y_inf],
        "predicted_label": [CLASS_NAMES[i] for i in inf_preds],
        "probability_active": inf_proba,
        "correct": [int(t == p) for t, p in zip(y_inf, inf_preds)],
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Outputs saved to   : {output_dir}")
