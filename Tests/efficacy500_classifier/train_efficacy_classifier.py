"""
train_efficacy_classifier.py

Train a binary classifier (XGBoost, CatBoost, or Gated ABMIL) on
embeddings_20ppm + efficacy.pt, then run inference on embeddings_100ppm
and evaluate against efficacy_500ppm.

Classifiers
-----------
  - xgboost  : mean-pools instance embeddings per compound, then XGBoost.
  - catboost : mean-pools instance embeddings per compound, then CatBoost.
  - abmil    : Gated Attention-Based MIL — learns instance-level attention
                weights over variable-length bags (no mean-pooling).
  - logsumexp: LogSumExp MIL — learnable smooth interpolation between
                mean- and max-pooling.

Workflow
--------
  1. TRAIN  — fit classifier on embeddings_20ppm / efficacy.pt
  2. INFER  — predict on embeddings_100ppm, evaluate vs efficacy_500ppm
             Logs: classification report, confusion matrix, AUROC curve,
             predictions CSV.

Usage
-----
  # XGBoost (default)
python TiltedVAEMyzus/Tests/efficacy500_classifier/train_efficacy_classifier.py --classifier xgboost --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/embeddings_20ppm.pt --efficacy TiltedVAEMyzus/Tests/efficacy500_classifier/efficacy.pt --inference_embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/embeddings_100ppm.pt --inference_efficacy TiltedVAEMyzus/Tests/efficacy500_classifier/compounds500ppm.csv

  # CatBoost
  python Tests/efficacy500_classifier/train_efficacy_classifier.py \\
      --classifier catboost \\
      --embeddings           Tests/efficacy500_classifier/embeddings_20ppm.pt \\
      --efficacy             Tests/efficacy500_classifier/efficacy.pt \\
      --inference_embeddings Tests/efficacy500_classifier/embeddings_100ppm.pt \\
      --inference_efficacy   Tests/efficacy500_classifier/efficacy_500ppm.csv

  # Gated ABMIL
  python Tests/efficacy500_classifier/train_efficacy_classifier.py \\
      --classifier abmil \\
      --embeddings           Tests/efficacy500_classifier/embeddings_20ppm.pt \\
      --efficacy             Tests/efficacy500_classifier/efficacy.pt \\
      --inference_embeddings Tests/efficacy500_classifier/embeddings_100ppm.pt \\
      --inference_efficacy   Tests/efficacy500_classifier/efficacy_500ppm.csv

  # Gated ABMIL with hyperparameter tuning
  python Tests/efficacy500_classifier/train_efficacy_classifier.py \\
      --classifier abmil --tune \\
      --abmil_tune_iter 50 \\
      --abmil_tune_epochs 50 \\
      --embeddings           Tests/efficacy500_classifier/embeddings_20ppm.pt \\
      --efficacy             Tests/efficacy500_classifier/efficacy.pt \\
      --inference_embeddings Tests/efficacy500_classifier/embeddings_100ppm.pt \\
      --inference_efficacy   Tests/efficacy500_classifier/efficacy_500ppm.csv

Output
------
  <output_dir>/
      classification_report.txt
      confusion_matrix.png
      auroc_curve.png
      predictions.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# ── project imports ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from .classifier_utils import (
        load_efficacy,
        binarize_efficacy,
        load_inference_labels,
        build_mil_bags,
        build_mean_latent_features,
        train_abmil,
        infer_abmil,
        train_logsumexp,
        infer_logsumexp,
        evaluate_and_report,
    )
    from .classifier_tuning import tune_abmil, tune_catboost, tune_xgboost, tune_logsumexp
except ImportError:
    from classifier_utils import (
        load_efficacy,
        binarize_efficacy,
        load_inference_labels,
        build_mil_bags,
        build_mean_latent_features,
        train_abmil,
        infer_abmil,
        train_logsumexp,
        infer_logsumexp,
        evaluate_and_report,
    )
    from classifier_tuning import tune_abmil, tune_catboost, tune_xgboost, tune_logsumexp


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary classifier: predict efficacy >= threshold "
        "(active) vs < threshold (inactive) from VAE embeddings.",
    )

    # ── Classifier choice ──
    p.add_argument(
        "--classifier",
        choices=["xgboost", "catboost", "abmil", "logsumexp"],
        default="xgboost",
        help="Classifier to use: 'xgboost', 'catboost' (mean-pooled), 'abmil' (gated attention MIL), or 'logsumexp' (LogSumExp MIL) (default: xgboost)",
    )

    # ── Training data ──
    p.add_argument(
        "--embeddings",
        default="Tests/efficacy500_classifier/embeddings_20ppm.pt",
        help="Training embeddings (default: Tests/efficacy500_classifier/embeddings_20ppm.pt)",
    )
    p.add_argument(
        "--efficacy",
        default="Tests/efficacy500_classifier/efficacy.pt",
        help="Training efficacy labels (default: Tests/efficacy500_classifier/efficacy.pt)",
    )
    p.add_argument(
        "--subtract_control",
        action="store_true",
        help="Subtract per-plate averaged control embedding from treated embeddings",
    )
    p.add_argument(
        "--normalize_before_subtract",
        action="store_true",
        help="L2-normalize treated and control embeddings before subtraction (requires --subtract_control)",
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes: undersample majority (XGBoost) or pos_weight (ABMIL)",
    )
    p.add_argument(
        "--scale_pos_weight",
        action="store_true",
        help="XGBoost: use scale_pos_weight=n_neg/n_pos instead of undersampling (requires --balance is NOT set)",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search before training",
    )
    p.add_argument("--tune_iter", type=int, default=100, help="Number of random search iterations (default: 100)")

    # ── Inference data ──
    p.add_argument(
        "--inference_embeddings",
        default="Tests/efficacy500_classifier/embeddings_100ppm.pt",
        help="Inference embeddings (default: Tests/efficacy500_classifier/embeddings_100ppm.pt)",
    )
    p.add_argument(
        "--inference_efficacy",
        default="Tests/efficacy500_classifier/efficacy_500ppm.csv",
        help="Ground-truth efficacy for inference evaluation CSV with 'Compound No' and 'Active' columns (default: Tests/efficacy500_classifier/efficacy_500ppm.csv)",
    )

    # ── Threshold ──
    p.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Efficacy threshold for binary classification: >= threshold -> active (default: 70)",
    )

    # ── XGBoost hyper-parameters ──
    p.add_argument("--xgb_n_estimators", type=int, default=1000, help="XGBoost rounds (default: 1000)")
    p.add_argument("--xgb_max_depth", type=int, default=2, help="XGBoost max depth (default: 2)")
    p.add_argument("--xgb_learning_rate", type=float, default=0.05, help="XGBoost lr (default: 0.05)")
    p.add_argument("--xgb_subsample", type=float, default=0.8, help="XGBoost row subsample (default: 0.8)")
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.7, help="XGBoost col subsample (default: 0.7)")
    p.add_argument("--xgb_min_child_weight", type=int, default=1, help="XGBoost min child weight (default: 1)")
    p.add_argument("--xgb_gamma", type=float, default=0.0, help="XGBoost gamma (default: 0)")
    p.add_argument("--xgb_reg_alpha", type=float, default=0.0, help="XGBoost L1 reg (default: 0)")
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0, help="XGBoost L2 reg (default: 1.0)")
    p.add_argument("--xgb_early_stopping", type=int, default=20, help="XGBoost early stopping (default: 20)")

    # ── CatBoost hyper-parameters ──
    p.add_argument("--cb_iterations", type=int, default=2000, help="CatBoost iterations (default: 2000)")
    p.add_argument("--cb_depth", type=int, default=8, help="CatBoost tree depth (default: 8)")
    p.add_argument("--cb_learning_rate", type=float, default=0.03, help="CatBoost lr (default: 0.03)")
    p.add_argument("--cb_l2_leaf_reg", type=float, default=5.0, help="CatBoost L2 regularisation (default: 5.0)")
    p.add_argument("--cb_subsample", type=float, default=0.8, help="CatBoost row subsample (default: 0.8)")
    p.add_argument("--cb_rsm", type=float, default=0.7, help="CatBoost random subspace method / col subsample (default: 0.7)")
    p.add_argument("--cb_early_stopping", type=int, default=50, help="CatBoost early stopping rounds (default: 50)")

    # ── ABMIL hyper-parameters ──
    p.add_argument("--abmil_hidden", type=int, default=256, help="ABMIL attention hidden dim (default: 256)")
    p.add_argument("--abmil_dropout", type=float, default=0.4, help="ABMIL dropout (default: 0.4)")
    p.add_argument("--abmil_lr", type=float, default=5e-4, help="ABMIL learning rate (default: 5e-4)")
    p.add_argument("--abmil_wd", type=float, default=1e-4, help="ABMIL weight decay (default: 1e-4)")
    p.add_argument("--abmil_epochs", type=int, default=20, help="ABMIL training epochs (default: 20)")
    p.add_argument("--abmil_patience", type=int, default=5, help="ABMIL early stopping patience (default: 5)")
    p.add_argument("--abmil_instance_dropout", type=float, default=0.2, help="Randomly drop this fraction of instances per bag during training (default: 0.2)")
    p.add_argument("--abmil_eval_every", type=int, default=1, help="Evaluate on test set every N epochs (default: 1)")
    p.add_argument("--abmil_tune_iter", type=int, default=50, help="Number of ABMIL hyperparameter search trials (default: 50)")
    p.add_argument("--abmil_tune_epochs", type=int, default=50, help="Max epochs per ABMIL tuning trial (default: 50)")

    # ── LogSumExp MIL hyper-parameters ──
    p.add_argument("--lse_hidden", type=int, default=256, help="LogSumExp classifier hidden dim (default: 256)")
    p.add_argument("--lse_dropout", type=float, default=0.4, help="LogSumExp dropout (default: 0.4)")
    p.add_argument("--lse_lr", type=float, default=5e-4, help="LogSumExp learning rate (default: 5e-4)")
    p.add_argument("--lse_wd", type=float, default=1e-4, help="LogSumExp weight decay (default: 1e-4)")
    p.add_argument("--lse_init_r", type=float, default=1.0, help="LogSumExp initial temperature r (default: 1.0)")
    p.add_argument("--lse_epochs", type=int, default=20, help="LogSumExp training epochs (default: 20)")
    p.add_argument("--lse_patience", type=int, default=5, help="LogSumExp early stopping patience (default: 5)")
    p.add_argument("--lse_instance_dropout", type=float, default=0.2, help="LogSumExp instance dropout fraction (default: 0.2)")
    p.add_argument("--lse_eval_every", type=int, default=1, help="Evaluate on test set every N epochs (default: 1)")
    p.add_argument("--lse_tune_iter", type=int, default=50, help="Number of LogSumExp hyperparameter search trials (default: 50)")
    p.add_argument("--lse_tune_epochs", type=int, default=50, help="Max epochs per LogSumExp tuning trial (default: 50)")

    # ── Misc ──
    p.add_argument(
        "--model_name",
        default="tilted_vae",
        help="Name of the model that produced the embeddings (included in output path and report)",
    )
    p.add_argument(
        "--output_dir",
        default="Tests/efficacy500_classifier/runs",
        help="Output directory (default: Tests/efficacy500_classifier/runs)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g. 'cuda:0', 'cuda:1', 'cpu' (default: cuda:0)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier runners
# ═══════════════════════════════════════════════════════════════════════════════


def _run_logsumexp(
    embeddings: Dict,
    cid2label: Dict[str, int],
    inf_embeddings: Dict,
    inf_cid2label: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Train LogSumExp MIL, run inference, return (preds, proba, y_true, cids, label)."""
    train_bags, train_labels, _ = build_mil_bags(
        embeddings, cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {len(train_bags)} training compounds (bags), feature dim {train_bags[0].shape[1]}.")
    if len(train_bags) == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    inf_bags, inf_labels, cids_inf = build_mil_bags(
        inf_embeddings, inf_cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    y_inf = np.array(inf_labels)
    print(f"  {len(inf_bags)} inference compounds (bags).")
    if len(inf_bags) == 0:
        raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

    if args.tune:
        best_params = tune_logsumexp(
            train_bags, train_labels, inf_bags, y_inf, args, device,
        )
        for k, v in best_params.items():
            setattr(args, f"lse_{k}", v)
        print(f"\n  Final LogSumExp config: hidden={args.lse_hidden}  dropout={args.lse_dropout}  "
              f"lr={args.lse_lr}  wd={args.lse_wd}  init_r={args.lse_init_r}  "
              f"instance_dropout={args.lse_instance_dropout}")
        params_path = output_dir / "best_tuning_params.txt"
        with open(params_path, "w") as f:
            f.write("LogSumExp best hyperparameters\n")
            f.write("=" * 40 + "\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"  Saved best params to {params_path}")

    model = train_logsumexp(
        train_bags, train_labels, args, device,
        eval_bags=inf_bags, eval_labels=y_inf, output_dir=output_dir,
    )

    inf_preds, inf_proba = infer_logsumexp(model, inf_bags, device)
    return inf_preds, inf_proba, y_inf, cids_inf, "LogSumExp"


def _run_abmil(
    embeddings: Dict,
    cid2label: Dict[str, int],
    inf_embeddings: Dict,
    inf_cid2label: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Train ABMIL, run inference, return (preds, proba, y_true, cids, label)."""
    train_bags, train_labels, _ = build_mil_bags(
        embeddings, cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {len(train_bags)} training compounds (bags), feature dim {train_bags[0].shape[1]}.")
    if len(train_bags) == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    # Build inference bags before training so we can evaluate during training
    inf_bags, inf_labels, cids_inf = build_mil_bags(
        inf_embeddings, inf_cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    y_inf = np.array(inf_labels)
    print(f"  {len(inf_bags)} inference compounds (bags).")
    if len(inf_bags) == 0:
        raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

    # ── Optional hyperparameter tuning ────────────────────────────────────
    if args.tune:
        best_params = tune_abmil(
            train_bags, train_labels, inf_bags, y_inf, args, device,
        )
        for k, v in best_params.items():
            setattr(args, f"abmil_{k}", v)
        print(f"\n  Final ABMIL config: hidden={args.abmil_hidden}  dropout={args.abmil_dropout}  "
              f"lr={args.abmil_lr}  wd={args.abmil_wd}  instance_dropout={args.abmil_instance_dropout}")
        params_path = output_dir / "best_tuning_params.txt"
        with open(params_path, "w") as f:
            f.write("ABMIL best hyperparameters\n")
            f.write("=" * 40 + "\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"  Saved best params to {params_path}")

    model = train_abmil(
        train_bags, train_labels, args, device,
        eval_bags=inf_bags, eval_labels=y_inf, output_dir=output_dir,
    )

    inf_preds, inf_proba = infer_abmil(model, inf_bags, device)
    return inf_preds, inf_proba, y_inf, cids_inf, "ABMIL"


def _run_xgboost(
    embeddings: Dict,
    cid2label: Dict[str, int],
    inf_embeddings: Dict,
    inf_cid2label: Dict[str, int],
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Train XGBoost, run inference, return (preds, proba, y_true, cids, label)."""
    if not _HAS_XGBOOST:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    # ── Build training features ──────────────────────────────────────────
    X_train, y_train, _ = build_mean_latent_features(
        embeddings, cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {X_train.shape[0]} training compounds, feature dim {X_train.shape[1]}.")

    if X_train.shape[0] == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    # ── Optionally balance training set (undersample majority) ───────────
    if args.balance:
        active_idx = np.where(y_train == 1)[0]
        inactive_idx = np.where(y_train == 0)[0]
        n_minority = min(len(active_idx), len(inactive_idx))
        rng = np.random.RandomState(args.seed)
        active_sampled = rng.choice(active_idx, size=n_minority, replace=False)
        inactive_sampled = rng.choice(inactive_idx, size=n_minority, replace=False)
        balanced_idx = np.sort(np.concatenate([active_sampled, inactive_sampled]))
        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]
        print(f"  Balanced training set: {n_minority} active + {n_minority} inactive = {len(y_train)} compounds.")

    # ── XGBoost parameters (defaults or from CLI) ────────────────────────
    xgb_params = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        min_child_weight=args.xgb_min_child_weight,
        gamma=args.xgb_gamma,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
    )

    # ── Optionally use scale_pos_weight ──────────────────────────────────
    if args.scale_pos_weight:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            spw = n_neg / n_pos
            xgb_params["scale_pos_weight"] = spw
            print(f"  XGBoost scale_pos_weight={spw:.3f} (neg={n_neg}, pos={n_pos})")

    # ── Inference features (built early so tuning can evaluate on them) ──
    X_inf, y_inf, cids_inf = build_mean_latent_features(
        inf_embeddings, inf_cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {X_inf.shape[0]} inference compounds, feature dim {X_inf.shape[1]}.")
    if X_inf.shape[0] == 0:
        raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

    # ── Optional hyperparameter tuning ───────────────────────────────────
    if args.tune:
        xgb_params = tune_xgboost(X_train, y_train, X_inf, y_inf, args)
        params_path = output_dir / "best_tuning_params.txt"
        with open(params_path, "w") as f:
            f.write("XGBoost best hyperparameters\n")
            f.write("=" * 40 + "\n")
            for k, v in xgb_params.items():
                f.write(f"{k}: {v}\n")
        print(f"  Saved best params to {params_path}")

    # ── 5-Fold Cross Validation ──────────────────────────────────────────
    print("\n5-Fold Cross Validation on training data ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_aurocs = []

    for seed in range(5):
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            fold_clf = xgb.XGBClassifier(
                **xgb_params,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=seed+fold_idx,
                device=args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu",
                early_stopping_rounds=args.xgb_early_stopping,
            )
            fold_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            va_proba = fold_clf.predict_proba(X_va)[:, 1]
            fold_aurocs.append(roc_auc_score(y_va, va_proba))
            print(f" Seed {seed} Fold {fold_idx}: AUROC={fold_aurocs[-1]:.4f}")

    print(f"  Mean : AUROC={np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")

    # ── Train final model on all training data ───────────────────────────
    clf = xgb.XGBClassifier(
        **xgb_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=args.seed,
        device=args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu",
        early_stopping_rounds=args.xgb_early_stopping,
    )
    print(f"\nTraining final XGBoost on all {X_train.shape[0]} training compounds ...")
    clf.fit(X_train, y_train, eval_set=[(X_inf, y_inf)], verbose=None)
    print("Training done.\n")

    inf_preds = clf.predict(X_inf)
    inf_proba = clf.predict_proba(X_inf)[:, 1]
    return inf_preds, inf_proba, y_inf, cids_inf, "XGBoost"


def _run_catboost(
    embeddings: Dict,
    cid2label: Dict[str, int],
    inf_embeddings: Dict,
    inf_cid2label: Dict[str, int],
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Train CatBoost, run inference, return (preds, proba, y_true, cids, label)."""
    if not _HAS_CATBOOST:
        raise ImportError("catboost is required. Install with: pip install catboost")

    # ── Build training features ──────────────────────────────────────────
    X_train, y_train, _ = build_mean_latent_features(
        embeddings, cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {X_train.shape[0]} training compounds, feature dim {X_train.shape[1]}.")

    if X_train.shape[0] == 0:
        raise RuntimeError("No compounds matched between embeddings and efficacy.")

    # ── Optionally balance training set (undersample majority) ───────────
    if args.balance:
        active_idx = np.where(y_train == 1)[0]
        inactive_idx = np.where(y_train == 0)[0]
        n_minority = min(len(active_idx), len(inactive_idx))
        rng = np.random.RandomState(args.seed)
        active_sampled = rng.choice(active_idx, size=n_minority, replace=False)
        inactive_sampled = rng.choice(inactive_idx, size=n_minority, replace=False)
        balanced_idx = np.sort(np.concatenate([active_sampled, inactive_sampled]))
        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]
        print(f"  Balanced training set: {n_minority} active + {n_minority} inactive = {len(y_train)} compounds.")

    # ── CatBoost parameters (defaults or from CLI) ───────────────────────
    cb_params = dict(
        iterations=args.cb_iterations,
        depth=args.cb_depth,
        learning_rate=args.cb_learning_rate,
        l2_leaf_reg=args.cb_l2_leaf_reg,
    )

    # ── Optionally use scale_pos_weight ──────────────────────────────────
    if args.scale_pos_weight:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            spw = n_neg / n_pos
            cb_params["scale_pos_weight"] = spw
            print(f"  CatBoost scale_pos_weight={spw:.3f} (neg={n_neg}, pos={n_pos})")

    # ── Inference features (built early so tuning can evaluate on them) ──
    X_inf, y_inf, cids_inf = build_mean_latent_features(
        inf_embeddings, inf_cid2label, args.subtract_control, args.normalize_before_subtract,
    )
    print(f"  {X_inf.shape[0]} inference compounds, feature dim {X_inf.shape[1]}.")
    if X_inf.shape[0] == 0:
        raise RuntimeError("No compounds matched between inference embeddings and efficacy.")

    # ── Optional hyperparameter tuning ───────────────────────────────────
    if args.tune:
        cb_params = tune_catboost(X_train, y_train, X_inf, y_inf, args)
        params_path = output_dir / "best_tuning_params.txt"
        with open(params_path, "w") as f:
            f.write("CatBoost best hyperparameters\n")
            f.write("=" * 40 + "\n")
            for k, v in cb_params.items():
                f.write(f"{k}: {v}\n")
        print(f"  Saved best params to {params_path}")

    # ── 5-Fold Cross Validation ──────────────────────────────────────────
    print("\n5-Fold Cross Validation on training data ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_aurocs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        fold_clf = CatBoostClassifier(
            **cb_params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.seed,
            verbose=0,
            task_type="GPU" if args.device.startswith("cuda") and torch.cuda.is_available() else "CPU",
            devices=args.device.split(":")[1] if args.device.startswith("cuda") and torch.cuda.is_available() else None,
            early_stopping_rounds=args.cb_early_stopping,
        )
        fold_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        va_proba = fold_clf.predict_proba(X_va)[:, 1]
        fold_aurocs.append(roc_auc_score(y_va, va_proba))
        print(f"  Fold {fold_idx}: AUROC={fold_aurocs[-1]:.4f}")

    print(f"  Mean : AUROC={np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")

    # ── Train final model on all training data ───────────────────────────
    clf = CatBoostClassifier(
        **cb_params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        verbose=500,
        task_type="GPU" if args.device.startswith("cuda") and torch.cuda.is_available() else "CPU",
        devices=args.device.split(":")[1] if args.device.startswith("cuda") and torch.cuda.is_available() else None,
        early_stopping_rounds=args.cb_early_stopping,
    )
    print(f"\nTraining final CatBoost on all {X_train.shape[0]} training compounds ...")
    clf.fit(X_train, y_train, eval_set=(X_train, y_train), verbose=500)
    print("Training done.\n")

    inf_preds = clf.predict(X_inf).astype(int).ravel()
    inf_proba = clf.predict_proba(X_inf)[:, 1]
    return inf_preds, inf_proba, y_inf, cids_inf, "CatBoost"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    if args.classifier == "xgboost" and not _HAS_XGBOOST:
        raise ImportError("xgboost is required. Install it with:  pip install xgboost")
    if args.classifier == "catboost" and not _HAS_CATBOOST:
        raise ImportError("catboost is required. Install it with:  pip install catboost")

    # ── Reproducibility ──────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    subtract_suffix = "subtract_control" if args.subtract_control else "no_subtract"
    output_dir = Path(args.output_dir) / args.model_name / args.classifier / subtract_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training data ───────────────────────────────────────────────
    print(f"Loading embeddings : {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds in embeddings.")

    print(f"Loading efficacy   : {args.efficacy}")
    efficacy = load_efficacy(args.efficacy)
    print(f"  {len(efficacy)} compounds in efficacy file.")

    cid2label = binarize_efficacy(efficacy, threshold=args.threshold)
    n_active = sum(v == 1 for v in cid2label.values())
    n_inactive = sum(v == 0 for v in cid2label.values())
    print(f"  Threshold: {args.threshold}  ->  {n_active} active, {n_inactive} inactive")

    # ── Load inference data ──────────────────────────────────────────────
    print(f"Loading inference embeddings : {args.inference_embeddings}")
    inf_embeddings = torch.load(args.inference_embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(inf_embeddings)} compounds in inference embeddings.")

    print(f"Loading inference efficacy   : {args.inference_efficacy}")
    inf_cid2label = load_inference_labels(args.inference_efficacy)
    print(f"  {len(inf_cid2label)} compounds in inference efficacy file.")

    # ── Run classifier ───────────────────────────────────────────────────
    if args.classifier == "abmil":
        inf_preds, inf_proba, y_inf, cids_inf, classifier_label = _run_abmil(
            embeddings, cid2label, inf_embeddings, inf_cid2label, args, device,
            output_dir=output_dir,
        )
    elif args.classifier == "logsumexp":
        inf_preds, inf_proba, y_inf, cids_inf, classifier_label = _run_logsumexp(
            embeddings, cid2label, inf_embeddings, inf_cid2label, args, device,
            output_dir=output_dir,
        )
    elif args.classifier == "catboost":
        inf_preds, inf_proba, y_inf, cids_inf, classifier_label = _run_catboost(
            embeddings, cid2label, inf_embeddings, inf_cid2label, args,
            output_dir=output_dir,
        )
    else:
        inf_preds, inf_proba, y_inf, cids_inf, classifier_label = _run_xgboost(
            embeddings, cid2label, inf_embeddings, inf_cid2label, args,
            output_dir=output_dir,
        )

    # ── Evaluate & save ──────────────────────────────────────────────────
    evaluate_and_report(
        y_inf, inf_preds, inf_proba, cids_inf,
        classifier_label, args, output_dir,
    )


if __name__ == "__main__":
    main()
