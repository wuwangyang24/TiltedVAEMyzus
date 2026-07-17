"""
train_efficacy_classifier.py

Train an XGBoost binary classifier on embeddings_20ppm + efficacy.pt,
then run inference on embeddings_100ppm and evaluate against efficacy_500ppm.

Workflow
--------
  1. TRAIN  — fit XGBoost on embeddings_20ppm / efficacy.pt
  2. INFER  — predict on embeddings_100ppm, evaluate vs efficacy_500ppm
             Logs: classification report, confusion matrix, AUROC curve,
             predictions CSV.

Usage
-----
<<<<<<< HEAD
  python Tests/efficacy500_classifier/train_efficacy_classifier.py \
      --embeddings           Tests/efficacy500_classifier/embeddings_20ppm.pt \
      --efficacy             Tests/efficacy500_classifier/efficacy.pt \
      --inference_embeddings Tests/efficacy500_classifier/embeddings_100ppm.pt \
=======
  # XGBoost (default)
python TiltedVAEMyzus/Tests/efficacy500_classifier/train_efficacy_classifier.py --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/embeddings_20ppm.pt --efficacy TiltedVAEMyzus/Tests/efficacy500_classifier/efficacy.pt --inference_embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/embeddings_100ppm.pt --inference_efficacy TiltedVAEMyzus/Tests/efficacy500_classifier/compounds500ppm.csv --subtract_control --normalize_before_subtract --scale_pos_weight

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
>>>>>>> d7308ad027d62cbabe8a43cbb955a7c2cd512553
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

import xgboost as xgb

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
        build_mean_latent_features,
        evaluate_and_report,
    )
    from .classifier_tuning import tune_xgboost
except ImportError:
    from classifier_utils import (
        load_efficacy,
        binarize_efficacy,
        load_inference_labels,
        build_mean_latent_features,
        evaluate_and_report,
    )
    from classifier_tuning import tune_xgboost


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary classifier: predict efficacy >= threshold "
        "(active) vs < threshold (inactive) from VAE embeddings.",
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
        help="Balance classes: undersample majority class",
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


def _run_xgboost(
    embeddings: Dict,
    cid2label: Dict[str, int],
    inf_embeddings: Dict,
    inf_cid2label: Dict[str, int],
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """Train XGBoost, run inference, return (preds, proba, y_true, cids, label)."""
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

    # ── Train on all training data and evaluate on test set ────────────
    clf = xgb.XGBClassifier(
        **xgb_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=args.seed,
        device=args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu",
        early_stopping_rounds=args.xgb_early_stopping,
    )
    print(f"\nTraining XGBoost on all {X_train.shape[0]} training compounds ...")
    clf.fit(X_train, y_train, eval_set=[(X_inf, y_inf)], verbose=False)
    print("Training done.\n")

    inf_preds = clf.predict(X_inf)
    inf_proba = clf.predict_proba(X_inf)[:, 1]
    return inf_preds, inf_proba, y_inf, cids_inf, "XGBoost"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    subtract_suffix = "subtract_control" if args.subtract_control else "no_subtract"
    output_dir = Path(args.output_dir) / args.model_name / "xgboost" / subtract_suffix
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

    # ── Run XGBoost classifier ───────────────────────────────────────────
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
