"""
train_efficacy_classifier_100_500_split.py

Train and evaluate an XGBoost binary classifier directly on the overlap between:
  - embeddings_100ppm.pt (features)
  - efficacy_500ppm.csv (labels: 'Compound No', 'Active')

Workflow
--------
  1. Build one feature vector per compound from 100ppm embeddings.
  2. Keep compounds present in both embeddings and 500ppm labels.
  3. Stratified split into train/test (default 80%/20%).
  4. Train on train split and evaluate on test split.

Example
-------
python TiltedVAEMyzus/Tests/efficacy500_classifier/train_efficacy_classifier_100_500_split.py --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/embeddings_100ppm.pt --labels_csv TiltedVAEMyzus/Tests/efficacy500_classifier/compounds500ppm.csv --test_size 0.2 --seed 42 --scale_pos_weight
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split

# project imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .classifier_utils import (
        load_inference_labels,
        load_inference_efficacy_values,
        build_mean_latent_features,
        evaluate_and_report,
    )
    from .classifier_tuning import tune_xgboost
except ImportError:
    from classifier_utils import (
        load_inference_labels,
        load_inference_efficacy_values,
        build_mean_latent_features,
        evaluate_and_report,
    )
    from classifier_tuning import tune_xgboost


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train/evaluate XGBoost on 100ppm embeddings + 500ppm efficacy labels "
            "using a stratified 80/20 split."
        )
    )

    p.add_argument(
        "--embeddings",
        default="Tests/efficacy500_classifier/embeddings_100ppm.pt",
        help="100ppm embeddings .pt",
    )
    p.add_argument(
        "--labels_csv",
        default="Tests/efficacy500_classifier/efficacy_500ppm.csv",
        help="CSV labels with columns 'Compound No' and 'Active'",
    )

    p.add_argument(
        "--subtract_control",
        action="store_true",
        help="Subtract per-plate averaged control embedding from treated embeddings",
    )
    p.add_argument(
        "--normalize_before_subtract",
        action="store_true",
        help="L2-normalize treated and control before subtraction (requires --subtract_control)",
    )

    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument(
        "--balance",
        action="store_true",
        help="Undersample majority class in training split",
    )
    p.add_argument(
        "--scale_pos_weight",
        action="store_true",
        help="XGBoost: set scale_pos_weight = n_neg/n_pos on training split",
    )

    p.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search before training",
    )
    p.add_argument("--tune_iter", type=int, default=100, help="Random search iterations")

    # XGBoost params
    p.add_argument("--xgb_n_estimators", type=int, default=1000)
    p.add_argument("--xgb_max_depth", type=int, default=2)
    p.add_argument("--xgb_learning_rate", type=float, default=0.05)
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.7)
    p.add_argument("--xgb_min_child_weight", type=int, default=1)
    p.add_argument("--xgb_gamma", type=float, default=0.0)
    p.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    p.add_argument("--xgb_early_stopping", type=int, default=20)

    p.add_argument(
        "--confidence_interval",
        action="store_true",
        help="Compute bootstrap confidence intervals for AUROC/F1/balanced accuracy",
    )
    p.add_argument("--ci_n_bootstraps", type=int, default=1000)
    p.add_argument("--ci_alpha", type=float, default=0.05)

    p.add_argument("--model_name", default="tilted_vae")
    p.add_argument(
        "--output_dir",
        default="Tests/efficacy500_classifier/runs",
        help="Base output directory",
    )
    p.add_argument("--device", type=str, default="cuda:0")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    subtract_suffix = "subtract_control" if args.subtract_control else "no_subtract"
    output_dir = (
        Path(args.output_dir)
        / args.model_name
        / "xgboost_100ppm500ppm_split"
        / subtract_suffix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings: {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds in embeddings")

    print(f"Loading labels CSV: {args.labels_csv}")
    cid2label_all: Dict[str, int] = load_inference_labels(args.labels_csv)
    print(f"  {len(cid2label_all)} compounds in labels CSV")
    cid2eff_all: Dict[str, float] = load_inference_efficacy_values(args.labels_csv)
    if cid2eff_all:
        print(f"  {len(cid2eff_all)} compounds with numeric efficacy values found for range analysis")
    else:
        print("  No numeric efficacy column found for range analysis; range TPR/TNR will be skipped")

    emb_ids = {str(k) for k in embeddings.keys()}
    label_ids = set(cid2label_all.keys())
    overlap = emb_ids & label_ids
    print(f"Overlap compounds: {len(overlap)}")
    if len(overlap) == 0:
        raise RuntimeError("No overlapping compound IDs between embeddings and labels CSV.")

    cid2label = {cid: cid2label_all[cid] for cid in overlap}

    X_all, y_all, cids_all = build_mean_latent_features(
        embeddings,
        cid2label,
        subtract_control=args.subtract_control,
        normalize_before_subtract=args.normalize_before_subtract,
    )
    print(f"Feature matrix: {X_all.shape}, labels: {y_all.shape}")

    n_active = int((y_all == 1).sum())
    n_inactive = int((y_all == 0).sum())
    print(f"Class distribution (all): active={n_active}, inactive={n_inactive}")

    if len(np.unique(y_all)) < 2:
        raise RuntimeError("Need at least two classes after overlap filtering.")

    idx = np.arange(len(y_all))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all,
    )

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_test, y_test = X_all[idx_test], y_all[idx_test]
    cids_test = [cids_all[i] for i in idx_test]

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(
        "Train class distribution: "
        f"active={(y_train == 1).sum()}, inactive={(y_train == 0).sum()}"
    )
    print(
        "Test class distribution: "
        f"active={(y_test == 1).sum()}, inactive={(y_test == 0).sum()}"
    )

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
        print(
            "Balanced train set: "
            f"active={n_minority}, inactive={n_minority}, total={len(y_train)}"
        )

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

    if args.scale_pos_weight:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            spw = n_neg / n_pos
            xgb_params["scale_pos_weight"] = spw
            print(f"XGBoost scale_pos_weight={spw:.3f} (neg={n_neg}, pos={n_pos})")

    if args.tune:
        print("Running XGBoost tuning on train split, evaluated on test split...")
        xgb_params = tune_xgboost(X_train, y_train, X_test, y_test, args)
        params_path = output_dir / "best_tuning_params.txt"
        with open(params_path, "w") as f:
            f.write("XGBoost best hyperparameters\n")
            f.write("=" * 40 + "\n")
            for k, v in xgb_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved best params: {params_path}")

    xgb_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    clf = xgb.XGBClassifier(
        **xgb_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=args.seed,
        device=xgb_device,
        early_stopping_rounds=args.xgb_early_stopping,
    )

    print("Training XGBoost...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("Training complete.")

    test_preds = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)[:, 1]

    # Map fields expected by evaluate_and_report for consistent report headers.
    args.efficacy = args.labels_csv
    args.inference_embeddings = args.embeddings
    args.inference_efficacy = args.labels_csv

    evaluate_and_report(
        y_test,
        test_preds,
        test_proba,
        cids_test,
        classifier_label="XGBoost",
        args=args,
        output_dir=output_dir,
        inf_efficacy_values=np.array([cid2eff_all.get(cid, np.nan) for cid in cids_test], dtype=float)
        if cid2eff_all else None,
    )


if __name__ == "__main__":
    main()
