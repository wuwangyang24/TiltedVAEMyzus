"""
train_chemical_class_classifier.py

Chemical-class prediction test for the TiltedVAE latent space.

Trains a **CatBoost** classifier to predict the chemical class of a compound
from its VAE latent embeddings.  The logic is ported directly from the
MyzusDINOAdapt synthesis-program classifier (CatBoost path only).

DATA FLOW
---------
For each compound:
  1. Collect all treated latent vectors across every plate    →  (M, D)
     (optionally subtract the per-plate averaged control embedding first)
  2. Compute the element-wise mean across M images            →  (D,)
  3. Feed the (N, D) feature matrix into CatBoost             →  num_classes

Inputs
------
  --embeddings   embeddings.pt from encode_embeddings.py:
                    { compound_id: { plate_id: {"treated": (N,D), "control": (D,)} } }

  --metadata     CSV / Excel file with at least two columns:
                    "compound"        (str)  — must match compound_id keys in the .pt file
                    "chemical_class"  (str)  — class label

Usage examples
--------------
  # 1) Encode images with a trained TiltedVAE checkpoint
  python Tests/chemical_class_classifier/encode_embeddings.py \
      --metadata   data/compound_images.json \
      --root_dir   ../DATA/Train/ \
      --output     Tests/chemical_class_classifier/embeddings.pt \
      --checkpoint results/checkpoints/last.ckpt \
      --model      tilted --latent_dim 128 --img_size 96

  # 2) Train + evaluate the CatBoost chemical-class classifier
python TiltedVAEMyzus/Tests/chemical_class_classifier/train_chemical_class_classifier.py --embeddings TiltedVAEMyzus/results/checkpoints/tilted-latent128_kl0.001_bestsofar/embeddings_best_balanced_acc.pt --metadata METADATA/synthesisprogram_compoundno.csv --save_predictions --label_col synthesis_program --min_compounds_per_class 30 --subtract_control --normalize_before_subtract --filter_by_efficacy 0

  # With control subtraction, softer class balancing and hyper-parameter tuning
  python Tests/chemical_class_classifier/train_chemical_class_classifier.py \
      --embeddings Tests/chemical_class_classifier/embeddings.pt \
      --metadata   data/compound_metadata.csv \
      --subtract_control \
      --cb_auto_class_weights SqrtBalanced \
      --cb_iterations 1000 --cb_depth 8 \
      --tune --tune_iter 50

Output
------
  <output_dir>/
      catboost_model_<stem>.cbm            — saved model
      label_encoder.json                    — { "classes": [...], "str2idx": {...} }
      classification_report_top1_<stem>.txt — top-1 report
      confusion_matrix_top1_<stem>.png      — top-1 confusion matrix
      predictions_<stem>.csv                — per-compound predictions (with --save_predictions)
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# This script lives in ``Tests/chemical_class_classifier/``; its own directory
# is on sys.path[0] when run as a script, so the sibling helper modules import
# directly.
from classifier_utils import (
    build_mean_latent_features,
    filter_rare_classes_array,
    build_label_encoder,
    save_label_encoder,
    save_results,
)
from classifier_tuning import _tune_catboost


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a CatBoost classifier for compound chemical classes."
    )

    # ---- Data ----
    p.add_argument("--embeddings", required=True,
                   help="Path to the .pt embeddings file from encode_embeddings.py")
    p.add_argument("--metadata", required=True,
                   help="CSV or Excel file with 'compound' and 'chemical_class' columns")
    p.add_argument("--compound_col", default="compound",
                   help="Name of the compound ID column in metadata. Default: compound")
    p.add_argument("--label_col", default="chemical_class",
                   help="Name of the chemical class column in metadata. Default: chemical_class")
    p.add_argument("--filter_by_efficacy", type=float, default=None,
                   help="Keep only compounds whose 'Efficacy' column in metadata is >= this value. "
                        "Requires an 'Efficacy' column in the metadata file.")
    p.add_argument("--subtract_control", action="store_true",
                   help="Subtract per-plate averaged control embedding from treated embeddings")
    p.add_argument("--normalize_before_subtract", action="store_true",
                   help="L2-normalize treated and control embeddings before subtraction "
                        "(requires --subtract_control)")
    p.add_argument("--test_split", type=float, default=0.2,
                   help="Fraction of compounds held out for final evaluation. Default: 0.2")
    p.add_argument("--min_compounds_per_class", type=int, default=2,
                   help="Drop chemical classes with fewer compounds than this. Default: 2")

    # ---- CatBoost hyper-parameters ----
    p.add_argument("--cb_iterations", type=int, default=300,
                   help="[CatBoost] Number of boosting iterations. Default: 300")
    p.add_argument("--cb_depth", type=int, default=5,
                   help="[CatBoost] Tree depth. Default: 5")
    p.add_argument("--cb_learning_rate", type=float, default=0.1,
                   help="[CatBoost] Learning rate. Default: 0.1")
    p.add_argument("--cb_l2_leaf_reg", type=float, default=1.0,
                   help="[CatBoost] L2 regularization. Default: 1.0")
    p.add_argument("--cb_auto_class_weights", choices=["None", "Balanced", "SqrtBalanced"],
                   default="Balanced",
                   help="[CatBoost] Auto class weighting. Default: Balanced")
    p.add_argument("--cb_early_stopping", type=int, default=50,
                   help="[CatBoost] Early stopping rounds (only used with --use_val_set). Default: 50")

    # ---- Tuning ----
    p.add_argument("--tune", action="store_true",
                   help="Run randomized hyperparameter search before final training "
                        "(implies --use_val_set)")
    p.add_argument("--tune_iter", type=int, default=50,
                   help="Number of random search iterations. Default: 50")
    p.add_argument("--use_val_set", action="store_true",
                   help="Use a 3-way train/val/test split with early stopping and "
                        "retrain on train+val. Without this flag the pipeline uses "
                        "a simple train/test split (matching the training callback).")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of compounds used for validation (only with --use_val_set). Default: 0.2")

    # ---- Misc ----
    p.add_argument("--output_dir", default="Tests/chemical_class_classifier/runs",
                   help="Directory for checkpoints and logs. "
                        "Default: Tests/chemical_class_classifier/runs")
    p.add_argument("--model_name", default=None,
                   help="Model name for the output directory. Defaults to the stem of the input file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--save_predictions", action="store_true",
                   help="Save test predictions + ground truth to predictions.csv")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5],
                   help="Top-k values for classification accuracy. Default: 1 3 5")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_catboost(
    args: argparse.Namespace,
    embeddings: Dict,
    df: pd.DataFrame,
    str2idx: Dict[str, int],
    classes: List[str],
    num_classes: int,
    output_dir: Path,
) -> None:
    """Train a CatBoost classifier on per-compound mean-latent features."""
    if not _HAS_CATBOOST:
        raise ImportError(
            "catboost is required for this test. Install it with:  pip install catboost"
        )

    from sklearn.model_selection import train_test_split

    # ── Build feature matrix ─────────────────────────────────────────────────
    X, y, cids = build_mean_latent_features(
        embeddings=embeddings,
        compound_col=df[args.compound_col],
        label_col=df[args.label_col],
        label2idx=str2idx,
        subtract_control=args.subtract_control,
        normalize_before_subtract=args.normalize_before_subtract,
    )
    print(f"  {X.shape[0]} compounds with valid features.")
    print(f"  Feature dim (D) : {X.shape[1]}")

    if X.shape[0] == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the embeddings file "
            "match the compound IDs in the metadata."
        )

    X, y, cids, classes, num_classes = filter_rare_classes_array(
        X, y, cids, classes, args.min_compounds_per_class,
    )

    # ── Class distribution ───────────────────────────────────────────────────
    for ci, cname in enumerate(classes):
        print(f"    {cname}: {(y == ci).sum()} compounds")

    # ── Train / test split ───────────────────────────────────────────────────
    use_val = args.use_val_set or args.tune
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, cids_train, cids_test = train_test_split(
        X, y, cids,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=strat,
    )

    X_val, y_val = None, None
    if use_val:
        strat_tv = y_train if len(np.unique(y_train)) > 1 else None
        relative_val = args.val_split / (1.0 - args.test_split)
        X_train, X_val, y_train, y_val, cids_train, _ = train_test_split(
            X_train, y_train, cids_train,
            test_size=relative_val,
            random_state=args.seed,
            stratify=strat_tv,
        )
        print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")
    else:
        print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")

    # ── Optional hyperparameter tuning (requires val set) ─────────────────────
    if args.tune:
        best_params = _tune_catboost(
            X_train, y_train, X_val, y_val, num_classes, args,
        )
        args.cb_iterations = best_params["iterations"]
        args.cb_depth = best_params["depth"]
        args.cb_learning_rate = best_params["learning_rate"]
        args.cb_l2_leaf_reg = best_params["l2_leaf_reg"]
        args.cb_auto_class_weights = best_params["auto_class_weights"]
        print(f"\n  Final CatBoost config: iterations={args.cb_iterations}  "
              f"depth={args.cb_depth}  lr={args.cb_learning_rate}  "
              f"l2_leaf_reg={args.cb_l2_leaf_reg}  "
              f"class_weights={args.cb_auto_class_weights}")

    # ── CatBoost model ───────────────────────────────────────────────────────
    auto_cw = None if args.cb_auto_class_weights == "None" else args.cb_auto_class_weights
    cb_params = dict(
        iterations=args.cb_iterations,
        depth=args.cb_depth,
        learning_rate=args.cb_learning_rate,
        auto_class_weights=auto_cw,
        loss_function="MultiClass" if num_classes > 2 else "Logloss",
        random_seed=args.seed,
        verbose=0,
    )

    if use_val:
        cb_params["l2_leaf_reg"] = args.cb_l2_leaf_reg
        cb_params["eval_metric"] = "TotalF1:average=Macro" if num_classes > 2 else "F1"
        cb_params["verbose"] = 50
        cb_params["early_stopping_rounds"] = args.cb_early_stopping

    clf = CatBoostClassifier(**cb_params)

    print(f"\nTraining CatBoost ({args.cb_iterations} iters, depth={args.cb_depth}, "
          f"lr={args.cb_learning_rate}, class_weights={auto_cw}) ...")

    if use_val:
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))

        # Retrain on train+val with best iteration count
        best_n = clf.get_best_iteration() + 1 if clf.get_best_iteration() is not None else args.cb_iterations
        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        print(f"\nRetraining CatBoost on train+val ({len(y_trainval)} compounds, {best_n} iterations) ...")
        cb_final_params = {k: v for k, v in cb_params.items() if k != 'early_stopping_rounds'}
        cb_final_params['iterations'] = best_n
        clf = CatBoostClassifier(**cb_final_params)
        clf.fit(X_trainval, y_trainval)
    else:
        clf.fit(X_train, y_train)

    # ── Evaluation on held-out test set ───────────────────────────────────────
    test_preds = clf.predict(X_test).astype(int).ravel()
    test_probs = clf.predict_proba(X_test)

    emb_stem = Path(args.embeddings).stem

    save_results(
        val_true=y_test,
        val_preds=test_preds,
        val_probs=test_probs,
        val_cids=cids_test,
        classes=classes,
        num_classes=num_classes,
        output_dir=output_dir,
        cm_title=f"Confusion Matrix — CatBoost — {emb_stem}",
        file_suffix=f"_{emb_stem}",
        report_header=(
            f"Classifier       : catboost\n"
            f"Embeddings       : {args.embeddings}\n"
            f"Subtract control : {args.subtract_control}\n"
            f"Normalize before subtract : {args.normalize_before_subtract}\n"
            f"Auto class weights : {auto_cw}\n\n"
        ),
        save_predictions=args.save_predictions,
        topk=tuple(args.topk),
    )

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = output_dir / f"catboost_model_{emb_stem}.cbm"
    clf.save_model(str(model_path))
    print(f"Model saved to     : {model_path}")

    # ── Training log ─────────────────────────────────────────────────────────
    evals = clf.get_evals_result()
    if evals and "validation" in evals:
        metrics = evals["validation"]
        first_key = next(iter(metrics))
        log_df = pd.DataFrame({
            "epoch": list(range(1, len(metrics[first_key]) + 1)),
            **{k: v for k, v in metrics.items()},
        })
        log_df.to_csv(output_dir / f"training_log_{emb_stem}.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Output directory ─────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_name = args.model_name if args.model_name else Path(args.embeddings).stem
    subtract_dir = "subtract_control" if args.subtract_control else "no_subtract_control"
    min_cpc_dir = f"minCPC{args.min_compounds_per_class}"
    output_dir = Path(args.output_dir) / date_str / model_name / "catboost" / subtract_dir / min_cpc_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ──────────────────────────────────────────────────────
    print(f"Loading embeddings : {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds found in embeddings file.")

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata   : {args.metadata}")
    meta_path = Path(args.metadata)
    if meta_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)

    required_cols = {args.compound_col, args.label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if args.filter_by_efficacy is not None:
        if "Efficacy" not in df.columns:
            raise ValueError(
                "--filter_by_efficacy requires an 'Efficacy' column in the metadata file. "
                f"Available columns: {list(df.columns)}"
            )
        before = len(df)
        df = df[df["Efficacy"] >= args.filter_by_efficacy]
        print(f"  Filtered by Efficacy >= {args.filter_by_efficacy}: {before} -> {len(df)} rows.")

    df = df[[args.compound_col, args.label_col]].dropna()
    print(f"  {len(df)} compound rows after dropping NaN.")

    # ── Label encoding ────────────────────────────────────────────────────────
    str2idx, classes = build_label_encoder(df[args.label_col])
    num_classes = len(classes)
    print(f"  {num_classes} chemical classes: {classes}")
    save_label_encoder(classes, str2idx, output_dir / "label_encoder.json")

    # ── Train CatBoost ────────────────────────────────────────────────────────
    _run_catboost(
        args=args,
        embeddings=embeddings,
        df=df,
        str2idx=str2idx,
        classes=classes,
        num_classes=num_classes,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
