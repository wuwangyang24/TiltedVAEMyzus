"""classifier_tuning.py

Randomized hyperparameter search for the CatBoost chemical-class classifier
(ported from the MyzusDINOAdapt synthesis-program classifier, CatBoost only).
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


def _tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    output_dir: Optional[Path] = None,
    file_suffix: str = "",
) -> Dict:
    """Random search over CatBoost hyperparameters, return best config."""
    param_space = {
        "iterations": [100],
        "depth": [9, 12, 15],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 5.0, 10.0, 20.0],
        "auto_class_weights": ["Balanced"],
        # "random_strength": [0.5, 1.0, 2.0],
        # "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    }

    rng = np.random.RandomState(args.seed)
    n_trials = args.tune_iter
    loss_fn = "MultiClass" if num_classes > 2 else "Logloss"
    eval_metric = "TotalF1:average=Macro" if num_classes > 2 else "F1"

    print(f"\nCatBoost hyperparameter tuning ({n_trials} trials) ...")

    best_acc = -1.0
    best_params = {}
    results = []

    for trial in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}
        auto_cw = None if config["auto_class_weights"] == "None" else config["auto_class_weights"]

        cb_params = dict(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            auto_class_weights=auto_cw,
            loss_function=loss_fn,
            eval_metric=eval_metric,
            random_seed=args.seed,
            verbose=0,
            early_stopping_rounds=args.cb_early_stopping,
            task_type=getattr(args, "cb_task_type", "CPU"),
        )
        if getattr(args, "cb_task_type", "CPU") == "GPU":
            cb_params["devices"] = getattr(args, "cb_devices", "0")

        clf = CatBoostClassifier(**cb_params)
        clf.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
        )

        val_preds = clf.predict(X_val).astype(int).ravel()
        trial_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        trial_acc = balanced_accuracy_score(y_val, val_preds)

        results.append({**config, "f1": trial_f1, "balanced_acc": trial_acc})
        is_best = trial_acc > best_acc
        if is_best:
            best_acc = trial_acc
            best_params = dict(config)

        print(f"  Trial {trial+1:3d}/{n_trials}  depth={config['depth']}  "
              f"lr={config['learning_rate']:.3f}  l2={config['l2_leaf_reg']:.1f}  "
              f"cw={config['auto_class_weights']:<12s}  "
              f"F1={trial_f1:.4f}  Acc={trial_acc:.4f}{'  * BEST' if is_best else ''}")

        # Release CatBoost GPU handles before building the next model to avoid
        # a CUDA handle leak when training many models in a loop.
        del clf
        gc.collect()

    print(f"\n  Best trial Acc: {best_acc:.4f}")
    print(f"  Best params: {best_params}")

    results_df = pd.DataFrame(results).sort_values("balanced_acc", ascending=False)
    print(f"\n  Top 5 configs:")
    print(results_df.head().to_string(index=False))

    best_config = {
        "iterations": int(best_params["iterations"]),
        "depth": int(best_params["depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
        "auto_class_weights": str(best_params["auto_class_weights"]),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"tuning_results{file_suffix}.csv"
        results_df.to_csv(results_path, index=False)
        best_path = output_dir / f"best_params{file_suffix}.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({**best_config, "balanced_acc": float(best_acc)}, f, indent=2)
        print(f"  Tuning results saved to : {results_path}")
        print(f"  Best params saved to    : {best_path}")

    return best_config
