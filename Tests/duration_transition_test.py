"""Duration-transition test for embedding quality.

Idea: if embeddings capture biologically meaningful temporal effects, the same
compound measured at different durations (e.g. day 5 vs day 8) should show a
consistent transition in latent space.

This script:
  1. Loads pre-computed embeddings from a .pt file produced by
     encode_embeddings.py.
  2. Loads a duration overlap CSV (default: day5day8overlap.csv) and maps each
     compound's plate IDs to durations.
  3. For compounds that have valid embeddings at all requested durations,
     computes one mean embedding per duration.
  4. Quantifies duration transition quality with:
     - per-compound transition distance and cosine shift
     - retrieval accuracy: does each day-8 embedding match its own day-5
       embedding more closely than other compounds?
     - within-compound vs between-compound day5->day8 distance comparison
  5. Saves per-compound metrics and a PCA trajectory plot.

Example:
python TiltedVAEMyzus/Tests/duration_transition_test.py \
  --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/embeddings_100ppm.pt \
  --metadata_csv TiltedVAEMyzus/Tests/day5day8overlap.csv \
  --durations 5 8 \
  --subtract_control \
  --normalize_before_subtract \
  --output_dir results/duration_transition_test
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test embedding transitions of the same compound across durations"
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to pre-computed embedding .pt file",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="Tests/day5day8overlap.csv",
        help="CSV containing Plate, Duration and compound IDs",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[5, 8],
        help="Duration values to test, in temporal order (default: 5 8)",
    )
    parser.add_argument(
        "--compound_col",
        type=str,
        default="Fraction_Number",
        help="Compound ID column in metadata CSV (default: Fraction_Number)",
    )
    parser.add_argument(
        "--plate_col",
        type=str,
        default="Plate",
        help="Plate ID column in metadata CSV (default: Plate)",
    )
    parser.add_argument(
        "--duration_col",
        type=str,
        default="Duration",
        help="Duration column in metadata CSV (default: Duration)",
    )

    parser.add_argument(
        "--subtract_control",
        action="store_true",
        help="Subtract plate-level control embedding from treated means",
    )
    parser.add_argument(
        "--normalize_before_subtract",
        action="store_true",
        help="L2-normalize treated/control means before subtraction",
    )
    parser.add_argument(
        "--max_compounds",
        type=int,
        default=None,
        help="Optional cap on number of compounds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/duration_transition_test",
        help="Directory to write plots and CSV outputs",
    )

    return parser.parse_args()


def _safe_l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - (np.dot(a, b) / (na * nb)))


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float(np.pi / 2.0)
    cos_sim = float(np.dot(a, b) / (na * nb))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim))


def load_duration_plate_map(
    metadata_csv: str,
    compound_col: str,
    plate_col: str,
    duration_col: str,
    durations: List[int],
) -> Dict[str, Dict[int, set[str]]]:
    """Build mapping: compound -> duration -> set(plate_id)."""
    df = pd.read_csv(metadata_csv)

    required = [compound_col, plate_col, duration_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in metadata CSV: {missing}. "
            f"Available: {list(df.columns)}"
        )

    keep_durations = set(int(d) for d in durations)
    sdf = df[[compound_col, plate_col, duration_col]].copy()
    sdf[compound_col] = sdf[compound_col].astype(str).str.strip()
    sdf[plate_col] = sdf[plate_col].astype(str).str.strip()
    sdf[duration_col] = pd.to_numeric(sdf[duration_col], errors="coerce")
    sdf = sdf.dropna(subset=[duration_col])
    sdf[duration_col] = sdf[duration_col].astype(int)
    sdf = sdf[sdf[duration_col].isin(keep_durations)]
    sdf = sdf[(sdf[compound_col] != "") & (sdf[plate_col] != "")]

    mapping: Dict[str, Dict[int, set[str]]] = {}
    for _, row in sdf.iterrows():
        compound = str(row[compound_col])
        plate_id = str(row[plate_col])
        duration = int(row[duration_col])

        if compound not in mapping:
            mapping[compound] = {d: set() for d in keep_durations}
        mapping[compound][duration].add(plate_id)

    return mapping


def build_duration_trajectories(
    embeddings_path: str,
    duration_plate_map: Dict[str, Dict[int, set[str]]],
    durations: List[int],
    subtract_control: bool,
    normalize_before_subtract: bool,
    max_compounds: int | None,
) -> Dict[str, np.ndarray]:
    """Build per-compound trajectory arrays of shape (n_durations, D)."""
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)

    trajectories: Dict[str, np.ndarray] = {}
    valid_compounds = sorted(set(data.keys()) & set(duration_plate_map.keys()))

    for compound in valid_compounds:
        compound_data = data[compound]
        per_duration_means: List[np.ndarray] = []
        has_all_durations = True

        for duration in durations:
            plate_ids = duration_plate_map[compound].get(duration, set())
            if not plate_ids:
                has_all_durations = False
                break

            plate_means: List[np.ndarray] = []
            for plate_id in plate_ids:
                plate_data = compound_data.get(str(plate_id))
                if plate_data is None:
                    continue

                treated = plate_data.get("treated")
                if treated is None or treated.numel() == 0:
                    continue

                well_mean = treated.mean(dim=0).numpy()

                if subtract_control:
                    control = plate_data.get("control")
                    if control is not None and control.numel() > 0:
                        ctrl_mean = control.numpy()
                        if ctrl_mean.ndim > 1:
                            ctrl_mean = ctrl_mean.mean(axis=0)
                        if normalize_before_subtract:
                            well_mean = _safe_l2_normalize(well_mean)
                            ctrl_mean = _safe_l2_normalize(ctrl_mean)
                        well_mean = well_mean - ctrl_mean

                plate_means.append(well_mean)

            if not plate_means:
                has_all_durations = False
                break

            per_duration_means.append(np.mean(plate_means, axis=0))

        if has_all_durations:
            trajectories[compound] = np.stack(per_duration_means, axis=0)

        if max_compounds is not None and len(trajectories) >= max_compounds:
            break

    return trajectories


def evaluate_duration_transition(
    trajectories: Dict[str, np.ndarray],
    durations: List[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    compounds = list(trajectories.keys())
    n_compounds = len(compounds)
    n_durations = len(durations)
    if n_compounds == 0:
        raise RuntimeError("No compounds with valid embeddings for all durations.")

    first = np.stack([trajectories[c][0] for c in compounds], axis=0)
    last = np.stack([trajectories[c][-1] for c in compounds], axis=0)

    per_rows: List[Dict[str, float | str]] = []
    within_distances: List[float] = []
    within_cosine: List[float] = []
    within_angular: List[float] = []

    for c in compounds:
        start = trajectories[c][0]
        end = trajectories[c][-1]

        euclidean = float(np.linalg.norm(end - start))
        cosine_dist = _cosine_distance(start, end)
        angular = _angular_distance(start, end)

        within_distances.append(euclidean)
        within_cosine.append(cosine_dist)
        within_angular.append(angular)

        per_rows.append(
            {
                "compound_id": c,
                f"norm_day{durations[0]}": float(np.linalg.norm(start)),
                f"norm_day{durations[-1]}": float(np.linalg.norm(end)),
                "delta_norm": float(np.linalg.norm(end) - np.linalg.norm(start)),
                "euclidean_shift": euclidean,
                "cosine_distance": cosine_dist,
                "angular_distance": angular,
            }
        )

    # Retrieval check: each end embedding should match its own start embedding.
    retrieval_correct = 0
    for i in range(n_compounds):
        dists = np.linalg.norm(first - last[i], axis=1)
        if int(np.argmin(dists)) == i:
            retrieval_correct += 1
    retrieval_top1 = retrieval_correct / n_compounds

    between_distances = []
    between_cosine = []
    between_angular = []
    for i in range(n_compounds):
        for j in range(n_compounds):
            if i == j:
                continue
            between_distances.append(float(np.linalg.norm(last[i] - first[j])))
            between_cosine.append(_cosine_distance(first[j], last[i]))
            between_angular.append(_angular_distance(first[j], last[i]))

    report = {
        "n_compounds": n_compounds,
        "n_durations": n_durations,
        "durations": durations,
        "mean_within_euclidean": float(np.mean(within_distances)),
        "mean_between_euclidean": float(np.mean(between_distances)),
        "within_between_euclidean_ratio": float(
            np.mean(within_distances) / (np.mean(between_distances) + 1e-8)
        ),
        "mean_within_cosine_distance": float(np.mean(within_cosine)),
        "mean_between_cosine_distance": float(np.mean(between_cosine)),
        "mean_within_angular_distance": float(np.mean(within_angular)),
        "mean_between_angular_distance": float(np.mean(between_angular)),
        "retrieval_top1_accuracy": float(retrieval_top1),
    }

    per_df = pd.DataFrame(per_rows)
    per_csv = os.path.join(output_dir, "per_compound_duration_transition.csv")
    per_df.to_csv(per_csv, index=False)

    report_df = pd.DataFrame([report])
    report_csv = os.path.join(output_dir, "duration_transition_summary.csv")
    report_df.to_csv(report_csv, index=False)

    print("=" * 72)
    print("DURATION TRANSITION TEST")
    print("=" * 72)
    print(f"Compounds with all durations: {n_compounds}")
    print(f"Durations: {durations}")
    print(f"Mean within-compound Euclidean shift : {report['mean_within_euclidean']:.4f}")
    print(f"Mean between-compound Euclidean shift: {report['mean_between_euclidean']:.4f}")
    print(f"Within/Between Euclidean ratio       : {report['within_between_euclidean_ratio']:.4f}")
    print(f"Retrieval top-1 accuracy             : {report['retrieval_top1_accuracy']:.4f}")
    print(f"Saved per-compound metrics           : {per_csv}")
    print(f"Saved summary metrics                : {report_csv}")
    print("=" * 72)


def plot_trajectories_pca(
    trajectories: Dict[str, np.ndarray],
    durations: List[int],
    output_dir: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    all_points = np.concatenate(list(trajectories.values()), axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    n_compounds = len(trajectories)
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral", n_compounds)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, (compound, traj) in enumerate(trajectories.items()):
        coords = pca.transform(traj)
        color = cmap(idx / max(n_compounds - 1, 1))

        ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.7, linewidth=1.2)
        sizes = np.linspace(35, 140, len(durations))
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=[color] * len(durations),
                   edgecolors="k", linewidths=0.3)

        for i in range(len(coords) - 1):
            ax.annotate(
                "",
                xy=(coords[i + 1, 0], coords[i + 1, 1]),
                xytext=(coords[i, 0], coords[i, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.65),
            )

    for i, duration in enumerate(durations):
        ax.scatter(
            [],
            [],
            s=np.linspace(35, 140, len(durations))[i],
            c="gray",
            edgecolors="k",
            linewidths=0.3,
            label=f"day {duration}",
        )

    ax.legend(title="Duration", fontsize=8, loc="best")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(
        f"Compound embedding transitions across durations (PCA)\n"
        f"{n_compounds} compounds x {len(durations)} durations"
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, "duration_trajectories_pca.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA trajectory plot             : {out_path}")


def main() -> None:
    args = parse_args()

    durations = [int(d) for d in args.durations]
    if len(durations) < 2:
        raise ValueError("Need at least 2 durations to evaluate transitions")

    duration_plate_map = load_duration_plate_map(
        metadata_csv=args.metadata_csv,
        compound_col=args.compound_col,
        plate_col=args.plate_col,
        duration_col=args.duration_col,
        durations=durations,
    )

    trajectories = build_duration_trajectories(
        embeddings_path=args.embeddings,
        duration_plate_map=duration_plate_map,
        durations=durations,
        subtract_control=args.subtract_control,
        normalize_before_subtract=args.normalize_before_subtract,
        max_compounds=args.max_compounds,
    )

    if not trajectories:
        raise RuntimeError(
            "No trajectories constructed. Check compound IDs, plate IDs, and durations "
            "between metadata CSV and embeddings file."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_duration_transition(trajectories, durations, args.output_dir)
    plot_trajectories_pca(trajectories, durations, args.output_dir)


if __name__ == "__main__":
    main()
