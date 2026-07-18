"""Concentration-transition test for embedding quality.

Idea: if the encoder captures biologically meaningful information, embeddings of
the same compound at different concentrations should form a smooth, monotonic
trajectory in latent space — i.e. increasing concentration should produce a
gradual shift in the embedding, not random jumps.

This script:
  1. Loads pre-computed embeddings from multiple concentration levels (one .pt
     file per concentration, produced by encode_embeddings.py).
  2. For each compound present in ALL concentration levels, computes the mean
     embedding per concentration (averaged across plates/wells).
  3. Optionally subtracts plate-level control embeddings before averaging.
  4. Visualizes the concentration trajectories using PCA, t-SNE, and UMAP:
     - Each compound is a line/trajectory colored by compound ID.
     - Points along the trajectory are sized/annotated by concentration.
  5. Quantifies transition smoothness:
     - Monotonicity score: fraction of compounds whose embedding norms increase
       monotonically with concentration.
     - Cosine alignment: for each compound, checks that successive concentration
       steps point in a consistent direction.
     - Path straightness: ratio of end-to-end distance to total path length.

Embedding .pt file structure (from encode_embeddings.py):
    {
        <compound_id (str)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # (N, D)
                "control": torch.Tensor    # (D,)
            }
        }
    }

Usage:
python TiltedVAEMyzus/Tests/concentration_transition_test.py --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/embeddings_4ppm.pt TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/embeddings_20ppm.pt TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/embeddings_100ppm.pt --concentrations 4 20 100 --subtract_control --normalize_before_subtract --output_dir results/concentration_transition_test
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add repo root to path so Models package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test that embeddings of the same compound at different "
                    "concentrations form smooth trajectories in latent space")

    # Embedding files (one per concentration)
    parser.add_argument("--embeddings", type=str, nargs="+", required=True,
                        help="Paths to pre-computed embedding .pt files, one per "
                             "concentration (in order of increasing concentration)")
    parser.add_argument("--concentrations", type=float, nargs="+", required=True,
                        help="Concentration values corresponding to each embedding "
                             "file (same order). Used as labels in plots.")

    # Processing options
    parser.add_argument("--subtract_control", action="store_true",
                        help="Subtract the plate-level mean control embedding from "
                             "each treated well embedding before computing means")
    parser.add_argument("--normalize_before_subtract", action="store_true",
                        help="L2-normalize treated and control mean embeddings "
                             "before subtracting control")
    parser.add_argument("--max_compounds", type=int, default=None,
                        help="Limit the number of compounds to visualize")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="results/concentration_transition_test",
                        help="Directory to save result plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=None,
                        help="Number of compounds to randomly sample for "
                             "trajectory plots. If None, plot all compounds. "
                             "Metrics are always computed on all compounds.")

    args = parser.parse_args()

    if len(args.embeddings) != len(args.concentrations):
        parser.error("--embeddings and --concentrations must have the same "
                     "number of entries")
    if len(args.embeddings) < 2:
        parser.error("Need at least 2 concentration levels to test transitions")

    return args


def load_compound_means(
    embedding_path: str,
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Dict[str, np.ndarray]:
    """Load a .pt embedding file and compute the mean embedding per compound.

    Averages over all plates/wells for each compound.

    Returns:
        Dictionary mapping compound_id -> mean embedding (D,).
    """
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)

    compound_means: Dict[str, np.ndarray] = {}

    for compound_id, plate_dict in data.items():
        compound_id = str(compound_id)
        well_embeddings: List[np.ndarray] = []

        for plate_id, plate_data in plate_dict.items():
            treated = plate_data.get("treated", None)
            if treated is None or treated.numel() == 0:
                continue

            well_mean = treated.mean(dim=0).numpy()

            if subtract_control:
                control = plate_data.get("control", None)
                if control is not None and control.numel() > 0:
                    ctrl_mean = control.numpy()
                    if ctrl_mean.ndim > 1:
                        ctrl_mean = ctrl_mean.mean(axis=0)
                    if normalize_before_subtract:
                        well_mean = well_mean / (np.linalg.norm(well_mean) + 1e-8)
                        ctrl_mean = ctrl_mean / (np.linalg.norm(ctrl_mean) + 1e-8)
                    well_mean = well_mean - ctrl_mean

            well_embeddings.append(well_mean)

        if well_embeddings:
            compound_means[compound_id] = np.mean(well_embeddings, axis=0)

    return compound_means


def build_trajectories(
    embedding_files: List[str],
    concentrations: List[float],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
    max_compounds: "int | None" = None,
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """Load embeddings from all concentration levels and build trajectories.

    Returns:
        trajectories: dict mapping compound_id -> (n_concentrations, D) array
            of mean embeddings ordered by concentration.
        concentrations: list of concentration values (sorted ascending).
    """
    # Sort by concentration
    sorted_pairs = sorted(zip(concentrations, embedding_files))
    sorted_concs = [c for c, _ in sorted_pairs]
    sorted_files = [f for _, f in sorted_pairs]

    # Load mean embeddings for each concentration
    all_means: List[Dict[str, np.ndarray]] = []
    for emb_path in sorted_files:
        print(f"  Loading {emb_path}...")
        means = load_compound_means(emb_path, subtract_control,
                                    normalize_before_subtract)
        all_means.append(means)
        print(f"    -> {len(means)} compounds")

    # Find compounds present in ALL concentration levels
    common_compounds = set(all_means[0].keys())
    for means in all_means[1:]:
        common_compounds &= set(means.keys())

    common_compounds = sorted(common_compounds)
    print(f"\n  Compounds present in all {len(sorted_concs)} levels: "
          f"{len(common_compounds)}")

    if max_compounds is not None and len(common_compounds) > max_compounds:
        common_compounds = common_compounds[:max_compounds]
        print(f"  Limited to {max_compounds} compounds")

    # Build trajectory array for each compound
    trajectories: Dict[str, np.ndarray] = {}
    for compound_id in common_compounds:
        traj = np.stack([means[compound_id] for means in all_means], axis=0)
        trajectories[compound_id] = traj

    return trajectories, sorted_concs


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothness metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_smoothness_metrics(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
) -> Dict[str, float]:
    """Compute metrics quantifying the smoothness of concentration transitions.

    Metrics:
      - norm_monotonicity: fraction of compounds whose embedding norms change
        monotonically (either increasing or decreasing) with concentration.
      - cosine_alignment: mean cosine similarity between consecutive step
        directions across compounds (1.0 = perfectly straight trajectory).
      - path_straightness: mean ratio of end-to-end Euclidean distance to total
        path length (1.0 = perfectly straight, 0.0 = returns to start).
    """
    n_concs = len(concentrations)
    monotonic_count = 0
    cosine_alignments: List[float] = []
    straightness_ratios: List[float] = []

    for compound_id, traj in trajectories.items():
        # traj shape: (n_concs, D)

        # --- Norm monotonicity ---
        norms = np.linalg.norm(traj, axis=1)
        diffs = np.diff(norms)
        is_monotone = np.all(diffs >= 0) or np.all(diffs <= 0)
        if is_monotone:
            monotonic_count += 1

        # --- Cosine alignment of consecutive steps ---
        if n_concs >= 3:
            steps = np.diff(traj, axis=0)  # (n_concs-1, D)
            cos_sims = []
            for i in range(len(steps) - 1):
                s1 = steps[i]
                s2 = steps[i + 1]
                n1 = np.linalg.norm(s1)
                n2 = np.linalg.norm(s2)
                if n1 > 1e-8 and n2 > 1e-8:
                    cos_sims.append(float(np.dot(s1, s2) / (n1 * n2)))
            if cos_sims:
                cosine_alignments.append(np.mean(cos_sims))

        # --- Path straightness ---
        end_to_end = np.linalg.norm(traj[-1] - traj[0])
        path_length = sum(
            np.linalg.norm(traj[i + 1] - traj[i]) for i in range(n_concs - 1)
        )
        if path_length > 1e-8:
            straightness_ratios.append(end_to_end / path_length)

    metrics = {
        "norm_monotonicity": monotonic_count / max(len(trajectories), 1),
        "cosine_alignment": float(np.mean(cosine_alignments)) if cosine_alignments else float("nan"),
        "path_straightness": float(np.mean(straightness_ratios)) if straightness_ratios else float("nan"),
        "n_compounds": len(trajectories),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectories_pca(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
    output_dir: str,
) -> None:
    """Visualize concentration trajectories using PCA (2D).

    Each compound is drawn as a connected trajectory with points sized by
    concentration and arrows showing direction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Stack all points for PCA fitting
    all_points = np.concatenate(list(trajectories.values()), axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    n_compounds = len(trajectories)
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral",
                           n_compounds)

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (compound_id, traj) in enumerate(trajectories.items()):
        coords = pca.transform(traj)  # (n_concs, 2)
        color = cmap(idx / max(n_compounds - 1, 1))

        # Draw trajectory line
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.2,
                alpha=0.7, zorder=1)

        # Draw arrows between consecutive points
        for i in range(len(coords) - 1):
            dx = coords[i + 1, 0] - coords[i, 0]
            dy = coords[i + 1, 1] - coords[i, 1]
            ax.annotate("", xy=(coords[i + 1, 0], coords[i + 1, 1]),
                        xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2, alpha=0.7))

        # Draw points sized by concentration rank
        sizes = np.linspace(30, 150, len(concentrations))
        ax.scatter(coords[:, 0], coords[:, 1], c=[color] * len(coords),
                   s=sizes, edgecolors="k", linewidths=0.4, zorder=2)

    # Add concentration legend (point sizes)
    for i, conc in enumerate(concentrations):
        ax.scatter([], [], s=np.linspace(30, 150, len(concentrations))[i],
                   c="gray", edgecolors="k", linewidths=0.4,
                   label=f"{conc} ppm")
    ax.legend(title="Concentration", loc="best", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Concentration trajectories (PCA)\n"
                 f"{n_compounds} compounds × {len(concentrations)} concentrations")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "concentration_trajectories_PCA.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved PCA trajectory plot to {out_path}")


def plot_trajectories_tsne(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
    output_dir: str,
    seed: int = 42,
) -> None:
    """Visualize concentration trajectories using t-SNE (2D)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    all_points = np.concatenate(list(trajectories.values()), axis=0)
    n_points = len(all_points)
    perplexity = min(30, max(2, n_points - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca", learning_rate="auto")
    all_coords = tsne.fit_transform(all_points)

    n_concs = len(concentrations)
    n_compounds = len(trajectories)
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral",
                           n_compounds)

    fig, ax = plt.subplots(figsize=(10, 8))

    offset = 0
    for idx, (compound_id, traj) in enumerate(trajectories.items()):
        coords = all_coords[offset:offset + n_concs]
        offset += n_concs
        color = cmap(idx / max(n_compounds - 1, 1))

        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.2,
                alpha=0.7, zorder=1)

        for i in range(len(coords) - 1):
            ax.annotate("", xy=(coords[i + 1, 0], coords[i + 1, 1]),
                        xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2, alpha=0.7))

        sizes = np.linspace(30, 150, n_concs)
        ax.scatter(coords[:, 0], coords[:, 1], c=[color] * n_concs,
                   s=sizes, edgecolors="k", linewidths=0.4, zorder=2)

    for i, conc in enumerate(concentrations):
        ax.scatter([], [], s=np.linspace(30, 150, n_concs)[i],
                   c="gray", edgecolors="k", linewidths=0.4,
                   label=f"{conc} ppm")
    ax.legend(title="Concentration", loc="best", fontsize=8)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Concentration trajectories (t-SNE)\n"
                 f"{n_compounds} compounds × {n_concs} concentrations")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "concentration_trajectories_tSNE.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved t-SNE trajectory plot to {out_path}")


def plot_trajectories_umap(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
    output_dir: str,
    seed: int = 42,
) -> None:
    """Visualize concentration trajectories using UMAP (2D)."""
    try:
        import umap
    except ImportError:
        print("[plot] umap-learn not installed, skipping UMAP visualization.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = np.concatenate(list(trajectories.values()), axis=0)
    n_points = len(all_points)

    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=min(15, n_points - 1), min_dist=0.1)
    all_coords = reducer.fit_transform(all_points)

    n_concs = len(concentrations)
    n_compounds = len(trajectories)
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral",
                           n_compounds)

    fig, ax = plt.subplots(figsize=(10, 8))

    offset = 0
    for idx, (compound_id, traj) in enumerate(trajectories.items()):
        coords = all_coords[offset:offset + n_concs]
        offset += n_concs
        color = cmap(idx / max(n_compounds - 1, 1))

        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.2,
                alpha=0.7, zorder=1)

        for i in range(len(coords) - 1):
            ax.annotate("", xy=(coords[i + 1, 0], coords[i + 1, 1]),
                        xytext=(coords[i, 0], coords[i, 1]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2, alpha=0.7))

        sizes = np.linspace(30, 150, n_concs)
        ax.scatter(coords[:, 0], coords[:, 1], c=[color] * n_concs,
                   s=sizes, edgecolors="k", linewidths=0.4, zorder=2)

    for i, conc in enumerate(concentrations):
        ax.scatter([], [], s=np.linspace(30, 150, n_concs)[i],
                   c="gray", edgecolors="k", linewidths=0.4,
                   label=f"{conc} ppm")
    ax.legend(title="Concentration", loc="best", fontsize=8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Concentration trajectories (UMAP)\n"
                 f"{n_compounds} compounds × {n_concs} concentrations")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "concentration_trajectories_UMAP.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved UMAP trajectory plot to {out_path}")


def plot_norm_vs_concentration(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
    output_dir: str,
) -> None:
    """Plot embedding norm vs concentration for each compound.

    Shows whether the magnitude of the embedding changes smoothly with
    concentration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_compounds = len(trajectories)
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral",
                           n_compounds)

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (compound_id, traj) in enumerate(trajectories.items()):
        norms = np.linalg.norm(traj, axis=1)
        color = cmap(idx / max(n_compounds - 1, 1))
        ax.plot(concentrations, norms, marker="o", color=color, alpha=0.6,
                linewidth=1.2, markersize=5)

    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("||μ|| (embedding norm)")
    ax.set_title(f"Embedding norm vs concentration\n({n_compounds} compounds)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "norm_vs_concentration.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved norm vs concentration plot to {out_path}")


def plot_pairwise_distance_vs_concentration_gap(
    trajectories: Dict[str, np.ndarray],
    concentrations: List[float],
    output_dir: str,
) -> None:
    """Plot average pairwise distance between concentration steps.

    For each step (conc_i -> conc_{i+1}), computes the mean Euclidean distance
    across all compounds. If transitions are smooth, larger concentration gaps
    should produce larger distances.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_concs = len(concentrations)
    step_labels = []
    mean_distances = []
    std_distances = []

    for i in range(n_concs - 1):
        dists = []
        for compound_id, traj in trajectories.items():
            d = np.linalg.norm(traj[i + 1] - traj[i])
            dists.append(d)
        step_labels.append(f"{concentrations[i]:.0f}→{concentrations[i+1]:.0f}")
        mean_distances.append(np.mean(dists))
        std_distances.append(np.std(dists))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(step_labels))
    ax.bar(x, mean_distances, yerr=std_distances, capsize=5,
           color="#1f77b4", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels)
    ax.set_xlabel("Concentration step (ppm)")
    ax.set_ylabel("Mean Euclidean distance in latent space")
    ax.set_title("Mean embedding distance per concentration step\n"
                 f"({len(trajectories)} compounds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "distance_vs_concentration_step.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved distance vs concentration step to {out_path}")


def plot_straightness_histogram(
    trajectories: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Histogram of path straightness ratios across compounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = []
    for traj in trajectories.values():
        end_to_end = np.linalg.norm(traj[-1] - traj[0])
        path_length = sum(
            np.linalg.norm(traj[i + 1] - traj[i]) for i in range(len(traj) - 1)
        )
        if path_length > 1e-8:
            ratios.append(end_to_end / path_length)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=30, alpha=0.7, color="#2ca02c", edgecolor="k",
            linewidth=0.5)
    ax.axvline(np.mean(ratios), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(ratios):.3f}")
    ax.set_xlabel("Path straightness (end-to-end / total path length)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory straightness across compounds\n"
                 "(1.0 = perfectly straight, 0.0 = returns to start)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(output_dir, "straightness_histogram.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved straightness histogram to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build trajectories ───────────────────────────────────────────────────
    print("Loading embeddings and building concentration trajectories...")
    trajectories, concentrations = build_trajectories(
        args.embeddings, args.concentrations,
        subtract_control=args.subtract_control,
        normalize_before_subtract=args.normalize_before_subtract,
        max_compounds=args.max_compounds,
    )

    if len(trajectories) == 0:
        print("ERROR: No compounds found in all concentration levels.")
        sys.exit(1)

    # ── Compute smoothness metrics ───────────────────────────────────────────
    print("\nComputing smoothness metrics...")
    metrics = compute_smoothness_metrics(trajectories, concentrations)

    print(f"\n{'='*70}")
    print("CONCENTRATION TRANSITION TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Compounds tested     : {metrics['n_compounds']}")
    print(f"  Concentration levels : {concentrations}")
    print(f"  Norm monotonicity    : {metrics['norm_monotonicity']:.3f} "
          f"({metrics['norm_monotonicity']*100:.1f}% of compounds)")
    print(f"  Cosine alignment     : {metrics['cosine_alignment']:.4f} "
          f"(1.0 = perfectly consistent direction)")
    print(f"  Path straightness    : {metrics['path_straightness']:.4f} "
          f"(1.0 = perfectly straight trajectory)")
    print(f"{'='*70}")

    # ── Visualizations ───────────────────────────────────────────────────────
    print("\nGenerating visualizations...")

    # Optionally subsample compounds for plotting (metrics use all data)
    plot_trajs = trajectories
    if args.n_sample is not None and args.n_sample < len(trajectories):
        rng = np.random.default_rng(args.seed)
        sampled_keys = rng.choice(
            sorted(trajectories.keys()), size=args.n_sample, replace=False,
        )
        plot_trajs = {k: trajectories[k] for k in sampled_keys}
        print(f"  Subsampled {args.n_sample} / {len(trajectories)} compounds for plots")

    plot_trajectories_pca(plot_trajs, concentrations, args.output_dir)
    plot_trajectories_tsne(plot_trajs, concentrations, args.output_dir,
                           seed=args.seed)
    plot_trajectories_umap(plot_trajs, concentrations, args.output_dir,
                           seed=args.seed)
    plot_norm_vs_concentration(plot_trajs, concentrations, args.output_dir)
    plot_pairwise_distance_vs_concentration_gap(
        plot_trajs, concentrations, args.output_dir)
    plot_straightness_histogram(plot_trajs, args.output_dir)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
