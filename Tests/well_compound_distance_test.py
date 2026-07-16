"""Well-vs-compound distance test for embedding quality.

Idea: if the encoder produces biologically meaningful embeddings, images from
different wells of the *same* compound (biological replicates) should have more
similar mean embeddings than images from *different* compounds.

This script:
  1. Loads a metadata JSON mapping compounds -> plates -> treated image paths.
  2. Encodes all treated images per well using the trained VAE/TiltedVAE encoder.
  3. Computes the mean embedding per well.
  4. Compares:
       - within-compound distances: pairwise distances between mean well
         embeddings that belong to the SAME compound.
       - between-compound distances: pairwise distances between mean well
         embeddings that belong to DIFFERENT compounds.
  5. Reports summary statistics and runs a Mann-Whitney U test to check that
     within-compound distances are significantly smaller.

Metadata format (same as encode_embeddings.py):
    [
        {
            "Compound": "1",
            "94000": {
                "treated": ["94000/well_2_1/treated/sample_1.png", ...],
                "control": [...]
            },
            "131000": { "treated": [...], "control": [...] }
        },
        ...
    ]

Each plate entry for a compound is treated as a separate "well" (biological
replicate). A compound must have at least 2 wells to contribute within-compound
distances.

Usage:
python Tests/well_compound_distance_test.py --metadata ../METADATA/metadata_compound_all100ppm.json --root_dir ../DATA_TEST/ --checkpoint results/checkpoints/tilted-latent128/best_balanced_acc.ckpt --model tilted --latent_dim 128 --img_size 96 --device cpu --max_compounds 10
"""
import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from scipy import stats as scipy_stats
from torchvision.io import ImageReadMode, read_image

# Add repo root to path so Models package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test that same-compound well embeddings are closer than "
                    "different-compound embeddings")

    # Data
    parser.add_argument("--metadata", type=str, required=True,
                        help="JSON metadata file mapping compounds -> plates -> paths")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Base directory prepended to relative image paths")

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained Lightning checkpoint (.ckpt) or state_dict (.pt/.pth)")
    parser.add_argument("--model", type=str, default="tilted",
                        choices=["vae", "tilted"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # Test config
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Distance metric for comparing mean embeddings")
    parser.add_argument("--max_compounds", type=int, default=None,
                        help="Limit the number of compounds to process (for speed)")
    parser.add_argument("--min_wells", type=int, default=2,
                        help="Minimum wells per compound to include it (default: 2)")
    parser.add_argument("--subtract_control", action="store_true",
                        help="Subtract the plate-level mean control embedding from "
                             "each treated well embedding before computing distances")
    parser.add_argument("--n_permutations", type=int, default=10000,
                        help="Number of permutations for the permutation test (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--output_dir", type=str, default="results/well_compound_test",
                        help="Directory to save result plots")

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "tilted":
        return TiltedVAE(
            in_channels=args.in_channels,
            latent_dim=args.latent_dim,
            tau=args.tau,
            img_size=args.img_size,
        )
    return VAE(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        img_size=args.img_size,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load weights from either a Lightning checkpoint or a raw state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load] Missing keys ({len(missing)}): {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load] Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
              f"{' ...' if len(unexpected) > 5 else ''}")


@torch.no_grad()
def encode_paths(
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
        mu, _ = model.encode(batch)
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0) if latents else torch.empty(0)


def compute_pairwise_distances(
    embeddings: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Compute pairwise distances between rows of an (N, D) array.
    Returns the upper-triangle distances as a 1D array."""
    from scipy.spatial.distance import pdist
    return pdist(embeddings, metric=metric)


def compute_well_mean_embeddings(
    metadata: List[dict],
    root_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    mode: ImageReadMode,
    batch_size: int,
    device: torch.device,
    min_wells: int,
    max_compounds: int | None,
    subtract_control: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Compute mean embedding per well for each compound.

    When ``subtract_control`` is True, the plate-level mean control embedding
    is subtracted from the treated well mean embedding, removing plate-specific
    batch effects.

    Returns:
        well_embeddings: (W, D) array of mean embeddings, one per well.
        well_compound_labels: list of compound IDs, one per well.
    """
    well_embeddings: List[np.ndarray] = []
    well_compound_labels: List[str] = []

    compounds_processed = 0
    for entry in metadata:
        compound_id = str(entry["Compound"])
        wells_for_compound: List[np.ndarray] = []

        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            treated_paths = plate_data.get("treated", [])
            if not treated_paths:
                continue

            latents = encode_paths(
                treated_paths, root_dir, model, transform, mode,
                batch_size, device,
            )
            if latents.numel() == 0:
                continue
            well_mean = latents.mean(dim=0).numpy()

            # Optionally subtract plate-level control mean
            if subtract_control:
                control_paths = plate_data.get("control", [])
                if control_paths:
                    ctrl_latents = encode_paths(
                        control_paths, root_dir, model, transform, mode,
                        batch_size, device,
                    )
                    if ctrl_latents.numel() > 0:
                        ctrl_mean = ctrl_latents.mean(dim=0).numpy()
                        well_mean = well_mean - ctrl_mean

            wells_for_compound.append(well_mean)

        if len(wells_for_compound) >= min_wells:
            for emb in wells_for_compound:
                well_embeddings.append(emb)
                well_compound_labels.append(compound_id)
            compounds_processed += 1

        if max_compounds is not None and compounds_processed >= max_compounds:
            break

    return np.array(well_embeddings), well_compound_labels


def compute_within_between_distances(
    well_embeddings: np.ndarray,
    well_compound_labels: List[str],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate pairwise distances into within-compound and between-compound.

    Returns:
        within_distances: distances between wells of the same compound.
        between_distances: distances between wells of different compounds.
    """
    n = len(well_compound_labels)
    within: List[float] = []
    between: List[float] = []

    for i, j in combinations(range(n), 2):
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            dot = np.dot(well_embeddings[i], well_embeddings[j])
            norm_i = np.linalg.norm(well_embeddings[i])
            norm_j = np.linalg.norm(well_embeddings[j])
            dist = 1.0 - dot / (norm_i * norm_j + 1e-8)
        else:
            dist = float(np.linalg.norm(well_embeddings[i] - well_embeddings[j]))

        if well_compound_labels[i] == well_compound_labels[j]:
            within.append(dist)
        else:
            between.append(dist)

    return np.array(within), np.array(between)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_mann_whitney(
    within: np.ndarray, between: np.ndarray
) -> Tuple[float, float]:
    """Mann-Whitney U test (one-sided: within < between).

    Returns:
        u_stat: U-statistic.
        p_value: one-sided p-value.
    """
    u_stat, p_value = scipy_stats.mannwhitneyu(
        within, between, alternative="less"
    )
    return float(u_stat), float(p_value)


def test_permutation(
    well_embeddings: np.ndarray,
    well_compound_labels: List[str],
    metric: str,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Permutation test: shuffle compound labels and recompute the difference
    in mean within-compound vs between-compound distances.

    The test statistic is: mean(between) - mean(within).  Under the null
    hypothesis (labels don't matter), this should be ~0.  A large observed
    value indicates that within-compound distances are genuinely smaller.

    Returns:
        observed_diff: observed mean(between) - mean(within).
        p_value: fraction of permutations with diff >= observed_diff.
        null_mean: mean of the null distribution.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(well_compound_labels)
    n = len(labels)

    # Pre-compute full pairwise distance matrix (upper triangle)
    from scipy.spatial.distance import pdist, squareform
    dist_vec = pdist(well_embeddings, metric=metric)
    dist_mat = squareform(dist_vec)

    # Helper: compute mean within and between from a label assignment
    def _compute_diff(lab: np.ndarray) -> float:
        within_sum = 0.0
        within_count = 0
        between_sum = 0.0
        between_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_mat[i, j]
                if lab[i] == lab[j]:
                    within_sum += d
                    within_count += 1
                else:
                    between_sum += d
                    between_count += 1
        w_mean = within_sum / within_count if within_count > 0 else 0.0
        b_mean = between_sum / between_count if between_count > 0 else 0.0
        return b_mean - w_mean

    observed_diff = _compute_diff(labels)

    # Permutation loop
    count_ge = 0
    null_diffs: List[float] = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        d = _compute_diff(perm_labels)
        null_diffs.append(d)
        if d >= observed_diff:
            count_ge += 1

    p_value = count_ge / n_permutations
    null_mean = float(np.mean(null_diffs))
    return observed_diff, p_value, null_mean


def test_welch_ttest(
    within: np.ndarray, between: np.ndarray
) -> Tuple[float, float]:
    """Welch's t-test (one-sided: within < between).

    Returns:
        t_stat: t-statistic.
        p_value: one-sided p-value.
    """
    t_stat, p_two = scipy_stats.ttest_ind(within, between, equal_var=False)
    # One-sided: we expect within < between => t_stat negative
    p_value = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2
    return float(t_stat), float(p_value)


def test_kolmogorov_smirnov(
    within: np.ndarray, between: np.ndarray
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (one-sided: within distribution is
    stochastically less than between distribution).

    Returns:
        ks_stat: KS statistic.
        p_value: one-sided p-value.
    """
    ks_stat, p_value = scipy_stats.ks_2samp(within, between, alternative="less")
    return float(ks_stat), float(p_value)


def plot_distance_distributions(
    within: np.ndarray,
    between: np.ndarray,
    metric: str,
    output_path: str,
) -> None:
    """Plot histograms of within-compound vs between-compound distances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(within, bins=50, alpha=0.6, label=f"Within-compound (n={len(within)})",
            color="#2ca02c", density=True)
    ax.hist(between, bins=50, alpha=0.6, label=f"Between-compound (n={len(between)})",
            color="#d62728", density=True)
    ax.axvline(within.mean(), color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"Within mean = {within.mean():.4f}")
    ax.axvline(between.mean(), color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Between mean = {between.mean():.4f}")
    ax.set_xlabel(f"{metric.capitalize()} distance")
    ax.set_ylabel("Density")
    ax.set_title("Well-level mean embedding distances:\nWithin-compound vs Between-compound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved distance histogram to {output_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model  : {args.model}  (latent dim {args.latent_dim})")

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(args.metadata) as f:
        metadata = json.load(f)
    print(f"Metadata: {len(metadata)} compounds")

    root_dir = Path(args.root_dir)
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ])
    mode = ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB

    # ── Compute mean embeddings per well ─────────────────────────────────────
    ctrl_msg = " (subtract_control=True)" if args.subtract_control else ""
    print(f"\nEncoding wells (min_wells={args.min_wells}){ctrl_msg}...")
    well_embeddings, well_labels = compute_well_mean_embeddings(
        metadata, root_dir, model, transform, mode,
        args.batch_size, device, args.min_wells, args.max_compounds,
        subtract_control=args.subtract_control,
    )

    n_wells = len(well_labels)
    n_compounds = len(set(well_labels))
    print(f"Encoded {n_wells} wells from {n_compounds} compounds")

    if n_wells < 3:
        print("ERROR: Not enough wells to run the test (need at least 3).")
        sys.exit(1)

    # ── Compute within/between distances ─────────────────────────────────────
    print(f"\nComputing pairwise {args.metric} distances...")
    within_dists, between_dists = compute_within_between_distances(
        well_embeddings, well_labels, args.metric,
    )

    if len(within_dists) == 0:
        print("ERROR: No within-compound pairs found. Need compounds with >= 2 wells.")
        sys.exit(1)

    # ── Run statistical tests ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS ({args.metric} distance)")
    print(f"{'='*70}")
    print(f"  Within-compound  : mean = {within_dists.mean():.4f} "
          f"+/- {within_dists.std():.4f}  (n = {len(within_dists)})")
    print(f"  Between-compound : mean = {between_dists.mean():.4f} "
          f"+/- {between_dists.std():.4f}  (n = {len(between_dists)})")
    print(f"  Ratio (within/between) : {within_dists.mean() / between_dists.mean():.4f}")

    results = {}  # test_name -> (stat, p_value)

    # 1. Mann-Whitney U test
    u_stat, mw_p = test_mann_whitney(within_dists, between_dists)
    results["Mann-Whitney U"] = (u_stat, mw_p)
    print(f"\n  [1] Mann-Whitney U test (within < between):")
    print(f"      U-statistic = {u_stat:.1f}")
    print(f"      p-value     = {mw_p:.2e}")
    print(f"      {'PASSED' if mw_p < 0.05 else 'FAILED'} (p < 0.05)")

    # 2. Permutation test
    print(f"\n  [2] Permutation test ({args.n_permutations:,} permutations)...")
    obs_diff, perm_p, null_mean = test_permutation(
        well_embeddings, well_labels, args.metric,
        n_permutations=args.n_permutations, seed=args.seed,
    )
    results["Permutation"] = (obs_diff, perm_p)
    print(f"      Observed diff (between-within) = {obs_diff:.4f}")
    print(f"      Null mean                      = {null_mean:.4f}")
    print(f"      p-value                        = {perm_p:.4f}")
    print(f"      {'PASSED' if perm_p < 0.05 else 'FAILED'} (p < 0.05)")

    # 3. Welch's t-test
    t_stat, tt_p = test_welch_ttest(within_dists, between_dists)
    results["Welch's t-test"] = (t_stat, tt_p)
    print(f"\n  [3] Welch's t-test (within < between):")
    print(f"      t-statistic = {t_stat:.4f}")
    print(f"      p-value     = {tt_p:.2e}")
    print(f"      {'PASSED' if tt_p < 0.05 else 'FAILED'} (p < 0.05)")

    # 4. Kolmogorov-Smirnov test
    ks_stat, ks_p = test_kolmogorov_smirnov(within_dists, between_dists)
    results["Kolmogorov-Smirnov"] = (ks_stat, ks_p)
    print(f"\n  [4] Kolmogorov-Smirnov test (within < between):")
    print(f"      KS-statistic = {ks_stat:.4f}")
    print(f"      p-value      = {ks_p:.2e}")
    print(f"      {'PASSED' if ks_p < 0.05 else 'FAILED'} (p < 0.05)")

    # ── Summary ──────────────────────────────────────────────────────────────
    all_passed = all(p < 0.05 for _, p in results.values())
    n_passed = sum(1 for _, p in results.values() if p < 0.05)
    print(f"\n{'─'*70}")
    print(f"  SUMMARY: {n_passed}/{len(results)} tests passed (p < 0.05)")
    print(f"  Overall: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*70}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"well_compound_distances_{args.metric}.png")
    plot_distance_distributions(within_dists, between_dists, args.metric, plot_path)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
