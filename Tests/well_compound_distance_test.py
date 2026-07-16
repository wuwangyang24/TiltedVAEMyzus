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
) -> Tuple[np.ndarray, List[str]]:
    """Compute mean embedding per well for each compound.

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
    print(f"\nEncoding wells (min_wells={args.min_wells})...")
    well_embeddings, well_labels = compute_well_mean_embeddings(
        metadata, root_dir, model, transform, mode,
        args.batch_size, device, args.min_wells, args.max_compounds,
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

    # ── Report statistics ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS ({args.metric} distance)")
    print(f"{'='*60}")
    print(f"  Within-compound  : mean = {within_dists.mean():.4f} "
          f"+/- {within_dists.std():.4f}  (n = {len(within_dists)})")
    print(f"  Between-compound : mean = {between_dists.mean():.4f} "
          f"+/- {between_dists.std():.4f}  (n = {len(between_dists)})")
    print(f"  Ratio (within/between) : {within_dists.mean() / between_dists.mean():.4f}")

    # Mann-Whitney U test (one-sided: within < between)
    u_stat, p_value = scipy_stats.mannwhitneyu(
        within_dists, between_dists, alternative="less"
    )
    print(f"\n  Mann-Whitney U test (within < between):")
    print(f"    U-statistic = {u_stat:.1f}")
    print(f"    p-value     = {p_value:.2e}")

    passed = p_value < 0.05
    print(f"\n  TEST {'PASSED' if passed else 'FAILED'}: "
          f"Within-compound distances are "
          f"{'significantly' if passed else 'NOT significantly'} "
          f"smaller than between-compound distances (p < 0.05).")
    print(f"{'='*60}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"well_compound_distances_{args.metric}.png")
    plot_distance_distributions(within_dists, between_dists, args.metric, plot_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
