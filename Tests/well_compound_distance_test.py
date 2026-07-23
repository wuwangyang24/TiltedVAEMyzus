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
python TiltedVAEMyzus/Tests/well_compound_distance_test.py --metadata METADATA/metadata_compound_all100ppm.json --root_dir DATA_TEST/ --embedding TiltedVAEMyzus/results/checkpoints/tilted-latent128_kl0.01/embeddings_best_balanced_acc.pt --model tilted --latent_dim 128 --img_size 96 --device cpu --max_compounds 1000 --subtract_control --normalize_before_subtract --metric radial 
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


class DinoV2Wrapper(torch.nn.Module):
    """Thin wrapper around pretrained DINOv2 with a VAE-like encode API."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMG_SIZE = 224

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN,
                                     std=self.IMAGENET_STD)

    def encode(self, x: torch.Tensor):
        x = self.normalize(x)
        features = self.backbone(x)  # (B, D), D=384 for vits14
        return features, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test that same-compound well embeddings are closer than "
                    "different-compound embeddings")

    # Data
    parser.add_argument("--metadata", type=str, required=True,
                        help="JSON metadata file mapping compounds -> plates -> paths")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Base directory prepended to relative image paths "
                             "(required unless --embedding is provided)")

    # Pre-computed embeddings (skip encoding)
    parser.add_argument("--embedding", type=str, default=None,
                        help="Path to a pre-computed embedding .pt file from "
                             "encode_embeddings.py. If provided, skips model loading "
                             "and encoding entirely.")

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained Lightning checkpoint (.ckpt) or state_dict (.pt/.pth) "
                            "(required unless --embedding is provided or --model dino)")
    parser.add_argument("--model", type=str, default="tilted",
                        choices=["vae", "tilted", "dino"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # Test config
    parser.add_argument("--metric", type=str, default="angular",
                        choices=["euclidean", "cosine", "angular", "radial"],
                        help="Distance metric for comparing mean embeddings. "
                             "'angular' (default) computes geodesic distance on the "
                             "hypersphere. 'radial' computes | ||mu_i|| - ||mu_j|| |, "
                             "the difference in radius from origin.")
    parser.add_argument("--max_compounds", type=int, default=None,
                        help="Limit the number of compounds to process (for speed)")
    parser.add_argument("--min_wells", type=int, default=2,
                        help="Minimum wells per compound to include it (default: 2)")
    parser.add_argument("--subtract_control", action="store_true",
                        help="Subtract the plate-level mean control embedding from "
                             "each treated well embedding before computing distances")
    parser.add_argument("--normalize_before_subtract", action="store_true",
                        help="L2-normalize treated and control mean embeddings "
                             "before subtracting control (only used with --subtract_control)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--output_dir", type=str, default="results/well_compound_test",
                        help="Directory to save result plots")

    args = parser.parse_args()

    # Validate: either --embedding or (--checkpoint + --root_dir) must be given
    if args.embedding is None:
        if args.model != "dino" and args.checkpoint is None:
            parser.error("--checkpoint is required when --embedding is not provided and --model is not dino")
        if args.root_dir is None:
            parser.error("--root_dir is required when --embedding is not provided")

    if args.model == "dino":
        if args.in_channels != 3:
            parser.error("--model dino requires --in_channels 3")
        args.img_size = DinoV2Wrapper.IMG_SIZE

    return args


def load_well_means_from_embedding(
    embedding_path: str,
    metadata: List[dict],
    min_wells: int,
    max_compounds: "int | None",
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load pre-computed embeddings from a .pt file and compute well means.

    The .pt file has the structure produced by encode_embeddings.py:
        { compound_id: { plate_id: { "treated": (N,D), "control": (D,) } } }

    Returns the same format as compute_well_mean_embeddings.
    """
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)

    well_embeddings: List[np.ndarray] = []
    well_compound_labels: List[str] = []
    well_plate_labels: List[str] = []

    compounds_processed = 0
    for entry in metadata:
        compound_id = str(entry["Compound"])
        if compound_id not in data:
            continue

        wells_for_compound: List[Tuple[np.ndarray, str]] = []
        compound_data = data[compound_id]

        for plate_id, plate_data in compound_data.items():
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

            wells_for_compound.append((well_mean, str(plate_id)))

        if len(wells_for_compound) >= min_wells:
            for emb, plate in wells_for_compound:
                well_embeddings.append(emb)
                well_compound_labels.append(compound_id)
                well_plate_labels.append(plate)
            compounds_processed += 1

        if max_compounds is not None and compounds_processed >= max_compounds:
            break

    return np.array(well_embeddings), well_compound_labels, well_plate_labels


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "dino":
        return DinoV2Wrapper()
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


def compute_well_mean_embeddings(
    metadata: List[dict],
    root_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    mode: ImageReadMode,
    batch_size: int,
    device: torch.device,
    min_wells: int,
    max_compounds: "int | None",
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Compute mean embedding per well for each compound.

    When ``subtract_control`` is True, the plate-level mean control embedding
    is subtracted from the treated well mean embedding, removing plate-specific
    batch effects.

    Returns:
        well_embeddings: (W, D) array of mean embeddings, one per well.
        well_compound_labels: list of compound IDs, one per well.
        well_plate_labels: list of plate IDs, one per well.
    """
    well_embeddings: List[np.ndarray] = []
    well_compound_labels: List[str] = []
    well_plate_labels: List[str] = []

    compounds_processed = 0
    for entry in metadata:
        compound_id = str(entry["Compound"])
        wells_for_compound: List[Tuple[np.ndarray, str]] = []

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
                        if normalize_before_subtract:
                            well_mean = well_mean / (np.linalg.norm(well_mean) + 1e-8)
                            ctrl_mean = ctrl_mean / (np.linalg.norm(ctrl_mean) + 1e-8)
                        well_mean = well_mean - ctrl_mean

            wells_for_compound.append((well_mean, str(plate_id)))

        if len(wells_for_compound) >= min_wells:
            for emb, plate in wells_for_compound:
                well_embeddings.append(emb)
                well_compound_labels.append(compound_id)
                well_plate_labels.append(plate)
            compounds_processed += 1

        if max_compounds is not None and compounds_processed >= max_compounds:
            break

    return np.array(well_embeddings), well_compound_labels, well_plate_labels


def compute_within_between_distances(
    well_embeddings: np.ndarray,
    well_compound_labels: List[str],
    well_plate_labels: List[str],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate pairwise distances into within-compound and between-compound.

    Within-compound pairs are always cross-plate (by construction: each plate
    gives one well per compound). To make the comparison fair, between-compound
    pairs are restricted to cross-plate pairs only.

    Returns:
        within_distances: distances between wells of the same compound (cross-plate).
        between_distances: distances between wells of different compounds (cross-plate only).
    """
    n = len(well_compound_labels)
    within: List[float] = []
    between: List[float] = []

    for i, j in combinations(range(n), 2):
        same_compound = well_compound_labels[i] == well_compound_labels[j]
        same_plate = well_plate_labels[i] == well_plate_labels[j]

        # Skip same-plate between-compound pairs to match the structural
        # constraint of within-compound pairs (which are always cross-plate).
        if not same_compound and same_plate:
            continue

        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            dot = np.dot(well_embeddings[i], well_embeddings[j])
            norm_i = np.linalg.norm(well_embeddings[i])
            norm_j = np.linalg.norm(well_embeddings[j])
            dist = 1.0 - dot / (norm_i * norm_j + 1e-8)
        elif metric == "angular":
            # Geodesic (arc) distance on hypersphere: arccos(cos_sim)
            # Returns angle in radians [0, pi]
            dot = np.dot(well_embeddings[i], well_embeddings[j])
            norm_i = np.linalg.norm(well_embeddings[i])
            norm_j = np.linalg.norm(well_embeddings[j])
            cos_sim = dot / (norm_i * norm_j + 1e-8)
            # Clamp for numerical stability
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            dist = float(np.arccos(cos_sim))
        elif metric == "radial":
            # Absolute difference in norms: | ||mu_i|| - ||mu_j|| |
            norm_i = np.linalg.norm(well_embeddings[i])
            norm_j = np.linalg.norm(well_embeddings[j])
            dist = float(abs(norm_i - norm_j))
        else:
            dist = float(np.linalg.norm(well_embeddings[i] - well_embeddings[j]))

        if same_compound:
            within.append(dist)
        else:
            between.append(dist)

    return np.array(within), np.array(between)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

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


def plot_dimension_reduction(
    well_embeddings: np.ndarray,
    well_compound_labels: List[str],
    output_dir: str,
    seed: int = 42,
) -> None:
    """Visualize well mean embeddings in 2D using UMAP and t-SNE.

    Produces one plot per method, with points colored by compound ID.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    numeric_labels = le.fit_transform(well_compound_labels)
    n_compounds = len(le.classes_)

    # Choose a colormap with enough distinct colors
    cmap = plt.cm.get_cmap("tab20" if n_compounds <= 20 else "nipy_spectral",
                           n_compounds)

    methods: Dict[str, np.ndarray] = {}

    # t-SNE
    from sklearn.manifold import TSNE
    perplexity = min(30, max(2, len(well_embeddings) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca", learning_rate="auto")
    methods["tSNE"] = tsne.fit_transform(well_embeddings)

    # UMAP (optional dependency)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=min(15, len(well_embeddings) - 1),
                            min_dist=0.1)
        methods["UMAP"] = reducer.fit_transform(well_embeddings)
    except ImportError:
        print("[dim-reduction] umap-learn not installed, skipping UMAP.")

    for method_name, coords_2d in methods.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=numeric_labels, cmap=cmap, s=60, alpha=0.8, edgecolors="k",
            linewidths=0.3,
        )

        # Legend (limit entries if too many compounds)
        if n_compounds <= 20:
            handles = []
            for idx, compound in enumerate(le.classes_):
                handles.append(plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=cmap(idx / max(n_compounds - 1, 1)),
                    markersize=8, label=compound,
                ))
            ax.legend(handles=handles, title="Compound", loc="best",
                      fontsize=7, ncol=max(1, n_compounds // 10))
        else:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Compound index")

        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        ax.set_title(f"Well mean embeddings – {method_name}\n"
                     f"({n_compounds} compounds, {len(well_embeddings)} wells)")
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"well_embeddings_{method_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] Saved {method_name} visualization to {out_path}")


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
        ax.hist(within, bins=50, alpha=0.6, label="Within-compound",
            color="#2ca02c", density=True)
        ax.hist(between, bins=50, alpha=0.6, label="Between-compound",
            color="#d62728", density=True)
    ax.axvline(within.mean(), color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"Within mean = {within.mean():.4f}")
    ax.axvline(between.mean(), color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Between mean = {between.mean():.4f}")
    ax.set_xlabel(f"{metric.capitalize()} distance")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved distance histogram to {output_path}")


def plot_radial_deviation(
    well_embeddings: np.ndarray,
    well_compound_labels: List[str],
    gamma: float,
    output_dir: str,
) -> None:
    """Diagnostic: plot distribution of ||mu|| - gamma per compound.

    Shows how far each well's mean embedding deviates from the expected
    hypersphere radius. Ideally all wells sit close to gamma.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norms = np.linalg.norm(well_embeddings, axis=1)
    deviations = norms - gamma

    unique_compounds = sorted(set(well_compound_labels))
    n_compounds = len(unique_compounds)

    # ── Summary statistics ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"RADIAL DEVIATION DIAGNOSTIC  (gamma = {gamma:.4f})")
    print(f"{'─'*70}")
    print(f"  ||mu|| : mean = {norms.mean():.4f}, std = {norms.std():.4f}")
    print(f"  ||mu|| - gamma : mean = {deviations.mean():.4f}, "
          f"std = {deviations.std():.4f}")
    print(f"  Range: [{deviations.min():.4f}, {deviations.max():.4f}]")

    # Per-compound radial stats
    print(f"\n  Per-compound ||mu|| - gamma (mean ± std):")
    compound_devs = {}
    for compound in unique_compounds:
        mask = [l == compound for l in well_compound_labels]
        c_devs = deviations[mask]
        compound_devs[compound] = c_devs
        if len(unique_compounds) <= 20:
            print(f"    Compound {compound:>6s}: {c_devs.mean():+.4f} ± {c_devs.std():.4f} "
                  f"(n={len(c_devs)})")
    print(f"{'─'*70}")

    # ── Plot 1: Overall deviation histogram ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deviations, bins=40, alpha=0.7, color="#1f77b4", edgecolor="k",
            linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5,
               label=f"γ = {gamma:.2f}")
    ax.axvline(deviations.mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"mean dev = {deviations.mean():.4f}")
    ax.set_xlabel("||μ|| − γ")
    ax.set_ylabel("Count")
    ax.set_title("Radial deviation of well mean embeddings from hypersphere")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "radial_deviation_histogram.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved radial deviation histogram to {out_path}")

    # ── Plot 2: Per-compound box plot ────────────────────────────────────────
    if n_compounds <= 30:
        fig, ax = plt.subplots(figsize=(max(8, n_compounds * 0.5), 5))
        data = [compound_devs[c] for c in unique_compounds]
        bp = ax.boxplot(data, labels=unique_compounds, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec7e8")
        ax.axhline(0, color="red", linestyle="--", linewidth=1, label="γ (expected radius)")
        ax.set_xlabel("Compound")
        ax.set_ylabel("||μ|| − γ")
        ax.set_title("Radial deviation per compound")
        ax.legend()
        if n_compounds > 10:
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out_path = os.path.join(output_dir, "radial_deviation_per_compound.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] Saved per-compound radial deviation to {out_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(args.metadata) as f:
        metadata = json.load(f)
    print(f"Metadata: {len(metadata)} compounds")

    # ── Compute mean embeddings per well ─────────────────────────────────────
    if args.embedding is not None:
        # Use pre-computed embeddings — skip model loading entirely
        print(f"\nLoading pre-computed embeddings from {args.embedding}...")
        well_embeddings, well_labels, well_plates = load_well_means_from_embedding(
            args.embedding, metadata, args.min_wells, args.max_compounds,
            subtract_control=args.subtract_control,
            normalize_before_subtract=args.normalize_before_subtract,
        )
        model = None
    else:
        # Build and load model, encode from images
        model = build_model(args)
        if args.model != "dino":
            load_checkpoint(model, args.checkpoint)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
        if args.model == "dino":
            print("Model  : DINOv2 vits14  (pretrained, latent dim 384)")
        else:
            print(f"Model  : {args.model}  (latent dim {args.latent_dim})")

        root_dir = Path(args.root_dir)
        transform = T.Compose([
            T.Resize((args.img_size, args.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
        ])
        mode = ImageReadMode.RGB if args.model == "dino" else (
            ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB
        )

        ctrl_msg = " (subtract_control=True)" if args.subtract_control else ""
        print(f"\nEncoding wells (min_wells={args.min_wells}){ctrl_msg}...")
        well_embeddings, well_labels, well_plates = compute_well_mean_embeddings(
            metadata, root_dir, model, transform, mode,
            args.batch_size, device, args.min_wells, args.max_compounds,
            subtract_control=args.subtract_control,
            normalize_before_subtract=args.normalize_before_subtract,
        )

    n_wells = len(well_labels)
    n_compounds = len(set(well_labels))
    n_plates = len(set(well_plates))
    print(f"Encoded {n_wells} wells from {n_compounds} compounds across {n_plates} plates")

    if n_wells < 3:
        print("ERROR: Not enough wells to run the test (need at least 3).")
        sys.exit(1)

    # ── Compute within/between distances ─────────────────────────────────────
    print(f"\nComputing pairwise {args.metric} distances...")
    within_dists, between_dists = compute_within_between_distances(
        well_embeddings, well_labels, well_plates, args.metric,
    )

    if len(within_dists) == 0:
        print("ERROR: No within-compound pairs found. Need compounds with >= 2 wells.")
        sys.exit(1)

    # Subsample between-compound distances to match within-compound count
    if len(between_dists) > len(within_dists):
        rng = np.random.default_rng(args.seed)
        subsample_idx = rng.choice(
            len(between_dists), size=len(within_dists), replace=False
        )
        between_dists = between_dists[subsample_idx]
        print(f"  Subsampled between-compound pairs to n={len(between_dists)} "
              f"(matching within-compound count)")

    # ── Run statistical test ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS ({args.metric} distance)")
    print(f"{'='*70}")
    print(f"  Within-compound  : mean = {within_dists.mean():.4f} "
          f"+/- {within_dists.std():.4f}  (n = {len(within_dists)})")
    print(f"  Between-compound : mean = {between_dists.mean():.4f} "
          f"+/- {between_dists.std():.4f}  (n = {len(between_dists)})")
    print(f"  Ratio (within/between) : {within_dists.mean() / between_dists.mean():.4f}")

    # Kolmogorov-Smirnov test (one-sided: within < between)
    ks_stat, ks_p = test_kolmogorov_smirnov(within_dists, between_dists)
    print(f"\n  Kolmogorov-Smirnov test (within < between):")
    print(f"      KS-statistic = {ks_stat:.4f}")
    print(f"      p-value      = {ks_p:.2e}")

    passed = ks_p < 0.05
    print(f"\n  TEST {'PASSED' if passed else 'FAILED'}: "
          f"Within-compound distances are "
          f"{'significantly' if passed else 'NOT significantly'} "
          f"smaller than between-compound distances (p < 0.05).")
    print(f"{'='*70}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"well_compound_distances_{args.metric}.png")
    plot_distance_distributions(within_dists, between_dists, args.metric, plot_path)

    # ── Radial deviation diagnostic (TiltedVAE) ─────────────────────────────
    if model is not None and hasattr(model, "gamma"):
        plot_radial_deviation(
            well_embeddings, well_labels, model.gamma, args.output_dir,
        )

    # ── Dimension reduction visualization ────────────────────────────────────
    print("\nGenerating dimension reduction plots...")
    plot_dimension_reduction(
        well_embeddings, well_labels, args.output_dir, seed=args.seed,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
