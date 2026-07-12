"""Color-permutation test for the biological meaningfulness of VAE latents.

Idea: a biologically meaningful embedding of an aphid image should be (largely)
invariant to modest changes in the specimen's apparent color intensity /
brightness. This script:

  1. Selects ``N`` images whose object color intensity (mean brightness of the
     non-black foreground pixels) is closest to a target point, so the specimens
     are comparably colored.
  2. Builds three views of each image, all at the model's native ``img_size``:
       - original : the image as the encoder normally sees it (baseline)
       - brighter : the image with color intensity increased by ``scale`` (~15%)
       - darker   : the image with color intensity decreased by ``scale`` (~15%)
  3. Encodes every view with the trained encoder (latent mean ``mu``).
  4. Projects the three latent groups to 2D (PCA / t-SNE / UMAP) and plots them.

If the latents are meaningful, the three views of the same specimen should land
close together (the clouds should overlap and the paired distances should be
small), rather than separating by color intensity.

Examples:
    python Tests/permutation_test_color.py --data_dir ../DATA/Train/ --checkpoint results/checkpoints/last.ckpt --model tilted --intensity_pool 1000 --method pca --scale 0.3 --device cpu
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

# This script lives in ``Tests/``; add the repo root to the path so the
# top-level ``Models`` package and ``dataset`` module can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models import VAE, TiltedVAE
from dataset import _scan_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Color-permutation test for VAE latent embeddings")

    # Data / model
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the image dataset (any nested folder layout)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained Lightning checkpoint (.ckpt) or a "
                             "raw model state_dict (.pt/.pth)")
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # Test configuration
    parser.add_argument("--scale", type=float, default=0.15,
                        help="Fractional color-intensity change for the brighter/"
                             "darker views (0.15 = 15%%)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Object-color matching (images have a black background, so the object
    # "color intensity" is the mean brightness of the non-black foreground pixels).
    parser.add_argument("--black_threshold", type=int, default=0,
                        help="A pixel counts as foreground if its max channel "
                             "value exceeds this (0-255)")
    parser.add_argument("--intensity_pool", type=int, default=300,
                        help="Number of images to select whose object color "
                             "intensity is closest to the target point")
    parser.add_argument("--intensity_percentile", type=float, default=50.0,
                        help="Target color intensity to match, as a percentile of "
                             "the dataset's measured foreground intensities (0-100)")

    # Dimensionality reduction / output
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne", "umap"],
                        help="2D projection method for visualization")
    parser.add_argument("--output_dir", type=str, default="results/permutation_test")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--draw_links", action="store_true",
                        help="Draw faint lines connecting the three views of each image")

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "tilted":
        model = TiltedVAE(
            in_channels=args.in_channels,
            latent_dim=args.latent_dim,
            tau=args.tau,
            img_size=args.img_size,
        )
    else:
        model = VAE(
            in_channels=args.in_channels,
            latent_dim=args.latent_dim,
            img_size=args.img_size,
        )
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load weights from either a Lightning checkpoint (keys prefixed with
    ``model.`` under ``state_dict``) or a raw model ``state_dict``."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip the LightningModule's ``model.`` prefix if present.
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


def foreground_intensity(path: str, in_channels: int, black_threshold: int) -> float:
    """Mean brightness of the non-black (foreground) pixels of an image, used as
    a proxy for the object's color intensity. The background is assumed to be
    black, so only pixels whose brightest channel exceeds ``black_threshold``
    are averaged. Returns 0 when there is no foreground."""
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB
    img = read_image(path, mode=mode).float()  # [C, H, W] in [0, 255]
    mask = img.amax(dim=0) > black_threshold   # [H, W] foreground pixels
    if not bool(mask.any()):
        return 0.0
    brightness = img.mean(dim=0)               # [H, W] mean over channels
    return float(brightness[mask].mean().item())


def sample_paths_by_intensity(data_dir: str, in_channels: int, black_threshold: int,
                              intensity_pool: int,
                              intensity_percentile: float) -> List[str]:
    """Select the ``intensity_pool`` images whose object color intensity (mean
    foreground brightness) is closest to a target point.

    Every image in the dataset is measured, the target intensity is taken as the
    ``intensity_percentile`` of those intensities, and the ``intensity_pool``
    images with the smallest absolute distance to that target are returned.
    """
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    print(f"[color] Measuring foreground intensity of {len(paths)} images...")
    intensities = np.array([
        foreground_intensity(p, in_channels, black_threshold) for p in paths
    ])

    # Pick the images whose intensity is closest to the target point (the
    # requested percentile of the measured intensities).
    n = min(intensity_pool, len(paths))
    target = np.percentile(intensities, intensity_percentile)
    window = np.argsort(np.abs(intensities - target))[:n]

    selected = intensities[window]
    print(f"[color] Selected {n} images closest to target intensity {target:.1f}: "
          f"intensity = {selected.mean():.1f} +/- {selected.std():.1f} "
          f"(min {selected.min():.1f}, max {selected.max():.1f})")

    return [paths[i] for i in window]


def build_transforms(scale: float):
    """Return three transforms (original / brighter / darker) that each take a
    square ``img_size`` float image in [0, 1] and produce one of the same shape.

    The color-intensity views multiply pixel values by ``1 +/- scale`` and clamp
    back to [0, 1]. Because the background is black (0), scaling leaves it black
    and only changes the intensity of the foreground.
    """
    def adjust(factor: float):
        return lambda img: torch.clamp(img * factor, 0.0, 1.0)

    original = T.Compose([])  # identity: base image is already the baseline view
    brighter = T.Lambda(adjust(1.0 + scale))
    darker = T.Lambda(adjust(1.0 - scale))
    return {"original": original, "brighter": brighter, "darker": darker}


def load_base_images(paths: List[str], img_size: int, in_channels: int) -> Tensor:
    """Load every image, resize to a square ``img_size`` float tensor in [0, 1].
    Returns a stacked tensor [N, C, img_size, img_size]."""
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB
    base_tf = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ])
    imgs = []
    for p in paths:
        img = read_image(p, mode=mode)
        imgs.append(base_tf(img))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def encode_group(model: torch.nn.Module, images: Tensor, transform,
                 batch_size: int, device: torch.device) -> np.ndarray:
    """Apply ``transform`` to each base image then encode to the latent mean.
    Returns an array [N, latent_dim]."""
    latents = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size]
        batch = torch.stack([transform(img) for img in batch], dim=0)
        batch = batch.to(device)
        mu, _ = model.encode(batch)
        latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0)


def reduce_dims(latents: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Project [M, D] latents to [M, 2] using the requested method."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(latents)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, (latents.shape[0] - 1) // 3))
        return TSNE(n_components=2, random_state=seed,
                    perplexity=perplexity, init="pca").fit_transform(latents)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(latents)
    raise ValueError(f"Unknown method: {method}")


def report_invariance(groups: dict) -> None:
    """Print how far the brighter/darker latents drift from the original view."""
    orig = groups["original"]

    def stats(other: np.ndarray):
        l2 = np.linalg.norm(other - orig, axis=1)
        cos = np.sum(other * orig, axis=1) / (
            np.linalg.norm(other, axis=1) * np.linalg.norm(orig, axis=1) + 1e-8)
        return l2.mean(), l2.std(), cos.mean()

    scale_ref = np.linalg.norm(orig, axis=1).mean()
    print("\n[invariance] Paired latent drift from the 'original' view "
          f"(mean ||mu||_orig = {scale_ref:.3f}):")
    for name in ("brighter", "darker"):
        l2_mean, l2_std, cos_mean = stats(groups[name])
        print(f"  {name:9s}: L2 = {l2_mean:.3f} +/- {l2_std:.3f}   "
              f"cosine = {cos_mean:.3f}")


def plot_projection(coords: np.ndarray, group_names: List[str], n: int,
                    method: str, draw_links: bool, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"original": "#1f77b4", "brighter": "#d62728", "darker": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 7))

    # Optionally connect the three views of each image (same row across groups).
    if draw_links:
        for i in range(n):
            pts = np.array([coords[g * n + i] for g in range(len(group_names))])
            ax.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.15,
                    linewidth=0.5, zorder=1)

    for g, name in enumerate(group_names):
        seg = coords[g * n:(g + 1) * n]
        ax.scatter(seg[:, 0], seg[:, 1], s=18, alpha=0.7,
                   c=colors.get(name, None), label=name, zorder=2)

    ax.set_title(f"Color-permutation test of VAE latents ({method.upper()})")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(title="view")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved projection to {out_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.output_dir, exist_ok=True)

    # Model + weights
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    # Select color-matched images and build the three intensity views
    paths = sample_paths_by_intensity(
        args.data_dir, args.in_channels, args.black_threshold,
        args.intensity_pool, args.intensity_percentile)
    n = len(paths)
    print(f"[data] Sampled {n} images from '{args.data_dir}'")

    base_images = load_base_images(paths, args.img_size, args.in_channels)
    transforms = build_transforms(args.scale)

    group_names = ["original", "brighter", "darker"]
    groups = {}
    for name in group_names:
        groups[name] = encode_group(
            model, base_images, transforms[name], args.batch_size, device)
        print(f"[encode] {name:9s}: latents {groups[name].shape}")

    # Quantitative invariance summary
    report_invariance(groups)

    # Save raw latents for further analysis
    npz_path = os.path.join(args.output_dir, "latents.npz")
    np.savez(npz_path, **groups, paths=np.array(paths))
    print(f"[save] Saved latents to {npz_path}")

    # 2D projection of all three groups jointly
    all_latents = np.concatenate([groups[name] for name in group_names], axis=0)
    coords = reduce_dims(all_latents, args.method, args.seed)

    out_path = os.path.join(args.output_dir, f"permutation_{args.method}.png")
    plot_projection(coords, group_names, n, args.method, args.draw_links, out_path)


if __name__ == "__main__":
    main()
