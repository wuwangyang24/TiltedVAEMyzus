"""Shape-permutation test for the biological meaningfulness of VAE latents.

Idea: a biologically meaningful embedding of an aphid image should be (largely)
invariant to modest changes in the specimen's apparent shape (the ratio between
its length and its width). This script:

  1. Selects ``N`` images whose object shape (length/width aspect ratio of the
     non-black foreground bounding box) is closest to a target point, so the
     specimens have comparable proportions.
  2. Builds three views of each image, all at the model's native ``img_size``:
       - original : the image as the encoder normally sees it (baseline)
       - stretched_length : stretched along the length (height) by ``scale`` (~15%)
       - stretched_width  : stretched along the width by ``scale`` (~15%)
  3. Encodes every view with the trained encoder (latent mean ``mu``).
  4. Projects the three latent groups to 2D (PCA / t-SNE / UMAP) and plots them.

If the latents are meaningful, the three views of the same specimen should land
close together (the clouds should overlap and the paired distances should be
small), rather than separating by shape.

Examples:
    python Tests/permutation_test_shape.py --data_dir ../DATA/Train/ --checkpoint results/checkpoints/last.ckpt --model vae --shape_pool 1000 --method pca --device cpu --black_threshold 0 --scale 0.2
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
        description="Shape-permutation test for VAE latent embeddings")

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
                        help="Fractional stretch for the length/width views "
                             "(0.15 = 15%%)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Object-shape matching (images have a black background, so the object
    # "shape" is approximated by the length/width ratio of the foreground
    # bounding box).
    parser.add_argument("--black_threshold", type=int, default=10,
                        help="A pixel counts as foreground if its max channel "
                             "value exceeds this (0-255)")
    parser.add_argument("--shape_pool", type=int, default=300,
                        help="Number of images to select whose object shape is "
                             "closest to the target point")
    parser.add_argument("--shape_percentile", type=float, default=50.0,
                        help="Target object aspect ratio to match, as a percentile "
                             "of the dataset's measured aspect ratios (0-100)")

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


def foreground_aspect_ratio(path: str, in_channels: int,
                            black_threshold: int) -> float:
    """Length/width aspect ratio of an image's non-black (foreground) bounding
    box, used as a proxy for the object's shape. The background is assumed to be
    black, so a pixel is treated as foreground when its brightest channel
    exceeds ``black_threshold``. Returns 0 when there is no foreground."""
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB
    img = read_image(path, mode=mode)          # uint8 [C, H, W]
    mask = img.amax(dim=0) > black_threshold   # [H, W] foreground pixels
    rows = torch.any(mask, dim=1)              # [H] rows with foreground
    cols = torch.any(mask, dim=0)              # [W] cols with foreground
    if not (bool(rows.any()) and bool(cols.any())):
        return 0.0
    row_idx = torch.where(rows)[0]
    col_idx = torch.where(cols)[0]
    length = float(row_idx[-1] - row_idx[0] + 1)   # bounding-box height
    width = float(col_idx[-1] - col_idx[0] + 1)    # bounding-box width
    return length / width if width > 0 else 0.0


def sample_paths_by_shape(data_dir: str, in_channels: int, black_threshold: int,
                          shape_pool: int, shape_percentile: float) -> List[str]:
    """Select the ``shape_pool`` images whose object shape (length/width aspect
    ratio) is closest to a target point.

    Every image in the dataset is measured, the target ratio is taken as the
    ``shape_percentile`` of those ratios, and the ``shape_pool`` images with the
    smallest absolute distance to that target are returned.
    """
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    print(f"[shape] Measuring foreground aspect ratio of {len(paths)} images...")
    ratios = np.array([
        foreground_aspect_ratio(p, in_channels, black_threshold) for p in paths
    ])

    # Pick the images whose aspect ratio is closest to the target point (the
    # requested percentile of the measured ratios).
    n = min(shape_pool, len(paths))
    target = np.percentile(ratios, shape_percentile)
    window = np.argsort(np.abs(ratios - target))[:n]

    selected = ratios[window]
    print(f"[shape] Selected {n} images closest to target ratio {target:.3f}: "
          f"ratio = {selected.mean():.3f} +/- {selected.std():.3f} "
          f"(min {selected.min():.3f}, max {selected.max():.3f})")

    return [paths[i] for i in window]


def build_transforms(img_size: int, scale: float):
    """Return three transforms (original / stretched_length / stretched_width)
    that each take a square ``img_size`` float image and produce an ``img_size``
    float image.

    Both stretched views scale one axis up by ``scale`` and then center-crop back
    to ``img_size`` so the output shape is unchanged:
      - stretched_length: upsample the height (length) by (1+scale), keep width.
      - stretched_width:  upsample the width by (1+scale), keep height.
    """
    stretched = int(round(img_size * (1.0 + scale)))

    original = T.Compose([])  # identity: base image is already the baseline view
    stretched_length = T.Compose([
        T.Resize((stretched, img_size), antialias=True),  # taller (H up)
        T.CenterCrop(img_size),
    ])
    stretched_width = T.Compose([
        T.Resize((img_size, stretched), antialias=True),  # wider (W up)
        T.CenterCrop(img_size),
    ])
    return {
        "original": original,
        "stretched_length": stretched_length,
        "stretched_width": stretched_width,
    }


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
    """Print how far the stretched latents drift from the original view."""
    orig = groups["original"]

    def stats(other: np.ndarray):
        l2 = np.linalg.norm(other - orig, axis=1)
        cos = np.sum(other * orig, axis=1) / (
            np.linalg.norm(other, axis=1) * np.linalg.norm(orig, axis=1) + 1e-8)
        return l2.mean(), l2.std(), cos.mean()

    scale_ref = np.linalg.norm(orig, axis=1).mean()
    print("\n[invariance] Paired latent drift from the 'original' view "
          f"(mean ||mu||_orig = {scale_ref:.3f}):")
    for name in ("stretched_length", "stretched_width"):
        l2_mean, l2_std, cos_mean = stats(groups[name])
        print(f"  {name:16s}: L2 = {l2_mean:.3f} +/- {l2_std:.3f}   "
              f"cosine = {cos_mean:.3f}")


def plot_projection(coords: np.ndarray, group_names: List[str], n: int,
                    method: str, draw_links: bool, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "original": "#1f77b4",
        "stretched_length": "#d62728",
        "stretched_width": "#2ca02c",
    }
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

    ax.set_title(f"Shape-permutation test of VAE latents ({method.upper()})")
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

    # Select shape-matched images and build the three stretched views
    paths = sample_paths_by_shape(
        args.data_dir, args.in_channels, args.black_threshold,
        args.shape_pool, args.shape_percentile)
    n = len(paths)
    print(f"[data] Sampled {n} images from '{args.data_dir}'")

    base_images = load_base_images(paths, args.img_size, args.in_channels)
    transforms = build_transforms(args.img_size, args.scale)

    group_names = ["original", "stretched_length", "stretched_width"]
    groups = {}
    for name in group_names:
        groups[name] = encode_group(
            model, base_images, transforms[name], args.batch_size, device)
        print(f"[encode] {name:16s}: latents {groups[name].shape}")

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
