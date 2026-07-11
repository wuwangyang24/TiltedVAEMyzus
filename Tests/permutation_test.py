"""Scale-permutation test for the biological meaningfulness of VAE latents.

Idea: a biologically meaningful embedding of an aphid image should be (largely)
invariant to the apparent size of the specimen in the frame. This script:

  1. Randomly samples ``N`` images from the dataset.
  2. Builds three views of each image, all at the model's native ``img_size``:
       - original : the image resized to ``img_size`` (baseline the encoder sees)
       - enlarged : the image zoomed in by ``scale`` (~15%) and center-cropped
       - shrunk   : the image zoomed out by ``scale`` (~15%) and center-cropped
  3. Encodes every view with the trained encoder (latent mean ``mu``).
  4. Projects the three latent groups to 2D (PCA / t-SNE / UMAP) and plots them.

If the latents are meaningful, the three views of the same specimen should land
close together (the clouds should overlap and the paired distances should be
small), rather than separating by scale.

Examples:
    python permutation_test.py --data_dir /path/to/images \
        --checkpoint results/checkpoints/last.ckpt --model tilted \
        --n_samples 300 --method umap
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
        description="Scale-permutation test for VAE latent embeddings")

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
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Number of images to randomly sample from the dataset")
    parser.add_argument("--scale", type=float, default=0.15,
                        help="Fractional zoom for the enlarge/shrink views (0.15 = 15%%)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Object-size filtering (images have a black background, so the object
    # "size" is approximated by the number of non-black foreground pixels).
    parser.add_argument("--match_size", action="store_true",
                        help="Only sample images whose object size (non-black "
                             "pixel count) is very similar")
    parser.add_argument("--black_threshold", type=int, default=10,
                        help="A pixel counts as foreground if its max channel "
                             "value exceeds this (0-255)")
    parser.add_argument("--size_pool", type=int, default=2000,
                        help="Number of random candidate images to measure when "
                             "--match_size is set (bounds the scanning cost)")
    parser.add_argument("--size_percentile", type=float, default=50.0,
                        help="Center of the object-size window to select, as a "
                             "percentile of the measured pool (0-100)")

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


def foreground_area(path: str, in_channels: int, black_threshold: int) -> int:
    """Count the non-black (foreground) pixels of an image as a proxy for the
    object's size. The background is assumed to be black, so a pixel is treated
    as foreground when its brightest channel exceeds ``black_threshold``."""
    mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB
    img = read_image(path, mode=mode)          # uint8 [C, H, W]
    brightness = img.amax(dim=0)               # [H, W] max over channels
    return int((brightness > black_threshold).sum().item())


def sample_paths(data_dir: str, n_samples: int, seed: int) -> List[str]:
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(paths))
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in idx]


def sample_paths_by_size(data_dir: str, n_samples: int, seed: int,
                         in_channels: int, black_threshold: int,
                         size_pool: int, size_percentile: float) -> List[str]:
    """Sample ``n_samples`` images whose object size (non-black pixel count) is
    as similar as possible.

    A random pool of up to ``size_pool`` images is measured, then the
    ``n_samples`` images whose areas are closest to the target point (the
    ``size_percentile`` of the measured pool) are returned.
    """
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    rng = np.random.default_rng(seed)
    pool_n = min(size_pool, len(paths))
    pool_idx = rng.choice(len(paths), size=pool_n, replace=False)
    pool_paths = [paths[i] for i in pool_idx]

    print(f"[size] Measuring foreground area of {pool_n} candidate images...")
    areas = np.array([
        foreground_area(p, in_channels, black_threshold) for p in pool_paths
    ])

    # Pick the n images whose area is closest to the target point (the requested
    # percentile of the measured pool).
    n = min(n_samples, pool_n)
    target = np.percentile(areas, size_percentile)
    window = np.argsort(np.abs(areas - target))[:n]

    selected_areas = areas[window]
    print(f"[size] Selected {n} images closest to target area {target:.0f} px: "
          f"area = {selected_areas.mean():.0f} +/- {selected_areas.std():.0f} px "
          f"(min {selected_areas.min()}, max {selected_areas.max()})")

    return [pool_paths[i] for i in window]


def build_transforms(img_size: int, scale: float):
    """Return three transforms (original / enlarged / shrunk) that each take a
    square ``img_size`` float image and produce an ``img_size`` float image.

    All views start from the same square base so they differ only in scale:
      - enlarged: upsample to img_size*(1+scale) then center-crop (zoom in).
      - shrunk:   downsample to img_size*(1-scale) then center-crop (zoom out,
                  the border is zero-padded back up to img_size).
    """
    enlarged_size = int(round(img_size * (1.0 + scale)))
    shrunk_size = int(round(img_size * (1.0 - scale)))

    original = T.Compose([])  # identity: base image is already the baseline view
    enlarged = T.Compose([
        T.Resize((enlarged_size, enlarged_size), antialias=True),
        T.CenterCrop(img_size),
    ])
    shrunk = T.Compose([
        T.Resize((shrunk_size, shrunk_size), antialias=True),
        T.CenterCrop(img_size),  # pads the smaller image back to img_size
    ])
    return {"original": original, "enlarged": enlarged, "shrunk": shrunk}


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
    """Print how far the enlarged/shrunk latents drift from the original view."""
    orig = groups["original"]

    def stats(other: np.ndarray):
        l2 = np.linalg.norm(other - orig, axis=1)
        cos = np.sum(other * orig, axis=1) / (
            np.linalg.norm(other, axis=1) * np.linalg.norm(orig, axis=1) + 1e-8)
        return l2.mean(), l2.std(), cos.mean()

    scale_ref = np.linalg.norm(orig, axis=1).mean()
    print("\n[invariance] Paired latent drift from the 'original' view "
          f"(mean ||mu||_orig = {scale_ref:.3f}):")
    for name in ("enlarged", "shrunk"):
        l2_mean, l2_std, cos_mean = stats(groups[name])
        print(f"  {name:9s}: L2 = {l2_mean:.3f} +/- {l2_std:.3f}   "
              f"cosine = {cos_mean:.3f}")


def plot_projection(coords: np.ndarray, group_names: List[str], n: int,
                    method: str, draw_links: bool, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"original": "#1f77b4", "enlarged": "#d62728", "shrunk": "#2ca02c"}
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

    ax.set_title(f"Scale-permutation test of VAE latents ({method.upper()})")
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

    # Sample images and build the three scaled views
    if args.match_size:
        paths = sample_paths_by_size(
            args.data_dir, args.n_samples, args.seed, args.in_channels,
            args.black_threshold, args.size_pool, args.size_percentile)
    else:
        paths = sample_paths(args.data_dir, args.n_samples, args.seed)
    n = len(paths)
    print(f"[data] Sampled {n} images from '{args.data_dir}'")

    base_images = load_base_images(paths, args.img_size, args.in_channels)
    transforms = build_transforms(args.img_size, args.scale)

    group_names = ["original", "enlarged", "shrunk"]
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
