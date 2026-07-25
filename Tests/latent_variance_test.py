"""Compute per-dimension latent variance from randomly selected dataset images.

This script samples a random number of images from the dataset, encodes them
with a VAE-family model, and reports the variance of each latent dimension.

python TiltedVAEMyzus/Tests/latent_variance_test.py --data_dir DATA/Train --model vae --latent_dim 128
python TiltedVAEMyzus/Tests/latent_variance_test.py --data_dir DATA/Train --model tilted --latent_dim 128 --embeddings TiltedVAEMyzus/Tests/efficacy500_classifier/tiltedvae/128/embeddings_100ppm.pt
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

# This script lives in ``Tests/``; add the repo root to the path so the
# top-level ``Models`` package and ``dataset`` module can be imported.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, ROOT)

from Models import TiltedVAE, VAE
from dataset import ImageFolderFlat, _scan_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute latent variance per dimension on random dataset images"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the image dataset (any nested folder layout)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained Lightning checkpoint (.ckpt) or a "
                             "raw model state_dict (.pt/.pth)")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to precomputed embeddings (.npy/.npz/.pt/.pth). "
                            "When provided, skips image sampling and model encoding")
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted"],
                        help="Model architecture used for encoding")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")
    parser.add_argument("--min_images", type=int, default=8,
                        help="Minimum random sample size")
    parser.add_argument("--max_images", type=int, default=64,
                        help="Maximum random sample size")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for image sampling")
    parser.add_argument("--output_dir", type=str, default="results/latent_variance",
                        help="Directory to save latent-variance plots")
    parser.add_argument("--hist_bins", type=int, default=20,
                        help="Number of bins for variance histogram")
    return parser.parse_args()


def sample_random_images(data_dir: str,
                         in_channels: int,
                         img_size: int,
                         min_images: int,
                         max_images: int,
                         seed: int) -> torch.Tensor:
    """Sample a random number of random images from ``data_dir``."""
    paths = _scan_images(data_dir)
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images in dataset to compute latent variance.")

    gen = torch.Generator().manual_seed(seed)
    max_pick = min(max_images, len(paths))
    min_pick = min(min_images, max_pick)
    if min_pick < 2:
        min_pick = 2

    num_images = int(torch.randint(low=min_pick, high=max_pick + 1, size=(1,), generator=gen).item())
    perm = torch.randperm(len(paths), generator=gen)[:num_images].tolist()
    selected_paths = [paths[i] for i in perm]

    transform = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ])
    dataset = ImageFolderFlat(selected_paths, transform=transform, in_channels=in_channels)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    return images


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
    """Load weights from a Lightning checkpoint or a raw model state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
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


def load_embeddings(path: str) -> torch.Tensor:
    """Load precomputed embeddings from .npy/.npz/.pt/.pth into [N, D] tensor."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        emb = torch.from_numpy(arr)
    elif ext == ".npz":
        data = np.load(path)
        if "embeddings" in data:
            arr = data["embeddings"]
        else:
            first_key = list(data.keys())[0] if data.files else None
            if first_key is None:
                raise RuntimeError("Empty .npz file: no arrays found")
            arr = data[first_key]
        emb = torch.from_numpy(arr)
    elif ext in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)

        rows = []

        def _collect_tensors(node):
            if isinstance(node, torch.Tensor):
                t = node.detach().cpu().float()
                if t.ndim == 1:
                    rows.append(t.unsqueeze(0))
                elif t.ndim == 2:
                    rows.append(t)
                return

            if isinstance(node, np.ndarray):
                t = torch.from_numpy(node).detach().cpu().float()
                if t.ndim == 1:
                    rows.append(t.unsqueeze(0))
                elif t.ndim == 2:
                    rows.append(t)
                return

            if isinstance(node, dict):
                for value in node.values():
                    _collect_tensors(value)
                return

            if isinstance(node, (list, tuple)):
                for item in node:
                    _collect_tensors(item)

        _collect_tensors(obj)

        if not rows:
            raise RuntimeError("No embedding tensors found in .pt/.pth file")

        dims = {row.shape[1] for row in rows}
        if len(dims) != 1:
            raise RuntimeError(f"Inconsistent embedding dimensions found in file: {sorted(dims)}")

        emb = torch.cat(rows, dim=0)
    else:
        raise RuntimeError("Unsupported embeddings file extension. Use .npy/.npz/.pt/.pth")

    if emb.ndim != 2:
        raise RuntimeError(f"Embeddings must be 2D [N, D], got shape {tuple(emb.shape)}")

    return emb.float().cpu()


def save_variance_plots(var_per_dim: torch.Tensor,
                        args: argparse.Namespace,
                        source_tag: str) -> None:
    """Save histogram and per-dimension plots for latent variances."""
    os.makedirs(args.output_dir, exist_ok=True)
    var_np = var_per_dim.cpu().numpy()

    # Histogram of variances across latent dimensions.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_np, bins=args.hist_bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvline(var_np.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={var_np.mean():.4f}")
    ax.set_title(f"Latent Variance Histogram ({source_tag})")
    ax.set_xlabel("Per-dimension variance")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, f"latent_variance_hist_{source_tag}.png")
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    # Variance value for each latent dimension index.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(var_np.shape[0])
    ax.bar(x, var_np, color="#4c78a8", width=0.85)
    ax.axhline(var_np.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={var_np.mean():.4f}")
    ax.set_title(f"Per-Dimension Latent Variance ({source_tag})")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Variance")
    ax.legend()
    fig.tight_layout()
    bar_path = os.path.join(args.output_dir, f"latent_variance_per_dim_{source_tag}.png")
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)

    print(f"[latent_var] saved_hist={hist_path}")
    print(f"[latent_var] saved_per_dim_plot={bar_path}")


def main() -> None:
    args = parse_args()

    if args.embeddings is None and not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")

    if args.min_images > args.max_images:
        raise ValueError("--min_images must be <= --max_images")

    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")

    if args.embeddings is not None and not os.path.isfile(args.embeddings):
        raise FileNotFoundError(f"Embeddings file does not exist: {args.embeddings}")

    sampled_images = None
    source_tag = "embeddings" if args.embeddings else "sampled_images"
    if args.embeddings:
        mu = load_embeddings(args.embeddings)
    else:
        images = sample_random_images(
            data_dir=args.data_dir,
            in_channels=args.in_channels,
            img_size=args.img_size,
            min_images=args.min_images,
            max_images=args.max_images,
            seed=args.seed,
        )
        sampled_images = images.shape[0]

        model = build_model(args)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint)

        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(images)

    if mu.shape[1] != args.latent_dim:
        raise RuntimeError(
            f"Expected embedding dim {args.latent_dim}, got {mu.shape[1]}"
        )

    var_per_dim = mu.var(dim=0, unbiased=False)
    if var_per_dim.shape != (args.latent_dim,):
        raise RuntimeError(
            f"Expected variance shape {(args.latent_dim,)}, got {tuple(var_per_dim.shape)}"
        )
    if not torch.isfinite(var_per_dim).all():
        raise RuntimeError("Latent variances contain non-finite values")
    if not (var_per_dim >= 0).all():
        raise RuntimeError("Latent variances contain negative values")

    print(f"[latent_var] model={args.model}")
    if args.embeddings:
        print(f"[latent_var] embeddings_file={args.embeddings}")
        print(f"[latent_var] embedding_rows={mu.shape[0]}")
    else:
        print(f"[latent_var] sampled_images={sampled_images}")
    print(f"[latent_var] latent_dim={args.latent_dim}")
    print(f"[latent_var] mean_variance={var_per_dim.mean().item():.6f}")
    print(f"[latent_var] min_variance={var_per_dim.min().item():.6f}")
    print(f"[latent_var] max_variance={var_per_dim.max().item():.6f}")
    print("[latent_var] per_dim_variance=")
    print(var_per_dim.cpu().numpy())
    save_variance_plots(var_per_dim, args, source_tag=source_tag)


if __name__ == "__main__":
    main()