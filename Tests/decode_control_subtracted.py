"""
decode_control_subtracted.py

Decode VAE embeddings with the plate-matched control mean subtracted to
visualize the *pure phenotypic change* induced by a compound — i.e. what the
compound adds on top of the DMSO/control background.

Two modes of operation:
  1. From pre-encoded embeddings (--embeddings): loads a .pt file produced by
     ``encode_embeddings.py``, subtracts the per-plate control mean from each
     treated embedding, then decodes the difference vectors.
  2. From raw images (--metadata / --root_dir): encodes on-the-fly, subtracts,
     and decodes.

Output: a grid of decoded images saved as PNG, plus optionally the raw
difference embeddings (.pt) for downstream analysis.

Usage:
# From pre-encoded embeddings
python TiltedVAEMyzus/Tests/decode_control_subtracted.py --embeddings TiltedVAEMyzus/results/checkpoints/tilted-latent128_kl0.001_bestsofar/embeddings_best_balanced_acc.pt --checkpoint TiltedVAEMyzus/results/checkpoints/tilted-latent128_kl0.001_bestsofar/best_balanced_acc.ckpt --model tilted --latent_dim 128 --img_size 96 --compounds BCS-AO37552 --max_images_per_compound 8 --output_dir results/control_subtracted/ --device cpu

    # From raw images (encodes on the fly)
    python Tests/decode_control_subtracted.py \
        --metadata ../METADATA/metadata_compound_all100ppm.json \
        --root_dir ../DATA_TEST/ \
        --checkpoint results/checkpoints/last.ckpt \
        --model tilted --latent_dim 128 --img_size 96 \
        --compounds 1 2 3 --max_images_per_compound 8 \
        --output_dir results/control_subtracted/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import make_grid, save_image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode control-subtracted embeddings to visualize compound effects."
    )

    # Input source (choose one)
    p.add_argument("--embeddings", default=None,
                   help=".pt file from encode_embeddings.py (pre-encoded embeddings)")
    p.add_argument("--metadata", default=None,
                   help="JSON metadata (for on-the-fly encoding)")
    p.add_argument("--root_dir", default=None,
                   help="Image root directory (used with --metadata)")

    # Model / checkpoint
    p.add_argument("--checkpoint", required=True,
                   help="Trained checkpoint (.ckpt or .pt/.pth)")
    p.add_argument("--model", default="tilted", choices=["vae", "tilted"])
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--tau", type=float, default=None)

    # Selection
    p.add_argument("--compounds", nargs="*", default=None,
                   help="Compound IDs to decode. If omitted, decodes all.")
    p.add_argument("--max_images_per_compound", type=int, default=8,
                   help="Max treated images to decode per compound (default: 8)")

    # Output
    p.add_argument("--output_dir", default="results/control_subtracted/",
                   help="Directory for output images and optional .pt file")
    p.add_argument("--save_embeddings", action="store_true",
                   help="Also save the control-subtracted latent vectors as .pt")
    p.add_argument("--nrow", type=int, default=8,
                   help="Number of images per row in the grid (default: 8)")
    p.add_argument("--also_decode_treated", action="store_true",
                   help="Additionally decode the raw treated embeddings for comparison")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v

    model.load_state_dict(cleaned, strict=False)


def _build_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ])


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


@torch.no_grad()
def decode_latents(
    z: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Decode latent vectors into images. Returns (N, C, H, W) on CPU."""
    decoded: List[torch.Tensor] = []
    for start in range(0, z.shape[0], batch_size):
        batch_z = z[start:start + batch_size].to(device)
        imgs = model.decode(batch_z)
        decoded.append(imgs.cpu())
    return torch.cat(decoded, dim=0)


def load_embeddings_from_file(
    emb_path: str,
    compounds: Optional[List[str]],
    max_per_compound: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load pre-encoded embeddings and compute control-subtracted vectors.

    Returns: {compound_id: {"subtracted": (N, D), "treated": (N, D), "control": (D,)}}
    """
    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    results = {}

    for compound_id, plates in data.items():
        if compounds is not None and compound_id not in compounds:
            continue

        all_treated = []
        all_subtracted = []
        control_sum = None
        control_count = 0

        for plate_id, plate_data in plates.items():
            treated = plate_data.get("treated")
            control = plate_data.get("control")
            if treated is None or control is None:
                continue
            if treated.ndim == 1:
                treated = treated.unsqueeze(0)
            if control.ndim == 1:
                pass  # (D,) as expected

            # Subtract plate-matched control from each treated embedding
            subtracted = treated - control.unsqueeze(0)
            all_treated.append(treated)
            all_subtracted.append(subtracted)

            control_sum = control if control_sum is None else control_sum + control
            control_count += 1

        if not all_subtracted:
            continue

        all_treated_cat = torch.cat(all_treated, dim=0)[:max_per_compound]
        all_subtracted_cat = torch.cat(all_subtracted, dim=0)[:max_per_compound]
        avg_control = control_sum / control_count if control_count > 0 else torch.zeros(all_subtracted_cat.shape[1])

        results[compound_id] = {
            "subtracted": all_subtracted_cat,
            "treated": all_treated_cat,
            "control": avg_control,
        }

    return results


def load_embeddings_from_metadata(
    metadata_path: str,
    root_dir: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Encode on the fly and compute control-subtracted vectors."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    transform = _build_transform(args.img_size)
    mode = ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB
    root = Path(root_dir)
    results = {}

    for entry in metadata:
        compound_id = str(entry["Compound"])
        if args.compounds is not None and compound_id not in args.compounds:
            continue

        all_treated = []
        all_subtracted = []
        control_sum = None
        control_count = 0

        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            treated_paths = plate_data.get("treated", [])
            control_paths = plate_data.get("control", [])
            if not treated_paths or not control_paths:
                continue

            treated_emb = encode_paths(
                treated_paths, root, model, transform, mode,
                args.batch_size, device,
            )
            control_emb = encode_paths(
                control_paths, root, model, transform, mode,
                args.batch_size, device,
            )
            if treated_emb.numel() == 0 or control_emb.numel() == 0:
                continue

            control_mean = control_emb.mean(dim=0)
            subtracted = treated_emb - control_mean.unsqueeze(0)
            all_treated.append(treated_emb)
            all_subtracted.append(subtracted)

            control_sum = control_mean if control_sum is None else control_sum + control_mean
            control_count += 1

        if not all_subtracted:
            continue

        all_treated_cat = torch.cat(all_treated, dim=0)[:args.max_images_per_compound]
        all_subtracted_cat = torch.cat(all_subtracted, dim=0)[:args.max_images_per_compound]
        avg_control = control_sum / control_count if control_count > 0 else torch.zeros(all_subtracted_cat.shape[1])

        results[compound_id] = {
            "subtracted": all_subtracted_cat,
            "treated": all_treated_cat,
            "control": avg_control,
        }

    return results


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Build and load model (decoder needed for all modes)
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model : {args.model}  (latent dim {args.latent_dim})")

    # Get embeddings
    if args.embeddings:
        print(f"Loading pre-encoded embeddings from: {args.embeddings}")
        compound_data = load_embeddings_from_file(
            args.embeddings, args.compounds, args.max_images_per_compound
        )
    elif args.metadata and args.root_dir:
        print(f"Encoding on the fly from: {args.metadata}")
        compound_data = load_embeddings_from_metadata(
            args.metadata, args.root_dir, model, args, device
        )
    else:
        raise ValueError("Provide either --embeddings or both --metadata and --root_dir")

    print(f"Compounds to decode: {len(compound_data)}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decode and save per-compound
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for compound_id, data in compound_data.items():
        subtracted = data["subtracted"]  # (N, D)
        treated = data["treated"]        # (N, D)
        control = data["control"]        # (D,)

        print(f"  Compound {compound_id}: {treated.shape[0]} images")

        # Decode treated embeddings in image space
        decoded_treated = decode_latents(treated, model, device, args.batch_size)
        decoded_treated = decoded_treated.clamp(0, 1)

        # Decode control embedding in image space
        decoded_control = decode_latents(control.unsqueeze(0), model, device)
        decoded_control = decoded_control.clamp(0, 1)  # (1, C, H, W)

        # Subtract reconstructed control image from each reconstructed treated image
        # in pixel space: (N, C, H, W) - (1, C, H, W)
        pixel_diff = decoded_treated - decoded_control

        # Save the latent-space subtracted grid (existing behavior)
        decoded_subtracted = decode_latents(subtracted, model, device, args.batch_size)
        decoded_subtracted = decoded_subtracted.clamp(0, 1)
        grid = make_grid(decoded_subtracted, nrow=args.nrow, padding=2)
        save_image(grid, out_dir / f"compound_{compound_id}_subtracted.png")

        # ── Mean heatmap: average pixel-space difference across all images ──
        # pixel_diff shape: (N, C, H, W) → mean over N → (C, H, W)
        mean_diff = pixel_diff.mean(dim=0)  # (C, H, W)
        # Convert to single-channel by averaging across color channels
        if mean_diff.shape[0] == 3:
            mean_gray = mean_diff.mean(dim=0).numpy()  # (H, W)
        else:
            mean_gray = mean_diff.squeeze(0).numpy()   # (H, W)

        fig, ax = plt.subplots(figsize=(6, 5))
        # Use diverging colormap: blue = control brighter, red = treated brighter
        vmax = max(abs(mean_gray.min()), abs(mean_gray.max()))
        im = ax.imshow(mean_gray, cmap="RdBu_r", interpolation="nearest",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"Compound {compound_id}\n"
                     f"Mean pixel diff: treated − control (N={treated.shape[0]})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        heatmap_path = out_dir / f"compound_{compound_id}_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"    Saved heatmap to {heatmap_path}")

        # Save treated and control decoded images
        if args.also_decode_treated:
            grid_treated = make_grid(decoded_treated, nrow=args.nrow, padding=2)
            save_image(grid_treated, out_dir / f"compound_{compound_id}_treated.png")
            save_image(decoded_control.squeeze(0), out_dir / f"compound_{compound_id}_control.png")

    # Optionally save all subtracted embeddings for downstream analysis
    if args.save_embeddings:
        emb_out = {}
        for compound_id, data in compound_data.items():
            emb_out[compound_id] = data["subtracted"]
        torch.save(emb_out, out_dir / "control_subtracted_embeddings.pt")
        print(f"Saved subtracted embeddings to: {out_dir / 'control_subtracted_embeddings.pt'}")

    print(f"Done. Decoded images saved to: {out_dir}")


if __name__ == "__main__":
    main()
