"""
encode_embeddings.py

Encode compound images with a trained TiltedVAE / VAE encoder into the
per-compound / per-plate embedding structure consumed by
``train_chemical_class_classifier.py``.

Adapted from the MyzusDINOAdapt ``encode_embeddings.py`` (custom-VAE path) to
use this repository's own ``Models`` (``VAE`` / ``TiltedVAE``) and its image
preprocessing (square resize to ``img_size``, pixels in ``[0, 1]``).

For each compound and each plate:
  - treated images are encoded individually and stored as a (N, D) tensor.
  - control images are encoded and averaged across all samples on that plate,
    stored as a single (D,) vector.

Metadata format (JSON, list of dicts, one per compound):
    [
        {
            "Compound": "1",
            "94000": {
                "treated": ["94000/well_2_1/treated/sample_1.png", ...],
                "control": ["94000/well_1_3/control/sample_1.png", ...]
            },
            "131000": { "treated": [...], "control": [...] }
        },
        { "Compound": "2", ... }
    ]

Output .pt file structure (dict):
    {
        <compound_id (str)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # (N, D) — one row per image (latent mean mu)
                "control": torch.Tensor    # (D,)   — averaged over all controls
            }
        }
    }

Usage:
    python Tests/chemical_class_classifier/encode_embeddings.py \
        --metadata   data/compound_images.json \
        --root_dir   ../DATA/Train/ \
        --output     Tests/chemical_class_classifier/embeddings.pt \
        --checkpoint results/checkpoints/last.ckpt \
        --model      tilted --latent_dim 128 --img_size 96 --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

# This script lives in ``Tests/chemical_class_classifier/``; add the repo root
# (two levels up) to the path so the top-level ``Models`` package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode compound images with a TiltedVAE/VAE encoder."
    )
    p.add_argument("--metadata", required=True,
                   help="JSON metadata file mapping compounds -> plates -> treated/control paths")
    p.add_argument("--root_dir", required=True,
                   help="Base directory prepended to every relative image path in the metadata")
    p.add_argument("--output", required=True,
                   help="Output .pt path for the encoded embeddings")

    # Model / checkpoint
    p.add_argument("--checkpoint", required=True,
                   help="Trained Lightning checkpoint (.ckpt) or raw state_dict (.pt/.pth)")
    p.add_argument("--model", default="tilted", choices=["vae", "tilted"],
                   help="Model architecture matching the checkpoint. Default: tilted")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--tau", type=float, default=None,
                   help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu)")
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
    """Load weights from either a Lightning checkpoint (keys prefixed with
    ``model.`` under ``state_dict``) or a raw model ``state_dict``."""
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


def _build_transform(img_size: int) -> T.Compose:
    """Square resize + scale to [0, 1] — matches the training preprocessing."""
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
    """Encode a list of image paths to a (N, D) float32 CPU tensor of latent means."""
    latents: List[torch.Tensor] = []
    for start in range(0, len(rel_paths), batch_size):
        batch_paths = rel_paths[start:start + batch_size]
        imgs = []
        for rel in batch_paths:
            img = read_image(str(root_dir / rel), mode=mode)
            imgs.append(transform(img))
        batch = torch.stack(imgs, dim=0).to(device)
        mu, _ = model.encode(batch)
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0) if latents else torch.empty(0)


def main() -> None:
    args = parse_args()

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

    root_dir = Path(args.root_dir)
    transform = _build_transform(args.img_size)
    mode = ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(args.metadata) as f:
        metadata = json.load(f)
    print(f"Metadata: {len(metadata)} compounds")

    embeddings = {}
    for entry in tqdm(metadata, desc="Encoding compounds"):
        compound_id = str(entry["Compound"])
        plate_dict = {}
        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            treated_paths = plate_data.get("treated", [])
            control_paths = plate_data.get("control", [])

            plate_entry = {}
            if treated_paths:
                plate_entry["treated"] = encode_paths(
                    treated_paths, root_dir, model, transform, mode,
                    args.batch_size, device,
                )
            if control_paths:
                control_latents = encode_paths(
                    control_paths, root_dir, model, transform, mode,
                    args.batch_size, device,
                )
                if control_latents.numel() > 0:
                    plate_entry["control"] = control_latents.mean(dim=0)

            if plate_entry:
                plate_dict[str(plate_id)] = plate_entry

        if plate_dict:
            embeddings[compound_id] = plate_dict

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} compounds to: {out_path}")


if __name__ == "__main__":
    main()
