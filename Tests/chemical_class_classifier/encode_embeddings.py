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
python Tests/chemical_class_classifier/encode_embeddings.py --metadata ../METADATA/metadata_compound_all100ppm.json --root_dir ../DATA_TEST/ --output Tests/chemical_class_classifier/embeddings.pt --checkpoint results/checkpoints/tilted-latent128_kl0.001/best_balanced_acc.ckpt --model tilted --latent_dim 128 --img_size 96 --device cpu --compound_col compound --label_col synthesis_program --min_compounds_per_class 30 --class_metadata ../METADATA/synthesisprogram_compoundno.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

# This script lives in ``Tests/chemical_class_classifier/``; add the repo root
# (two levels up) to the path so the top-level ``Models`` package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE, DinoV2LoRA


class DinoV2Wrapper(torch.nn.Module):
    """Thin wrapper around a pretrained DINOv2 backbone that exposes an
    ``.encode()`` method compatible with VAE/TiltedVAE."""

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    def encode(self, x: torch.Tensor):
        features = self.backbone(x)          # (B, D)  D=384 for vits14
        return features, None                # no log_var


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
    p.add_argument("--checkpoint", default=None,
                   help="Trained Lightning checkpoint (.ckpt) or raw state_dict (.pt/.pth). "
                        "Not required for --model dino.")
    p.add_argument("--model", default="tilted", choices=["vae", "tilted", "dino", "dino_lora"],
                   help="Model architecture. 'dino' uses pretrained DINOv2 vits14. "
                        "'dino_lora' uses DinoV2LoRA with a checkpoint. Default: tilted")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=96,
                   help="Image size for VAE/TiltedVAE. Ignored for dino (uses 224). "
                        "For dino_lora must be a multiple of 14 (default 224).")
    p.add_argument("--tau", type=float, default=None,
                   help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # DinoV2LoRA-specific arguments
    p.add_argument("--dino_backbone", type=str, default="vit_small_patch14_dinov2",
                   help="DINOv2 backbone variant for dino_lora")
    p.add_argument("--embedding_dim", type=int, default=256,
                   help="Output embedding dimension for dino_lora")
    p.add_argument("--proj_hidden_dim", type=int, default=2048,
                   help="Projection head hidden dim for dino_lora")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, nargs="+", default=["qkv"],
                   help="Leaf module names to adapt with LoRA")
    p.add_argument("--use_proj_head", action="store_true", default=True,
                   help="Use projection head (default: True)")
    p.add_argument("--no_proj_head", action="store_true",
                   help="Disable projection head (output backbone features directly)")

    # Pre-filtering by class membership
    p.add_argument("--class_metadata", default=None,
                   help="Optional CSV/Excel file with compound and class columns "
                        "(same format as train_chemical_class_classifier.py --metadata). "
                        "Required when using --min_compounds_per_class.")
    p.add_argument("--compound_col", default="compound",
                   help="Compound ID column in --class_metadata. Default: compound")
    p.add_argument("--label_col", default="chemical_class",
                   help="Class label column in --class_metadata. Default: chemical_class")
    p.add_argument("--min_compounds_per_class", type=int, default=None,
                   help="Only encode compounds belonging to classes with at least this "
                        "many compounds. Requires --class_metadata.")
    p.add_argument("--filter_by_efficacy", type=float, default=0,
                   help="Keep only compounds with Efficacy >= this value (requires "
                        "an 'Efficacy' column in --class_metadata). Default: 0")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu)")

    args = p.parse_args()

    if args.model not in ("dino",) and args.checkpoint is None:
        p.error("--checkpoint is required for --model vae/tilted/dino_lora")

    if args.no_proj_head:
        args.use_proj_head = False

    if args.model == "dino_lora" and args.img_size % 14 != 0:
        args.img_size = 224

    return args


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "dino":
        return DinoV2Wrapper()
    if args.model == "dino_lora":
        return DinoV2LoRA(
            backbone=args.dino_backbone,
            img_size=args.img_size,
            embedding_dim=args.embedding_dim,
            proj_hidden_dim=args.proj_hidden_dim,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
            use_proj_head=args.use_proj_head,
        )
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


def _build_transform(img_size: int, imagenet_normalize: bool = False) -> T.Compose:
    """Square resize + scale to [0, 1].  Optionally add ImageNet normalization
    (required for DINOv2)."""
    transforms = [
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
    ]
    if imagenet_normalize:
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        )
    return T.Compose(transforms)


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
        out = model.encode(batch)
        mu = out if isinstance(out, torch.Tensor) else out[0]
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
    if args.model == "dino":
        print("Model  : DINOv2 vits14  (pretrained, latent dim 384)")
    elif args.model == "dino_lora":
        load_checkpoint(model, args.checkpoint)
        dim = args.embedding_dim if args.use_proj_head else "backbone"
        print(f"Model  : DinoV2LoRA  (backbone={args.dino_backbone}, "
              f"embedding_dim={dim}, img_size={args.img_size})")
    else:
        load_checkpoint(model, args.checkpoint)
        print(f"Model  : {args.model}  (latent dim {args.latent_dim})")
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    root_dir = Path(args.root_dir)
    if args.model in ("dino", "dino_lora"):
        img_size = args.img_size if args.model == "dino_lora" else 224
        transform = _build_transform(img_size, imagenet_normalize=True)
        mode = ImageReadMode.RGB
    else:
        img_size = args.img_size
        transform = _build_transform(img_size)
        mode = ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(args.metadata) as f:
        metadata = json.load(f)
    print(f"Metadata: {len(metadata)} compounds")

    # ── Pre-filter by efficacy and min_compounds_per_class ────────────────
    if (args.min_compounds_per_class is not None
            or (args.filter_by_efficacy and args.filter_by_efficacy > 0)):
        if args.class_metadata is None:
            raise ValueError("--class_metadata is required when using "
                             "--min_compounds_per_class or --filter_by_efficacy")
        ext = Path(args.class_metadata).suffix.lower()
        if ext in (".xls", ".xlsx"):
            class_df = pd.read_excel(args.class_metadata)
        else:
            class_df = pd.read_csv(args.class_metadata)
        class_df[args.compound_col] = class_df[args.compound_col].astype(str)
        class_df[args.label_col] = class_df[args.label_col].astype(str)

        if args.filter_by_efficacy and "Efficacy" in class_df.columns:
            before_eff = len(class_df)
            class_df = class_df[class_df["Efficacy"] >= args.filter_by_efficacy]
            print(f"Efficacy filter: kept {len(class_df)}/{before_eff} rows "
                  f"(Efficacy >= {args.filter_by_efficacy})")

        valid_compounds: Optional[Set[str]] = None
        if args.min_compounds_per_class is not None:
            min_cpc = max(args.min_compounds_per_class, 2)
            compounds_per_class = (
                class_df.groupby(args.label_col)[args.compound_col].nunique()
            )
            valid_classes = set(
                compounds_per_class[compounds_per_class >= min_cpc].index
            )
            valid_compounds = set(
                class_df.loc[class_df[args.label_col].isin(valid_classes), args.compound_col]
            )
        else:
            valid_compounds = set(class_df[args.compound_col])

        before = len(metadata)
        metadata = [e for e in metadata if str(e["Compound"]) in valid_compounds]
        print(f"Pre-filter: kept {len(metadata)}/{before} compounds")

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
