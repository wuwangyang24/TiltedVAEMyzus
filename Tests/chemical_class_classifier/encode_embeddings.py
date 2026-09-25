"""
encode_embeddings.py

Encode compound images with a trained TiltedVAE / VAE encoder into the
per-compound / per-plate embedding structure consumed by
``train_chemical_class_classifier.py``.

Uses this repository's own ``Models`` (``VAE`` / ``TiltedVAE``) and its image
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
python TiltedVAEMyzus/Tests/chemical_class_classifier/encode_embeddings.py --metadata METADATA/metadata_compound_all100ppm.json --root_dir DATA_TEST/ --output results/embeddings_best_knn.pt --checkpoint 'results/checkpoints/tilted-latent128/best.ckpt' --model tilted --latent_dim 128 --img_size 96 --device cuda --compound_col compound --label_col synthesis_program --min_compounds_per_class 30 --filter_by_efficacy 0 --class_metadata METADATA/synthesisprogram_compoundno.csv
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
from torch.utils.data import DataLoader, Dataset
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
                   help="Trained Lightning checkpoint (.ckpt) or raw state_dict (.pt/.pth).")
    p.add_argument("--model", default="tilted", choices=["vae", "tilted"],
                   help="Model architecture. Default: tilted")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=96,
                   help="Image size for VAE/TiltedVAE.")
    p.add_argument("--tau", type=float, default=None,
                   help="Tilt parameter for TiltedVAE (only used with --model tilted)")

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
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for parallel image loading. Default: 4")
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu)")

    args = p.parse_args()

    return args


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


def _build_transform(img_size: int, imagenet_normalize: bool = False) -> T.Compose:
    """Square resize + scale to [0, 1].  Optionally add ImageNet normalization
    (required for ImageNet-pretrained backbones)."""
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


class _ImagePathDataset(Dataset):
    """Dataset that loads and transforms images from a list of relative paths."""

    def __init__(self, rel_paths: List[str], root_dir: Path,
                 transform: T.Compose, mode: ImageReadMode):
        self.paths = [root_dir / p for p in rel_paths]
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        path = self.paths[idx]
        if not path.exists():
            return None
        img = read_image(str(path), mode=self.mode)
        return self.transform(img)


def _collate_skip_none(batch):
    """Collate that filters out None entries (missing files)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)


@torch.no_grad()
def encode_paths(
    rel_paths: List[str],
    root_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    mode: ImageReadMode,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> torch.Tensor:
    """Encode a list of image paths to a (N, D) float32 CPU tensor of latent means."""
    if not rel_paths:
        return torch.empty(0)

    dataset = _ImagePathDataset(rel_paths, root_dir, transform, mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none,
    )

    latents: List[torch.Tensor] = []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device, non_blocking=True)
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
    load_checkpoint(model, args.checkpoint)
    print(f"Model  : {args.model}  (latent dim {args.latent_dim})")
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "cuda":
        model = torch.compile(model)

    root_dir = Path(args.root_dir)
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
    # ── Flatten all paths into a single list for one DataLoader pass ─────
    all_paths: List[str] = []
    # Each entry: (compound_id, plate_id, "treated"|"control", start_idx, count)
    index_map: List[tuple] = []

    for entry in metadata:
        compound_id = str(entry["Compound"])
        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            for role in ("treated", "control"):
                paths = plate_data.get(role, [])
                if paths:
                    index_map.append((compound_id, str(plate_id), role,
                                      len(all_paths), len(paths)))
                    all_paths.extend(paths)

    print(f"Total images to encode: {len(all_paths)}")

    dataset = _ImagePathDataset(all_paths, root_dir, transform, mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none,
    )

    # Encode everything in one pass
    all_latents: List[torch.Tensor] = []
    encode_kwargs = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            if batch is None:
                all_latents.append(torch.empty(0))
                continue
            batch = batch.to(device, non_blocking=True)
            out = model.encode(batch, **encode_kwargs)
            mu = out if isinstance(out, torch.Tensor) else out[0]
            all_latents.append(mu.cpu())

    all_encoded = torch.cat(all_latents, dim=0) if all_latents else torch.empty(0)

    # Scatter results back into compound/plate/role structure
    for compound_id, plate_id, role, start, count in index_map:
        chunk = all_encoded[start:start + count]
        if chunk.numel() == 0:
            continue
        if compound_id not in embeddings:
            embeddings[compound_id] = {}
        if plate_id not in embeddings[compound_id]:
            embeddings[compound_id][plate_id] = {}
        if role == "control":
            embeddings[compound_id][plate_id]["control_mean"] = chunk.mean(dim=0)
            embeddings[compound_id][plate_id]["control_median"] = chunk.median(dim=0).values
        else:
            embeddings[compound_id][plate_id]["treated"] = chunk

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} compounds to: {out_path}")


if __name__ == "__main__":
    main()
