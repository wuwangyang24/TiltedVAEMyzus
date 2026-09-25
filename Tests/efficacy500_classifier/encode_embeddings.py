"""
encode_embeddings.py

Encode compound images with a trained TiltedVAE / VAE encoder into the
per-compound / per-plate embedding structure consumed by
``train_efficacy_classifier.py``.

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

Usage (VAE/TiltedVAE):
python TiltedVAEMyzus/Tests/efficacy500_classifier/encode_embeddings.py --metadata METADATA/metadata_compound_all100ppm.json --root_dir DATA_TEST/ --output TiltedVAEMyzus/Tests/efficacy500_classifier/embeddings_100ppm.pt --checkpoint TiltedVAEMyzus/results/checkpoints/tilted-latent256_kld0.01/best_balanced_acc.ckpt --model tilted --latent_dim 256 --img_size 96 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

# This script lives in ``Tests/efficacy500_classifier/``; add the repo root
# (two levels up) to the path so the top-level ``Models`` package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE


class ImagePathDataset(Dataset):
    """Dataset that loads images by path and returns (global_index, image_tensor).
    Invalid/missing images are skipped via a collate function."""

    def __init__(self, paths: List[str], root_dir: Path, transform: T.Compose,
                 mode: ImageReadMode):
        self.paths = paths
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        full_path = self.root_dir / self.paths[idx]
        if not full_path.exists():
            return None
        img = read_image(str(full_path), mode=self.mode)
        return idx, self.transform(img)


def _collate_skip_none(batch):
    """Collate that filters out None entries (missing files)."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    indices, imgs = zip(*batch)
    return list(indices), torch.stack(imgs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode compound images with a TiltedVAE/VAE encoder "
                    "for efficacy classification."
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

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader workers for parallel image loading")
    p.add_argument("--fp16", action="store_true",
                   help="Use mixed-precision (fp16) inference for faster encoding on GPU")
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
    """Square resize + scale to [0, 1].  Optionally add ImageNet normalization."""
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

    root_dir = Path(args.root_dir)
    img_size = args.img_size
    transform = _build_transform(img_size)
    mode = ImageReadMode.GRAY if args.in_channels == 1 else ImageReadMode.RGB

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(args.metadata) as f:
        metadata = json.load(f)
    print(f"Metadata: {len(metadata)} compounds")

    # ── Collect all image paths first, then encode in large contiguous batches ─
    # This avoids per-compound tiny batches that starve the GPU.
    all_paths: List[str] = []
    path_to_global_idx: dict = {}  # rel_path -> index in all_paths

    # Track structure: (compound_id, plate_id, "treated"/"control", [global_indices])
    structure: List[tuple] = []

    for entry in metadata:
        compound_id = str(entry["Compound"])
        for plate_id, plate_data in entry.items():
            if plate_id == "Compound":
                continue
            for role in ("treated", "control"):
                paths = plate_data.get(role, [])
                if not paths:
                    continue
                indices = []
                for p in paths:
                    if p not in path_to_global_idx:
                        path_to_global_idx[p] = len(all_paths)
                        all_paths.append(p)
                    indices.append(path_to_global_idx[p])
                structure.append((compound_id, str(plate_id), role, indices))

    print(f"Total images to encode: {len(all_paths)}")

    # ── Encode all images in large batches via DataLoader ───────────────────
    dataset = ImagePathDataset(all_paths, root_dir, transform, mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_skip_none,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    all_embeddings = torch.zeros(len(all_paths), 1)  # placeholder
    encoded_mask = np.zeros(len(all_paths), dtype=bool)
    latent_chunks: List[Tuple[List[int], torch.Tensor]] = []

    for batch_data in tqdm(loader, desc="Encoding images"):
        if batch_data is None:
            continue
        valid_indices, imgs = batch_data
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=args.fp16):
            out = model.encode(imgs)
        mu = out if isinstance(out, torch.Tensor) else out[0]
        latent_chunks.append((valid_indices, mu.float().cpu()))

    # Assemble into a single tensor
    if latent_chunks:
        dim = latent_chunks[0][1].size(1)
        all_embeddings = torch.zeros(len(all_paths), dim)
        for valid_indices, mu in latent_chunks:
            for local_i, global_i in enumerate(valid_indices):
                all_embeddings[global_i] = mu[local_i]
                encoded_mask[global_i] = True

    # ── Reassemble per-compound structure ────────────────────────────────────
    embeddings = {}
    for compound_id, plate_id, role, indices in structure:
        valid = [i for i in indices if encoded_mask[i]]
        if not valid:
            continue
        emb = all_embeddings[valid]
        if compound_id not in embeddings:
            embeddings[compound_id] = {}
        if plate_id not in embeddings[compound_id]:
            embeddings[compound_id][plate_id] = {}
        if role == "treated":
            embeddings[compound_id][plate_id]["treated"] = emb
        else:
            embeddings[compound_id][plate_id]["control_mean"] = emb.mean(dim=0)
            embeddings[compound_id][plate_id]["control_median"] = emb.median(dim=0).values

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} compounds to: {out_path}")


if __name__ == "__main__":
    main()
