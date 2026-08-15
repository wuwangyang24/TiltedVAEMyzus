"""Per-layer kNN accuracy test for the DINOv2 backbone.

For a given dataset and ``--test_cat`` taxonomy level, this encodes the
validation images through *every* transformer block of the (LoRA-adapted)
DINOv2 backbone and computes the top-k kNN accuracy of each layer's pooled
features. The output ranks the layers so you can see which block produces the
most linearly/metrically separable representation for the chosen ``test_cat``.

The kNN metric matches the validation ``val_knn_acc`` used during training
(cosine similarity, leave-one-out over the full val set, top-k hit if any of
the k nearest neighbours shares the label).

Usage:
    python Tests/layer_knn_acc_test.py \
        --checkpoint results/checkpoints/.../best.ckpt \
        --val_metadata inat2021/val.json \
        --val_image_dir inat2021 \
        --test_cat family \
        --superclass Insects \
        --dino_backbone vit_small_patch14_dinov2 \
        --lora_rank 8 --lora_alpha 16 --lora_targets qkv \
        --pooling cls --img_size 224 --batch_size 128 --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import DinoV2LoRA

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Dataset ──────────────────────────────────────────────────────────────────

class InatValDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        return self.transform(img), label


def parse_inat_json(
    metadata_path: str, image_dir: str, test_cat: str,
    superclass: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse iNat2021 metadata, return (path, test_cat_value) pairs + classes."""
    with open(metadata_path) as f:
        data = json.load(f)

    cat_map: Dict[int, Dict[str, str]] = {c["id"]: c for c in data["categories"]}
    img_map: Dict[int, str] = {i["id"]: i["file_name"] for i in data["images"]}

    sc = superclass.lower() if superclass else None
    samples = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in img_map or cat_id not in cat_map:
            continue
        cat_info = cat_map[cat_id]
        if sc is not None and str(cat_info.get("supercategory", "")).lower() != sc:
            continue
        test_val = cat_info.get(test_cat)
        if test_val is None:
            continue
        samples.append((os.path.join(image_dir, img_map[img_id]), str(test_val)))

    classes = sorted(set(s[1] for s in samples))
    return samples, classes


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(args: argparse.Namespace) -> DinoV2LoRA:
    model = DinoV2LoRA(
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
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
                   for k, v in state_dict.items()}
        # Resize DCL memory-bank buffers to match the checkpoint before loading.
        for key in ("dcl_sigreg_loss.class_means", "dcl_sigreg_loss.initialized"):
            if key in cleaned:
                parent = model
                parts = key.split(".")
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                buf = cleaned[key]
                if getattr(parent, parts[-1]).shape != buf.shape:
                    parent.register_buffer(parts[-1], torch.empty_like(buf))
        incompatible = model.load_state_dict(cleaned, strict=False)

        # strict=False silently drops mismatched keys. If the LoRA adapters in
        # the checkpoint don't match the ones built here (wrong rank/alpha/
        # targets/backbone), they stay at zero-init (identity) and the test
        # ends up measuring the *frozen pretrained* backbone. Surface that.
        ckpt_lora = [k for k in cleaned if "lora_" in k]
        missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
        unexpected_lora = [k for k in incompatible.unexpected_keys if "lora_" in k]
        print(f"[load_checkpoint] loaded {len(cleaned)} tensors; "
              f"missing={len(incompatible.missing_keys)}, "
              f"unexpected={len(incompatible.unexpected_keys)}")
        print(f"[load_checkpoint] LoRA tensors in checkpoint: {len(ckpt_lora)}; "
              f"missing_lora={len(missing_lora)}, unexpected_lora={len(unexpected_lora)}")
        if unexpected_lora or (ckpt_lora and missing_lora):
            print("[load_checkpoint] WARNING: LoRA keys did not match the built "
                  "model. The adapters are NOT loaded (identity), so results "
                  "reflect the pretrained backbone. Check --lora_targets/"
                  "--lora_rank/--lora_alpha/--dino_backbone against the checkpoint.")
    return model


# ── Per-layer encoding ───────────────────────────────────────────────────────

def _pool(patch_tokens: torch.Tensor, prefix_tokens: torch.Tensor,
          pooling: str) -> torch.Tensor:
    """Reduce block token outputs to a single vector per image.

    patch_tokens:  (N, num_patches, D)
    prefix_tokens: (N, num_prefix, D)  -- prefix[:, 0] is the CLS token.
    """
    cls = prefix_tokens[:, 0]
    if pooling == "cls":
        return cls
    if pooling == "mean":
        return patch_tokens.mean(dim=1)
    if pooling == "cls_mean":
        return torch.cat([cls, patch_tokens.mean(dim=1)], dim=-1)
    raise ValueError(f"Unknown pooling '{pooling}'")


@torch.no_grad()
def encode_layers(
    model: DinoV2LoRA, dataset: Dataset, layer_indices: List[int],
    pooling: str, batch_size: int, device: torch.device, num_workers: int = 4,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Encode the dataset, returning {layer_idx: embeddings (N, D)} and labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    backbone = model.backbone
    per_layer: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
    all_labels: List[torch.Tensor] = []

    for imgs, labels in tqdm(loader, desc="Encoding layers"):
        # get_intermediate_layers returns, for each requested block, a
        # (patch_tokens, prefix_tokens) pair with the final norm applied.
        outputs = backbone.get_intermediate_layers(
            imgs.to(device), n=tuple(layer_indices),
            reshape=False, return_prefix_tokens=True, norm=True,
        )
        for idx, (patch_tokens, prefix_tokens) in zip(layer_indices, outputs):
            per_layer[idx].append(_pool(patch_tokens, prefix_tokens, pooling).cpu())
        all_labels.append(labels)

    embeddings = {i: torch.cat(v, dim=0) for i, v in per_layer.items()}
    return embeddings, torch.cat(all_labels, dim=0)


# ── kNN accuracy (mirrors ContrastiveExperiment._full_set_knn_accuracy) ───────

@torch.no_grad()
def full_set_knn_accuracy(
    embeddings: torch.Tensor, labels: torch.Tensor,
    ks: Tuple[int, ...] = (1, 3, 5), chunk_size: int = 1024,
) -> Dict[int, float]:
    """Top-k kNN accuracy over the full set (cosine, leave-one-out)."""
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = labels.view(-1)
    n = embeddings.size(0)
    max_k = min(max(ks), n - 1)
    if max_k < 1:
        return {k: 1.0 for k in ks}

    hits = {k: torch.zeros(n, dtype=torch.bool, device=embeddings.device) for k in ks}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = embeddings[start:end] @ embeddings.t()
        rows = torch.arange(end - start, device=embeddings.device)
        sim[rows, torch.arange(start, end, device=embeddings.device)] = float("-inf")
        topk_idx = sim.topk(max_k, dim=1).indices
        match = labels[topk_idx] == labels[start:end].unsqueeze(1)
        for k in ks:
            hits[k][start:end] = match[:, :min(k, max_k)].any(dim=1)
    return {k: float(hits[k].float().mean().item()) for k in ks}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-layer kNN accuracy of the DINOv2 backbone for a test_cat")

    p.add_argument("--checkpoint", default=None,
                   help="Model checkpoint (.ckpt). Omit to probe the pretrained backbone.")
    p.add_argument("--val_metadata", required=True, help="iNat2021 val metadata JSON")
    p.add_argument("--val_image_dir", required=True, help="Image directory for val set")
    p.add_argument("--test_cat", required=True,
                   help="Taxonomy level used as the kNN label (e.g. family, genus)")
    p.add_argument("--superclass", default=None,
                   help="Filter to this supercategory (e.g. Insects, Birds)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on the number of val images (random subset)")

    # Feature pooling per layer
    p.add_argument("--pooling", default="cls", choices=["cls", "mean", "cls_mean"],
                   help="How to pool each block's tokens into one vector")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Specific block indices to probe (default: all blocks)")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5],
                   help="Values of k for top-k kNN accuracy")

    # Model architecture (must match checkpoint)
    p.add_argument("--dino_backbone", default="vit_small_patch14_dinov2")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--proj_hidden_dim", type=int, default=2048)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, nargs="*", default=["qkv"])
    p.add_argument("--use_proj_head", action="store_true")

    # Runtime
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    val_raw, val_classes = parse_inat_json(
        args.val_metadata, args.val_image_dir, args.test_cat, args.superclass)
    class2idx = {c: i for i, c in enumerate(sorted(val_classes))}
    samples = [(path, class2idx[label]) for path, label in val_raw]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(samples)
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]

    print(f"Val set: {len(samples)} images, {len(class2idx)} '{args.test_cat}' classes"
          + (f" (superclass={args.superclass})" if args.superclass else ""))

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = InatValDataset(samples, transform)

    model = load_model(args).to(device).eval()

    depth = len(model.backbone.blocks)
    layer_indices = args.layers if args.layers else list(range(depth))
    layer_indices = [i for i in layer_indices if 0 <= i < depth]
    print(f"Backbone '{args.dino_backbone}': {depth} blocks; "
          f"probing layers {layer_indices} with pooling='{args.pooling}'\n")

    embeddings, labels = encode_layers(
        model, dataset, layer_indices, args.pooling,
        args.batch_size, device, args.num_workers)
    labels = labels.to(device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ks = tuple(sorted(set(args.topk)))
    results: Dict[int, Dict[int, float]] = {}
    for idx in layer_indices:
        results[idx] = full_set_knn_accuracy(
            embeddings[idx].to(device), labels, ks=ks)

    # ── Report ────────────────────────────────────────────────────────────────
    header = "Layer  " + "  ".join(f"top{k:>2}" for k in ks)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for idx in layer_indices:
        row = f"{idx:>5}  " + "  ".join(f"{results[idx][k]:.4f}" for k in ks)
        print(row)
    print("=" * len(header))

    primary_k = ks[0]
    best_layer = max(layer_indices, key=lambda i: results[i][primary_k])
    print(f"\nBest layer for '{args.test_cat}' (top-{primary_k} kNN acc): "
          f"layer {best_layer} = {results[best_layer][primary_k]:.4f}")


if __name__ == "__main__":
    main()
