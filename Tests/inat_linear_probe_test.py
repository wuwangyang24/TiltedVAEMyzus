"""Linear probing test for iNat embeddings.

Encodes the iNat validation set using a trained checkpoint, then trains a
linear classifier on the embeddings using --test_cat labels (the taxonomy
level reserved for evaluation). Reports top-1 and top-5 accuracy with a
train/test split of the val set embeddings.

Usage:
    python Tests/inat_linear_probe_test.py \
        --checkpoint results/checkpoints/.../best.ckpt \
        --val_metadata inat2021/val.json \
        --val_image_dir inat2021 \
        --test_cat family \
        --superclass Insects \
        --dino_backbone vit_small_patch14_dinov2 \
        --lora_rank 8 --lora_alpha 16 --lora_targets qkv \
        --img_size 224 --batch_size 128 --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
    """Parse iNat2021 metadata, return (path, test_cat_value) pairs."""
    with open(metadata_path) as f:
        data = json.load(f)

    cat_map: Dict[int, Dict[str, str]] = {}
    for cat in data["categories"]:
        cat_map[cat["id"]] = cat

    img_map: Dict[int, str] = {}
    for img in data["images"]:
        img_map[img["id"]] = img["file_name"]

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
        full_path = os.path.join(image_dir, img_map[img_id])
        samples.append((full_path, str(test_val)))

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
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
               for k, v in state_dict.items()}
    # Resize DCL memory-bank buffers to match the checkpoint before loading.
    for key in ("dcl_sigreg_loss.class_means", "dcl_sigreg_loss.initialized"):
        if key in cleaned:
            buf = cleaned[key]
            param = model
            for attr in key.split("."):
                param = getattr(param, attr, None)
                if param is None:
                    break
            if param is not None and param.shape != buf.shape:
                # Replace the registered buffer with the correct size
                parent = model
                parts = key.split(".")
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                parent.register_buffer(parts[-1], torch.empty_like(buf))
    model.load_state_dict(cleaned, strict=False)
    return model


# ── Encoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(
    model: DinoV2LoRA, dataset: Dataset, batch_size: int,
    device: torch.device, num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_embs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Encoding"):
        embs = model.encode(imgs.to(device), normalize=True)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


# ── Linear probe ─────────────────────────────────────────────────────────────

def linear_probe(
    train_embs: torch.Tensor, train_labels: torch.Tensor,
    test_embs: torch.Tensor, test_labels: torch.Tensor,
    num_classes: int, device: torch.device,
    lr: float = 0.01, epochs: int = 100, batch_size: int = 256,
) -> Dict[str, float]:
    """Train a linear classifier and return top-1/top-5 accuracy on test."""
    dim = train_embs.size(1)
    classifier = torch.nn.Linear(dim, num_classes).to(device)
    optimizer = torch.optim.LBFGS(classifier.parameters(), lr=lr, max_iter=20)

    train_embs = train_embs.to(device)
    train_labels = train_labels.to(device)
    test_embs = test_embs.to(device)
    test_labels = test_labels.to(device)

    # LBFGS full-batch (embeddings fit in memory)
    def closure():
        optimizer.zero_grad()
        logits = classifier(train_embs)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        return loss

    for _ in range(epochs):
        optimizer.step(closure)

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_embs)
        top1 = (logits.argmax(dim=1) == test_labels).float().mean().item()
        k = min(5, num_classes)
        top5_pred = logits.topk(k, dim=1).indices
        top5 = (top5_pred == test_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"top1_acc": top1, "top5_acc": top5}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probing test on iNat val embeddings")

    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.ckpt)")
    p.add_argument("--train_metadata", default=None, help="iNat2021 train metadata JSON (if omitted, val set is split)")
    p.add_argument("--train_image_dir", default=None, help="Image directory for train set")
    p.add_argument("--val_metadata", required=True, help="iNat2021 val metadata JSON")
    p.add_argument("--val_image_dir", required=True, help="Image directory for val set")
    p.add_argument("--train_fraction", type=float, default=0.8,
                   help="Fraction of val set used for training when no train set is provided")
    p.add_argument("--test_cat", required=True,
                   help="Taxonomy level for linear probe labels (e.g. family, genus)")
    p.add_argument("--superclass", default=None,
                   help="Filter to this supercategory (e.g. Insects, Birds)")

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

    # Probe settings
    p.add_argument("--probe_lr", type=float, default=0.1)
    p.add_argument("--probe_epochs", type=int, default=100,
                   help="LBFGS iterations for the linear probe")

    # Runtime
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Parse metadata
    val_raw, val_classes = parse_inat_json(
        args.val_metadata, args.val_image_dir, args.test_cat, args.superclass)

    if args.train_metadata and args.train_image_dir:
        train_raw, train_classes = parse_inat_json(
            args.train_metadata, args.train_image_dir, args.test_cat, args.superclass)
        all_classes = sorted(set(train_classes + val_classes))
        class2idx = {c: i for i, c in enumerate(all_classes)}
        train_samples = [(path, class2idx[label]) for path, label in train_raw]
        val_samples = [(path, class2idx[label]) for path, label in val_raw]
    else:
        all_classes = sorted(set(val_classes))
        class2idx = {c: i for i, c in enumerate(all_classes)}
        indexed = [(path, class2idx[label]) for path, label in val_raw]
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(indexed))
        split = int(len(indexed) * args.train_fraction)
        train_samples = [indexed[i] for i in perm[:split]]
        val_samples = [indexed[i] for i in perm[split:]]

    num_classes = len(all_classes)
    print(f"Train set: {len(train_samples)} images, {num_classes} {args.test_cat} classes")
    print(f"Val set:   {len(val_samples)} images")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = InatValDataset(train_samples, transform)
    val_dataset = InatValDataset(val_samples, transform)

    # Load model and encode both splits
    model = load_model(args).to(device).eval()
    print("Encoding train set...")
    train_embs, train_labels = encode_dataset(model, train_dataset, args.batch_size,
                                              device, args.num_workers)
    print("Encoding val set...")
    test_embs, test_labels = encode_dataset(model, val_dataset, args.batch_size,
                                            device, args.num_workers)
    del model
    torch.cuda.empty_cache()

    # Linear probe
    print("Training linear probe...")
    results = linear_probe(
        train_embs, train_labels, test_embs, test_labels,
        num_classes=num_classes, device=device,
        lr=args.probe_lr, epochs=args.probe_epochs,
    )

    print(f"\n{'=' * 50}")
    print(f"Linear probe results (test_cat={args.test_cat}, "
          f"{num_classes} classes, train={len(train_samples)}, val={len(val_samples)}):")
    print(f"  Top-1 accuracy: {results['top1_acc']:.4f}")
    print(f"  Top-5 accuracy: {results['top5_acc']:.4f}")


if __name__ == "__main__":
    main()
