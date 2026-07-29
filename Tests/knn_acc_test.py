"""KNN accuracy test for embedding quality.

Randomly selects n compounds and evaluates top-k KNN accuracy on their
embeddings, mirroring the batch-level KNN accuracy computed during validation.

For each image embedding, the k nearest neighbours (by cosine similarity) are
found among all other embeddings. The image is considered correctly classified
if *any* of its k nearest neighbours shares the same compound label (top-k
hit). The overall accuracy is the fraction of images with a correct top-k hit.

Supports both pre-computed embeddings (from encode_embeddings.py) and on-the-fly
encoding with a trained model checkpoint.

Usage (pre-computed embeddings):
python TiltedVAEMyzus/Tests/knn_acc_test.py --metadata METADATA/metadata_compound_all100ppm.json --embedding results/checkpoints/DINO_LoRA(qkv&proj)_R32_A64_P64_K8_NoProj_T0.05_Comp/best_val_knn_acc/embeddings_best_val_knn_acc.pt --n_compounds 50 --topk 1 5 10 --seed 42

Usage (on-the-fly encoding):
    python Tests/knn_acc_test.py \
        --metadata METADATA/metadata_compound_all100ppm.json \
        --root_dir DATA_TEST/ \
        --checkpoint results/checkpoints/.../best.ckpt \
        --model tilted --latent_dim 128 --img_size 96 \
        --n_compounds 50 --topk 1 5 10 --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import VAE, TiltedVAE


# ── Model helpers (same pattern as other tests) ──────────────────────────────

class DinoV2Wrapper(torch.nn.Module):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMG_SIZE = 224

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    def encode(self, x: torch.Tensor):
        x = self.normalize(x)
        return self.backbone(x), None


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "dino":
        return DinoV2Wrapper()
    if args.model == "tilted":
        return TiltedVAE(in_channels=args.in_channels, latent_dim=args.latent_dim,
                         tau=args.tau, img_size=args.img_size)
    return VAE(in_channels=args.in_channels, latent_dim=args.latent_dim,
               img_size=args.img_size)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_metadata(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def select_compounds(metadata: List[dict], n: int, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    if n >= len(metadata):
        return metadata
    indices = rng.choice(len(metadata), size=n, replace=False)
    return [metadata[i] for i in indices]


@torch.no_grad()
def encode_paths(
    rel_paths: List[str], root_dir: Path, model: torch.nn.Module,
    transform: T.Compose, mode: ImageReadMode, batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    for start in range(0, len(rel_paths), batch_size):
        batch_paths = rel_paths[start:start + batch_size]
        imgs = []
        for rel in batch_paths:
            full = root_dir / rel
            if not full.exists():
                continue
            imgs.append(transform(read_image(str(full), mode=mode)))
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        mu, _ = model.encode(batch)
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0) if latents else torch.empty(0)


def gather_embeddings_from_file(
    embedding_path: str, selected: List[dict],
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load pre-computed embeddings and gather per-image embeddings + labels."""
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)
    all_embs, all_labels = [], []
    label_map: Dict[str, int] = {}

    for entry in selected:
        cid = str(entry["Compound"])
        if cid not in data:
            continue
        if cid not in label_map:
            label_map[cid] = len(label_map)
        lab = label_map[cid]
        for plate_data in data[cid].values():
            treated = plate_data.get("treated", None)
            if treated is None or treated.numel() == 0:
                continue
            embs = treated.clone()
            if subtract_control:
                control = plate_data.get("control", None)
                if control is not None and control.numel() > 0:
                    ctrl = control.float()
                    if ctrl.ndim > 1:
                        ctrl = ctrl.mean(dim=0)
                    if normalize_before_subtract:
                        embs = torch.nn.functional.normalize(embs, dim=1)
                        ctrl = ctrl / (ctrl.norm() + 1e-8)
                    embs = embs - ctrl.unsqueeze(0)
            all_embs.append(embs)
            all_labels.extend([lab] * embs.size(0))

    if not all_embs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embs, dim=0), torch.tensor(all_labels, dtype=torch.long)


def gather_embeddings_from_model(
    selected: List[dict], args: argparse.Namespace,
    model: torch.nn.Module, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode images on the fly and return per-image embeddings + labels."""
    root_dir = Path(args.root_dir)
    img_size = args.img_size
    mode = ImageReadMode.RGB if args.in_channels == 3 else ImageReadMode.GRAY
    transform = T.Compose([T.Resize((img_size, img_size), antialias=True),
                           T.ConvertImageDtype(torch.float32)])

    all_embs, all_labels = [], []
    label_map: Dict[str, int] = {}

    for entry in selected:
        cid = str(entry["Compound"])
        if cid not in label_map:
            label_map[cid] = len(label_map)
        lab = label_map[cid]
        for key, plate_data in entry.items():
            if key == "Compound" or not isinstance(plate_data, dict):
                continue
            treated_paths = plate_data.get("treated", [])
            if not treated_paths:
                continue
            embs = encode_paths(treated_paths, root_dir, model, transform, mode,
                                args.batch_size, device)
            if embs.numel() == 0:
                continue
            if args.subtract_control:
                ctrl_paths = plate_data.get("control", [])
                if ctrl_paths:
                    ctrl_embs = encode_paths(ctrl_paths, root_dir, model, transform,
                                             mode, args.batch_size, device)
                    if ctrl_embs.numel() > 0:
                        ctrl = ctrl_embs.mean(dim=0)
                        if args.normalize_before_subtract:
                            embs = torch.nn.functional.normalize(embs, dim=1)
                            ctrl = ctrl / (ctrl.norm() + 1e-8)
                        embs = embs - ctrl.unsqueeze(0)
            all_embs.append(embs)
            all_labels.extend([lab] * embs.size(0))

    if not all_embs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embs, dim=0), torch.tensor(all_labels, dtype=torch.long)


# ── KNN accuracy (mirrors DinoV2LoRA._batch_knn_accuracy, generalized to top-k) ─

@torch.no_grad()
def topk_knn_accuracy(
    embeddings: torch.Tensor, labels: torch.Tensor, k: int,
) -> float:
    """Fraction of samples whose top-k nearest neighbours contain a same-label sample.

    Uses cosine similarity (embeddings are L2-normalized first).
    """
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.t()
    # Mask self-similarity
    sim.fill_diagonal_(float("-inf"))

    n = embeddings.size(0)
    actual_k = min(k, n - 1)
    if actual_k < 1:
        return 0.0

    _, topk_idx = sim.topk(actual_k, dim=1)  # (N, k)
    topk_labels = labels[topk_idx]            # (N, k)
    hits = (topk_labels == labels.unsqueeze(1)).any(dim=1)  # (N,)
    return float(hits.float().mean().item())


def bootstrap_ci_by_compound(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    compound_ids: np.ndarray,
    topk: List[int],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[int, Tuple[float, float, float]]:
    """Bootstrap confidence intervals by resampling compounds.

    Each bootstrap iteration samples compounds with replacement, gathers all
    their images, and recomputes top-k KNN accuracy. This respects the
    within-compound correlation structure.

    Returns {k: (mean, lower, upper)} for each k in topk.
    """
    rng = np.random.default_rng(seed)
    unique_compounds = np.unique(compound_ids)
    n_compounds = len(unique_compounds)

    # Pre-compute per-compound image indices for speed
    compound_to_idx: Dict[int, np.ndarray] = {}
    for cid in unique_compounds:
        compound_to_idx[cid] = np.where(compound_ids == cid)[0]

    boot_accs: Dict[int, List[float]] = {k: [] for k in topk}

    for _ in range(n_bootstraps):
        # Resample compounds with replacement
        sampled = rng.choice(unique_compounds, size=n_compounds, replace=True)

        # Gather indices; duplicates of the same compound share one label
        unique_sampled = np.unique(sampled)
        label_map = {cid: i for i, cid in enumerate(unique_sampled)}

        all_idx, new_labels = [], []
        for cid in sampled:
            idx = compound_to_idx[cid]
            all_idx.append(idx)
            new_labels.extend([label_map[cid]] * len(idx))

        all_idx = np.concatenate(all_idx)
        boot_embs = embeddings[all_idx]
        boot_labels = torch.tensor(new_labels, dtype=torch.long,
                                   device=embeddings.device)

        for k in topk:
            boot_accs[k].append(topk_knn_accuracy(boot_embs, boot_labels, k))

    alpha = 1.0 - ci_level
    results = {}
    for k in topk:
        vals = np.array(boot_accs[k])
        results[k] = (
            float(vals.mean()),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))),
        )
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-k KNN accuracy test on randomly selected compounds")

    p.add_argument("--metadata", required=True, help="JSON metadata file")
    p.add_argument("--embedding", default=None, help="Pre-computed embedding .pt file")
    p.add_argument("--root_dir", default=None, help="Image root directory")

    p.add_argument("--checkpoint", default=None, help="Model checkpoint")
    p.add_argument("--model", default="tilted", choices=["vae", "tilted", "dino"])
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--tau", type=float, default=None)

    p.add_argument("--n_compounds", type=int, default=50,
                   help="Number of compounds to randomly select")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10],
                   help="Values of k for top-k KNN accuracy")
    p.add_argument("--n_trials", type=int, default=1,
                   help="Number of random selections to average over")
    p.add_argument("--subtract_control", action="store_true",
                   help="Subtract the plate-level mean control embedding from "
                        "each treated image embedding before computing KNN")
    p.add_argument("--normalize_before_subtract", action="store_true",
                   help="L2-normalize treated and control embeddings before "
                        "subtracting control (only used with --subtract_control)")

    p.add_argument("--n_bootstraps", type=int, default=1000,
                   help="Number of bootstrap resamples for CI estimation")
    p.add_argument("--ci_level", type=float, default=0.95,
                   help="Confidence level for bootstrap CI (default: 0.95)")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    args = p.parse_args()

    if args.embedding is None:
        if args.model != "dino" and args.checkpoint is None:
            p.error("--checkpoint is required when --embedding is not provided and --model is not dino")
        if args.root_dir is None:
            p.error("--root_dir is required when --embedding is not provided")

    if args.model == "dino":
        args.img_size = DinoV2Wrapper.IMG_SIZE

    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = load_metadata(args.metadata)

    # Load model if encoding on-the-fly
    model = None
    if args.embedding is None:
        model = build_model(args)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint)
        model.to(device).eval()

    # When using pre-computed embeddings, filter metadata to only compounds
    # present in the embedding file (typically the validation set) so that
    # random selection draws from available compounds only.
    if args.embedding is not None:
        emb_data = torch.load(args.embedding, map_location="cpu", weights_only=False)
        emb_compounds = set(emb_data.keys())
        metadata = [e for e in metadata if str(e["Compound"]) in emb_compounds]
        print(f"Embedding file: {len(emb_compounds)} compounds in total")
        print(f"Metadata: {len(metadata)} compounds (filtered to embedding file)")
    else:
        print(f"Metadata: {len(metadata)} compounds")

    print(f"Selecting {args.n_compounds} compounds, top-k = {args.topk}, "
          f"trials = {args.n_trials}\n")

    # Run trials with different random seeds
    results_per_k: Dict[int, List[float]] = {k: [] for k in args.topk}
    ci_per_k: Dict[int, List[Tuple[float, float, float]]] = {k: [] for k in args.topk}

    for trial in tqdm(range(args.n_trials), desc="Trials", disable=args.n_trials == 1):
        trial_seed = args.seed + trial
        selected = select_compounds(metadata, args.n_compounds, trial_seed)

        if args.embedding is not None:
            embeddings, labels = gather_embeddings_from_file(
                args.embedding, selected,
                subtract_control=args.subtract_control,
                normalize_before_subtract=args.normalize_before_subtract,
            )
        else:
            embeddings, labels = gather_embeddings_from_model(selected, args, model, device)

        if embeddings.size(0) < 2:
            print(f"Trial {trial + 1}: not enough images, skipping")
            continue

        n_unique = labels.unique().size(0)

        for k in args.topk:
            acc = topk_knn_accuracy(embeddings, labels, k)
            results_per_k[k].append(acc)

        # Bootstrap CI by resampling compounds
        compound_ids = labels.numpy()
        ci_results = bootstrap_ci_by_compound(
            embeddings, labels, compound_ids, args.topk,
            n_bootstraps=args.n_bootstraps, ci_level=args.ci_level,
            seed=trial_seed,
        )
        for k in args.topk:
            ci_per_k[k].append(ci_results[k])

    # Summary
    pct = int(args.ci_level * 100)
    print("\n" + "=" * 60)
    print(f"Summary (mean +/- std across {args.n_trials} trial(s)):")
    for k in args.topk:
        vals = results_per_k[k]
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            # Average CI bounds across trials
            ci_los = [ci[1] for ci in ci_per_k[k]]
            ci_his = [ci[2] for ci in ci_per_k[k]]
            ci_lo = np.mean(ci_los)
            ci_hi = np.mean(ci_his)
            print(f"  top-{k}: {mean:.4f} +/- {std:.4f}  "
                  f"{pct}% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f"  top-{k}: no valid trials")


if __name__ == "__main__":
    main()
