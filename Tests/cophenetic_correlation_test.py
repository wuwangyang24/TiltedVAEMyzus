"""Cophenetic correlation test: do learned embeddings preserve the taxonomy?

We treat the iNaturalist Linnaean taxonomy
(kingdom > phylum > class > order > family > genus > species) as a rooted tree
whose leaves are species. The *taxonomic cophenetic distance* between two leaves
is the number of ranks between the leaves and their lowest common ancestor
(0 = same species, 1 = same genus/different species, ..., 7 = different kingdom).
This is an ultrametric ground-truth distance derived purely from the labels.

For the embeddings we compute the pairwise distance matrix (cosine by default)
and quantify how faithfully it preserves the taxonomy via:

  1. Cophenetic correlation with the ground-truth taxonomy (headline metric):
     Spearman & Pearson correlation between the embedding pairwise distances and
     the taxonomic cophenetic distances (a Mantel-style comparison of two
     distance matrices). Higher = the embedding geometry mirrors the taxonomy.

  2. Dendrogram recovery: build an agglomerative dendrogram from the embedding
     distances (scipy linkage), take its cophenetic distances, and correlate them
     with the taxonomy. Also reports the classic cophenetic correlation
     coefficient (dendrogram vs. its own input distances) as a tree-ness
     diagnostic of the embedding space.

  3. Per-rank distance profile: mean/std embedding distance grouped by taxonomic
     distance level, which should increase monotonically if the hierarchy is
     preserved.

By default samples are aggregated to species centroids (the taxonomy's leaves);
pass ``--level sample`` for a per-image comparison.

Usage:
    python Tests/cophenetic_correlation_test.py \
        --checkpoints_root results/checkpoints \
        --val_metadata inat2021/val.json \
        --val_image_dir inat2021 \
        --backbone vit_small_patch16_224 \
        --superclass Birds --train_cat order \
        --test_cat family genus specific_epithet \
        --level species --metric cosine --linkage average \
        --img_size 224 --batch_size 128 --device cuda

The test globs every run folder matching the backbone / dataset / train_cat /
test_cat (e.g. ``FFT_ViTs16_..._SupConSoftPos-Tau4.0-Sinkhorn5_inat_order->
family-genus-specific_epithet_Birds``), one per Tau value, and reports the
cophenetic metrics for each so the Tau sweep can be compared side by side.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models import Backbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fully fine-tuned timm backbones supported by ``Models.Backbone``.
SUPPORTED_BACKBONES = [
    "resnet18", "resnet50", "vit_small_patch16_224",
    "swin_tiny_patch4_window7_224", "convnext_tiny",
]

# Backbone -> short tag used in the training-run folder name (see train.py).
BACKBONE_TAG = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "vit_small_patch16_224": "ViTs16",
    "swin_tiny_patch4_window7_224": "SwinT",
    "convnext_tiny": "ConvNeXtT",
}

# Linnaean ranks from root to leaf. iNat stores the species epithet under
# "specific_epithet"; together with the ancestor ranks it identifies a leaf.
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus",
         "specific_epithet"]


# ── Dataset ──────────────────────────────────────────────────────────────────

class InatValDataset(Dataset):
    """Returns (image, leaf_id) where leaf_id indexes the species (leaf)."""

    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, leaf_id = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        return self.transform(img), leaf_id


def parse_inat_taxonomy(
    metadata_path: str, image_dir: str, superclass: Optional[str] = None,
) -> Tuple[List[Tuple[str, Tuple[str, ...]]], Dict[Tuple[str, ...], int]]:
    """Parse iNat2021 metadata into (path, taxonomy_tuple) samples.

    The taxonomy tuple holds one string per entry in ``RANKS`` (root -> leaf).
    """
    with open(metadata_path) as f:
        data = json.load(f)

    cat_map: Dict[int, Dict[str, str]] = {c["id"]: c for c in data["categories"]}
    img_map: Dict[int, str] = {i["id"]: i["file_name"] for i in data["images"]}

    sc = superclass.lower() if superclass else None
    samples: List[Tuple[str, Tuple[str, ...]]] = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in img_map or cat_id not in cat_map:
            continue
        cat_info = cat_map[cat_id]
        if sc is not None and str(cat_info.get("supercategory", "")).lower() != sc:
            continue
        taxonomy = tuple(str(cat_info.get(rank, "")) for rank in RANKS)
        samples.append((os.path.join(image_dir, img_map[img_id]), taxonomy))

    leaf_ids: Dict[Tuple[str, ...], int] = {}
    for _, taxonomy in samples:
        leaf_ids.setdefault(taxonomy, len(leaf_ids))
    return samples, leaf_ids


# ── Model loading ────────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace, use_proj_head: bool) -> torch.nn.Module:
    return Backbone(
        backbone=args.backbone,
        img_size=args.img_size,
        embedding_dim=args.embedding_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        use_proj_head=use_proj_head,
        pretrained=False,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    backbone_missing = [k for k in incompatible.missing_keys if k.startswith("backbone.")]
    print(f"[load_checkpoint] loaded {len(cleaned)} tensors; "
          f"missing={len(incompatible.missing_keys)}, "
          f"unexpected={len(incompatible.unexpected_keys)}")
    if backbone_missing:
        print(f"[load_checkpoint] WARNING: {len(backbone_missing)} backbone weights "
              "were not loaded; check --backbone matches the checkpoint.")


def discover_runs(args: argparse.Namespace) -> Tuple[List[Tuple[str, str]], str]:
    """Glob every run folder matching the backbone / dataset / train_cat / test_cat.

    One folder is expected per Tau value; the middle of the name (batch size,
    temperature, loss config incl. Tau) is matched with a wildcard.
    """
    tag = BACKBONE_TAG.get(args.backbone, args.backbone)
    test_tag = "-".join(args.test_cat)
    pattern = f"FFT_{tag}_*inat_{args.train_cat}->{test_tag}_{args.superclass}"
    runs: List[Tuple[str, str]] = []
    for run_dir in sorted(glob.glob(os.path.join(args.checkpoints_root, pattern))):
        if not os.path.isdir(run_dir):
            continue
        ckpt = os.path.join(run_dir, args.ckpt_name)
        if os.path.exists(ckpt):
            runs.append((run_dir, ckpt))
    return runs, pattern


def parse_tau(run_name: str) -> str:
    """Extract the soft-positive Tau value from a run folder name."""
    m = re.search(r"LinearTau([\d.]+)to([\d.]+)", run_name)
    if m:
        return f"{m.group(1)}->{m.group(2)}"
    m = re.search(r"-Tau([\d.]+)", run_name) or re.search(r"Tau([\d.]+)", run_name)
    return m.group(1) if m else "NA"


def tau_sort_key(tau: str):
    try:
        return (0, float(tau))
    except ValueError:
        return (1, 0.0, tau)


# ── Encoding ─────────────────────────────────────────────────────────────────

def build_transform(args: argparse.Namespace) -> T.Compose:
    return T.Compose([
        T.Resize((args.img_size, args.img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def encode_dataset(
    model: torch.nn.Module, dataset: Dataset,
    batch_size: int, device: torch.device, num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    embs: List[torch.Tensor] = []
    leaf_ids: List[torch.Tensor] = []
    for imgs, ids in tqdm(loader, desc="Encoding"):
        z = model.encode(imgs.to(device), normalize=True)
        embs.append(z.cpu())
        leaf_ids.append(ids)
    return torch.cat(embs, dim=0), torch.cat(leaf_ids, dim=0)


# ── Taxonomic (ground-truth) cophenetic distances ────────────────────────────

def taxonomic_cophenetic_matrix(tax_tuples: List[Tuple[str, ...]]) -> np.ndarray:
    """Ultrametric distance from the taxonomy tree.

    distance(i, j) = R - depth(LCA(i, j)), where the depth is the number of
    leading ranks (root -> leaf) whose full ancestral path matches. Paths are
    compared cumulatively so identical rank labels under different ancestors do
    not spuriously merge.
    """
    n_ranks = len(RANKS)
    m = len(tax_tuples)
    lca_depth = np.zeros((m, m), dtype=np.int32)
    for r in range(n_ranks):
        path_to_code: Dict[Tuple[str, ...], int] = {}
        codes = np.empty(m, dtype=np.int64)
        for i, tax in enumerate(tax_tuples):
            prefix = tax[:r + 1]
            codes[i] = path_to_code.setdefault(prefix, len(path_to_code))
        lca_depth += (codes[:, None] == codes[None, :]).astype(np.int32)
    coph = (n_ranks - lca_depth).astype(np.float64)
    np.fill_diagonal(coph, 0.0)
    return coph


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(
    embeddings: torch.Tensor, leaf_ids: torch.Tensor,
    leaf_taxonomy: Dict[int, Tuple[str, ...]], level: str,
) -> Tuple[np.ndarray, List[Tuple[str, ...]]]:
    """Return (X, tax_tuples): points to compare and their taxonomy tuples."""
    if level == "sample":
        tax = [leaf_taxonomy[int(i)] for i in leaf_ids.tolist()]
        return embeddings.numpy(), tax

    # Species centroids: mean embedding per leaf.
    centroids: List[torch.Tensor] = []
    tax = []
    for lid in sorted(set(leaf_ids.tolist())):
        mask = leaf_ids == lid
        centroids.append(embeddings[mask].mean(dim=0))
        tax.append(leaf_taxonomy[int(lid)])
    return torch.stack(centroids, dim=0).numpy(), tax


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cophenetic correlation between embeddings and the taxonomy")

    p.add_argument("--checkpoints_root", default="results/checkpoints",
                   help="Root directory holding the per-run checkpoint folders")
    p.add_argument("--ckpt_name", default="best-val-loss.ckpt",
                   help="Checkpoint file to load inside each run folder")
    p.add_argument("--val_metadata", required=True, help="iNat2021 val metadata JSON")
    p.add_argument("--val_image_dir", required=True, help="Image directory for val set")
    p.add_argument("--superclass", required=True,
                   help="Dataset / supercategory name, e.g. Birds, Insects "
                        "(also the trailing run-name token)")
    p.add_argument("--train_cat", required=True,
                   help="Taxonomy level used for training labels (run-name token)")
    p.add_argument("--test_cat", nargs="+", required=True,
                   help="Taxonomy level(s) used for evaluation (dash-joined run token)")

    # What to compare and how
    p.add_argument("--level", choices=["species", "sample"], default="species",
                   help="Aggregate to species centroids (default) or use raw images")
    p.add_argument("--metric", default="cosine",
                   choices=["cosine", "euclidean", "correlation"],
                   help="Pairwise embedding distance metric")
    p.add_argument("--linkage", default="average",
                   choices=["average", "complete", "single", "ward"],
                   help="Agglomerative linkage for the embedding dendrogram")
    p.add_argument("--max_species", type=int, default=None,
                   help="Cap the number of distinct species/leaves (random subset)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the number of images encoded (random subset)")

    # Fine-tuned backbone / architecture (must match the checkpoints)
    p.add_argument("--backbone", default="vit_small_patch16_224",
                   choices=SUPPORTED_BACKBONES,
                   help="timm backbone fine-tuned by Models.Backbone")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--proj_hidden_dim", type=int, default=2048)

    # Runtime / output
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None,
                   help="If set, save summary.json, per_rank.csv and a scatter plot")

    return p.parse_args()


def select_samples(
    samples: List[Tuple[str, Tuple[str, ...]]],
    leaf_ids: Dict[Tuple[str, ...], int],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, int]], Dict[int, Tuple[str, ...]]]:
    rng = np.random.RandomState(args.seed)

    keep_taxa = set(leaf_ids)
    if args.max_species is not None and args.max_species < len(leaf_ids):
        all_taxa = sorted(leaf_ids)
        idx = rng.choice(len(all_taxa), size=args.max_species, replace=False)
        keep_taxa = {all_taxa[i] for i in idx}

    filtered = [(path, tax) for path, tax in samples if tax in keep_taxa]
    rng.shuffle(filtered)
    if args.max_samples is not None and args.max_samples < len(filtered):
        filtered = filtered[:args.max_samples]

    # Re-index leaves contiguously over the surviving samples.
    leaf_index: Dict[Tuple[str, ...], int] = {}
    indexed: List[Tuple[str, int]] = []
    for path, tax in filtered:
        lid = leaf_index.setdefault(tax, len(leaf_index))
        indexed.append((path, lid))
    leaf_taxonomy = {lid: tax for tax, lid in leaf_index.items()}
    return indexed, leaf_taxonomy


def per_rank_profile(
    emb_condensed: np.ndarray, tax_condensed: np.ndarray,
) -> List[Tuple[int, int, float, float]]:
    """Group embedding distances by integer taxonomic distance level."""
    levels = np.rint(tax_condensed).astype(int)
    profile: List[Tuple[int, int, float, float]] = []
    for lvl in range(len(RANKS) + 1):
        vals = emb_condensed[levels == lvl]
        if vals.size == 0:
            continue
        profile.append((lvl, int(vals.size), float(vals.mean()), float(vals.std())))
    return profile


def maybe_save(
    out_dir: str, metric: str, summary: dict,
    profile: List[Tuple[int, int, float, float]],
    tax_condensed: np.ndarray, emb_condensed: np.ndarray,
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "per_rank.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taxonomic_distance", "num_pairs",
                         "mean_embedding_distance", "std_embedding_distance"])
        writer.writerows(profile)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        levels = np.rint(tax_condensed).astype(int)
        data = [emb_condensed[levels == lvl] for lvl, *_ in profile]
        ax.boxplot(data, labels=[str(lvl) for lvl, *_ in profile], showfliers=False)
        ax.set_xlabel("Taxonomic cophenetic distance")
        ax.set_ylabel(f"Embedding distance ({metric})")
        ax.set_title(f"Spearman={summary['spearman_r']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "distance_by_rank.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[plot] skipped ({exc})")


def cophenetic_metrics(
    X: np.ndarray, tax_tuples: List[Tuple[str, ...]], metric: str, linkage_method: str,
) -> Tuple[dict, list, np.ndarray, np.ndarray]:
    """Compute the cophenetic metrics for one embedding set against the taxonomy."""
    tax_condensed = squareform(taxonomic_cophenetic_matrix(tax_tuples), checks=False)
    emb_condensed = pdist(X, metric=metric)

    spearman_r, spearman_p = spearmanr(emb_condensed, tax_condensed)
    pearson_r, pearson_p = pearsonr(emb_condensed, tax_condensed)

    Z = linkage(emb_condensed, method=linkage_method)
    cpcc, coph_dists = cophenet(Z, emb_condensed)
    dendro_tax_r, _ = spearmanr(coph_dists, tax_condensed)

    profile = per_rank_profile(emb_condensed, tax_condensed)
    metrics = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "classic_cpcc": float(cpcc),
        "dendrogram_taxonomy_spearman": float(dendro_tax_r),
    }
    return metrics, profile, tax_condensed, emb_condensed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    runs, pattern = discover_runs(args)
    if not runs:
        raise SystemExit(
            f"No checkpoints found. Looked for '{args.ckpt_name}' in folders matching:\n"
            f"  {os.path.join(args.checkpoints_root, pattern)}")
    print(f"Found {len(runs)} run(s) matching {pattern}\n")

    # Parse + subsample the evaluation set ONCE; it is shared across all runs.
    samples, leaf_ids = parse_inat_taxonomy(
        args.val_metadata, args.val_image_dir, args.superclass)
    if not samples:
        raise SystemExit("No samples parsed; check --val_metadata / --superclass.")
    indexed, leaf_taxonomy = select_samples(samples, leaf_ids, args)
    print(f"Encoding {len(indexed)} images over {len(leaf_taxonomy)} species "
          f"(superclass={args.superclass})\n")

    transform = build_transform(args)
    dataset = InatValDataset(indexed, transform)

    rows: List[dict] = []
    for run_dir, ckpt_path in runs:
        run_name = os.path.basename(run_dir)
        tau = parse_tau(run_name)
        use_proj_head = "NoProj" not in run_name
        print("=" * 70)
        print(f"RUN  Tau={tau}  proj_head={use_proj_head}")
        print(f"     {run_name}")

        model = build_model(args, use_proj_head)
        load_checkpoint(model, ckpt_path)
        model = model.to(device).eval()

        embeddings, sample_leaf_ids = encode_dataset(
            model, dataset, args.batch_size, device, args.num_workers)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        X, tax_tuples = aggregate(embeddings, sample_leaf_ids, leaf_taxonomy, args.level)
        if X.shape[0] < 3:
            print(f"     Skipped: need >=3 points to correlate, got {X.shape[0]}.")
            continue

        metrics, profile, tax_condensed, emb_condensed = cophenetic_metrics(
            X, tax_tuples, args.metric, args.linkage)
        print(f"     Spearman={metrics['spearman_r']:.4f}  "
              f"Pearson={metrics['pearson_r']:.4f}  "
              f"CPCC={metrics['classic_cpcc']:.4f}  "
              f"dendro-tax={metrics['dendrogram_taxonomy_spearman']:.4f}")

        row = {
            "run_name": run_name,
            "tau": tau,
            "use_proj_head": use_proj_head,
            "num_points": int(X.shape[0]),
            "num_images": len(indexed),
            "num_species": len(leaf_taxonomy),
            **metrics,
        }
        rows.append(row)

        if args.output_dir:
            maybe_save(os.path.join(args.output_dir, run_name), args.metric,
                       row, profile, tax_condensed, emb_condensed)

    if not rows:
        raise SystemExit("No run produced enough points to correlate.")

    rows.sort(key=lambda r: tau_sort_key(r["tau"]))

    # ── Comparison report across Tau values ───────────────────────────────────
    print("\n" + "=" * 78)
    print(f"COPHENETIC CORRELATION vs TAXONOMY  "
          f"({args.backbone}, {args.superclass}, {args.train_cat}->"
          f"{'-'.join(args.test_cat)}, {args.level}-level, {args.metric})")
    print("-" * 78)
    header = (f"{'Tau':>10}  {'Spearman':>9}  {'Pearson':>9}  "
              f"{'CPCC':>7}  {'DendroTax':>9}  {'proj':>5}")
    print(header)
    print("-" * 78)
    for r in rows:
        print(f"{r['tau']:>10}  {r['spearman_r']:>9.4f}  {r['pearson_r']:>9.4f}  "
              f"{r['classic_cpcc']:>7.4f}  {r['dendrogram_taxonomy_spearman']:>9.4f}  "
              f"{str(r['use_proj_head']):>5}")
    best = max(rows, key=lambda r: r["spearman_r"])
    print("-" * 78)
    print(f"Best Spearman: Tau={best['tau']} ({best['spearman_r']:.4f})")
    print("=" * 78)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "tau_report.json"), "w") as f:
            json.dump({
                "backbone": args.backbone,
                "superclass": args.superclass,
                "train_cat": args.train_cat,
                "test_cat": args.test_cat,
                "level": args.level,
                "metric": args.metric,
                "linkage": args.linkage,
                "runs": rows,
            }, f, indent=2)
        with open(os.path.join(args.output_dir, "tau_report.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
