"""Taxonomic benchmark on the iNat test set for a set of frozen backbones.

For every (superclass x test_cat x backbone) combination this encodes the iNat
test images once with the *frozen* backbone and reports six metrics:

  * kNN-1  : top-1 accuracy of a 1-nearest-neighbour classifier (cosine,
             leave-one-out over the full set) at the ``test_cat`` level.
  * kNN-5  : top-1 hit within the 5 nearest neighbours.
  * LinProbe: linear-probe top-1 accuracy (LBFGS logistic regression on the
             frozen embeddings, train/test split of the encoded set).
  * Spearman / Pearson : rank / linear correlation between the pairwise
             embedding distances (``test_cat`` centroids) and the taxonomic
             cophenetic distances (Mantel-style comparison of two distance
             matrices).
  * Dendrogram: Spearman correlation between the cophenetic distances of an
             agglomerative dendrogram built on the embeddings and the taxonomy.

The results are printed and (optionally) written as one LaTeX table per metric,
laid out like the paper template: rows grouped by ``test_cat`` (multirow),
methods as sub-rows, superclasses as columns; the largest value across methods
per (test_cat, superclass) cell is bolded.

Everything is driven by a JSON config (``--config``). Only frozen
ImageNet-pretrained backbones are supported: ``resnet50``,
``vit_small_patch16_224`` (ViT-S/16) and ``swin_tiny_patch4_window7_224``
(Swin-T).

    {
      "test_metadata": "inat2021/val.json",
      "test_image_dir": "inat2021",
      "train_metadata": null,          // optional; if null the test set is split
      "train_image_dir": null,
      "superclasses": ["Mollusks", "Mammals", "Fishes", "Insects", "Birds",
                        "Reptiles", "Amphibians", "Fungi", "Plants"],
      "test_cats": [
        {"rank": "order",            "label": "O"},
        {"rank": "family",           "label": "F"},
        {"rank": "genus",            "label": "G"},
        {"rank": "specific_epithet", "label": "S"}
      ],
      "backbones": [
        {
          "name": "ResNet50",
          "backbone": "resnet50",
          "img_size": 224, "use_proj_head": false
        },
        {
          "name": "ViT-S/16",
          "backbone": "vit_small_patch16_224",
          "img_size": 224, "use_proj_head": false
        },
        {
          "name": "Swin-T",
          "backbone": "swin_tiny_patch4_window7_224",
          "img_size": 224, "use_proj_head": false
        }
      ]
    }

Usage:
    python Tests/inat_taxonomic_benchmark_test.py \
        --config Tests/benchmark_config.json \
        --output_dir results/benchmark_tables \
        --batch_size 128 --device cuda
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

# Frozen ImageNet-pretrained backbones supported by ``Models.Backbone``.
SUPPORTED_BACKBONES = (
    "resnet50", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
)

# Linnaean ranks from root to leaf (iNat stores the species epithet field name
# as "specific_epithet"). A rank's *class* is the cumulative path to that rank,
# so identical names under different ancestors never collapse together.
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus",
         "specific_epithet"]

# Metric key -> (table label, human caption fragment). Order defines the tables.
METRICS: List[Tuple[str, str, str]] = [
    ("knn1", "kNN-1", "1-nearest-neighbour top-1 accuracy"),
    ("knn5", "kNN-5", "5-nearest-neighbour top-1 accuracy"),
    ("linprobe", "LinProbe", "linear-probe top-1 accuracy"),
    ("spearman", "Spearman", "Spearman correlation between embedding and taxonomic distances"),
    ("pearson", "Pearson", "Pearson correlation between embedding and taxonomic distances"),
    ("dendrogram", "Dendro", "dendrogram cophenetic correlation with the taxonomy"),
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class InatDataset(Dataset):
    """Returns (image, sample_index) so labels/taxonomy stay outside the loader."""

    def __init__(self, paths: List[str], transform: T.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = read_image(self.paths[index], mode=ImageReadMode.RGB)
        return self.transform(img), index


def parse_inat_taxonomy(
    metadata_path: str, image_dir: str, superclass: Optional[str],
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Parse iNat metadata into (paths, taxonomy_tuples) for one superclass."""
    with open(metadata_path) as f:
        data = json.load(f)

    cat_map: Dict[int, Dict[str, str]] = {c["id"]: c for c in data["categories"]}
    img_map: Dict[int, str] = {i["id"]: i["file_name"] for i in data["images"]}

    sc = superclass.lower() if superclass else None
    paths: List[str] = []
    taxa: List[Tuple[str, ...]] = []
    for ann in data["annotations"]:
        img_id, cat_id = ann["image_id"], ann["category_id"]
        if img_id not in img_map or cat_id not in cat_map:
            continue
        cat_info = cat_map[cat_id]
        if sc is not None and str(cat_info.get("supercategory", "")).lower() != sc:
            continue
        paths.append(os.path.join(image_dir, img_map[img_id]))
        taxa.append(tuple(str(cat_info.get(rank, "")) for rank in RANKS))
    return paths, taxa


# ── Frozen encoders ──────────────────────────────────────────────────────────

def _clean_state_dict(state_dict: dict) -> dict:
    return {(k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()}


def _resize_dcl_buffers(model: torch.nn.Module, cleaned: dict) -> None:
    """Match DCL memory-bank buffer shapes to the checkpoint before loading."""
    for key in ("dcl_sigreg_loss.class_means", "dcl_sigreg_loss.initialized"):
        if key not in cleaned:
            continue
        parts = key.split(".")
        parent = model
        ok = True
        for attr in parts[:-1]:
            parent = getattr(parent, attr, None)
            if parent is None:
                ok = False
                break
        if ok and getattr(parent, parts[-1]).shape != cleaned[key].shape:
            parent.register_buffer(parts[-1], torch.empty_like(cleaned[key]))


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = _clean_state_dict(sd)
    _resize_dcl_buffers(model, cleaned)
    model.load_state_dict(cleaned, strict=False)


def build_encoder(cfg: dict) -> torch.nn.Module:
    """Instantiate a frozen ImageNet-pretrained backbone from a config dict."""
    backbone = cfg.get("backbone")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from {list(SUPPORTED_BACKBONES)}.")

    model = Backbone(
        backbone=backbone,
        img_size=cfg.get("img_size", 224),
        embedding_dim=cfg.get("embedding_dim", 256),
        proj_hidden_dim=cfg.get("proj_hidden_dim", 2048),
        use_proj_head=cfg.get("use_proj_head", False),
        pretrained=cfg.get("pretrained", True),
    )
    if cfg.get("checkpoint"):
        _load_checkpoint(model, cfg["checkpoint"])
    return model


def build_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def encode_paths(
    model: torch.nn.Module, paths: List[str], transform: T.Compose,
    batch_size: int, device: torch.device, num_workers: int,
) -> torch.Tensor:
    dataset = InatDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    collected: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    for imgs, idx in tqdm(loader, desc="  encoding", leave=False):
        collected.append(model.encode(imgs.to(device), normalize=True).cpu())
        indices.append(idx)
    if not collected:
        return torch.empty(0)
    z = torch.cat(collected, dim=0)
    order = torch.cat(indices, dim=0)
    # Restore original sample order (loader is not shuffled, but be robust).
    result = torch.empty_like(z)
    result[order] = z
    return result


# ── Taxonomy / metric helpers ────────────────────────────────────────────────

def taxonomic_cophenetic_matrix(tax_tuples: List[Tuple[str, ...]]) -> np.ndarray:
    """Ultrametric taxonomic distance: n_ranks - depth(LCA)."""
    n_ranks = len(tax_tuples[0])
    m = len(tax_tuples)
    lca_depth = np.zeros((m, m), dtype=np.int32)
    for r in range(n_ranks):
        path_to_code: Dict[Tuple[str, ...], int] = {}
        codes = np.empty(m, dtype=np.int64)
        for i, tax in enumerate(tax_tuples):
            codes[i] = path_to_code.setdefault(tax[:r + 1], len(path_to_code))
        lca_depth += (codes[:, None] == codes[None, :]).astype(np.int32)
    coph = (n_ranks - lca_depth).astype(np.float64)
    np.fill_diagonal(coph, 0.0)
    return coph


def labels_at_rank(taxa: List[Tuple[str, ...]], rank_idx: int) -> torch.Tensor:
    """Integer labels using the cumulative path up to ``rank_idx`` as the class."""
    class_ids: Dict[Tuple[str, ...], int] = {}
    labels = [class_ids.setdefault(t[:rank_idx + 1], len(class_ids)) for t in taxa]
    return torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def knn_accuracy(
    embeddings: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...],
    chunk_size: int = 1024,
) -> Dict[int, float]:
    """Top-k kNN accuracy over the full set (cosine, leave-one-out)."""
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.view(-1)
    n = embeddings.size(0)
    max_k = min(max(ks), n - 1)
    if max_k < 1:
        return {k: float("nan") for k in ks}
    hits = {k: torch.zeros(n, dtype=torch.bool) for k in ks}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = embeddings[start:end] @ embeddings.t()
        rows = torch.arange(end - start)
        sim[rows, torch.arange(start, end)] = float("-inf")
        topk_idx = sim.topk(max_k, dim=1).indices
        match = labels[topk_idx] == labels[start:end].unsqueeze(1)
        for k in ks:
            hits[k][start:end] = match[:, :min(k, max_k)].any(dim=1)
    return {k: float(hits[k].float().mean().item()) for k in ks}


def linear_probe_top1(
    embeddings: torch.Tensor, labels: torch.Tensor, device: torch.device,
    train_fraction: float, seed: int, lr: float = 0.1, epochs: int = 100,
) -> float:
    """LBFGS logistic-regression probe; returns top-1 accuracy on the held-out split."""
    num_classes = int(labels.max().item()) + 1
    if num_classes < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(embeddings.size(0))
    split = int(embeddings.size(0) * train_fraction)
    tr, te = perm[:split], perm[split:]
    if len(tr) < num_classes or len(te) == 0:
        return float("nan")

    x_tr = embeddings[tr].to(device)
    y_tr = labels[tr].to(device)
    x_te = embeddings[te].to(device)
    y_te = labels[te].to(device)

    clf = torch.nn.Linear(embeddings.size(1), num_classes).to(device)
    opt = torch.optim.LBFGS(clf.parameters(), lr=lr, max_iter=20)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(x_tr), y_tr)
        loss.backward()
        return loss.detach()

    for _ in range(epochs):
        opt.step(closure)

    clf.eval()
    with torch.no_grad():
        top1 = (clf(x_te).argmax(dim=1) == y_te).float().mean().item()
    return float(top1)


def correlation_metrics(
    embeddings: torch.Tensor, taxa: List[Tuple[str, ...]], rank_idx: int,
    metric: str = "cosine", linkage_method: str = "average",
) -> Dict[str, float]:
    """Aggregate to rank centroids and correlate with the taxonomy."""
    class_ids: Dict[Tuple[str, ...], int] = {}
    for t in taxa:
        class_ids.setdefault(t[:rank_idx + 1], len(class_ids))
    if len(class_ids) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "dendrogram": float("nan")}

    sums = torch.zeros(len(class_ids), embeddings.size(1))
    counts = torch.zeros(len(class_ids))
    for emb, t in zip(embeddings, taxa):
        cid = class_ids[t[:rank_idx + 1]]
        sums[cid] += emb
        counts[cid] += 1
    centroids = (sums / counts.unsqueeze(1)).numpy()
    tax_tuples = [key for key, _ in sorted(class_ids.items(), key=lambda kv: kv[1])]

    tax_condensed = squareform(taxonomic_cophenetic_matrix(tax_tuples), checks=False)
    emb_condensed = pdist(centroids, metric=metric)

    spearman_r, _ = spearmanr(emb_condensed, tax_condensed)
    pearson_r, _ = pearsonr(emb_condensed, tax_condensed)
    Z = linkage(emb_condensed, method=linkage_method)
    _, coph_dists = cophenet(Z, emb_condensed)
    dendro_r, _ = spearmanr(coph_dists, tax_condensed)
    return {"spearman": float(spearman_r), "pearson": float(pearson_r),
            "dendrogram": float(dendro_r)}


# ── LaTeX table ──────────────────────────────────────────────────────────────

def format_value(value: Optional[float], is_best: bool, scale: float,
                 decimals: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    txt = f"{value * scale:.{decimals}f}"
    return f"\\textbf{{{txt}}}" if is_best else txt


def build_latex_table(
    metric_key: str, table_label: str, caption_fragment: str,
    results: dict, test_cats: List[dict], methods: List[str],
    superclasses: List[str], scale: float, decimals: int,
) -> str:
    col_spec = "ll" + "c" * len(superclasses)
    lines = [
        r"\begin{table*}[ht]\centering\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Metric & Method & " + " & ".join(superclasses) + r" \\",
        r"\midrule",
    ]

    for gi, tc in enumerate(test_cats):
        label = tc["label"]
        # Best (max) value per superclass column within this test_cat group.
        best_per_col: List[Optional[float]] = []
        for sc in superclasses:
            vals = [results[metric_key][label][m].get(sc) for m in methods]
            vals = [v for v in vals if v is not None and not
                    (isinstance(v, float) and np.isnan(v))]
            best_per_col.append(max(vals) if vals else None)

        lines.append(rf"\multirow{{{len(methods)}}}{{*}}{{{label}}}")
        for method in methods:
            cells = []
            for ci, sc in enumerate(superclasses):
                v = results[metric_key][label][method].get(sc)
                is_best = (best_per_col[ci] is not None and v is not None
                           and not (isinstance(v, float) and np.isnan(v))
                           and abs(v - best_per_col[ci]) < 1e-12)
                cells.append(format_value(v, is_best, scale, decimals))
            method_cell = method.replace("_", r"\_")
            lines.append(f"& {method_cell:<14} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule" if gi == len(test_cats) - 1 else r"\midrule")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{caption_fragment}. Values are multiplied by {int(scale)} and "
        rf"rounded to {decimals} decimals. Bold indicates the largest value across "
        r"methods for each metric row-group and taxonomic group.}",
        rf"\label{{tab:inat_{metric_key}}}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


# ── CLI / main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Taxonomic benchmark (kNN / linear-probe / cophenetic) on iNat")
    p.add_argument("--config", required=True, help="JSON benchmark config (see docstring)")
    p.add_argument("--output_dir", default=None,
                   help="If set, write one <metric>.tex table and results.json here")
    p.add_argument("--train_fraction", type=float, default=0.8,
                   help="Fraction of the encoded set used to train the linear probe")
    p.add_argument("--metric", default="cosine",
                   choices=["cosine", "euclidean", "correlation"],
                   help="Pairwise embedding distance for the correlation metrics")
    p.add_argument("--linkage", default="average",
                   choices=["average", "complete", "single", "ward"])
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on images per (backbone, superclass)")
    p.add_argument("--probe_lr", type=float, default=0.1)
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = json.load(f)

    superclasses: List[str] = cfg["superclasses"]
    test_cats: List[dict] = cfg["test_cats"]
    backbones: List[dict] = cfg["backbones"]
    methods = [b["name"] for b in backbones]
    for tc in test_cats:
        if tc["rank"] not in RANKS:
            raise SystemExit(f"test_cat rank '{tc['rank']}' not in {RANKS}")

    # results[metric][test_cat_label][method][superclass] = value
    results: dict = {
        mk: {tc["label"]: {m: {} for m in methods} for tc in test_cats}
        for mk, _, _ in METRICS
    }

    for backbone_cfg in backbones:
        name = backbone_cfg["name"]
        img_size = backbone_cfg.get("img_size", 224)
        transform = build_transform(img_size)
        print("\n" + "=" * 74)
        print(f"BACKBONE: {name}  (backbone={backbone_cfg.get('backbone')}, "
              f"img_size={img_size})")

        model = build_encoder(backbone_cfg).to(device).eval()

        for sc in superclasses:
            paths, taxa = parse_inat_taxonomy(
                cfg["test_metadata"], cfg["test_image_dir"], sc)
            if not paths:
                print(f"  [{sc}] no images, skipping")
                continue
            if args.max_samples and len(paths) > args.max_samples:
                rng = np.random.RandomState(args.seed)
                keep = rng.choice(len(paths), size=args.max_samples, replace=False)
                paths = [paths[i] for i in keep]
                taxa = [taxa[i] for i in keep]

            print(f"  [{sc}] encoding {len(paths)} images...")
            embeddings = encode_paths(model, paths, transform, args.batch_size,
                                      device, args.num_workers)

            for tc in test_cats:
                rank_idx = RANKS.index(tc["rank"])
                labels = labels_at_rank(taxa, rank_idx)

                knn = knn_accuracy(embeddings, labels, ks=(1, 5))
                results["knn1"][tc["label"]][name][sc] = knn[1]
                results["knn5"][tc["label"]][name][sc] = knn[5]

                results["linprobe"][tc["label"]][name][sc] = linear_probe_top1(
                    embeddings, labels, device, args.train_fraction, args.seed,
                    lr=args.probe_lr, epochs=args.probe_epochs)

                corr = correlation_metrics(embeddings, taxa, rank_idx,
                                           args.metric, args.linkage)
                results["spearman"][tc["label"]][name][sc] = corr["spearman"]
                results["pearson"][tc["label"]][name][sc] = corr["pearson"]
                results["dendrogram"][tc["label"]][name][sc] = corr["dendrogram"]

                print(f"    {tc['label']} ({tc['rank']}): "
                      f"kNN1={knn[1]:.3f} kNN5={knn[5]:.3f} "
                      f"LinProbe={results['linprobe'][tc['label']][name][sc]:.3f} "
                      f"Spearman={corr['spearman']:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Emit LaTeX tables ─────────────────────────────────────────────────────
    tables: Dict[str, str] = {}
    for metric_key, table_label, caption in METRICS:
        table = build_latex_table(
            metric_key, table_label, caption, results, test_cats, methods,
            superclasses, scale=100.0, decimals=2)
        tables[metric_key] = table
        print("\n" + "#" * 74)
        print(f"# {table_label} table")
        print("#" * 74)
        print(table)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for metric_key, table in tables.items():
            with open(os.path.join(args.output_dir, f"{metric_key}.tex"), "w") as f:
                f.write(table + "\n")
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(tables)} tables + results.json to {args.output_dir}")


if __name__ == "__main__":
    main()
