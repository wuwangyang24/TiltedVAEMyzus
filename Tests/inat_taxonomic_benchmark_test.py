"""Taxonomic benchmark on the iNat (or FGVC-Aircraft) test set for frozen backbones.

For every (superclass x test_cat x backbone) combination this encodes the
test images once with the *frozen* backbone and reports six metrics:

  * kNN-1  : top-1 accuracy of a 1-nearest-neighbour classifier (cosine,
             leave-one-out over the full set) at the ``test_cat`` level.
  * kNN-5  : top-1 hit within the 5 nearest neighbours.
  * LinProbe: linear-probe top-1 accuracy (LBFGS logistic regression on the
             frozen embeddings, train/test split of the encoded set).
  * Spearman / Pearson / CPCC / Dendrogram : Mantel-style comparison of the
             pairwise embedding distances against the taxonomic cophenetic
             distances, computed once per superclass. This mirrors the
             validation-epoch metric in ``contrastive_experiment``: sample-level
             (no centroids), capped at 2048 randomly drawn images (seed 42),
             cosine distances, average linkage, and a hierarchy built only from
             the ``test_cats`` ranks.

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

FGVC-Aircraft is supported with ``"dataset": "aircraft"``; its hierarchy is
``manufacturer`` -> ``family`` -> ``variant`` and ``superclasses`` (optional,
defaults to ``["Aircraft"]``) are manufacturer names used to split the columns:

    {
      "dataset": "aircraft",
      "aircraft_root": "data/fgvc",
      "aircraft_split": "test",
      "aircraft_download": false,
      "superclasses": ["Aircraft"],
      "test_cats": [
        {"rank": "manufacturer", "label": "M"},
        {"rank": "family",       "label": "F"},
        {"rank": "variant",      "label": "V"}
      ],
      "backbones": [...]
    }

Usage:
    python Tests/inat_taxonomic_benchmark_test.py \\
        --config Tests/benchmark_config.json \\
        --output_dir results/benchmark_tables \\
        --batch_size 128 --device cuda
"""

import argparse
import contextlib
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

# FGVC-Aircraft hierarchy (coarse -> fine), matching ``dataset.AIRCRAFT_LEVELS``.
AIRCRAFT_RANKS = ["manufacturer", "family", "variant"]

# Column name used when an aircraft config does not split by manufacturer.
AIRCRAFT_ALL_SUPERCLASS = "Aircraft"

# Per-``test_cat`` accuracy metrics -> (table label, caption fragment).
ACC_METRICS: List[Tuple[str, str, str]] = [
    ("knn1", "kNN-1", "1-nearest-neighbour top-1 accuracy"),
    ("knn5", "kNN-5", "5-nearest-neighbour top-1 accuracy"),
    ("linprobe", "LinProbe", "linear-probe top-1 accuracy"),
]

# Per-superclass correlation metrics (computed once over the test_cat ranks).
CORR_METRICS: List[Tuple[str, str, str]] = [
    ("spearman", "Spearman", "Spearman correlation between embedding and taxonomic distances"),
    ("pearson", "Pearson", "Pearson correlation between embedding and taxonomic distances"),
    ("cpcc", "CPCC", "cophenetic correlation coefficient of the embedding dendrogram"),
    ("dendrogram", "Dendro", "dendrogram cophenetic correlation with the taxonomy"),
]

# Defaults hard-coded in ``contrastive_experiment._cophenetic_correlation``.
COPH_MAX_SAMPLES = 2048
COPH_SEED = 42


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
    required_ranks: Optional[List[str]] = None,
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
        # ``InatDataModule._parse_inat_json`` drops samples missing a test_cat rank.
        if required_ranks and any(cat_info.get(r) is None for r in required_ranks):
            continue
        paths.append(os.path.join(image_dir, img_map[img_id]))
        taxa.append(tuple(str(cat_info.get(rank, "")) for rank in RANKS))
    return paths, taxa


def parse_aircraft_taxonomy(
    root: str, split: str, superclass: Optional[str], download: bool = False,
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Parse FGVC-Aircraft into (paths, taxonomy_tuples) over ``AIRCRAFT_RANKS``.

    ``superclass`` (when not the catch-all column) filters by manufacturer.
    """
    from torchvision.datasets import FGVCAircraft

    per_level: Dict[str, Dict[str, str]] = {}
    files: Optional[List[str]] = None
    for level in AIRCRAFT_RANKS:
        ds = FGVCAircraft(root=root, split=split, annotation_level=level,
                          download=download)
        per_level[level] = {str(p): ds.classes[l]
                            for p, l in zip(ds._image_files, ds._labels)}
        if files is None:
            files = [str(p) for p in ds._image_files]

    sc = superclass.lower() if superclass else None
    if sc == AIRCRAFT_ALL_SUPERCLASS.lower():
        sc = None

    paths: List[str] = []
    taxa: List[Tuple[str, ...]] = []
    for path in (files or []):
        tax = tuple(per_level[r][path] for r in AIRCRAFT_RANKS)
        if sc is not None and tax[0].lower() != sc:
            continue
        paths.append(path)
        taxa.append(tax)
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
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    dataset = InatDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    collected: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    for imgs, idx in tqdm(loader, desc="  encoding", leave=False):
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else contextlib.nullcontext())
        with ctx:
            z = model.encode(imgs.to(device), normalize=True)
        # Keep the reduced-precision rounding, but accumulate in fp32.
        collected.append(z.float().cpu())
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


def select_subset(
    paths: List[str], taxa: List[Tuple[str, ...]],
    max_species: Optional[int], max_samples: Optional[int], seed: int,
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Cap species then images, matching ``cophenetic_correlation_test.select_samples``."""
    rng = np.random.RandomState(seed)

    all_taxa = sorted(set(taxa))
    keep_taxa = set(all_taxa)
    if max_species is not None and max_species < len(all_taxa):
        idx = rng.choice(len(all_taxa), size=max_species, replace=False)
        keep_taxa = {all_taxa[i] for i in idx}

    filtered = [(p, t) for p, t in zip(paths, taxa) if t in keep_taxa]
    rng.shuffle(filtered)
    if max_samples is not None and max_samples < len(filtered):
        filtered = filtered[:max_samples]
    return [p for p, _ in filtered], [t for _, t in filtered]


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
    embeddings: torch.Tensor, taxa: List[Tuple[str, ...]], rank_indices: List[int],
    metric: str = "cosine", linkage_method: str = "average",
    max_samples: int = COPH_MAX_SAMPLES, seed: int = COPH_SEED,
) -> Dict[str, float]:
    """Sample-level cophenetic correlation, mirroring the validation-epoch metric.

    ``rank_indices`` selects the ``test_cat`` ranks (coarse -> fine), matching the
    columns of ``val_test_labels`` in ``contrastive_experiment``.
    """
    nan_result = {"spearman": float("nan"), "pearson": float("nan"),
                  "cpcc": float("nan"), "dendrogram": float("nan")}
    n, num_levels = len(taxa), len(rank_indices)
    if n < 3 or num_levels < 2:
        return nan_result

    tax_tuples = [tuple(t[r] for r in rank_indices) for t in taxa]
    X = embeddings.detach().cpu().float().numpy()
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        tax_tuples = [tax_tuples[i] for i in idx]

    tax_condensed = squareform(taxonomic_cophenetic_matrix(tax_tuples), checks=False)
    emb_condensed = pdist(X, metric=metric)
    if not np.isfinite(emb_condensed).all() or emb_condensed.std() == 0:
        return nan_result

    spearman_r, _ = spearmanr(emb_condensed, tax_condensed)
    pearson_r, _ = pearsonr(emb_condensed, tax_condensed)
    Z = linkage(emb_condensed, method=linkage_method)
    cpcc, coph_dists = cophenet(Z, emb_condensed)
    dendro_r, _ = spearmanr(coph_dists, tax_condensed)
    return {"spearman": float(spearman_r), "pearson": float(pearson_r),
            "cpcc": float(cpcc), "dendrogram": float(dendro_r)}


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


def build_latex_table_flat(
    metric_key: str, caption_fragment: str, results: dict,
    methods: List[str], superclasses: List[str], scale: float, decimals: int,
) -> str:
    """One row per method, one column per superclass (no test_cat grouping)."""
    col_spec = "l" + "c" * len(superclasses)
    lines = [
        r"\begin{table*}[ht]\centering\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Method & " + " & ".join(superclasses) + r" \\",
        r"\midrule",
    ]

    best_per_col: List[Optional[float]] = []
    for sc in superclasses:
        vals = [results[metric_key][m].get(sc) for m in methods]
        vals = [v for v in vals if v is not None and not
                (isinstance(v, float) and np.isnan(v))]
        best_per_col.append(max(vals) if vals else None)

    for method in methods:
        cells = []
        for ci, sc in enumerate(superclasses):
            v = results[metric_key][method].get(sc)
            is_best = (best_per_col[ci] is not None and v is not None
                       and not (isinstance(v, float) and np.isnan(v))
                       and abs(v - best_per_col[ci]) < 1e-12)
            cells.append(format_value(v, is_best, scale, decimals))
        method_cell = method.replace("_", r"\_")
        lines.append(f"{method_cell:<14} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{caption_fragment}, computed per superclass over the "
        rf"{{\ttfamily test\_cats}} ranks at the sample level. Values are "
        rf"multiplied by {int(scale)} and rounded to "
        rf"{decimals} decimals. Bold indicates the largest value across methods "
        r"per taxonomic group.}",
        rf"\label{{tab:inat_{metric_key}}}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


# ── CLI / main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Taxonomic benchmark (kNN / linear-probe / cophenetic) on iNat or FGVC-Aircraft")
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
    p.add_argument("--coph_max_samples", type=int, default=COPH_MAX_SAMPLES,
                   help="Images subsampled for the correlation metrics")
    p.add_argument("--coph_seed", type=int, default=COPH_SEED,
                   help="Seed for the correlation-metric subsample")
    p.add_argument("--max_species", type=int, default=None,
                   help="Optional cap on distinct leaf classes per superclass (random subset)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on images per (backbone, superclass)")
    p.add_argument("--probe_lr", type=float, default=0.1)
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None,
                   help="'cuda', 'cpu', 'cuda:1' or a bare GPU index such as '0'")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Autocast dtype for encoding (bf16 matches training)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_device(spec: Optional[str]) -> torch.device:
    if not spec:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(f"cuda:{spec}" if spec.isdigit() else spec)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": None}[args.precision]

    with open(args.config) as f:
        cfg = json.load(f)

    test_cats: List[dict] = cfg["test_cats"]
    backbones: List[dict] = cfg["backbones"]
    methods = [b["name"] for b in backbones]

    dataset_kind = str(cfg.get("dataset", "inat")).lower()
    if dataset_kind not in ("inat", "aircraft"):
        raise SystemExit(f"Unknown dataset '{dataset_kind}' (use 'inat' or 'aircraft')")
    if dataset_kind == "aircraft":
        ranks = AIRCRAFT_RANKS
        superclasses = cfg.get("superclasses") or [AIRCRAFT_ALL_SUPERCLASS]
    else:
        ranks = RANKS
        superclasses = cfg["superclasses"]

    for tc in test_cats:
        if tc["rank"] not in ranks:
            raise SystemExit(f"test_cat rank '{tc['rank']}' not in {ranks}")

    # Correlation ranks = the test_cat columns of ``val_test_labels``; the config
    # order is preserved because the LCA depth assumes coarse -> fine.
    corr_rank_indices = [ranks.index(tc["rank"]) for tc in test_cats]

    # results_acc[metric][test_cat_label][method][superclass] = value
    results_acc: dict = {
        mk: {tc["label"]: {m: {} for m in methods} for tc in test_cats}
        for mk, _, _ in ACC_METRICS
    }
    # results_corr[metric][method][superclass] = value  (one per superclass)
    results_corr: dict = {mk: {m: {} for m in methods} for mk, _, _ in CORR_METRICS}

    for backbone_cfg in backbones:
        name = backbone_cfg["name"]
        img_size = backbone_cfg.get("img_size", 224)
        transform = build_transform(img_size)
        print("\n" + "=" * 74)
        print(f"BACKBONE: {name}  (backbone={backbone_cfg.get('backbone')}, "
              f"img_size={img_size})")

        model = build_encoder(backbone_cfg).to(device).eval()

        for sc in superclasses:
            if dataset_kind == "aircraft":
                paths, taxa = parse_aircraft_taxonomy(
                    cfg["aircraft_root"], cfg.get("aircraft_split", "test"), sc,
                    download=bool(cfg.get("aircraft_download", False)))
            else:
                paths, taxa = parse_inat_taxonomy(
                    cfg["test_metadata"], cfg["test_image_dir"], sc,
                    required_ranks=[tc["rank"] for tc in test_cats])
            if not paths:
                print(f"  [{sc}] no images, skipping")
                continue
            paths, taxa = select_subset(paths, taxa, args.max_species,
                                        args.max_samples, args.seed)
            if not paths:
                print(f"  [{sc}] no images after subsampling, skipping")
                continue

            print(f"  [{sc}] encoding {len(paths)} images over "
                  f"{len(set(taxa))} classes...")
            embeddings = encode_paths(model, paths, transform, args.batch_size,
                                      device, args.num_workers, amp_dtype)

            corr = correlation_metrics(embeddings, taxa, corr_rank_indices,
                                       args.metric, args.linkage,
                                       args.coph_max_samples, args.coph_seed)
            for mk, _, _ in CORR_METRICS:
                results_corr[mk][name][sc] = corr[mk]

            for tc in test_cats:
                rank_idx = ranks.index(tc["rank"])
                labels = labels_at_rank(taxa, rank_idx)

                knn = knn_accuracy(embeddings, labels, ks=(1, 5))
                results_acc["knn1"][tc["label"]][name][sc] = knn[1]
                results_acc["knn5"][tc["label"]][name][sc] = knn[5]

                results_acc["linprobe"][tc["label"]][name][sc] = linear_probe_top1(
                    embeddings, labels, device, args.train_fraction, args.seed,
                    lr=args.probe_lr, epochs=args.probe_epochs)

                print(f"    {tc['label']} ({tc['rank']}): "
                      f"kNN1={knn[1]:.3f} kNN5={knn[5]:.3f} "
                      f"LinProbe={results_acc['linprobe'][tc['label']][name][sc]:.3f}")

            print(f"    corr (test_cat ranks): Spearman={corr['spearman']:.3f} "
                  f"Pearson={corr['pearson']:.3f} CPCC={corr['cpcc']:.3f} "
                  f"Dendro={corr['dendrogram']:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Emit LaTeX tables ─────────────────────────────────────────────────────
    tables: Dict[str, str] = {}
    for metric_key, table_label, caption in ACC_METRICS:
        tables[metric_key] = build_latex_table(
            metric_key, table_label, caption, results_acc, test_cats, methods,
            superclasses, scale=100.0, decimals=2)
    for metric_key, table_label, caption in CORR_METRICS:
        tables[metric_key] = build_latex_table_flat(
            metric_key, caption, results_corr, methods,
            superclasses, scale=100.0, decimals=2)

    for metric_key, table_label, _ in ACC_METRICS + CORR_METRICS:
        print("\n" + "#" * 74)
        print(f"# {table_label} table")
        print("#" * 74)
        print(tables[metric_key])

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for metric_key, table in tables.items():
            with open(os.path.join(args.output_dir, f"{metric_key}.tex"), "w") as f:
                f.write(table + "\n")
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({"accuracy": results_acc, "correlation": results_corr},
                      f, indent=2)
        print(f"\nWrote {len(tables)} tables + results.json to {args.output_dir}")


if __name__ == "__main__":
    main()
