import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

# DINOv2 was pretrained with ImageNet normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Common raster image extensions to pick up when walking the dataset folder.
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _scan_images(data_dir: str) -> List[str]:
    """Recursively collect image file paths under ``data_dir`` (any nested
    folder layout), sorted for a deterministic ordering."""
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


class ImageFolderFlat(Dataset):
    """Loads images from a flat list of file paths.

    Each item is returned as ``(image_tensor, 0)`` so the batch matches the
    ``images, _ = batch`` unpacking used by the training loop. Images are
    converted to ``in_channels`` channels and scaled to ``[0, 1]`` (matching
    the model's final Sigmoid activation).
    """

    def __init__(self, paths: List[str], transform: T.Compose,
                 in_channels: int = 3) -> None:
        self.paths = paths
        self.transform = transform
        self.mode = ImageReadMode.GRAY if in_channels == 1 else ImageReadMode.RGB

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # read_image returns a uint8 [C, H, W] tensor; the transform resizes,
        # crops, and converts it to a float tensor in [0, 1].
        img = read_image(self.paths[index], mode=self.mode)
        tensor = self.transform(img)
        return tensor, 0


class VAEDataModule(pl.LightningDataModule):
    """LightningDataModule that serves images for VAE training.

    Recursively scans ``data_dir`` for images (optionally caching the file
    list to ``index_cache`` to avoid re-walking huge datasets), then splits
    them into train/validation subsets.

    Args:
        data_dir: root folder to scan for images (any nested layout).
        img_size: square size images are resized/cropped to.
        batch_size: mini-batch size for both loaders.
        num_workers: DataLoader worker processes.
        val_split: fraction of the data used for validation.
        index_cache: optional ``.npy`` path caching the scanned image list.
        max_val_samples: optional cap on the validation subset size.
        in_channels: number of image channels (1 grayscale, 3 RGB).
        seed: RNG seed for the train/val split shuffle.
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 96,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 index_cache: Optional[str] = None,
                 max_val_samples: Optional[int] = None,
                 in_channels: int = 3,
                 seed: int = 42) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.index_cache = index_cache
        self.max_val_samples = max_val_samples
        self.in_channels = in_channels
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def _build_transform(self) -> T.Compose:
        # Resize directly to a fixed square so the encoder always sees
        # ``img_size x img_size`` inputs. ConvertImageDtype scales uint8 pixels
        # to [0, 1], matching the decoder's Sigmoid output.
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
        ])

    def _load_paths(self) -> List[str]:
        # Reuse a cached image list when available to skip walking the tree.
        if self.index_cache and os.path.isfile(self.index_cache):
            return np.load(self.index_cache, allow_pickle=True).tolist()

        paths = _scan_images(self.data_dir)
        if not paths:
            raise RuntimeError(
                f"No images found under '{self.data_dir}'. Supported "
                f"extensions: {', '.join(IMG_EXTENSIONS)}"
            )

        if self.index_cache:
            os.makedirs(os.path.dirname(self.index_cache) or ".", exist_ok=True)
            np.save(self.index_cache, np.array(paths))

        return paths

    def setup(self, stage: Optional[str] = None) -> None:
        paths = self._load_paths()

        # Deterministic shuffle so the split is reproducible across ranks/runs.
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(paths))

        n_val = int(len(paths) * self.val_split)
        if self.max_val_samples is not None:
            n_val = min(n_val, self.max_val_samples)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]

        transform = self._build_transform()
        self.train_dataset = ImageFolderFlat(train_paths, transform, self.in_channels)
        self.val_dataset = ImageFolderFlat(val_paths, transform, self.in_channels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


class PKBatchSampler(Sampler):
    """Class-balanced (P x K) batch sampler for supervised contrastive training.

    Each yielded batch contains ``classes_per_batch`` (P) distinct synthesis
    programs with ``samples_per_class`` (K) images each, so every batch is
    guaranteed to hold multiple positives per program (same class) and multiple
    negatives (different classes). The effective batch size is ``P * K``.

    Classes with fewer than K samples are sampled with replacement. The number
    of batches per epoch defaults to ``len(labels) // (P * K)``.
    """

    def __init__(self, labels: List[int], classes_per_batch: int,
                 samples_per_class: int, num_batches: Optional[int] = None,
                 seed: int = 0) -> None:
        super().__init__(None)
        self.labels = np.asarray(labels)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self._epoch = 0

        self.label_to_indices: Dict[int, np.ndarray] = {}
        for idx, lab in enumerate(self.labels):
            self.label_to_indices.setdefault(int(lab), []).append(idx)
        self.label_to_indices = {
            lab: np.asarray(idxs) for lab, idxs in self.label_to_indices.items()
        }
        self.unique_labels = list(self.label_to_indices.keys())

        # Can't draw more distinct classes than exist.
        self.classes_per_batch = min(classes_per_batch, len(self.unique_labels))
        if self.classes_per_batch < 2:
            raise ValueError(
                "PKBatchSampler needs at least 2 synthesis programs to form "
                f"contrastive batches, found {len(self.unique_labels)}."
            )

        batch_size = self.classes_per_batch * self.samples_per_class
        if num_batches is None:
            num_batches = len(self.labels) // batch_size
        self.num_batches = max(1, num_batches)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        # Vary the shuffle each epoch while staying reproducible.
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.num_batches):
            chosen = rng.choice(
                self.unique_labels, size=self.classes_per_batch, replace=False)
            batch: List[int] = []
            for lab in chosen:
                idxs = self.label_to_indices[int(lab)]
                replace = len(idxs) < self.samples_per_class
                picked = rng.choice(idxs, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in picked)
            yield batch


class ContrastiveImageDataset(Dataset):
    """Loads RGB images labelled by synthesis program for contrastive training.

    Each item is ``(image_tensor, label_idx)`` where ``label_idx`` is the
    integer-encoded synthesis-program class. Images are resized, scaled to
    ``[0, 1]``, and normalized with ImageNet statistics (matching DINOv2).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        return self.transform(img), label


def build_ssl_transform(img_size: int, rotation: float = 30.0,
                        translate: float = 0.1,
                        min_scale: float = 0.5,
                        gaussian_blur: float = 0.5) -> T.Compose:
    """Stochastic augmentation pipeline producing one random view of an image
    for self-supervised (LeJEPA) training.

    Uses geometric augmentations and optional Gaussian blur: a random-resized
    crop (scale in ``[min_scale, 1.0]``) that makes the two views differ in
    framing/zoom, plus random rotation and translation, and Gaussian blur
    applied with probability ``gaussian_blur``. Drawing the transform ``V``
    times from the same image gives ``V`` correlated views. Output is a float
    tensor normalized with ImageNet statistics (DINOv2).
    """
    transforms = [
        T.RandomResizedCrop(
            (img_size, img_size), scale=(min_scale, 1.0), antialias=True),
        T.RandomAffine(degrees=rotation, translate=(translate, translate)),
    ]
    if gaussian_blur > 0:
        kernel_size = img_size // 20 * 2 + 1  # odd kernel ~ 5% of image size
        transforms.append(
            T.RandomApply([T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))],
                          p=gaussian_blur))
    transforms += [
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


class MultiViewImageDataset(Dataset):
    """Loads images and returns ``num_views`` independently augmented views of
    each image for self-supervised (LeJEPA) training.

    Each item is ``(views, label)`` where ``views`` is a stacked tensor of shape
    ``[num_views, C, H, W]``. The label is retained only for optional monitoring;
    the LeJEPA objective itself is label-free.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose,
                 num_views: int = 2) -> None:
        self.samples = samples
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        views = torch.stack([self.transform(img) for _ in range(self.num_views)])
        return views, label


class CompoundViewDataset(Dataset):
    """SSL dataset that uses different images of the same compound as views.

    Instead of augmenting a single image multiple times, this samples
    ``num_views`` distinct images from the same compound. Each image still
    receives the stochastic transform (crop/rotation) but the views are
    fundamentally different biological replicates.

    When a compound has fewer images than ``num_views``, images are resampled
    with replacement.
    """

    def __init__(self, compound_groups: List[Tuple[List[str], int]],
                 transform: T.Compose, num_views: int = 2) -> None:
        """
        Args:
            compound_groups: list of (image_paths, label) per compound.
            transform: stochastic augmentation applied to each sampled image.
            num_views: number of images to sample per compound per item.
        """
        self.compound_groups = compound_groups
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.compound_groups)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        paths, label = self.compound_groups[index]
        # Sample num_views paths (with replacement if fewer available).
        indices = np.random.choice(len(paths), size=self.num_views,
                                   replace=len(paths) < self.num_views)
        views = []
        for i in indices:
            img = read_image(paths[i], mode=ImageReadMode.RGB)
            views.append(self.transform(img))
        return torch.stack(views), label


class ContrastiveDataModule(pl.LightningDataModule):
    """LightningDataModule serving synthesis-program-labelled images for
    supervised contrastive (InfoNCE / SupCon) training of the DINOv2+LoRA model.

    Labels are derived by joining an image-metadata JSON (compound -> plates ->
    image paths, same format as the classifier callback) with a label CSV/Excel
    mapping each compound to a ``synthesis_program`` class.

    Args:
        image_metadata_json: JSON mapping compounds to plate/image paths.
        label_metadata_csv: CSV/Excel with compound -> synthesis-program labels.
        root_dir: base directory prepended to the relative image paths.
        img_size: square image size (must be a multiple of the DINOv2 patch, 14).
        batch_size: mini-batch size for both loaders.
        num_workers: DataLoader worker processes.
        val_split: fraction of images held out for validation.
        compound_col: compound-ID column in the label CSV.
        label_col: synthesis-program column in the label CSV.
        min_compounds_per_class: drop classes with fewer distinct compounds.
        filter_by_efficacy: keep only compounds with ``Efficacy`` >= this value
            (ignored if the column is absent or the value is 0/None).
        use_control: also include per-plate control images as training samples.
        classes_per_batch: P for P x K class-balanced sampling. When > 0 (with
            ``samples_per_class`` > 0), each train batch holds this many distinct
            synthesis programs, guaranteeing positives and negatives per batch.
        samples_per_class: K images per program for P x K sampling.
        compound_level: derive contrastive labels at the compound level instead
            of the synthesis-program level. Each compound becomes its own class,
            so positives are images of the same compound (across plates /
            replicates) rather than of the same synthesis program.
        seed: RNG seed for the train/val split.
    """

    def __init__(self,
                 image_metadata_json,
                 label_metadata_csv: str,
                 root_dir: str,
                 img_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 compound_col: str = "compound",
                 label_col: str = "synthesis_program",
                 min_compounds_per_class: int = 2,
                 filter_by_efficacy: Optional[float] = 0,
                 use_control: bool = False,
                 classes_per_batch: int = 0,
                 samples_per_class: int = 0,
                 compound_level: bool = False,
                 ssl_mode: bool = False,
                 ssl_views: int = 2,
                 ssl_rotation: float = 30.0,
                 ssl_translate: float = 0.1,
                 ssl_min_scale: float = 0.5,
                 ssl_gaussian_blur: float = 0.5,
                 ssl_compound_views: bool = False,
                 seed: int = 42) -> None:
        super().__init__()
        self.image_metadata_json = image_metadata_json
        self.label_metadata_csv = label_metadata_csv
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.compound_col = compound_col
        self.label_col = label_col
        self.min_compounds_per_class = min_compounds_per_class
        self.filter_by_efficacy = filter_by_efficacy
        self.use_control = use_control
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.compound_level = compound_level
        self.ssl_mode = ssl_mode
        self.ssl_views = ssl_views
        self.ssl_rotation = ssl_rotation
        self.ssl_translate = ssl_translate
        self.ssl_min_scale = ssl_min_scale
        self.ssl_gaussian_blur = ssl_gaussian_blur
        self.ssl_compound_views = ssl_compound_views
        self.seed = seed

        self.classes: List[str] = []
        self._train_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def use_pk_sampler(self) -> bool:
        return (not self.ssl_mode
                and self.classes_per_batch > 0 and self.samples_per_class > 0)

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_compound_labels(self) -> Dict[str, str]:
        """Return a {compound_id: synthesis_program} map, after optional
        efficacy filtering and dropping classes with too few compounds."""
        suffix = os.path.splitext(self.label_metadata_csv)[1].lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(self.label_metadata_csv)
        else:
            df = pd.read_csv(self.label_metadata_csv)

        if (self.filter_by_efficacy and self.filter_by_efficacy > 0
                and "Efficacy" in df.columns):
            df = df[df["Efficacy"] >= self.filter_by_efficacy]

        df = df[[self.compound_col, self.label_col]].dropna()
        df[self.compound_col] = df[self.compound_col].astype(str)
        df[self.label_col] = df[self.label_col].astype(str)

        # Drop classes with fewer than the required number of distinct compounds.
        min_cpc = max(self.min_compounds_per_class, 2)
        counts = df.groupby(self.label_col)[self.compound_col].nunique()
        valid_classes = set(counts[counts >= min_cpc].index)
        df = df[df[self.label_col].isin(valid_classes)]

        return dict(zip(df[self.compound_col], df[self.label_col]))

    def _build_samples(self) -> List[Tuple[str, int]]:
        # Support single path or list of paths for metadata JSONs.
        paths = self.image_metadata_json
        if isinstance(paths, str):
            paths = [paths]
        metadata = []
        for p in paths:
            print(f"[ContrastiveDataModule] Loading metadata: {p} ...", flush=True)
            with open(p) as f:
                metadata.extend(json.load(f))
        print(f"[ContrastiveDataModule] Loaded {len(metadata)} entries from {len(paths)} file(s)", flush=True)

        comp2label = self._load_compound_labels()
        print(f"[ContrastiveDataModule] Label map: {len(comp2label)} compounds", flush=True)
        if not comp2label:
            raise RuntimeError(
                "No compounds with valid synthesis-program labels remained after "
                "filtering. Check --contrastive_labels / --contrastive_min_per_class."
            )

        if self.compound_level:
            # Each compound is its own contrastive class; positives are images
            # of the same compound (across plates / replicates).
            self.classes = sorted(comp2label.keys())
            label2idx = {c: i for i, c in enumerate(self.classes)}

            def label_for(compound_id: str) -> int:
                return label2idx[compound_id]
        else:
            self.classes = sorted(set(comp2label.values()))
            label2idx = {c: i for i, c in enumerate(self.classes)}

            def label_for(compound_id: str) -> int:
                return label2idx[comp2label[compound_id]]

        subsets = ("treated", "control") if self.use_control else ("treated",)
        samples: List[Tuple[str, int]] = []
        # Also track compound membership for ssl_compound_views grouping.
        compound_to_sample_indices: Dict[str, List[int]] = {}
        for entry in metadata:
            cid = str(entry["Compound"])
            if cid not in comp2label:
                continue
            label_idx = label_for(cid)
            for plate_id, plate_data in entry.items():
                if plate_id == "Compound":
                    continue
                for subset in subsets:
                    for rel in plate_data.get(subset, []):
                        compound_to_sample_indices.setdefault(cid, []).append(len(samples))
                        samples.append((os.path.join(self.root_dir, rel), label_idx))

        if not samples:
            raise RuntimeError(
                "No labelled images found. Check --contrastive_metadata / "
                "--contrastive_root_dir and the compound-ID join."
            )

        # Recompute classes to only include those with actual images.
        actual_labels = sorted(set(label for _, label in samples))
        if self.compound_level:
            self.classes = [self.classes[i] for i in actual_labels]
        else:
            self.classes = [self.classes[i] for i in actual_labels]
        # Remap labels to contiguous 0..K-1
        old2new = {old: new for new, old in enumerate(actual_labels)}
        samples = [(path, old2new[label]) for path, label in samples]

        # Rebuild compound groups with remapped labels.
        compound_groups: Dict[str, Tuple[List[str], int]] = {}
        for cid, idxs in compound_to_sample_indices.items():
            paths_for_compound = [samples[i][0] for i in idxs if i < len(samples)]
            if paths_for_compound:
                label = samples[idxs[0]][1]
                compound_groups[cid] = (paths_for_compound, label)

        return samples, compound_groups

    def setup(self, stage: Optional[str] = None) -> None:
        samples, compound_groups = self._build_samples()

        rng = np.random.default_rng(self.seed)

        if self.compound_level:
            # Split at the compound level: all images of a given compound go
            # entirely to train or val to avoid data leakage.
            label_to_indices: Dict[int, List[int]] = {}
            for idx, (_, label) in enumerate(samples):
                label_to_indices.setdefault(label, []).append(idx)
            all_labels = list(label_to_indices.keys())
            rng.shuffle(all_labels)
            n_val_classes = max(1, int(len(all_labels) * self.val_split))
            val_labels = set(all_labels[:n_val_classes])
            train_idx = [i for lab, idxs in label_to_indices.items()
                         if lab not in val_labels for i in idxs]
            val_idx = [i for lab in val_labels for i in label_to_indices[lab]]
        else:
            indices = rng.permutation(len(samples))
            n_val = int(len(samples) * self.val_split)
            val_idx = indices[:n_val].tolist()
            train_idx = indices[n_val:].tolist()

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        # Shuffle val samples so that the val dataloader (shuffle=False) does
        # not serve compound-grouped batches, which would inflate batch-level
        # metrics like kNN accuracy.
        rng.shuffle(val_samples)

        if self.ssl_mode:
            # Self-supervised (LeJEPA): each item yields ``ssl_views`` randomly
            # augmented views; labels are ignored by the objective.
            ssl_transform = build_ssl_transform(
                self.img_size,
                rotation=self.ssl_rotation,
                translate=self.ssl_translate,
                min_scale=self.ssl_min_scale,
                gaussian_blur=self.ssl_gaussian_blur,
            )

            if self.ssl_compound_views:
                # Use different images from the same compound as views.
                # Split compound groups into train/val.
                all_cids = list(compound_groups.keys())
                rng.shuffle(all_cids)
                n_val_compounds = max(1, int(len(all_cids) * self.val_split))
                val_cids = set(all_cids[:n_val_compounds])
                train_groups = [compound_groups[c] for c in all_cids
                                if c not in val_cids]
                val_groups = [compound_groups[c] for c in all_cids
                              if c in val_cids]
                self.train_dataset = CompoundViewDataset(
                    train_groups, ssl_transform, num_views=self.ssl_views)
                self.val_dataset = CompoundViewDataset(
                    val_groups, ssl_transform, num_views=self.ssl_views)
                self._train_labels = [label for _, label in train_groups]
                self._val_labels = [label for _, label in val_groups]
                print(
                    f"[ContrastiveDataModule] SSL/LeJEPA compound-view mode: "
                    f"{self.ssl_views} images/compound, "
                    f"{len(train_groups)} train / {len(val_groups)} val compounds",
                    flush=True,
                )
            else:
                self.train_dataset = MultiViewImageDataset(
                    train_samples, ssl_transform, num_views=self.ssl_views)
                self.val_dataset = MultiViewImageDataset(
                    val_samples, ssl_transform, num_views=self.ssl_views)
                self._train_labels = [label for _, label in train_samples]
                self._val_labels = [label for _, label in val_samples]
                print(
                    f"[ContrastiveDataModule] SSL/LeJEPA mode: {self.ssl_views} views "
                    f"per image, {len(train_samples)} train / {len(val_samples)} val",
                    flush=True,
                )
            return

        transform = self._build_transform()
        self.train_dataset = ContrastiveImageDataset(train_samples, transform)
        self.val_dataset = ContrastiveImageDataset(val_samples, transform)
        self._train_labels = [label for _, label in train_samples]
        self._val_labels = [label for _, label in val_samples]
        print(
            f"[ContrastiveDataModule] {len(samples)} images, "
            f"{self.num_classes} "
            f"{'compounds' if self.compound_level else 'synthesis programs'} "
            f"(train={len(train_samples)}, val={len(val_samples)})",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        # Class-balanced P x K sampling guarantees positives and negatives per
        # batch; falls back to plain random shuffling when disabled.
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._train_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._val_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# iNaturalist 2021 dataset support
# ──────────────────────────────────────────────────────────────────────────────


class InatContrastiveDataset(Dataset):
    """iNaturalist dataset returning (image, train_label, test_label).

    ``train_label`` is used for contrastive loss and ``test_label`` is used for
    kNN evaluation on a different taxonomy level.
    """

    def __init__(self, samples: List[Tuple[str, int, int]], transform: T.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, train_label, test_label = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        return self.transform(img), train_label, test_label


class InatDataModule(pl.LightningDataModule):
    """LightningDataModule for iNaturalist 2021 mini dataset.

    Loads train/val metadata JSONs (COCO-style with ``images``, ``annotations``,
    ``categories``), assigns contrastive labels from ``train_cat`` taxonomy level,
    and evaluation labels from ``test_cat`` taxonomy level.

    Args:
        train_metadata: path to train_mini.json
        val_metadata: path to val.json
        train_image_dir: path to training images (e.g. inat2021/train_mini)
        val_image_dir: path to val images (e.g. inat2021/val)
        train_cat: taxonomy column for contrastive training (e.g. 'class')
        test_cat: taxonomy column for kNN evaluation (e.g. 'phylum')
        img_size: square image size
        batch_size: mini-batch size
        num_workers: DataLoader workers
        classes_per_batch: P for P x K sampling (0 disables)
        samples_per_class: K for P x K sampling
        seed: RNG seed
    """

    def __init__(self,
                 train_metadata: str,
                 val_metadata: str,
                 train_image_dir: str,
                 val_image_dir: str,
                 train_cat: str = "class",
                 test_cat: str = "phylum",
                 img_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 classes_per_batch: int = 0,
                 samples_per_class: int = 0,
                 superclass: Optional[str] = None,
                 seed: int = 42) -> None:
        super().__init__()
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.train_cat = train_cat
        self.test_cat = test_cat
        self.superclass = superclass
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed

        self.train_classes: List[str] = []
        self.test_classes: List[str] = []
        self._train_labels: List[int] = []
        self._val_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_train_classes(self) -> int:
        return len(self.train_classes)

    @property
    def num_test_classes(self) -> int:
        return len(self.test_classes)

    @property
    def use_pk_sampler(self) -> bool:
        return self.classes_per_batch > 0 and self.samples_per_class > 0

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _parse_inat_json(metadata_path: str, image_dir: str,
                         train_cat: str, test_cat: str,
                         superclass: Optional[str] = None
                         ) -> List[Tuple[str, str, str]]:
        """Parse iNat2021 COCO-style JSON, return (path, train_cat_value, test_cat_value).

        If ``superclass`` is given, only categories whose ``supercategory``
        field matches it (case-insensitive) are kept.
        """
        with open(metadata_path) as f:
            data = json.load(f)

        # Build category_id -> taxonomy mapping
        cat_map: Dict[int, Dict[str, str]] = {}
        for cat in data["categories"]:
            cat_map[cat["id"]] = cat

        # Build image_id -> file_name mapping
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
            train_val = cat_info.get(train_cat)
            test_val = cat_info.get(test_cat)
            if train_val is None or test_val is None:
                continue
            file_name = img_map[img_id]
            full_path = os.path.join(image_dir, file_name)
            samples.append((full_path, str(train_val), str(test_val)))

        return samples

    def setup(self, stage: Optional[str] = None) -> None:
        train_raw = self._parse_inat_json(
            self.train_metadata, self.train_image_dir,
            self.train_cat, self.test_cat, self.superclass)
        val_raw = self._parse_inat_json(
            self.val_metadata, self.val_image_dir,
            self.train_cat, self.test_cat, self.superclass)

        # Build unified label encodings across both splits
        all_train_cats = sorted(set(s[1] for s in train_raw + val_raw))
        all_test_cats = sorted(set(s[2] for s in train_raw + val_raw))
        self.train_classes = all_train_cats
        self.test_classes = all_test_cats
        train_cat2idx = {c: i for i, c in enumerate(all_train_cats)}
        test_cat2idx = {c: i for i, c in enumerate(all_test_cats)}

        train_samples = [(p, train_cat2idx[tc], test_cat2idx[ec])
                         for p, tc, ec in train_raw]
        val_samples = [(p, train_cat2idx[tc], test_cat2idx[ec])
                       for p, tc, ec in val_raw]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(val_samples)

        self._train_labels = [s[1] for s in train_samples]
        self._val_labels = [s[1] for s in val_samples]

        transform = self._build_transform()
        self.train_dataset = InatContrastiveDataset(train_samples, transform)
        self.val_dataset = InatContrastiveDataset(val_samples, transform)

        print(
            f"[InatDataModule] superclass={self.superclass}, "
            f"train_cat='{self.train_cat}' ({self.num_train_classes} classes), "
            f"test_cat='{self.test_cat}' ({self.num_test_classes} classes), "
            f"images: train={len(train_samples)}, val={len(val_samples)}",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._train_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._val_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
