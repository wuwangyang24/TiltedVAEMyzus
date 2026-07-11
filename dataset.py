import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as T

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
        self.mode = "L" if in_channels == 1 else "RGB"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as img:
            img = img.convert(self.mode)
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
        # Resize the shorter side then center-crop to a fixed square so the
        # encoder always sees ``img_size x img_size`` inputs. ToTensor scales
        # pixels to [0, 1], matching the decoder's Sigmoid output range.
        return T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
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
