import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image, ImageFile

# Tolerate slightly truncated files instead of hanging/erroring on read.
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _scan_images(root: str) -> list:
    """Recursively collect image paths (relative to ``root``) using os.scandir.

    Follows directory symlinks (common on SageMaker/mounted storage) while
    guarding against symlink loops via visited real paths.
    """
    paths = []
    seen_dirs = set()
    stack = [root]
    while stack:
        current = stack.pop()
        real = os.path.realpath(current)
        if real in seen_dirs:
            continue
        seen_dirs.add(real)
        with os.scandir(current) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=True):
                    stack.append(entry.path)
                elif entry.name.lower().endswith(IMG_EXTENSIONS):
                    paths.append(os.path.relpath(entry.path, root))
    paths.sort()
    return paths


class RecursiveImageDataset(Dataset):
    """
    Dataset that recursively collects every image under a root directory,
    regardless of folder layout or nesting depth. Labels are not used.

    For very large datasets (millions of images), the scanned file list can be
    cached to ``index_cache`` so subsequent runs skip the directory walk. Paths
    are stored relative to ``root`` in a compact NumPy array to keep memory and
    per-worker pickling cost low.

    Args:
        root: root directory to search for images
        transform: optional torchvision transform applied to each image
        index_cache: optional path to a .npy file used to cache the file list
    """

    def __init__(self, root: str, transform=None,
                 index_cache: Optional[str] = None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform

        if index_cache and os.path.isfile(index_cache):
            paths = np.load(index_cache, allow_pickle=False)
        else:
            paths = None

        # Ignore an empty/stale cache and (re)scan the directory.
        if paths is None or len(paths) == 0:
            if not os.path.isdir(root):
                raise RuntimeError(
                    f"Data directory does not exist: {root!r} "
                    f"(resolved to {os.path.abspath(root)!r}, cwd={os.getcwd()!r}). "
                    f"Pass an absolute --data_dir."
                )
            paths = np.asarray(_scan_images(root))
            # Only cache a non-empty result so we never persist a bad scan.
            if index_cache and len(paths) > 0:
                os.makedirs(os.path.dirname(index_cache) or ".", exist_ok=True)
                np.save(index_cache, paths)

        # Paths are relative to ``root``; rejoined in __getitem__.
        self.paths = paths

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found under {root!r} "
                f"(resolved to {os.path.abspath(root)!r}). "
                f"Supported extensions: {', '.join(IMG_EXTENSIONS)}."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, str(self.paths[index]))
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            # Don't hang/crash the whole run on one bad file: log it and fall
            # back to the next image so training can continue.
            print(f"[RecursiveImageDataset] Skipping unreadable image "
                  f"{path!r}: {exc}", flush=True)
            return self.__getitem__((index + 1) % len(self.paths))
        if self.transform is not None:
            image = self.transform(image)
        # Return a dummy label of 0 to stay compatible with (images, _) unpacking.
        return image, 0


class VAEDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading raw RGB images for VAE training.

    Recursively collects every image under ``data_dir``, independent of the
    folder structure or nesting depth (any subfolders are searched). Labels
    are ignored by the VAE; only the images are used.

    Images are resized to `img_size` x `img_size` and scaled to [0, 1]
    (matching the Sigmoid output of the decoder).

    Args:
        data_dir: root directory of the image dataset
        img_size: spatial size images are resized to (square)
        batch_size: training/validation batch size
        num_workers: number of DataLoader workers
        val_split: fraction of the dataset used for validation
        pin_memory: whether to use pinned memory in the DataLoader
        index_cache: optional .npy path to cache the scanned file list so
            large datasets are not re-walked on every run
        max_val_samples: optional cap on the validation subset size. With huge
            datasets a small fixed val set (e.g. 20k) makes validation fast
            while still giving stable loss / KL / AU estimates.
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 96,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 pin_memory: bool = True,
                 index_cache: Optional[str] = None,
                 max_val_samples: Optional[int] = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.index_cache = index_cache
        self.max_val_samples = max_val_samples

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None

    def _build_transform(self) -> transforms.Compose:
        # Scale raw RGB images to [0, 1] to match the decoder's Sigmoid output.
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        transform = self._build_transform()
        full_dataset = RecursiveImageDataset(
            root=self.data_dir,
            transform=transform,
            index_cache=self.index_cache,
        )

        val_len = int(len(full_dataset) * self.val_split)
        if self.max_val_samples is not None:
            val_len = min(val_len, self.max_val_samples)
        train_len = len(full_dataset) - val_len

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
