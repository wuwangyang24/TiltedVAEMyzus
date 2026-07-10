import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder


class VAEDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading raw RGB images for VAE training.

    Expects a directory of images readable by torchvision's ImageFolder,
    i.e. images organized into one or more class subfolders:

        data_dir/
            class_a/xxx.png
            class_b/yyy.jpg
            ...

    Labels are ignored by the VAE; only the images are used.

    Images are resized to `img_size` x `img_size` and scaled to [0, 1]
    (matching the Sigmoid output of the decoder).

    Args:
        data_dir: root directory of the image dataset
        img_size: spatial size images are resized to (square)
        batch_size: training/validation batch size
        num_workers: number of DataLoader workers
        val_split: fraction of the dataset used for validation
        pin_memory: whether to use pinned memory in the DataLoader
    """

    def __init__(self,
                 data_dir: str,
                 img_size: int = 96,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 pin_memory: bool = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory

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
        full_dataset = ImageFolder(root=self.data_dir, transform=transform)

        val_len = int(len(full_dataset) * self.val_split)
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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
