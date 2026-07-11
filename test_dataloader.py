"""Standalone dataloader test — no model, no training.

Exercises the VAEDataModule in isolation to diagnose hangs/slowness during
data loading. Two modes:

  1. Loader mode (default): iterate the DataLoader exactly as training does,
     printing throughput and timing per batch. Reproduces multi-worker hangs.

  2. Scan mode (--scan): read/decode every image sequentially in the main
     process (num_workers=0) and report any file that fails or is unusually
     slow. Use this to find the specific corrupt/oversized image that stalls
     a worker.

Examples:
    python test_dataloader.py --data_dir /path/to/images
    python test_dataloader.py --data_dir /path/to/images --scan
    python test_dataloader.py --data_dir /path/to/images --num_workers 0
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader

from dataset import VAEDataModule, ImageFolderFlat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the VAE dataloader in isolation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--index_cache", type=str, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop after this many batches (default: full epoch)")
    parser.add_argument("--slow_threshold", type=float, default=2.0,
                        help="Warn when a batch/image takes longer than this (seconds)")
    parser.add_argument("--scan", action="store_true",
                        help="Sequentially decode every image to find bad files")
    return parser.parse_args()


def build_datamodule(args: argparse.Namespace) -> VAEDataModule:
    dm = VAEDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        index_cache=args.index_cache,
        max_val_samples=args.max_val_samples,
        in_channels=args.in_channels,
    )
    dm.setup()
    return dm


def scan_images(args: argparse.Namespace) -> None:
    """Decode every training image one-by-one to surface failures/slow reads."""
    dm = build_datamodule(args)
    dataset: ImageFolderFlat = dm.train_dataset
    n = len(dataset)
    print(f"[scan] Decoding {n} training images sequentially (main process)...")

    failures = []
    slow = []
    t_start = time.perf_counter()
    for i in range(n):
        path = dataset.paths[i]
        t0 = time.perf_counter()
        try:
            img, _ = dataset[i]
        except Exception as exc:  # noqa: BLE001 - we want to report any decode error
            failures.append((path, repr(exc)))
            print(f"[scan] FAIL  #{i}  {path}\n         {exc!r}")
            continue
        dt = time.perf_counter() - t0

        if not torch.isfinite(img).all():
            failures.append((path, "non-finite pixel values"))
            print(f"[scan] FAIL  #{i}  {path}  (non-finite pixels)")
        if dt > args.slow_threshold:
            slow.append((path, dt))
            print(f"[scan] SLOW  #{i}  {dt:6.2f}s  {path}")

        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.perf_counter() - t_start)
            print(f"[scan] {i + 1}/{n}  ({rate:.1f} img/s)")

    total = time.perf_counter() - t_start
    print("\n[scan] Done.")
    print(f"[scan] {n} images in {total:.1f}s "
          f"({n / total:.1f} img/s)")
    print(f"[scan] {len(failures)} failed, {len(slow)} slow "
          f"(> {args.slow_threshold}s)")
    if failures:
        print("[scan] First failures:")
        for path, err in failures[:20]:
            print(f"         {path}  ->  {err}")


def iterate_loader(args: argparse.Namespace) -> None:
    """Iterate the DataLoader like training does, timing every batch."""
    dm = build_datamodule(args)
    loader: DataLoader = dm.train_dataloader()
    n_batches = len(loader)
    print(f"[loader] train images: {len(dm.train_dataset)}  "
          f"val images: {len(dm.val_dataset)}")
    print(f"[loader] iterating {n_batches} batches "
          f"(batch_size={args.batch_size}, num_workers={args.num_workers})")

    t_start = time.perf_counter()
    t_prev = t_start
    for i, (images, _) in enumerate(loader):
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now

        if i == 0:
            print(f"[loader] first batch shape={tuple(images.shape)} "
                  f"dtype={images.dtype} min={images.min():.3f} "
                  f"max={images.max():.3f}")
        if dt > args.slow_threshold:
            print(f"[loader] SLOW batch #{i}: {dt:.2f}s")
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (now - t_start)
            print(f"[loader] {i + 1}/{n_batches}  ({rate:.2f} it/s)")

        if args.max_batches is not None and (i + 1) >= args.max_batches:
            print(f"[loader] stopping early after {i + 1} batches")
            break

    total = time.perf_counter() - t_start
    print(f"\n[loader] Done. Iterated in {total:.1f}s "
          f"({(i + 1) / total:.2f} it/s)")


def main() -> None:
    args = parse_args()
    if args.scan:
        scan_images(args)
    else:
        iterate_loader(args)


if __name__ == "__main__":
    main()
