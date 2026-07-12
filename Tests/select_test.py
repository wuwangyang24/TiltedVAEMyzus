"""Select a group of images matched in BOTH object size and color, then run
both permutation tests (scale and color) on that single shared group.

Idea: the scale- and color-permutation tests each pick their own images (one by
foreground area, the other by foreground brightness). To compare the two tests
on exactly the same specimens, this script selects one group of images that are
simultaneously close to a target object *size* and a target object *color
intensity*, then feeds that fixed group to both tests.

Pipeline:
  1. Scan every image and measure two properties:
       - size  : number of non-black foreground pixels (area proxy)
       - color : mean brightness of the non-black foreground pixels
  2. Standardize both measures (z-scores) and select the ``pool`` images whose
     combined distance to the target (size percentile, color percentile) is
     smallest -- i.e. images that are similar in *both* size and color.
  3. Run the scale-permutation test on that group (original/enlarged/shrunk).
  4. Run the color-permutation test on that group (original/brighter/darker).

Both tests reuse the exact functions from ``permutation_test_size.py`` and
``permutation_test_color.py`` so their behaviour is unchanged; only the image
group is shared.

Examples:
    python Tests/select_test.py --data_dir ../DATA/Train/ --checkpoint results/checkpoints/last.ckpt --model tilted --pool 500 --method pca --device cpu
"""
import argparse
import importlib.util
import os
import sys
from types import ModuleType
from typing import List

import numpy as np
import torch

# This script lives in ``Tests/``; add the repo root to the path so the
# top-level ``Models`` package and ``dataset`` module can be imported.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, ROOT)

from dataset import _scan_images


def _load_test_module(filename: str, name: str) -> ModuleType:
    """Import a sibling script from ``Tests/`` by file path (the folder is not a
    package, so it cannot be imported with a normal ``import`` statement)."""
    path = os.path.join(TESTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


size_test = _load_test_module("permutation_test_size.py", "permutation_test_size")
color_test = _load_test_module("permutation_test_color.py", "permutation_test_color")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a size- and color-matched image group and run both "
                    "permutation tests on it")

    # Data / model
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the image dataset (any nested folder layout)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained Lightning checkpoint (.ckpt) or a "
                             "raw model state_dict (.pt/.pth)")
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # Group selection (matched in both size and color)
    parser.add_argument("--pool", type=int, default=300,
                        help="Number of images to select whose object size AND "
                             "color intensity are both closest to the target")
    parser.add_argument("--size_percentile", type=float, default=50.0,
                        help="Target object size to match, as a percentile of the "
                             "dataset's measured foreground areas (0-100)")
    parser.add_argument("--color_percentile", type=float, default=50.0,
                        help="Target color intensity to match, as a percentile of "
                             "the dataset's measured foreground brightnesses (0-100)")
    parser.add_argument("--black_threshold", type=int, default=0,
                        help="A pixel counts as foreground if its max channel "
                             "value exceeds this (0-255)")

    # Per-test view strength
    parser.add_argument("--size_scale", type=float, default=0.15,
                        help="Fractional zoom for the enlarge/shrink views")
    parser.add_argument("--color_scale", type=float, default=0.15,
                        help="Fractional color-intensity change for the "
                             "brighter/darker views")

    # Shared test configuration
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne", "umap"],
                        help="2D projection method for visualization")
    parser.add_argument("--output_dir", type=str, default="results/permutation_test")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--draw_links", action="store_true",
                        help="Draw faint lines connecting the views of each image")

    return parser.parse_args()


def select_matched_paths(data_dir: str, in_channels: int, black_threshold: int,
                         pool: int, size_percentile: float,
                         color_percentile: float, output_dir: str) -> List[str]:
    """Select the ``pool`` images that are simultaneously closest to the target
    object size and target color intensity.

    Both properties are standardized (z-scored) so they contribute comparably,
    and images are ranked by the Euclidean distance of their (size, color)
    z-scores to the target point. The selected paths are also written to
    ``<output_dir>/selected_paths.txt`` for reference.
    """
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    print(f"[select] Measuring size and color of {len(paths)} images...")
    areas = np.array([
        size_test.foreground_area(p, in_channels, black_threshold) for p in paths
    ], dtype=np.float64)
    intensities = np.array([
        color_test.foreground_intensity(p, in_channels, black_threshold)
        for p in paths
    ], dtype=np.float64)

    # Standardize both measures so distances in each are comparable.
    area_std = areas.std() or 1.0
    int_std = intensities.std() or 1.0
    area_z = (areas - areas.mean()) / area_std
    int_z = (intensities - intensities.mean()) / int_std

    # Target point: the requested percentile in each property, in z-space.
    area_target = (np.percentile(areas, size_percentile) - areas.mean()) / area_std
    int_target = (np.percentile(intensities, color_percentile) - intensities.mean()) / int_std

    # Combined distance to the target in the (size, color) z-plane.
    dist = np.sqrt((area_z - area_target) ** 2 + (int_z - int_target) ** 2)

    n = min(pool, len(paths))
    window = np.argsort(dist)[:n]

    sel_area = areas[window]
    sel_int = intensities[window]
    print(f"[select] Selected {n} size+color-matched images:")
    print(f"[select]   size  = {sel_area.mean():.0f} +/- {sel_area.std():.0f} px "
          f"(min {sel_area.min():.0f}, max {sel_area.max():.0f})")
    print(f"[select]   color = {sel_int.mean():.1f} +/- {sel_int.std():.1f} "
          f"(min {sel_int.min():.1f}, max {sel_int.max():.1f})")

    selected = [paths[i] for i in window]

    os.makedirs(output_dir, exist_ok=True)
    list_path = os.path.join(output_dir, "selected_paths.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected))
    print(f"[select] Wrote selected image list to {list_path}")

    return selected


def run_permutation_test(test_module: ModuleType, model: torch.nn.Module,
                         paths: List[str], group_names: List[str], transforms: dict,
                         args: argparse.Namespace, device: torch.device,
                         label: str, output_dir: str) -> None:
    """Run one permutation test (scale or color) on a fixed image group, reusing
    the encoding / reduction / plotting helpers from that test's module."""
    print(f"\n[{label}] Running {label}-permutation test on {len(paths)} images")
    os.makedirs(output_dir, exist_ok=True)

    base_images = test_module.load_base_images(paths, args.img_size, args.in_channels)

    groups = {}
    for name in group_names:
        groups[name] = test_module.encode_group(
            model, base_images, transforms[name], args.batch_size, device)
        print(f"[{label}] {name:9s}: latents {groups[name].shape}")

    test_module.report_invariance(groups)

    npz_path = os.path.join(output_dir, "latents.npz")
    np.savez(npz_path, **groups, paths=np.array(paths))
    print(f"[{label}] Saved latents to {npz_path}")

    all_latents = np.concatenate([groups[name] for name in group_names], axis=0)
    coords = test_module.reduce_dims(all_latents, args.method, args.seed)

    out_path = os.path.join(output_dir, f"permutation_{args.method}.png")
    test_module.plot_projection(
        coords, group_names, len(paths), args.method, args.draw_links, out_path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.output_dir, exist_ok=True)

    # Model + weights (built/loaded once and shared by both tests).
    model = size_test.build_model(args)
    size_test.load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    # Select the shared, size- and color-matched image group.
    paths = select_matched_paths(
        args.data_dir, args.in_channels, args.black_threshold, args.pool,
        args.size_percentile, args.color_percentile, args.output_dir)

    # Scale-permutation test on the shared group.
    size_transforms = size_test.build_transforms(args.img_size, args.size_scale)
    run_permutation_test(
        size_test, model, paths,
        ["original", "enlarged", "shrunk"], size_transforms,
        args, device, "size", os.path.join(args.output_dir, "size"))

    # Color-permutation test on the shared group.
    color_transforms = color_test.build_transforms(args.color_scale)
    run_permutation_test(
        color_test, model, paths,
        ["original", "brighter", "darker"], color_transforms,
        args, device, "color", os.path.join(args.output_dir, "color"))

    print("\n[done] Both permutation tests completed on the shared image group.")


if __name__ == "__main__":
    main()
