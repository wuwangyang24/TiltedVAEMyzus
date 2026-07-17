"""Select a group of images matched in object size, color AND shape, then run
all three permutation tests (scale, color and shape) on that single shared group.

Idea: the scale-, color- and shape-permutation tests each pick their own images
(by foreground area, foreground brightness, and foreground aspect ratio
respectively). To compare the three tests on exactly the same specimens, this
script selects one group of images that are simultaneously close to a target
object *size*, *color intensity* and *shape*, then feeds that fixed group to all
three tests.

Pipeline:
  1. Scan every image and measure three properties:
       - size  : number of non-black foreground pixels (area proxy)
       - color : mean brightness of the non-black foreground pixels
       - shape : length/width aspect ratio of the foreground bounding box
  2. Standardize all measures (z-scores) and select the ``pool`` images whose
     combined distance to the target (size, color, shape percentiles) is
     smallest -- i.e. images that are similar in size, color *and* shape.
  3. Run the scale-permutation test on that group (original/enlarged/shrunk).
  4. Run the color-permutation test on that group (original/brighter/darker).
  5. Run the shape-permutation test on that group
     (original/stretched_length/stretched_width).

All tests reuse the exact functions from ``permutation_test_size.py``,
``permutation_test_color.py`` and ``permutation_test_shape.py`` so their
behaviour is unchanged; only the image group is shared.

Examples:
python TiltedVAEMyzus/Tests/select_test.py --embedding TiltedVAEMyzus/results/checkpoints/tilted-latent128_kl0.001_bestsofar/embeddings_best_balanced_acc.pt --model tilted --pool 500 --method pca --device cpu --subtract_control --normalize_before_subtract
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
shape_test = _load_test_module("permutation_test_shape.py", "permutation_test_shape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a size- and color-matched image group and run both "
                    "permutation tests on it")

    # Data / model
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the image dataset (required unless --embedding is provided)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained Lightning checkpoint (.ckpt) or a "
                             "raw model state_dict (.pt/.pth) "
                             "(required unless --embedding is provided)")
    parser.add_argument("--embedding", type=str, default=None,
                        help="Path to a pre-computed embedding .pt file from "
                             "encode_embeddings.py. When provided, skips model "
                             "loading and image encoding.")
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted"],
                        help="Model architecture matching the checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for TiltedVAE (only used with --model tilted)")

    # Group selection (matched in size, color and shape)
    parser.add_argument("--pool", type=int, default=300,
                        help="Number of images to select whose object size, "
                             "color intensity AND shape are all closest to the target")
    parser.add_argument("--size_percentile", type=float, default=50.0,
                        help="Target object size to match, as a percentile of the "
                             "dataset's measured foreground areas (0-100)")
    parser.add_argument("--color_percentile", type=float, default=50.0,
                        help="Target color intensity to match, as a percentile of "
                             "the dataset's measured foreground brightnesses (0-100)")
    parser.add_argument("--shape_percentile", type=float, default=50.0,
                        help="Target object shape to match, as a percentile of the "
                             "dataset's measured foreground aspect ratios (0-100)")
    parser.add_argument("--black_threshold", type=int, default=0,
                        help="A pixel counts as foreground if its max channel "
                             "value exceeds this (0-255)")

    # Per-test view strength
    parser.add_argument("--size_scale", type=float, default=0.2,
                        help="Fractional zoom for the enlarge/shrink views")
    parser.add_argument("--color_scale", type=float, default=0.3,
                        help="Fractional color-intensity change for the "
                             "brighter/darker views")
    parser.add_argument("--shape_scale", type=float, default=0.2,
                        help="Fractional stretch for the length/width views")

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
    parser.add_argument("--shared_projection", action="store_true",
                        help="Fit a single dimensionality reducer jointly on all "
                             "seven view groups (original/enlarged/shrunk/brighter/"
                             "darker/stretched_length/stretched_width) so the "
                             "'original' points land at identical coordinates in "
                             "all plots")
    parser.add_argument("--subtract_control", action="store_true",
                        help="Subtract per-plate averaged control embedding from "
                             "treated embeddings (only used with --embedding)")
    parser.add_argument("--normalize_before_subtract", action="store_true",
                        help="L2-normalize treated and control embeddings before "
                             "subtraction (only used with --subtract_control)")

    args = parser.parse_args()

    if args.embedding is not None:
        pass  # no extra requirements
    else:
        if args.data_dir is None:
            parser.error("--data_dir is required when --embedding is not provided")
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --embedding is not provided")

    return args


def select_matched_paths(data_dir: str, in_channels: int, black_threshold: int,
                         pool: int, size_percentile: float,
                         color_percentile: float, shape_percentile: float,
                         output_dir: str) -> List[str]:
    """Select the ``pool`` images that are simultaneously closest to the target
    object size, target color intensity and target shape.

    All properties are standardized (z-scored) so they contribute comparably,
    and images are ranked by the Euclidean distance of their (size, color, shape)
    z-scores to the target point. The selected paths are also written to
    ``<output_dir>/selected_paths.txt`` for reference.
    """
    paths = _scan_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    print(f"[select] Measuring size, color and shape of {len(paths)} images...")
    areas = np.array([
        size_test.foreground_area(p, in_channels, black_threshold) for p in paths
    ], dtype=np.float64)
    intensities = np.array([
        color_test.foreground_intensity(p, in_channels, black_threshold)
        for p in paths
    ], dtype=np.float64)
    ratios = np.array([
        shape_test.foreground_aspect_ratio(p, in_channels, black_threshold)
        for p in paths
    ], dtype=np.float64)

    # Standardize all measures so distances in each are comparable.
    area_std = areas.std() or 1.0
    int_std = intensities.std() or 1.0
    ratio_std = ratios.std() or 1.0
    area_z = (areas - areas.mean()) / area_std
    int_z = (intensities - intensities.mean()) / int_std
    ratio_z = (ratios - ratios.mean()) / ratio_std

    # Target point: the requested percentile in each property, in z-space.
    area_target = (np.percentile(areas, size_percentile) - areas.mean()) / area_std
    int_target = (np.percentile(intensities, color_percentile) - intensities.mean()) / int_std
    ratio_target = (np.percentile(ratios, shape_percentile) - ratios.mean()) / ratio_std

    # Combined distance to the target in the (size, color, shape) z-space.
    dist = np.sqrt((area_z - area_target) ** 2 + (int_z - int_target) ** 2
                   + (ratio_z - ratio_target) ** 2)

    n = min(pool, len(paths))
    window = np.argsort(dist)[:n]

    sel_area = areas[window]
    sel_int = intensities[window]
    sel_ratio = ratios[window]
    print(f"[select] Selected {n} size+color+shape-matched images:")
    print(f"[select]   size  = {sel_area.mean():.0f} +/- {sel_area.std():.0f} px "
          f"(min {sel_area.min():.0f}, max {sel_area.max():.0f})")
    print(f"[select]   color = {sel_int.mean():.1f} +/- {sel_int.std():.1f} "
          f"(min {sel_int.min():.1f}, max {sel_int.max():.1f})")
    print(f"[select]   shape = {sel_ratio.mean():.3f} +/- {sel_ratio.std():.3f} "
          f"(min {sel_ratio.min():.3f}, max {sel_ratio.max():.3f})")

    selected = [paths[i] for i in window]

    os.makedirs(output_dir, exist_ok=True)
    list_path = os.path.join(output_dir, "selected_paths.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected))
    print(f"[select] Wrote selected image list to {list_path}")

    return selected


def load_latents_from_embedding(
    embedding_path: str,
    subtract_control: bool = False,
    normalize_before_subtract: bool = False,
) -> np.ndarray:
    """Load per-image treated latents from a pre-computed .pt embedding file.

    Returns a (N, D) array of all treated latent vectors across all compounds
    and plates, optionally with plate-level control subtraction.
    """
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)

    all_latents: List[np.ndarray] = []
    for compound_id, compound_data in data.items():
        for plate_id, plate_data in compound_data.items():
            treated = plate_data.get("treated", None)
            if treated is None or treated.numel() == 0:
                continue
            treated = treated.float()
            if subtract_control:
                control = plate_data.get("control", None)
                if control is not None and control.numel() > 0:
                    control = control.float()
                    if normalize_before_subtract:
                        treated = treated / (treated.norm(dim=-1, keepdim=True) + 1e-8)
                        control = control / (control.norm(dim=-1, keepdim=True) + 1e-8)
                    if control.ndim == 1:
                        control = control.unsqueeze(0)
                    treated = treated - control
            all_latents.append(treated.numpy())

    if not all_latents:
        raise RuntimeError(
            "No treated embeddings found in the .pt file."
        )
    return np.concatenate(all_latents, axis=0)


def encode_groups(test_module: ModuleType, model: torch.nn.Module,
                  paths: List[str], group_names: List[str], transforms: dict,
                  args: argparse.Namespace, device: torch.device,
                  label: str, output_dir: str) -> dict:
    """Encode every view of the image group to latents, print the invariance
    summary, and save the raw latents. Returns the per-view latent dict."""
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

    return groups


def plot_groups(test_module: ModuleType, coords: np.ndarray,
                group_names: List[str], n: int, args: argparse.Namespace,
                output_dir: str) -> None:
    """Plot the 2D projection of a test's groups (coords laid out group-by-group,
    each group occupying ``n`` contiguous rows)."""
    out_path = os.path.join(output_dir, f"permutation_{args.method}.png")
    test_module.plot_projection(
        coords, group_names, n, args.method, args.draw_links, out_path)


def run_permutation_test(test_module: ModuleType, model: torch.nn.Module,
                         paths: List[str], group_names: List[str], transforms: dict,
                         args: argparse.Namespace, device: torch.device,
                         label: str, output_dir: str) -> None:
    """Run one permutation test (scale or color) on a fixed image group with its
    own independently-fit projection."""
    groups = encode_groups(test_module, model, paths, group_names, transforms,
                           args, device, label, output_dir)
    all_latents = np.concatenate([groups[name] for name in group_names], axis=0)
    coords = test_module.reduce_dims(all_latents, args.method, args.seed)
    plot_groups(test_module, coords, group_names, len(paths), args, output_dir)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.embedding is not None:
        # ── Pre-computed embeddings mode ─────────────────────────────────────
        print(f"Loading pre-computed embeddings from {args.embedding}...")
        latents = load_latents_from_embedding(
            args.embedding,
            subtract_control=args.subtract_control,
            normalize_before_subtract=args.normalize_before_subtract,
        )
        print(f"Loaded {latents.shape[0]} latent vectors (dim={latents.shape[1]})")

        # Save latents
        npz_path = os.path.join(args.output_dir, "latents_embedding.npz")
        np.savez(npz_path, latents=latents)
        print(f"Saved latents to {npz_path}")

        # Dimensionality reduction and plot
        print(f"\nReducing to 2D with {args.method}...")
        coords = size_test.reduce_dims(latents, args.method, args.seed)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.6, edgecolors="k",
                   linewidths=0.2)
        ax.set_xlabel(f"{args.method.upper()} 1")
        ax.set_ylabel(f"{args.method.upper()} 2")
        ctrl_msg = " (control-subtracted)" if args.subtract_control else ""
        ax.set_title(f"Pre-computed embeddings{ctrl_msg}\n"
                     f"({latents.shape[0]} images, dim={latents.shape[1]})")
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, f"embeddings_{args.method}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_path}")
        print("\n[done] Embedding visualization complete.")
        return

    # ── Standard mode: encode from images ────────────────────────────────────
    # Model + weights (built/loaded once and shared by both tests).
    model = size_test.build_model(args)
    size_test.load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    # Select the shared, size-, color- and shape-matched image group.
    paths = select_matched_paths(
        args.data_dir, args.in_channels, args.black_threshold, args.pool,
        args.size_percentile, args.color_percentile, args.shape_percentile,
        args.output_dir)
    n = len(paths)

    size_names = ["original", "enlarged", "shrunk"]
    color_names = ["original", "brighter", "darker"]
    shape_names = ["original", "stretched_length", "stretched_width"]
    size_dir = os.path.join(args.output_dir, "size")
    color_dir = os.path.join(args.output_dir, "color")
    shape_dir = os.path.join(args.output_dir, "shape")
    size_transforms = size_test.build_transforms(args.img_size, args.size_scale)
    color_transforms = color_test.build_transforms(args.color_scale)
    shape_transforms = shape_test.build_transforms(args.img_size, args.shape_scale)

    if args.shared_projection:
        # Encode all tests, then fit ONE reducer jointly across all seven view
        # groups so the identical 'original' latents map to identical 2D points.
        size_groups = encode_groups(
            size_test, model, paths, size_names, size_transforms,
            args, device, "size", size_dir)
        color_groups = encode_groups(
            color_test, model, paths, color_names, color_transforms,
            args, device, "color", color_dir)
        shape_groups = encode_groups(
            shape_test, model, paths, shape_names, shape_transforms,
            args, device, "shape", shape_dir)

        # 'original' is identical in all tests (same images, identity view).
        original = size_groups["original"]
        combined = np.concatenate([
            original,
            size_groups["enlarged"], size_groups["shrunk"],
            color_groups["brighter"], color_groups["darker"],
            shape_groups["stretched_length"], shape_groups["stretched_width"],
        ], axis=0)

        print("\n[shared] Fitting one joint projection across all seven view groups")
        coords_all = size_test.reduce_dims(combined, args.method, args.seed)
        o = coords_all[0:n]
        en, sh = coords_all[n:2 * n], coords_all[2 * n:3 * n]
        br, da = coords_all[3 * n:4 * n], coords_all[4 * n:5 * n]
        sl, sw = coords_all[5 * n:6 * n], coords_all[6 * n:7 * n]

        # Reassemble per-test coords group-by-group; 'original' slice is shared.
        plot_groups(size_test, np.concatenate([o, en, sh], axis=0),
                    size_names, n, args, size_dir)
        plot_groups(color_test, np.concatenate([o, br, da], axis=0),
                    color_names, n, args, color_dir)
        plot_groups(shape_test, np.concatenate([o, sl, sw], axis=0),
                    shape_names, n, args, shape_dir)
    else:
        # Each test fits its own projection (original points differ per plot).
        run_permutation_test(
            size_test, model, paths, size_names, size_transforms,
            args, device, "size", size_dir)
        run_permutation_test(
            color_test, model, paths, color_names, color_transforms,
            args, device, "color", color_dir)
        run_permutation_test(
            shape_test, model, paths, shape_names, shape_transforms,
            args, device, "shape", shape_dir)

    print("\n[done] All permutation tests completed on the shared image group.")


if __name__ == "__main__":
    main()
