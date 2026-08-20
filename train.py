import argparse
import hashlib
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class BestValLossReporter(Callback):
    """Track the epoch with the lowest ``val_loss`` and, at the end of
    training, print a table of the kNN / linear-probe metrics recorded at that
    epoch together with the positive-weight temperature and Sinkhorn settings."""

    def __init__(self, pos_weight_tau, sinkhorn: bool, sinkhorn_iters: int,
                 stats_dir: str, report_name: str) -> None:
        super().__init__()
        self.pos_weight_tau = pos_weight_tau
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.stats_dir = stats_dir
        self.report_name = report_name
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_metrics: dict = {}

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = int(trainer.current_epoch)
            self.best_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if ("knn_top" in k or "linprobe_top" in k)
            }

    def on_fit_end(self, trainer, pl_module) -> None:
        if self.best_epoch is None:
            return

        # Group the recorded metrics by their label-set prefix so each row of
        # the table corresponds to one evaluation (e.g. val_test_phylum).
        prefixes = sorted({
            k.rsplit("_", 2)[0] for k in self.best_metrics
        })

        def fmt(value):
            return f"{value:.4f}" if value is not None else "-"

        header = ["Eval", "kNN@1", "kNN@5", "LinProbe@1", "LinProbe@5"]
        rows = []
        for prefix in prefixes:
            rows.append([
                prefix,
                fmt(self.best_metrics.get(f"{prefix}_knn_top1")),
                fmt(self.best_metrics.get(f"{prefix}_knn_top5")),
                fmt(self.best_metrics.get(f"{prefix}_linprobe_top1")),
                fmt(self.best_metrics.get(f"{prefix}_linprobe_top5")),
            ])

        widths = [
            max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
            for i in range(len(header))
        ]

        def render(cells):
            return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        tau_str = (
            f"{self.pos_weight_tau:.4g}"
            if isinstance(self.pos_weight_tau, (int, float))
            else str(self.pos_weight_tau)
        )
        sinkhorn_str = f"yes ({self.sinkhorn_iters} iters)" if self.sinkhorn else "no"

        lines = [
            "",
            "=" * max(60, sum(widths) + 3 * (len(widths) - 1)),
            f"Best val_loss: {self.best_val_loss:.6f} @ epoch {self.best_epoch}",
            f"Positive-weight Tau: {tau_str}",
            f"Sinkhorn: {sinkhorn_str}",
            "-" * max(60, sum(widths) + 3 * (len(widths) - 1)),
        ]
        if rows:
            lines.append(render(header))
            lines.append("-+-".join("-" * w for w in widths))
            lines.extend(render(r) for r in rows)
        else:
            lines.append("(no kNN / linear-probe metrics were recorded)")
        lines.append("=" * max(60, sum(widths) + 3 * (len(widths) - 1)))
        lines.append("")
        report = "\n".join(lines)
        print(report)

        # Persist the same report next to the checkpoints under a 'stats' folder.
        os.makedirs(self.stats_dir, exist_ok=True)
        with open(os.path.join(self.stats_dir, self.report_name),
                  "w", encoding="utf-8") as f:
            f.write(report)

from Models import VAE, TiltedVAE, DinoV2LoRA, Backbone
from dataset import VAEDataModule, ContrastiveDataModule, InatDataModule
from experiment import VAEExperiment
from contrastive_experiment import ContrastiveExperiment, LeJEPAExperiment

# Use file-system based tensor sharing to avoid /dev/shm exhaustion, which
# otherwise hangs DataLoader workers in containers with a small shared-memory
# mount (e.g. Docker/SageMaker default of 64MB).
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Convolutional VAE with PyTorch Lightning + W&B")

    # Data
    parser.add_argument("--dataset", type=str, default="myzus",
                        choices=["myzus", "inat"],
                        help="Dataset to use: 'myzus' (default synthesis-program dataset) "
                             "or 'inat' (iNaturalist 2021 mini)")
    parser.add_argument("--train_cat", type=str, default="class",
                        help="Taxonomy level for contrastive training labels (inat only). "
                             "Options: kingdom, phylum, class, order, family, genus")
    parser.add_argument("--test_cat", type=str, nargs="+", default=["phylum"],
                        help="Taxonomy level(s) for kNN / linear-probe evaluation "
                             "labels (inat only). One or more of: kingdom, phylum, "
                             "class, order, family, genus. When several are given, "
                             "kNN and linear-probe metrics are logged for each.")
    parser.add_argument("--superclass", type=str, default=None,
                        help="Keep only iNat categories whose 'supercategory' matches "
                             "this value (e.g. Plants, Insects). inat only.")
    parser.add_argument("--inat_train_metadata", type=str, default="train_mini.json",
                        help="Path to iNat2021 train metadata JSON")
    parser.add_argument("--inat_val_metadata", type=str, default="val.json",
                        help="Path to iNat2021 val metadata JSON")
    parser.add_argument("--inat_train_dir", type=str, default="inat2021/train_mini",
                        help="Directory containing iNat2021 training images")
    parser.add_argument("--inat_val_dir", type=str, default="inat2021/val",
                        help="Directory containing iNat2021 validation images")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the image dataset (any nested folder layout). "
                             "Required for the VAE models; ignored for --model dino_lora, "
                             "which uses --contrastive_metadata instead.")
    parser.add_argument("--img_size", type=int, default=96, help="Square image size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--index_cache", type=str, default=None,
                        help="Optional .npy path to cache the scanned image list "
                             "(avoids re-walking huge datasets each run)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Cap the validation subset size (e.g. 20000) to keep "
                             "validation fast on very large datasets")

    # Model
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted", "dino_lora", "backbone"],
                        help="Which model to train: 'vae' (standard VAE), "
                             "'tilted' (TiltedVAE with an exponentially tilted prior), "
                             "'dino_lora' (LoRA-adapted DINOv2 trained with a "
                             "supervised InfoNCE/SupCon loss over synthesis programs), "
                             "or 'backbone' (a fully fine-tuned backbone trained with the "
                             "same supervised contrastive losses, no LoRA; pick the "
                             "architecture with --backbone)")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for the TiltedVAE prior (only used when "
                             "--model tilted). Defaults to sqrt(2 * latent_dim)")

    # DINOv2 + LoRA contrastive model (only used when --model dino_lora)
    parser.add_argument("--dino_backbone", type=str, default="vit_small_patch14_dinov2",
                        choices=["vit_small_patch14_dinov2",
                                 "vit_base_patch14_dinov2",
                                 "vit_large_patch14_dinov2"],
                        help="DINOv2 backbone variant to adapt with LoRA")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_small_patch16_224",
                                 "swin_tiny_patch4_window7_224"],
                        help="Backbone to fully fine-tune when --model backbone: "
                             "'resnet18', 'resnet50', 'vit_small_patch16_224' or "
                             "'swin_tiny_patch4_window7_224'")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Projected embedding dimension for the contrastive head")
    parser.add_argument("--proj_hidden_dim", type=int, default=2048,
                        help="Hidden width of the 2-layer projection MLP")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout applied to the LoRA input")
    parser.add_argument("--lora_targets", type=str, nargs="*", default=["qkv"],
                        help="Leaf module names in the backbone to adapt with LoRA "
                             "(e.g. qkv proj)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for the InfoNCE/SupCon loss")
    parser.add_argument("--use_proj_head", action="store_true",
                        help="Use the projection head on top of the backbone features. "
                             "If not set, the backbone features are directly L2-normalized.")

    # Contrastive dataset (synthesis-program labels; only used when --model dino_lora)
    parser.add_argument("--contrastive_metadata", type=str, nargs="+", default=None,
                        help="JSON metadata (compounds -> plates -> image paths) for "
                             "the contrastive dataset. Multiple files can be provided "
                             "and will be merged. Required for --model dino_lora.")
    parser.add_argument("--contrastive_labels", type=str, default=None,
                        help="CSV/Excel mapping compounds to synthesis-program labels. "
                             "Required for --model dino_lora.")
    parser.add_argument("--contrastive_root_dir", type=str, default=None,
                        help="Root directory prepended to image paths in the JSON. "
                             "Required for --model dino_lora.")
    parser.add_argument("--contrastive_compound_col", type=str, default="compound",
                        help="Compound-ID column in the label CSV. Default: compound")
    parser.add_argument("--contrastive_label_col", type=str, default="synthesis_program",
                        help="Synthesis-program column in the label CSV. "
                             "Default: synthesis_program")
    parser.add_argument("--contrastive_min_per_class", type=int, default=2,
                        help="Drop synthesis programs with fewer distinct compounds. "
                             "Default: 2")
    parser.add_argument("--contrastive_filter_efficacy", type=float, default=0,
                        help="Keep only compounds with Efficacy >= this value")
    parser.add_argument("--contrastive_use_control", action="store_true",
                        help="Also include per-plate control images as training samples")
    parser.add_argument("--contrastive_classes_per_batch", type=int, default=0,
                        help="P for class-balanced P x K sampling: distinct synthesis "
                             "programs per batch. When > 0 (with "
                             "--contrastive_samples_per_class), guarantees positives and "
                             "negatives in every batch; effective batch size = P * K "
                             "(overrides --batch_size for training).")
    parser.add_argument("--contrastive_samples_per_class", type=int, default=0,
                        help="K for class-balanced P x K sampling: images per synthesis "
                             "program per batch.")
    parser.add_argument("--compound_level", action="store_true",
                        help="Compute the contrastive loss at the compound level "
                             "instead of the synthesis-program level: each compound "
                             "becomes its own class, so positives are images of the "
                             "same compound (across plates/replicates).")
    parser.add_argument("--contrastive_sigreg_loss", action="store_true",
                        help="Replace negatives in the contrastive loss with SIGReg "
                             "regularization for collapse prevention.")
    parser.add_argument("--dcl_sigreg_loss", action="store_true",
                        help="Use Decoupled Contrastive Loss with SIGReg: "
                             "loss = pos + lambda*neg + (1-lambda)*SIGReg.")
    parser.add_argument("--dcl_ema_momentum", type=float, default=0.9,
                        help="EMA momentum for the DCL-SIGReg class-mean memory bank. "
                             "Default: 0.9")
    parser.add_argument("--dcl_suspicion_tau", type=float, default=0.1,
                        help="Temperature of the DCL-SIGReg suspicion sigmoid. "
                             "Default: 0.1")
    parser.add_argument("--dcl_suspicion_bias", type=float, default=0.5,
                        help="Similarity threshold (bias) of the DCL-SIGReg suspicion "
                             "sigmoid. Default: 0.5")
    parser.add_argument("--dcl_suspicion_standardize", action="store_true",
                        help="Z-score the class-mean similarities over the batch's "
                             "negatives before the suspicion sigmoid (avoids "
                             "saturation; --dcl_suspicion_bias is then in std units, "
                             "try ~0 with --dcl_suspicion_tau ~1).")
    parser.add_argument("--normal_dcl", action="store_true",
                        help="With--dcl_sigreg_loss, use plain DCL+SIGReg: all "
                             "negatives weighted equally (no suspicion memory bank) "
                             "and SIGReg applied to the batch embeddings.")
    parser.add_argument("--dcl_soft_pos_loss", action="store_true",
                        help="Use DCL with similarity-weighted positives: positive "
                             "pairs that are more similar get larger weight, forming "
                             "tighter sub-clusters. Negatives are plain DCL.")
    parser.add_argument("--supcon_soft_pos_loss", action="store_true",
                        help="Use supervised contrastive (SupCon) loss with "
                             "similarity-weighted positives (same soft-positive "
                             "scheme as --dcl_soft_pos_loss, but with the coupled "
                             "SupCon denominator plus SIGReg regularization).")
    parser.add_argument("--supcon_soft_pos_tau", type=float, default=0.1,
                        help="Temperature for the positive-pair softmax weighting in "
                             "--supcon_soft_pos_loss (lower = more weight on closest "
                             "positives). Default: 0.1")
    parser.add_argument("--denominator_pos_weight", action="store_true",
                        help="With --supcon_soft_pos_loss, also re-weight the positive "
                             "terms inside the SupCon denominator by the soft positive "
                             "weights (negatives stay at weight 1).")
    parser.add_argument("--tau_annealing", action="store_true",
                        help="Linearly anneal the SupCon soft-positive temperature from "
                             "--supcon_tau_start to --supcon_tau_end over training.")
    parser.add_argument("--supcon_tau_start", type=float, default=0.1,
                        help="Initial SupCon soft-positive temperature when "
                             "--tau_annealing is enabled. Default: 0.1")
    parser.add_argument("--supcon_tau_end", type=float, default=0.1,
                        help="Final SupCon soft-positive temperature when "
                             "--tau_annealing is enabled. Default: 0.1")
    parser.add_argument("--no_pos_weight_epoch", type=int, default=0,
                        help="Number of initial epochs that use uniform positive "
                            "weights before SupCon/InfoNCE soft-positive weighting "
                            "starts. "
                             "Default: 0")
    parser.add_argument("--vanilla_dcl", action="store_true",
                        help="Use the plain Decoupled Contrastive Loss (no SIGReg, no "
                             "suspicion re-weighting): supervised single-view DCL.")
    parser.add_argument("--vanilla_supcon", action="store_true",
                        help="Use the plain Supervised Contrastive (SupCon) loss (no "
                             "SIGReg, no soft positives): supervised single-view "
                             "SupCon with the coupled InfoNCE denominator.")
    parser.add_argument("--infonce_softpos", action="store_true",
                        help="Use InfoNCE/SupCon with similarity-weighted positives "
                             "(same soft-positive scheme as --dcl_soft_pos_loss, but "
                             "with the coupled InfoNCE denominator and no SIGReg).")
    parser.add_argument("--pos_weight_tau", type=float, default=0.1,
                        help="Temperature for the positive-pair softmax weighting in "
                             "--infonce_softpos (lower = more weight on closest "
                             "positives). Default: 0.1")
    parser.add_argument("--dcl_soft_pos_tau", type=float, default=0.1,
                        help="Temperature for the positive-pair softmax weighting. "
                             "Lower values concentrate weight on closest positives. "
                             "Default: 0.1")
    parser.add_argument("--sinkhorn", action="store_true",
                        help="Use Sinkhorn-Knopp iterations to produce a doubly-stochastic "
                             "positive weight matrix instead of row-wise softmax. "
                             "Only used with --dcl_soft_pos_loss.")
    parser.add_argument("--sinkhorn_iters", type=int, default=5,
                        help="Number of Sinkhorn-Knopp iterations. Default: 5")

    # LeJEPA self-supervised training (only used when --model dino_lora)
    parser.add_argument("--ssl_lejepa", action="store_true",
                        help="Train the DINOv2+LoRA model with the label-free LeJEPA "
                             "self-supervised objective (multi-view prediction + SIGReg) "
                             "instead of supervised contrastive learning.")
    parser.add_argument("--ssl_views", type=int, default=2,
                        help="Number of augmented views per image for LeJEPA. Default: 2")
    parser.add_argument("--ssl_rotation", type=float, default=30.0,
                        help="Max random rotation (degrees) for LeJEPA augmentations.")
    parser.add_argument("--ssl_translate", type=float, default=0.1,
                        help="Max random translation (fraction of image size) for "
                             "LeJEPA augmentations.")
    parser.add_argument("--ssl_min_scale", type=float, default=0.5,
                        help="Min RandomResizedCrop scale for LeJEPA views (crop covers "
                             "[min_scale, 1.0] of the image area). Default: 0.5")
    parser.add_argument("--ssl_gaussian_blur", type=float, default=0.5,
                        help="Probability of applying Gaussian blur to each LeJEPA view. "
                             "0 disables blur. Default: 0.5")
    parser.add_argument("--ssl_compound_views", action="store_true",
                        help="Use different images from the same compound as views "
                             "instead of augmenting a single image multiple times. "
                             "Only used with --ssl_lejepa.")
    parser.add_argument("--sigreg_weight", type=float, default=0.05,
                        help="Lambda in [0,1] for the convex LeJEPA loss "
                             "(1-lambda)*prediction + lambda*SIGReg. Paper default: 0.05")
    parser.add_argument("--sigreg_slices", type=int, default=512,
                        help="Number of random 1-D projections for SIGReg. Default: 512")
    parser.add_argument("--sigreg_num_freqs", type=int, default=33,
                        help="Quadrature points for the SIGReg Epps-Pulley integral.")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kld_weight", type=float, default=0.005,
                        help="Weight for the KL term (M_N); ~ batch_size / dataset_size")
    parser.add_argument("--anneal_kld", action="store_true",
                        help="Enable sigmoid annealing of the KL weight over training steps")
    parser.add_argument("--anneal_k", type=float, default=0.0025,
                        help="Steepness of the sigmoid KL annealing schedule")
    parser.add_argument("--anneal_x0", type=int, default=2500,
                        help="Global step at which the sigmoid schedule reaches its midpoint")
    parser.add_argument("--au_threshold", type=float, default=0.01,
                        help="Posterior-mean variance threshold for counting active units (AU)")
    parser.add_argument("--scheduler_gamma", type=float, default=0.95)
    parser.add_argument("--scheduler", type=str, default="exponential",
                        choices=["exponential", "cosine"],
                        help="LR scheduler: 'exponential' (decay by gamma each epoch) "
                             "or 'cosine' (anneal to ~0 over all epochs). Default: exponential")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs before the main schedule. Default: 0")
    parser.add_argument("--epochs", type=int, default=100)

    # Trainer / hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Lightning precision (e.g. 16-mixed, bf16-mixed, 32-true)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic algorithms (reproducible but slower)")
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N training epochs (default: 1)")
    parser.add_argument("--seed", type=int, default=42)

    # Logging / checkpoints
    parser.add_argument("--project", type=str, default="tilted-vae-myzus",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--entity", type=str, default="fm_val",
                        help="W&B entity (team or username)")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="Optional W&B run tags, space separated")
    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


def ensure_wandb_login() -> None:
    """Ensure W&B is authenticated via the WANDB_API_KEY env var or a prior
    `wandb login`. Raises a clear error if no credentials are available."""
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        return

    # Fall back to cached credentials (e.g. from `wandb login`).
    if wandb.api.api_key:
        return

    raise RuntimeError(
        "Weights & Biases is not authenticated. Set the WANDB_API_KEY "
        "environment variable or run `wandb login` before training."
    )


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    ensure_wandb_login()

    # Inputs are fixed-size, so let cuDNN pick the fastest conv algorithms.
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True

    is_dino = args.model == "dino_lora"
    is_backbone = args.model == "backbone"
    is_contrastive = is_dino or is_backbone

    if is_contrastive:
        # DINOv2 expects 3-channel, patch14-compatible inputs. Force a valid
        # image size (multiple of 14) and RGB regardless of the VAE defaults.
        # The fully fine-tuned backbones handle their own sizing, so only the
        # DINOv2 path is constrained here.
        if is_dino and args.img_size % 14 != 0:
            args.img_size = 224
            print(f"[dino_lora] img_size must be a multiple of 14; using {args.img_size}")
        args.in_channels = 3

        if args.dataset == "inat":
            # iNaturalist 2021 dataset
            datamodule = InatDataModule(
                train_metadata=args.inat_train_metadata,
                val_metadata=args.inat_val_metadata,
                train_image_dir=args.inat_train_dir,
                val_image_dir=args.inat_val_dir,
                train_cat=args.train_cat,
                test_cat=args.test_cat,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                classes_per_batch=args.contrastive_classes_per_batch,
                samples_per_class=args.contrastive_samples_per_class,
                superclass=args.superclass,
                seed=args.seed,
            )
        else:
            # Myzus (default) dataset
            missing = [name for name, val in (
                ("--contrastive_metadata", args.contrastive_metadata),
                ("--contrastive_labels", args.contrastive_labels),
                ("--contrastive_root_dir", args.contrastive_root_dir),
            ) if not val]
            if missing:
                raise ValueError(
                    f"--model dino_lora requires {', '.join(missing)} to build the "
                    "synthesis-program-labelled contrastive dataset."
                )

            datamodule = ContrastiveDataModule(
                image_metadata_json=args.contrastive_metadata,
                label_metadata_csv=args.contrastive_labels,
                root_dir=args.contrastive_root_dir,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                val_split=args.val_split,
                compound_col=args.contrastive_compound_col,
                label_col=args.contrastive_label_col,
                min_compounds_per_class=args.contrastive_min_per_class,
                filter_by_efficacy=args.contrastive_filter_efficacy,
                use_control=args.contrastive_use_control,
                classes_per_batch=args.contrastive_classes_per_batch,
                samples_per_class=args.contrastive_samples_per_class,
                compound_level=args.compound_level,
                ssl_mode=args.ssl_lejepa,
                ssl_views=args.ssl_views,
                ssl_rotation=args.ssl_rotation,
                ssl_translate=args.ssl_translate,
                ssl_min_scale=args.ssl_min_scale,
                ssl_gaussian_blur=args.ssl_gaussian_blur,
                ssl_compound_views=args.ssl_compound_views,
                seed=args.seed,
            )

        if is_backbone:
            model = Backbone(
                backbone=args.backbone,
                img_size=args.img_size,
                embedding_dim=args.embedding_dim,
                proj_hidden_dim=args.proj_hidden_dim,
                temperature=args.temperature,
                use_proj_head=args.use_proj_head,
                dcl_ema_momentum=args.dcl_ema_momentum,
                dcl_suspicion_tau=args.dcl_suspicion_tau,
                dcl_suspicion_bias=args.dcl_suspicion_bias,
                dcl_suspicion_standardize=args.dcl_suspicion_standardize,
                dcl_normal=args.normal_dcl,
                dcl_soft_pos=args.dcl_soft_pos_loss,
                dcl_soft_pos_tau=args.dcl_soft_pos_tau,
                supcon_soft_pos=args.supcon_soft_pos_loss,
                supcon_soft_pos_tau=args.supcon_soft_pos_tau,
                supcon_denom_pos_weight=args.denominator_pos_weight,
                sinkhorn=args.sinkhorn,
                sinkhorn_iters=args.sinkhorn_iters,
            )
        else:
            model = DinoV2LoRA(
                backbone=args.dino_backbone,
                img_size=args.img_size,
                embedding_dim=args.embedding_dim,
                proj_hidden_dim=args.proj_hidden_dim,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_targets=args.lora_targets,
                temperature=args.temperature,
                use_proj_head=args.use_proj_head,
                dcl_ema_momentum=args.dcl_ema_momentum,
                dcl_suspicion_tau=args.dcl_suspicion_tau,
                dcl_suspicion_bias=args.dcl_suspicion_bias,
                dcl_suspicion_standardize=args.dcl_suspicion_standardize,
                dcl_normal=args.normal_dcl,
                dcl_soft_pos=args.dcl_soft_pos_loss,
                dcl_soft_pos_tau=args.dcl_soft_pos_tau,
                supcon_soft_pos=args.supcon_soft_pos_loss,
                supcon_soft_pos_tau=args.supcon_soft_pos_tau,
                supcon_denom_pos_weight=args.denominator_pos_weight,
                sinkhorn=args.sinkhorn,
                sinkhorn_iters=args.sinkhorn_iters,
            )

        if args.ssl_lejepa:
            if not is_dino:
                raise ValueError("--ssl_lejepa is only supported with --model dino_lora.")
            experiment = LeJEPAExperiment(
                model=model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sigreg_weight=args.sigreg_weight,
                sigreg_slices=args.sigreg_slices,
                sigreg_num_freqs=args.sigreg_num_freqs,
                scheduler_gamma=args.scheduler_gamma,
                scheduler=args.scheduler,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs,
            )
        else:
            experiment = ContrastiveExperiment(
                model=model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                scheduler_gamma=args.scheduler_gamma,
                scheduler=args.scheduler,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs,
                contrastive_sigreg_loss=args.contrastive_sigreg_loss,
                dcl_sigreg_loss=args.dcl_sigreg_loss,
                dcl_soft_pos_loss=args.dcl_soft_pos_loss,
                vanilla_dcl=args.vanilla_dcl,
                vanilla_supcon=args.vanilla_supcon,
                infonce_softpos=args.infonce_softpos,
                supcon_softpos=args.supcon_soft_pos_loss,
                supcon_soft_pos_tau=args.supcon_soft_pos_tau,
                denom_pos_weight=args.denominator_pos_weight,
                tau_annealing=args.tau_annealing,
                supcon_tau_start=args.supcon_tau_start,
                supcon_tau_end=args.supcon_tau_end,
                no_pos_weight_epoch=args.no_pos_weight_epoch,
                pos_weight_tau=args.pos_weight_tau,
                sinkhorn=args.sinkhorn,
                sinkhorn_iters=args.sinkhorn_iters,
                sigreg_weight=args.sigreg_weight,
                sigreg_slices=args.sigreg_slices,
                train_cat=args.train_cat,
                test_cats=args.test_cat,
            )
    else:
        if not args.data_dir:
            raise ValueError(f"--data_dir is required for --model {args.model}.")
        # Data
        datamodule = VAEDataModule(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            index_cache=args.index_cache,
            max_val_samples=args.max_val_samples,
        )

        # Model
        if args.model == "tilted":
            model = TiltedVAE(
                in_channels=args.in_channels,
                latent_dim=args.latent_dim,
                tau=args.tau,
                img_size=args.img_size,
            )
        else:
            model = VAE(
                in_channels=args.in_channels,
                latent_dim=args.latent_dim,
                img_size=args.img_size,
            )

        experiment = VAEExperiment(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            kld_weight=args.kld_weight,
            scheduler_gamma=args.scheduler_gamma,
            anneal_kld=args.anneal_kld,
            anneal_k=args.anneal_k,
            anneal_x0=args.anneal_x0,
            au_threshold=args.au_threshold,
        )

    # Build checkpoint suffix (also used as default W&B run name).
    if is_contrastive:
        proj_tag = "Proj" if args.use_proj_head else "NoProj"
        p_val = args.contrastive_classes_per_batch
        k_val = args.contrastive_samples_per_class
        level_tag = "_Comp" if args.compound_level else ""
        test_cat_tag = "-".join(args.test_cat)
        dataset_tag = f"_inat_{args.train_cat}->{test_cat_tag}" if args.dataset == "inat" else ""
        if args.dataset == "inat" and args.superclass:
            dataset_tag += f"_{args.superclass}"
        if is_backbone:
            model_prefix = "Backbone"
        else:
            targets_tag = "_".join(args.lora_targets)
            model_prefix = (
                f"DINO_LoRA_{targets_tag}"
                f"_R{args.lora_rank}_A{args.lora_alpha}_D{args.lora_dropout}"
            )
        if args.ssl_lejepa:
            cv_tag = "_CompViews" if args.ssl_compound_views else ""
            aug_tag = (f"_Aug-R{args.ssl_rotation:.0f}T{args.ssl_translate}"
                       f"S{args.ssl_min_scale}B{args.ssl_gaussian_blur}")
            ckpt_suffix = (
                f"{model_prefix}"
                f"_BS{args.batch_size}"
                f"_{proj_tag}"
                f"_LeJEPA_Views{args.ssl_views}_SW{args.sigreg_weight}"
                f"{cv_tag}{aug_tag}"
                f"{dataset_tag}"
            )
        else:
            sigreg_tag = f"_SIGReg{args.sigreg_weight}" if args.contrastive_sigreg_loss else ""
            dcl_variant = "-Normal" if args.normal_dcl else ""
            if args.dcl_sigreg_loss and not args.normal_dcl:
                std_tag = "-Std" if args.dcl_suspicion_standardize else ""
                susp_tag = f"_EMA{args.dcl_ema_momentum}_B{args.dcl_suspicion_bias}_Tau{args.dcl_suspicion_tau}{std_tag}"
            else:
                susp_tag = ""
            dcl_tag = f"_DCL{dcl_variant}-SIGReg{args.sigreg_weight}{susp_tag}" if args.dcl_sigreg_loss else ""
            softpos_tag = f"_DCLSoftPos-Tau{args.dcl_soft_pos_tau}{'-Sinkhorn' + str(args.sinkhorn_iters) if args.sinkhorn else ''}" if args.dcl_soft_pos_loss else ""
            vanilla_dcl_tag = "_VanillaDCL" if args.vanilla_dcl else ""
            vanilla_supcon_tag = "_VanillaSupCon" if args.vanilla_supcon else ""
            infonce_softpos_tag = (
                f"_InfoNCESoftPos-Tau{args.pos_weight_tau}"
                f"{'-NoPosWeight' + str(args.no_pos_weight_epoch) if args.no_pos_weight_epoch else ''}"
                f"{'-DenomPosW' if args.denominator_pos_weight else ''}"
                f"{'-Sinkhorn' + str(args.sinkhorn_iters) if args.sinkhorn else ''}"
            ) if args.infonce_softpos else ""
            supcon_softpos_tag = (
                f"_SupConSoftPos-{'LinearTau' + str(args.supcon_tau_start) + 'to' + str(args.supcon_tau_end) if args.tau_annealing else 'Tau' + str(args.supcon_soft_pos_tau)}"
                f"{'-NoPosWeight' + str(args.no_pos_weight_epoch) if args.no_pos_weight_epoch else ''}"
                f"{'-DenomPosW' if args.denominator_pos_weight else ''}"
                f"{'-Sinkhorn' + str(args.sinkhorn_iters) if args.sinkhorn else ''}"
            ) if args.supcon_soft_pos_loss else ""
            ckpt_suffix = (
                f"{model_prefix}"
                f"_P{p_val}_K{k_val}_BS{args.batch_size}"
                f"_{proj_tag}"
                f"_T{args.temperature}"
                f"{level_tag}"
                f"{sigreg_tag}"
                f"{dcl_tag}"
                f"{softpos_tag}"
                f"{vanilla_dcl_tag}"
                f"{vanilla_supcon_tag}"
                f"{infonce_softpos_tag}"
                f"{supcon_softpos_tag}"
                f"{dataset_tag}"
            )
    else:
        ckpt_suffix = f"{args.model}-latent{args.latent_dim}-kld{args.kld_weight}-BS{args.batch_size}"

    # Resume from a previous run if a `last.ckpt` already exists for this config.
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", ckpt_suffix)
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_from = resume_ckpt if os.path.exists(resume_ckpt) else None

    # Only pin a deterministic W&B run id when we are actually resuming from a
    # checkpoint, so the resumed training continues logging to the same run.
    # When starting fresh we let W&B generate a new id: reusing a fixed id fails
    # if that id was previously created and then deleted on the W&B server.
    if resume_from is not None:
        wandb_run_id = hashlib.sha1(
            f"{args.project}/{args.run_name or ckpt_suffix}".encode()
        ).hexdigest()[:16]
        print(f"[resume] Found {resume_ckpt}; resuming W&B run {wandb_run_id}")
    else:
        wandb_run_id = None

    # Logger (Weights & Biases)
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or ckpt_suffix,
        entity=args.entity,
        tags=args.tags,
        save_dir=args.output_dir,
        log_model=False,
        id=wandb_run_id,
        resume="allow",
    )
    wandb_logger.log_hyperparams(vars(args))

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor]

    # Keep the last epoch's checkpoint (last.ckpt) for all models.
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=0))

    # Also keep the checkpoint with the lowest validation loss.
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-val-loss",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    ))

    # Report the best-val-loss epoch's metrics as a table at the end of
    # training (kNN / linear-probe accuracies are only produced by the
    # contrastive models).
    if is_contrastive:
        # The active soft-positive loss determines which temperature governs the
        # positive-pair weighting.
        if args.supcon_soft_pos_loss:
            pos_weight_tau = (
                f"{args.supcon_tau_start}->{args.supcon_tau_end}"
                if args.tau_annealing else args.supcon_soft_pos_tau
            )
        elif args.infonce_softpos:
            pos_weight_tau = args.pos_weight_tau
        elif args.dcl_soft_pos_loss:
            pos_weight_tau = args.dcl_soft_pos_tau
        else:
            pos_weight_tau = "n/a"
        callbacks.append(BestValLossReporter(
            pos_weight_tau=pos_weight_tau,
            sinkhorn=args.sinkhorn,
            sinkhorn_iters=args.sinkhorn_iters,
            stats_dir=os.path.join(args.output_dir, "reports"),
            report_name=f"{ckpt_suffix}_best_val_loss_report.txt",
        ))

    # Trainer
    # Parse --devices: "auto" stays as-is; comma-separated digits become a
    # list of ints so Lightning selects the right GPU(s) (e.g. "0" -> [0]).
    devices = args.devices
    if devices != "auto":
        try:
            devices = [int(d) for d in devices.split(",")]
        except ValueError:
            pass  # let Lightning handle unexpected values
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.val_every_n_epochs,
        deterministic=args.deterministic,
    )

    trainer.fit(experiment, datamodule=datamodule, ckpt_path=resume_from)


if __name__ == "__main__":
    main()
