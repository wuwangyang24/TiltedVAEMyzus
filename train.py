import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Models import VAE, TiltedVAE, DinoTiltedVAE
from dataset import VAEDataModule
from experiment import VAEExperiment
from Tests.chemical_class_classifier.classifier_callback import ChemicalClassClassifierCallback

# Use file-system based tensor sharing to avoid /dev/shm exhaustion, which
# otherwise hangs DataLoader workers in containers with a small shared-memory
# mount (e.g. Docker/SageMaker default of 64MB).
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Convolutional VAE with PyTorch Lightning + W&B")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the image dataset (any nested folder layout)")
    parser.add_argument("--img_size", type=int, default=96, help="Square image size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--index_cache", type=str, default=None,
                        help="Optional .npy path to cache the scanned image list "
                             "(avoids re-walking huge datasets each run)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Cap the validation subset size (e.g. 20000) to keep "
                             "validation fast on very large datasets")

    # Model
    parser.add_argument("--model", type=str, default="vae",
                        choices=["vae", "tilted", "dino_tilted"],
                        help="Which model to train: 'vae' (standard VAE), "
                             "'tilted' (TiltedVAE with an exponentially tilted prior), "
                             "or 'dino_tilted' (DINOv2 encoder + TiltedVAE)")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--tau", type=float, default=None,
                        help="Tilt parameter for the TiltedVAE prior (only used when "
                             "--model tilted). Defaults to sqrt(2 * latent_dim)")

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
    parser.add_argument("--epochs", type=int, default=100)

    # Trainer / hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Lightning precision (e.g. 16-mixed, bf16-mixed, 32-true)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic algorithms (reproducible but slower)")
    parser.add_argument("--seed", type=int, default=42)

    # Logging / checkpoints
    parser.add_argument("--project", type=str, default="tilted-vae-myzus",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (team or username)")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="Optional W&B run tags, space separated")
    parser.add_argument("--output_dir", type=str, default="results")

    # Chemical-class classifier callback
    parser.add_argument("--cls_image_metadata", type=str, default=None,
                        help="JSON metadata file for the classifier callback "
                             "(compounds -> plates -> image paths). If not set, "
                             "the classifier callback is disabled.")
    parser.add_argument("--cls_label_metadata", type=str, default=None,
                        help="CSV/Excel with compound labels for the classifier callback")
    parser.add_argument("--cls_root_dir", type=str, default=None,
                        help="Root directory prepended to image paths in the JSON metadata")
    parser.add_argument("--cls_every_n_epochs", type=int, default=5,
                        help="Run the classifier test every N validation epochs. Default: 5")
    parser.add_argument("--cls_compound_col", type=str, default="compound",
                        help="Compound ID column in the label CSV. Default: compound")
    parser.add_argument("--cls_label_col", type=str, default="synthesis_program",
                        help="Class label column in the label CSV. Default: synthesis_program")
    parser.add_argument("--cls_subtract_control", action="store_true",
                        help="Subtract per-plate control embedding before classification")
    parser.add_argument("--cls_normalize_before_subtract", action="store_true",
                        help="L2-normalize embeddings before control subtraction (requires --cls_subtract_control)")
    parser.add_argument("--cls_filter_by_efficacy", type=float, default=0,
                        help="Keep only compounds with Efficacy >= this value")
    parser.add_argument("--cls_min_compounds_per_class", type=int, default=30,
                        help="Drop classes with fewer compounds. Default: 30")
    parser.add_argument("--cls_cb_iterations", type=int, default=300,
                        help="CatBoost iterations for the callback classifier. Default: 300")

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
    if args.model == "dino_tilted":
        model = DinoTiltedVAE(
            latent_dim=args.latent_dim,
            tau=args.tau,
        )
    elif args.model == "tilted":
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

    # Logger (Weights & Biases)
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        tags=args.tags,
        save_dir=args.output_dir,
        log_model=False,
    )
    wandb_logger.log_hyperparams(vars(args))

    # Callbacks
    ckpt_dir = os.path.join(
        args.output_dir, "checkpoints", f"{args.model}-latent{args.latent_dim}-kld{args.kld_weight}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=args.model + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    # Optional: chemical-class classifier callback
    if args.cls_image_metadata and args.cls_label_metadata and args.cls_root_dir:
        cls_callback = ChemicalClassClassifierCallback(
            image_metadata_json=args.cls_image_metadata,
            label_metadata_csv=args.cls_label_metadata,
            root_dir=args.cls_root_dir,
            eval_every_n_epochs=args.cls_every_n_epochs,
            compound_col=args.cls_compound_col,
            label_col=args.cls_label_col,
            subtract_control=args.cls_subtract_control,
            normalize_before_subtract=args.cls_normalize_before_subtract,
            filter_by_efficacy=args.cls_filter_by_efficacy,
            min_compounds_per_class=args.cls_min_compounds_per_class,
            img_size=args.img_size,
            in_channels=args.in_channels,
            batch_size=args.batch_size,
            cb_iterations=args.cls_cb_iterations,
            seed=args.seed,
            output_dir=args.output_dir,
            ckpt_subdir=f"{args.model}-latent{args.latent_dim}-kld{args.kld_weight}",
        )
        callbacks.append(cls_callback)
        print(f"[ClassifierCallback] Enabled — evaluating every {args.cls_every_n_epochs} epochs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=args.deterministic,
    )

    trainer.fit(experiment, datamodule=datamodule)


if __name__ == "__main__":
    main()
