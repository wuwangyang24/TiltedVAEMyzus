import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Models import VAE
from dataset import VAEDataModule
from experiment import VAEExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Convolutional VAE with PyTorch Lightning + W&B")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the image dataset (ImageFolder layout)")
    parser.add_argument("--img_size", type=int, default=96, help="Square image size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Model
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)

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
    parser.add_argument("--precision", type=str, default="32-true")
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

    # Data
    datamodule = VAEDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # Model
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
        log_model=True,
    )
    wandb_logger.log_hyperparams(vars(args))

    # Callbacks
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="vae-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(experiment, datamodule=datamodule)


if __name__ == "__main__":
    main()
