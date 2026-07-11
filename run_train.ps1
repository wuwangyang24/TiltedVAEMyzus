# Training launch script for the Convolutional VAE (PowerShell).
#
# Usage:
#   1. Set your data directory below (or pass -DataDir).
#   2. Provide your W&B key once:  $env:WANDB_API_KEY = "xxxx"   (or run `wandb login`)
#   3. Run:  .\run_train.ps1
#
# Override any default from the command line, e.g.:
#   .\run_train.ps1 -DataDir "D:\images" -BatchSize 128 -Epochs 50

param(
    [string]$DataDir     = "C:\Users\gosnn\OneDrive - Bayer\Personal Data\TiltedVAEMyzus\data",
    [int]   $ImgSize     = 96,
    [int]   $BatchSize   = 64,
    [int]   $NumWorkers  = 8,
    [double]$ValSplit    = 0.05,
    [int]   $MaxValSamples = 20000,
    [int]   $LatentDim   = 128,
    [double]$Lr          = 1e-3,
    [int]   $Epochs      = 100,
    [string]$Precision   = "16-mixed",
    [string]$Project     = "tilted-vae-myzus",
    [string]$Entity      = "your-wandb-entity",
    [string]$RunName     = "vae-run",
    [string]$OutputDir   = "results"
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script.
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$IndexCache = Join-Path $ScriptDir "cache\image_index.npy"

if (-not $env:WANDB_API_KEY) {
    Write-Warning "WANDB_API_KEY is not set. Set it with `$env:WANDB_API_KEY = '<key>'` or run `wandb login` first."
}

python "$ScriptDir\train.py" `
    --data_dir        "$DataDir" `
    --img_size        $ImgSize `
    --batch_size      $BatchSize `
    --num_workers     $NumWorkers `
    --val_split       $ValSplit `
    --max_val_samples $MaxValSamples `
    --index_cache     "$IndexCache" `
    --latent_dim      $LatentDim `
    --lr              $Lr `
    --epochs          $Epochs `
    --precision       $Precision `
    --anneal_kld `
    --anneal_end      1.0 `
    --anneal_k        0.0025 `
    --anneal_x0       2500 `
    --au_threshold    0.01 `
    --project         "$Project" `
    --entity          "$Entity" `
    --run_name        "$RunName" `
    --output_dir      "$OutputDir"
