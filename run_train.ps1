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
    [string]$DataDir     = "../DATA/",
    [int]   $ImgSize     = 96,
    [int]   $BatchSize   = 512,
    [int]   $NumWorkers  = 8,
    [double]$ValSplit    = 0.05,
    [int]   $MaxValSamples = 500000,
    [int]   $LatentDim   = 128,
    [double]$Lr          = 1e-4,
    [int]   $Epochs      = 100,
    [string]$Precision   = "16-mixed",
    [string]$Project     = "tiltedvae-myzus",
    [string]$Entity      = "wangyang-wu-bayer",
    [string]$RunName     = "vae-run",
    [string]$OutputDir   = "results",
    # SECURITY: this key is hard-coded below. Do NOT commit this file to git
    # once you fill it in (add run_train.ps1 to .gitignore).
    [string]$WandbApiKey = "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"
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script.
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$IndexCache = Join-Path $ScriptDir "cache\image_index.npy"

# Set the W&B API key for this run if provided via the parameter.
if ($WandbApiKey) {
    $env:WANDB_API_KEY = $WandbApiKey
}

if (-not $env:WANDB_API_KEY) {
    Write-Warning "WANDB_API_KEY is not set. Pass -WandbApiKey '<key>', set `$env:WANDB_API_KEY, or run `wandb login` first."
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
    --anneal_k        3.5e-5 `
    --anneal_x0       200000 `
    --au_threshold    0.01 `
    --project         "$Project" `
    --entity          "$Entity" `
    --run_name        "$RunName" `
    --output_dir      "$OutputDir"
