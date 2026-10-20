# Train the blank model from random initialization using small Hugging Face samples.
# Run from the repository root in Windows PowerShell.
# All datasets and checkpoints are written under Downloads, not into the Git repo.

param(
    [ValidateSet("blank", "bigbang", "supernova")]
    [string]$Model = "blank",
    [ValidateRange(1, 1000000)]
    [int]$Steps = 2000,
    [ValidateRange(100, 100000)]
    [int]$TinyStoriesSamples = 3000,
    [ValidateRange(100, 100000)]
    [int]$FineWebSamples = 1000,
    [ValidateRange(100, 100000)]
    [int]$CodeFeedbackSamples = 1500,
    [ValidateRange(100, 100000)]
    [int]$CodeAlpacaSamples = 1000,
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$RunRoot = Join-Path $env:USERPROFILE "Downloads\blank-training-run"
$DataRoot = Join-Path $RunRoot "data"
$CheckpointRoot = Join-Path $RunRoot "checkpoints\$Model"

if (-not (Test-Path (Join-Path $RepoRoot "blank_model\train.py"))) {
    throw "Run this script from the root of blank-code-training-repo."
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
Write-Host "Run folder: $RunRoot"
Write-Host "Model: $Model | steps: $Steps"
Write-Host "The model is initialized from random weights; no pretrained weights are loaded."

python -m pip install -r requirements-data.txt
if ($LASTEXITCODE -ne 0) { throw "Installing Hugging Face dataset dependencies failed." }

$jobs = @(
    @{ Name = "tinystories"; Samples = $TinyStoriesSamples },
    @{ Name = "fineweb-edu"; Samples = $FineWebSamples },
    @{ Name = "codefeedback"; Samples = $CodeFeedbackSamples },
    @{ Name = "codealpaca"; Samples = $CodeAlpacaSamples }
)

foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "Preparing $($job.Name): $($job.Samples) samples"
    python scripts/prepare_datasets.py --dataset $job.Name --max-samples $job.Samples --output-dir $DataRoot --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed for '$($job.Name)'. Check its Hugging Face dataset card and try again."
    }
}

$TrainPath = Join-Path $DataRoot "train"
$ValidationPath = Join-Path $DataRoot "validation"
if (-not (Test-Path $TrainPath) -or -not (Get-ChildItem $TrainPath -Filter "*.txt" -File)) {
    throw "No training text files were created."
}

Write-Host ""
Write-Host "Starting from-scratch training. Keep this window open."
Write-Host "Training data: $TrainPath"
Write-Host "Validation data: $ValidationPath"
Write-Host "Checkpoints: $CheckpointRoot"

python -m blank_model.train --config $Model --data $TrainPath --validation-data $ValidationPath --steps $Steps --seed $Seed --output $CheckpointRoot
if ($LASTEXITCODE -ne 0) { throw "Training failed. Review the traceback above before retrying." }

Write-Host ""
Write-Host "Training process finished."
Write-Host "Checkpoint: $(Join-Path $CheckpointRoot 'latest.pt')"
Write-Host "Run metadata: $(Join-Path $CheckpointRoot 'run.json')"
Write-Host "This small experiment is a starting point, not proof of coding ability. Inspect validation loss and generated samples."
