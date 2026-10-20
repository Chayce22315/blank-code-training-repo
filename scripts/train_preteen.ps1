# Continue blank-child v1 on coding-focused instruction data.
# Run from the repository root in PowerShell. Keeps the original checkpoint untouched.

param(
    [ValidateRange(100, 1000000)]
    [int]$Steps = 2000,
    [ValidateRange(100, 100000)]
    [int]$CodeFeedbackSamples = 2000,
    [ValidateRange(100, 100000)]
    [int]$OpenCodeGeneticSamples = 1500,
    [ValidateRange(100, 100000)]
    [int]$CodeAlpacaSamples = 1500,
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$RunRoot = Join-Path $env:USERPROFILE "Downloads\blank-training-run"
$StageData = Join-Path $RunRoot "preteen-data"
$ParentCheckpoint = Join-Path $RunRoot "checkpoints\blank\latest.pt"
$Output = Join-Path $RunRoot "checkpoints\blank-preteen"

if (-not (Test-Path (Join-Path $RepoRoot "blank_model\train.py"))) {
    throw "Run this script from the root of blank-code-training-repo."
}
if (-not (Test-Path $ParentCheckpoint)) {
    throw "Could not find blank-child v1 checkpoint: $ParentCheckpoint. Finish the 5,000-step run first."
}

New-Item -ItemType Directory -Force -Path $StageData | Out-Null
Write-Host "Parent checkpoint: $ParentCheckpoint"
Write-Host "Stage data: $StageData"
Write-Host "New checkpoint output: $Output"
Write-Host "The original blank-child checkpoint will not be overwritten."
Write-Host "This is coding-focused continued next-token training, not full supervised fine-tuning."

python -m pip install -r requirements-data.txt
if ($LASTEXITCODE -ne 0) { throw "Installing Hugging Face dataset dependencies failed." }

$jobs = @(
    @{ Name = "codefeedback"; Samples = $CodeFeedbackSamples },
    @{ Name = "opencodegenetic"; Samples = $OpenCodeGeneticSamples },
    @{ Name = "codealpaca"; Samples = $CodeAlpacaSamples }
)

foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "Preparing $($job.Name): $($job.Samples) samples"
    python scripts/prepare_datasets.py --dataset $job.Name --max-samples $job.Samples --output-dir $StageData --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed for '$($job.Name)'. Check the dataset card and retry."
    }
}

$TrainPath = Join-Path $StageData "train"
$ValidationPath = Join-Path $StageData "validation"
if (-not (Test-Path $TrainPath) -or -not (Get-ChildItem $TrainPath -Filter "*.txt" -File)) {
    throw "No coding-stage training text files were created."
}
if (-not (Test-Path $ValidationPath) -or -not (Get-ChildItem $ValidationPath -Filter "*.txt" -File)) {
    throw "No coding-stage validation text files were created."
}

Write-Host ""
Write-Host "Starting the pre-teen stage for $Steps additional optimizer steps."
python -m blank_model.train --config blank --data $TrainPath --validation-data $ValidationPath --steps $Steps --seed $Seed --resume $ParentCheckpoint --output $Output
if ($LASTEXITCODE -ne 0) { throw "Continued training failed. Review the traceback above." }

Write-Host ""
Write-Host "Pre-teen stage finished."
Write-Host "New checkpoint: $(Join-Path $Output 'latest.pt')"
Write-Host "Stage metadata: $(Join-Path $Output 'run.json')"
Write-Host "Original checkpoint remains at: $ParentCheckpoint"
