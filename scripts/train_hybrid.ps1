# Hybrid stage: coding + arithmetic + story continuation.
# Run from the repository root in PowerShell. Keeps the pre-teen checkpoint untouched.

param(
    [ValidateRange(100, 1000000)]
    [int]$Steps = 2000,
    [ValidateRange(100, 100000)]
    [int]$CodingSamples = 2000,
    [ValidateRange(100, 100000)]
    [int]$StorySamples = 1500,
    [ValidateRange(100, 100000)]
    [int]$MathSamples = 1500,
    [ValidateRange(0.000001, 0.01)]
    [double]$LearningRate = 0.0001,
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$RunRoot = Join-Path $env:USERPROFILE "Downloads\blank-training-run"
$StageData = Join-Path $RunRoot "hybrid-data"
$ParentCheckpoint = Join-Path $RunRoot "checkpoints\blank-preteen\latest.pt"
$Output = Join-Path $RunRoot "checkpoints\blank-hybrid"

if (-not (Test-Path (Join-Path $RepoRoot "blank_model\sft_train.py"))) {
    throw "Run this script from the updated blank-code-training-repo root."
}
if (-not (Test-Path $ParentCheckpoint)) {
    throw "Could not find the pre-teen checkpoint: $ParentCheckpoint. Finish pre-teen training first."
}

New-Item -ItemType Directory -Force -Path $StageData | Out-Null
Write-Host "Parent checkpoint: $ParentCheckpoint"
Write-Host "Hybrid data: $StageData"
Write-Host "New checkpoint: $Output"
Write-Host "The child and pre-teen checkpoints will remain untouched."
Write-Host "Task mix: coding=$CodingSamples, stories=$StorySamples, arithmetic=$MathSamples"
Write-Host "Training objective: supervised fine-tuning; prompt tokens are masked from the loss."

python -m pip install -r requirements-data.txt
if ($LASTEXITCODE -ne 0) { throw "Installing Hugging Face dataset dependencies failed." }

python scripts/prepare_hybrid_data.py --output-dir $StageData --coding-samples $CodingSamples --story-samples $StorySamples --math-samples $MathSamples --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Hybrid dataset preparation failed. Review the traceback above." }

$TrainFile = Join-Path $StageData "train.jsonl"
$ValidationFile = Join-Path $StageData "validation.jsonl"
if (-not (Test-Path $TrainFile) -or -not (Test-Path $ValidationFile)) {
    throw "Hybrid train/validation JSONL files were not created."
}

Write-Host ""
Write-Host "Starting hybrid supervised fine-tuning for $Steps additional optimizer steps."
python -m blank_model.sft_train --config blank --data $TrainFile --validation-data $ValidationFile --steps $Steps --seed $Seed --resume $ParentCheckpoint --output $Output --learning-rate $LearningRate
if ($LASTEXITCODE -ne 0) { throw "Hybrid training failed. Review the traceback above." }

Write-Host ""
Write-Host "Hybrid stage finished."
Write-Host "Checkpoint: $(Join-Path $Output 'latest.pt')"
Write-Host "Manifest: $(Join-Path $StageData 'manifest.json')"
Write-Host "Parent pre-teen checkpoint remains at: $ParentCheckpoint"
