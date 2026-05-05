param(
    [switch]$Test,
    [string]$PythonExe = "",
    [string]$VisualFeatDir = ".\fis-v\swintx_avg_fps25_clip32",
    [string]$TrainLabel = ".\fis-v\train.txt",
    [string]$TestLabel = ".\fis-v\test.txt",
    [string]$MmFeatDir = "..\PAMFN_Reproduce\data\features",
    [string]$PamfnCkptDir = "..\PAMFN_Reproduce\pretrained_models\feats1",
    [string]$PamfnJointCkpt = "..\PAMFN_Reproduce\pretrained_models\feats1\PCS_multimodal.pth",
    [string]$OutCkpt = ".\ckpt\joint_fisv_best.pkl",
    [string]$ErrorReport = ".\logs\joint_fisv_error_report.txt",
    [string]$LogDir = ".\logs\joint_fisv",
    [string]$Device = "cuda",
    [int]$Epoch = 160,
    [int]$WarmupEpochs = 40,
    [int]$Batch = 16,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"

function Resolve-PathSafe {
    param([string]$PathValue)
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $PathValue))
}

function Assert-Exists {
    param(
        [string]$PathValue,
        [string]$Description
    )
    if (-not (Test-Path $PathValue)) {
        throw "$Description not found: $PathValue"
    }
}

function Resolve-PythonExe {
    param([string]$PythonValue)

    if ($PythonValue -and $PythonValue.Trim() -ne "") {
        return $PythonValue
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return $condaPython
        }
    }

    return "python"
}

$visualFeatDir = Resolve-PathSafe $VisualFeatDir
$trainLabel = Resolve-PathSafe $TrainLabel
$testLabel = Resolve-PathSafe $TestLabel
$mmFeatDir = Resolve-PathSafe $MmFeatDir
$pamfnCkptDir = Resolve-PathSafe $PamfnCkptDir
$pamfnJointCkpt = Resolve-PathSafe $PamfnJointCkpt
$outCkpt = Resolve-PathSafe $OutCkpt
$errorReport = Resolve-PathSafe $ErrorReport
$logDir = Resolve-PathSafe $LogDir
$pythonExe = Resolve-PythonExe $PythonExe

Assert-Exists $visualFeatDir "Visual feature directory"
Assert-Exists $trainLabel "Train label"
Assert-Exists $testLabel "Test label"
Assert-Exists $mmFeatDir "PAMFN multimodal feature directory"
Assert-Exists $pamfnCkptDir "PAMFN checkpoint directory"

$requiredFiles = @(
    (Join-Path $mmFeatDir "FISV_rgb_VST.npy"),
    (Join-Path $mmFeatDir "FISV_flow_I3D.npy"),
    (Join-Path $mmFeatDir "FISV_audio_AST.npy"),
    (Join-Path $pamfnCkptDir "PCS_rgb_VST.pth"),
    (Join-Path $pamfnCkptDir "PCS_flow_I3D.pth"),
    (Join-Path $pamfnCkptDir "PCS_audio_AST.pth")
)

foreach ($file in $requiredFiles) {
    Assert-Exists $file "Required asset"
}

$commonArgs = @(
    "joint_fisv_main.py",
    "--visual-feat-dir", $visualFeatDir,
    "--mm-feat-dir", $mmFeatDir,
    "--joint-train-label", $trainLabel,
    "--joint-test-label", $testLabel,
    "--pamfn-ckpt-dir", $pamfnCkptDir,
    "--pamfn-dataset-name", "PCS",
    "--ckpt-path", $outCkpt,
    "--error-report-path", $errorReport,
    "--log-dir", $logDir,
    "--device", $Device,
    "--batch", $Batch,
    "--num-workers", $NumWorkers
)

if (Test-Path $pamfnJointCkpt) {
    $commonArgs += @("--pamfn-ckpt", $pamfnJointCkpt)
} else {
    Write-Host "PAMFN multimodal checkpoint not found, warm-starting from single-modality checkpoints only:" -ForegroundColor Yellow
    Write-Host "  $pamfnJointCkpt" -ForegroundColor Yellow
}

if ($Test) {
    $argsList = $commonArgs + @("--test", "--ckpt", $outCkpt)
} else {
    $argsList = $commonArgs + @(
        "--epoch", $Epoch,
        "--warmup-epochs", $WarmupEpochs
    )
}

Write-Host "Running:" -ForegroundColor Cyan
Write-Host "Python executable: $pythonExe" -ForegroundColor Cyan
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "Conda env: $($env:CONDA_DEFAULT_ENV)" -ForegroundColor Cyan
}
Write-Host "$pythonExe $($argsList -join ' ')" -ForegroundColor Cyan
& $pythonExe @argsList
