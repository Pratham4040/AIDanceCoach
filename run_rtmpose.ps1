param(
    [string]$VideoPath = "C:\MACHINE_LEARNING\DL\testvideo\827ee98cea73a9e430af191644d20051_720w.mp4",
    [ValidateSet('cpu', 'cuda', 'auto')]
    [string]$Device = 'auto',
    [int]$Camera = 0,
    [ValidateSet('auto', 'default', 'msmf', 'dshow')]
    [string]$CameraBackend = 'auto',
    [ValidateSet('side-by-side', 'top-bottom')]
    [string]$Layout = 'side-by-side',
    [switch]$SaveVideo,
    [string]$VideoOutput = 'output_annotated.mp4',
    [switch]$SkipRender,
    [string]$RenderedOutputDir = 'data/rendered_videos',
    [string]$ExtractedStepsDir = 'data/extracted_steps',
    [string]$ReferenceOutputVideosDir = 'data/output_videos',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Auto-detect GPU if Device='auto'
if ($Device -eq 'auto') {
    Write-Host "Detecting GPU availability..." -ForegroundColor Cyan
    $cudaCheck = & {
        $pythonDefault = 'python'
        $pythonExes = @(
            (Join-Path $scriptDir '.venv\Scripts\python.exe'),
            (Join-Path (Split-Path $scriptDir -Parent) '.venv\Scripts\python.exe'),
            'python'
        )
        
        foreach ($py in $pythonExes) {
            $result = & $py -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $result
            }
        }
        return 'cpu'
    }
    
    $Device = $cudaCheck
    if ($Device -eq 'cuda') {
        Write-Host "[OK] GPU detected - using CUDA" -ForegroundColor Green
    } else {
        Write-Host "[OK] No GPU detected - using CPU" -ForegroundColor Yellow
    }
}
$venvActivateCandidates = @(
    (Join-Path $scriptDir '.venv\Scripts\Activate.ps1'),
    (Join-Path (Split-Path $scriptDir -Parent) '.venv\Scripts\Activate.ps1')
)

$venvPythonCandidates = @(
    (Join-Path $scriptDir '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path $scriptDir -Parent) '.venv\Scripts\python.exe')
)

$pythonExe = 'python'
foreach ($candidate in $venvPythonCandidates) {
    if (-not (Test-Path $candidate)) {
        continue
    }

    & $candidate -c "import mmpose, mmdet" *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = $candidate
        break
    }

    if ($pythonExe -eq 'python') {
        $pythonExe = $candidate
    }
}

$activated = $false
foreach ($candidate in $venvActivateCandidates) {
    if (Test-Path $candidate) {
        & $candidate
        $activated = $true
        break
    }
}

if (-not $activated) {
    Write-Warning 'No virtual environment activation script found. Using discovered python executable.'
}

$venvRoot = Split-Path -Parent (Split-Path -Parent $pythonExe)
$packagedPoseConfig = Join-Path $venvRoot 'Lib\site-packages\mmpose\.mim\configs\body_2d_keypoint\rtmo\body7\rtmo-l_16xb16-600e_body7-640x640.py'
$packagedDetConfig = Join-Path $venvRoot 'Lib\site-packages\mmdet\.mim\configs\yolox\yolox_l_8xb8-300e_coco.py'

$poseConfigPath = if (Test-Path $packagedPoseConfig) { $packagedPoseConfig } else { 'models\rtmo-l_16xb16-600e_body7-640x640.py' }
$detConfigPath = if (Test-Path $packagedDetConfig) { $packagedDetConfig } else { 'models\yolox_l_8x8_300e_coco.py' }

$requiredFiles = @(
    'models\rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth',
    'models\yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
)

if (-not (Test-Path $poseConfigPath)) {
    $requiredFiles += $poseConfigPath
}
if (-not (Test-Path $detConfigPath)) {
    $requiredFiles += $detConfigPath
}

$missing = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host 'Missing required model files for RTMPose:' -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }

    if (Test-Path 'download_models.py') {
        Write-Host 'Attempting automatic model download...' -ForegroundColor Cyan
        & $pythonExe download_models.py

        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Automatic model download failed.' -ForegroundColor Red
            Write-Host 'Run this manually and retry: python download_models.py' -ForegroundColor Yellow
            exit $LASTEXITCODE
        }

        $missing = @()
        foreach ($file in $requiredFiles) {
            if (-not (Test-Path $file)) {
                $missing += $file
            }
        }

        if ($missing.Count -gt 0) {
            Write-Host 'Some required model files are still missing after download:' -ForegroundColor Red
            $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
            exit 1
        }
    }
    else {
        Write-Host 'download_models.py not found in project root.' -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $VideoPath)) {
    Write-Host "Video file not found: $VideoPath" -ForegroundColor Red
    exit 1
}

$cmdArgs = @(
    'main.py',
    '--url', $VideoPath,
    '--camera', $Camera,
    '--camera-backend', $CameraBackend,
    '--device', $Device,
    '--layout', $Layout,
    '--rendered-output-dir', $RenderedOutputDir,
    '--extracted-steps-dir', $ExtractedStepsDir,
    '--reference-output-videos-dir', $ReferenceOutputVideosDir,
    '--pose-config', $poseConfigPath,
    '--pose-ckpt', 'models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth',
    '--det-config', $detConfigPath,
    '--det-ckpt', 'models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
)

if ($SkipRender) {
    $cmdArgs += '--skip-render'
}

if ($SaveVideo) {
    $cmdArgs += @('--save-video', '--video-output', $VideoOutput)
}

if ($DryRun) {
    Write-Host 'Dry run command:' -ForegroundColor Cyan
    Write-Host ($pythonExe + ' ' + ($cmdArgs -join ' '))
    exit 0
}

& $pythonExe @cmdArgs
exit $LASTEXITCODE
