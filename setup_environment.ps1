# ==============================================================================
# AI Dance Coach - Automated Environment Setup Script
# ==============================================================================
param(
    [ValidateSet('121', '118')]
    [string]$CudaVersion = '121'
)

$ErrorColor = 'Red'
$SuccessColor = 'Green'
$InfoColor = 'Cyan'
$WarningColor = 'Yellow'

function Write-Step {
    param([string]$Message)
    Write-Host "`n--- $Message ---" -ForegroundColor $InfoColor
}

function Write-SuccessMsg {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor $SuccessColor
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor
}

function Write-WarningMsg {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor
}

Write-Host "============================================================================" -ForegroundColor $InfoColor
Write-Host "              AI Dance Coach - Environment Setup                            " -ForegroundColor $InfoColor
Write-Host "                                                                            " -ForegroundColor $InfoColor
Write-Host "  This script will install:                                                 " -ForegroundColor $InfoColor
Write-Host "  - PyTorch 2.1.2 with CUDA $CudaVersion support                           " -ForegroundColor $InfoColor
Write-Host "  - MMCV 2.1.0 (pre-built wheel)                                            " -ForegroundColor $InfoColor
Write-Host "  - MMPose, MMAction2, MMDetection                                          " -ForegroundColor $InfoColor
Write-Host "  - MediaPipe and other dependencies                                        " -ForegroundColor $InfoColor
Write-Host "============================================================================" -ForegroundColor $InfoColor

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-ErrorMsg "No virtual environment detected!"
    Write-Host "Please activate your virtual environment first:"
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-SuccessMsg "Virtual environment detected: $env:VIRTUAL_ENV"

# Check Python version
Write-Step "Checking Python version"
$pythonVersion = python --version 2>&1
Write-Host "Current: $pythonVersion"

if ($pythonVersion -match "Python 3\.9") {
    Write-SuccessMsg "Python 3.9 detected - Perfect!"
} elseif ($pythonVersion -match "Python 3\.(8|10|11)") {
    Write-WarningMsg "Python $($matches[0]) detected. Python 3.9 is recommended for best compatibility."
    $continue = Read-Host "Continue anyway? (yes/no)"
    if ($continue -ne 'yes') {
        Write-Host "Setup cancelled. See SETUP_GUIDE.md for instructions on using Python 3.9"
        exit 1
    }
} else {
    Write-ErrorMsg "Python version $pythonVersion may have compatibility issues."
    Write-Host "Python 3.9 is strongly recommended."
    $continue = Read-Host "Continue at your own risk? (yes/no)"
    if ($continue -ne 'yes') {
        exit 1
    }
}

# Check NVIDIA GPU
Write-Step "Checking for NVIDIA GPU"
try {
    $nvidiaOutput = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-SuccessMsg "NVIDIA GPU detected"
        Write-Host $nvidiaOutput | Select-String "CUDA Version"
    } else {
        Write-WarningMsg "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    }
} catch {
    Write-WarningMsg "Could not detect NVIDIA GPU. Installation will continue but GPU may not work."
}

# Confirm installation
Write-Host "`n"
$confirm = Read-Host "Ready to begin installation? This will take 5-15 minutes. (yes/no)"
if ($confirm -ne 'yes') {
    Write-Host "Setup cancelled."
    exit 0
}

$ErrorActionPreference = "Stop"

try {
    # Step 1: Upgrade pip
    Write-Step "Step 1/6: Upgrading pip"
    python -m pip install --upgrade pip setuptools wheel
    Write-SuccessMsg "pip upgraded"

    # Step 2: Install PyTorch
    Write-Step "Step 2/6: Installing PyTorch 2.1.2 with CUDA $CudaVersion"
    Write-Host "This may take a few minutes..."
    $torchUrl = "https://download.pytorch.org/whl/cu$CudaVersion"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url $torchUrl
    Write-SuccessMsg "PyTorch installed"

    # Verify PyTorch
    Write-Host "Verifying PyTorch installation..."
    $torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" 2>&1
    Write-Host $torchCheck
    
    if ($torchCheck -match "CUDA Available: True") {
        Write-SuccessMsg "PyTorch with CUDA support verified!"
    } elseif ($torchCheck -match "CUDA Available: False") {
        Write-WarningMsg "PyTorch installed but CUDA is not available. GPU acceleration will not work."
    }

    # Step 3: Install OpenMIM
    Write-Step "Step 3/6: Installing OpenMIM"
    pip install -U openmim
    Write-SuccessMsg "OpenMIM installed"

    # Step 4: Install MMEngine
    Write-Step "Step 4/6: Installing MMEngine 0.10.5"
    mim install mmengine==0.10.5
    Write-SuccessMsg "MMEngine installed"

    # Step 5: Install MMCV
    Write-Step "Step 5/6: Installing MMCV 2.1.0 (pre-built wheel)"
    Write-Host "This is the critical step - downloading pre-built wheel..."
    $mmcvUrl = "https://download.openmmlab.com/mmcv/dist/cu$CudaVersion/torch2.1.0/index.html"
    pip install mmcv==2.1.0 -f $mmcvUrl
    Write-SuccessMsg "MMCV installed"

    # Verify MMCV
    Write-Host "Verifying MMCV installation..."
    $mmcvCheck = python -c "import mmcv; print(f'MMCV {mmcv.__version__}')" 2>&1
    Write-Host $mmcvCheck
    Write-SuccessMsg "MMCV verified!"

    # Step 6: Install remaining dependencies
    Write-Step "Step 6/6: Installing remaining dependencies from requirements.txt"
    Write-Host "This will install MMPose, MMAction2, MediaPipe, and other packages..."
    pip install -r requirements.txt
    Write-SuccessMsg "All dependencies installed"

    Write-Step "Final Verification"
    
    $packages = @(
        @{Name="torch"; Import="import torch; torch.__version__"},
        @{Name="mmcv"; Import="import mmcv; mmcv.__version__"},
        @{Name="mmengine"; Import="import mmengine; mmengine.__version__"},
        @{Name="mmpose"; Import="import mmpose; mmpose.__version__"},
        @{Name="mmaction2"; Import="import mmaction; mmaction.__version__"},
        @{Name="mmdet"; Import="import mmdet; mmdet.__version__"},
        @{Name="mediapipe"; Import="import mediapipe; mediapipe.__version__"},
        @{Name="cv2"; Import="import cv2; cv2.__version__"}
    )

    $allSuccess = $true
    foreach ($pkg in $packages) {
        try {
            $version = python -c $pkg.Import 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-SuccessMsg "$($pkg.Name): $version"
            } else {
                Write-ErrorMsg "$($pkg.Name): Failed to import"
                $allSuccess = $false
            }
        } catch {
            Write-ErrorMsg "$($pkg.Name): Failed to import"
            $allSuccess = $false
        }
    }

    if ($allSuccess) {
        Write-Host "`n============================================================================" -ForegroundColor $SuccessColor
        Write-Host "                    INSTALLATION COMPLETE!                                  " -ForegroundColor $SuccessColor
        Write-Host "                                                                            " -ForegroundColor $SuccessColor
        Write-Host "  Your AI Dance Coach environment is ready to use.                          " -ForegroundColor $SuccessColor
        Write-Host "  You can now run: python main.py                                           " -ForegroundColor $SuccessColor
        Write-Host "============================================================================`n" -ForegroundColor $SuccessColor
    } else {
        Write-WarningMsg "Some packages failed verification. Check errors above."
    }

} catch {
    Write-ErrorMsg "Installation failed: $_"
    Write-Host "`nTroubleshooting tips:"
    Write-Host "1. Check your internet connection"
    Write-Host "2. Make sure you have enough disk space (5GB+ free)"
    Write-Host "3. Verify CUDA toolkit is correctly installed (nvcc --version)"
    Write-Host "4. See SETUP_GUIDE.md for detailed troubleshooting"
    exit 1
}

# Optional: Check GPU
Write-Host "`n"
$testGpu = Read-Host "Would you like to test GPU detection? (yes/no)"
if ($testGpu -eq 'yes') {
    Write-Step "GPU Detection Test"
    python -c @"
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    # Quick GPU test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print('[OK] GPU computation test passed!')
else:
    print('No CUDA GPU detected')
"@
}

Write-Host "`nSetup complete! Happy coding!" -ForegroundColor $InfoColor
