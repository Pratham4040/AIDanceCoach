# AI Dance Coach - Environment Setup Guide

## ⚠️ Critical Notice: Installation Order Matters!

This project uses OpenMMLab ecosystem (MMCV, MMPose, MMAction2) with MediaPipe. **You CANNOT simply run `pip install -r requirements.txt`** because MMCV requires custom CUDA operators that must be compiled or downloaded as pre-built wheels matching your exact PyTorch and CUDA versions.

## Prerequisites

- **Python 3.9** (Recommended - best compatibility)
- **NVIDIA GPU** with CUDA 12.1 or 11.8
- **CUDA Toolkit** installed on your system
- **Windows 10/11** (adjust commands for Linux/Mac)

## Current Environment Status

⚠️ **Your current Python version is 3.12.2**, but Python 3.9 is recommended for maximum compatibility with pre-built wheels.

## Option A: Recreate Environment with Python 3.9 (Recommended)

### Step 1: Install Python 3.9

Download Python 3.9 from [python.org](https://www.python.org/downloads/) or use `pyenv`:

```powershell
# If you have pyenv-win installed:
pyenv install 3.9.13
pyenv local 3.9.13
```

### Step 2: Create New Virtual Environment

```powershell
# Navigate to project directory
cd C:\MACHINE_LEARNING\DL\AIDanceCoach

# Remove old environment
Remove-Item -Recurse -Force .venv

# Create new environment with Python 3.9
py -3.9 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

### Step 3: Verify Python Version

```powershell
python --version
# Should output: Python 3.9.x
```

### Step 4: Install PyTorch with CUDA Support

**For CUDA 12.1:**
```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install OpenMIM and MMEngine

```powershell
pip install openmim
mim install mmengine==0.10.5
```

### Step 6: Install MMCV (Pre-built Wheel)

**For CUDA 12.1:**
```powershell
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

**For CUDA 11.8:**
```powershell
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

### Step 7: Install Remaining Dependencies

```powershell
pip install -r requirements.txt
```

### Step 8: Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"
python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"
```

## Option B: Continue with Python 3.12 (Not Recommended)

If you absolutely must use Python 3.12, note that:
- You'll need to build MMCV from source (no pre-built wheels)
- Compilation takes 20-40 minutes and may fail
- Many compatibility issues with OpenMMLab

To try anyway:
```powershell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install openmim
mim install mmengine==0.10.5
pip install mmcv==2.1.0  # This will try to build from source
pip install -r requirements.txt
```

## Troubleshooting

### Error: "CUDA not available" or torch.cuda.is_available() returns False

1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Check CUDA toolkit version: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version

### Error: "ImportError: DLL load failed" (Windows)

Install Visual C++ Redistributable:
- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Error: MMCV compilation fails

- Use pre-built wheel with correct CUDA version (see Step 6)
- Ensure PyTorch is installed BEFORE MMCV
- Check that CUDA toolkit matches your PyTorch CUDA version

### Error: "No module named 'mmcv._ext'"

MMCV was not properly installed. Uninstall and reinstall:
```powershell
pip uninstall mmcv mmcv-full -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

## Version Matrix Summary

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.9.x | Sweet spot for compatibility |
| PyTorch | 2.1.2 | Stable with CUDA 12.1/11.8 |
| torchvision | 0.16.2 | Matches PyTorch 2.1.2 |
| MMCV | 2.1.0 | Do NOT use 2.2.0 |
| MMEngine | 0.10.5 | |
| MMPose | 1.3.2 | |
| MMAction2 | 1.2.0 | |
| MMDetection | 3.3.0 | Optional but recommended |
| MediaPipe | >=0.10.9 | |
| NumPy | <2.0.0 | NumPy 2.x breaks C++ modules |

## Quick Setup Script

We've provided an automated setup script: `setup_environment.ps1`

Run it with:
```powershell
.\setup_environment.ps1
```

This will guide you through the installation process interactively.
