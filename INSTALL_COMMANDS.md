# Quick Installation Commands - AI Dance Coach

## For CUDA 12.1

```powershell
# Step 1: Upgrade pip
python -m pip install --upgrade pip

# Step 2: Install PyTorch 2.1.2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install OpenMIM and MMEngine
pip install openmim
mim install mmengine==0.10.5

# Step 4: Install MMCV (pre-built wheel)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# Step 5: Install remaining dependencies
pip install -r requirements.txt

# Step 6: Verify installation
python verify_installation.py
```

## For CUDA 11.8

```powershell
# Step 1: Upgrade pip
python -m pip install --upgrade pip

# Step 2: Install PyTorch 2.1.2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install OpenMIM and MMEngine
pip install openmim
mim install mmengine==0.10.5

# Step 4: Install MMCV (pre-built wheel)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# Step 5: Install remaining dependencies
pip install -r requirements.txt

# Step 6: Verify installation
python verify_installation.py
```

## Automated Setup

Use the PowerShell script for automated installation:

```powershell
# For CUDA 12.1 (default)
.\setup_environment.ps1

# For CUDA 11.8
.\setup_environment.ps1 -CudaVersion 118
```

## Check Your CUDA Version

```powershell
# Check NVIDIA driver and CUDA version
nvidia-smi

# Check CUDA toolkit version
nvcc --version
```

## If You Need Python 3.9

```powershell
# Create new environment with Python 3.9
py -3.9 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Verify
python --version
```
