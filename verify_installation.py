"""
AI Dance Coach - Installation Verification Script
Run this after installation to verify all components are working correctly.
"""

import sys
from typing import Tuple, List

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_package(name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and return version.
    
    Args:
        name: Display name of the package
        import_name: Actual module name (if different from display name)
    
    Returns:
        Tuple of (success, version_or_error)
    """
    if import_name is None:
        import_name = name.lower().replace('-', '_')
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"

def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            return True, f"CUDA {cuda_version}, {device_count} GPU(s): {device_name}"
        else:
            return False, "CUDA not available (CPU only)"
    except Exception as e:
        return False, f"Error: {e}"

def test_gpu_computation() -> Tuple[bool, str]:
    """Test basic GPU computation."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No CUDA GPU available"
        
        # Simple matrix multiplication
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        return True, "GPU computation test passed"
    except Exception as e:
        return False, f"GPU test failed: {e}"

def check_mmcv_ops() -> Tuple[bool, str]:
    """Check if MMCV custom CUDA operators are available."""
    try:
        import mmcv
        from mmcv.ops import get_compiling_cuda_version, get_compiler_version
        
        cuda_version = get_compiling_cuda_version()
        compiler = get_compiler_version()
        
        return True, f"CUDA {cuda_version}, Compiler: {compiler}"
    except ImportError:
        return False, "MMCV not installed"
    except Exception as e:
        return False, f"MMCV ops not available: {e}"

def main():
    print_header("AI Dance Coach - Installation Verification")
    
    print("\n📋 System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Core packages to check
    packages = [
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
    ]
    
    # OpenMMLab packages
    openmmlab_packages = [
        ("MMCV", "mmcv"),
        ("MMEngine", "mmengine"),
        ("MMPose", "mmpose"),
        ("MMAction2", "mmaction"),
        ("MMDetection", "mmdet"),
    ]
    
    # Computer vision packages
    cv_packages = [
        ("MediaPipe", "mediapipe"),
        ("OpenCV", "cv2"),
    ]
    
    # Audio/video packages
    media_packages = [
        ("Librosa", "librosa"),
        ("SoundFile", "soundfile"),
    ]
    
    # Utility packages
    util_packages = [
        ("ONNX Runtime", "onnxruntime"),
        ("tqdm", "tqdm"),
        ("PyYAML", "yaml"),
    ]
    
    all_success = True
    results: List[Tuple[str, bool, str]] = []
    
    # Check all packages
    print_header("Core Packages")
    for name, import_name in packages:
        success, info = check_package(name, import_name)
        results.append((name, success, info))
        status = "✓" if success else "✗"
        print(f"{status} {name:20s} {info}")
        if not success:
            all_success = False
    
    # Check OpenMMLab packages
    print_header("OpenMMLab Ecosystem")
    for name, import_name in openmmlab_packages:
        success, info = check_package(name, import_name)
        results.append((name, success, info))
        status = "✓" if success else "✗"
        print(f"{status} {name:20s} {info}")
        if not success:
            all_success = False
    
    # Check computer vision packages
    print_header("Computer Vision Libraries")
    for name, import_name in cv_packages:
        success, info = check_package(name, import_name)
        results.append((name, success, info))
        status = "✓" if success else "✗"
        print(f"{status} {name:20s} {info}")
        if not success:
            all_success = False
    
    # Check media packages
    print_header("Media Processing Libraries")
    for name, import_name in media_packages:
        success, info = check_package(name, import_name)
        results.append((name, success, info))
        status = "✓" if success else "✗"
        print(f"{status} {name:20s} {info}")
        if not success:
            all_success = False
    
    # Check utilities
    print_header("Utility Libraries")
    for name, import_name in util_packages:
        success, info = check_package(name, import_name)
        results.append((name, success, info))
        status = "✓" if success else "✗"
        print(f"{status} {name:20s} {info}")
        # Don't fail on optional utilities
    
    # CUDA checks
    print_header("CUDA & GPU Status")
    
    cuda_success, cuda_info = check_torch_cuda()
    status = "✓" if cuda_success else "⚠"
    print(f"{status} PyTorch CUDA:      {cuda_info}")
    
    if cuda_success:
        compute_success, compute_info = test_gpu_computation()
        status = "✓" if compute_success else "✗"
        print(f"{status} GPU Computation:   {compute_info}")
        
        mmcv_ops_success, mmcv_ops_info = check_mmcv_ops()
        status = "✓" if mmcv_ops_success else "⚠"
        print(f"{status} MMCV CUDA Ops:     {mmcv_ops_info}")
    else:
        print("⚠ Warning: No CUDA GPU detected. The system will run on CPU only.")
        print("  This will be significantly slower for pose estimation and video processing.")
    
    # Final summary
    print_header("Summary")
    
    failed_packages = [name for name, success, _ in results if not success]
    
    if all_success and cuda_success:
        print("\n🎉 SUCCESS! All packages installed correctly with GPU support.")
        print("   Your AI Dance Coach environment is ready to use!")
        print("\n   Next steps:")
        print("   - Run: python main.py --help")
        print("   - See README.md for usage examples")
    elif all_success and not cuda_success:
        print("\n⚠ All packages installed, but GPU is not available.")
        print("  The system will work but will be slower on CPU.")
        print("\n  To enable GPU:")
        print("  1. Install NVIDIA drivers: nvidia-smi should work")
        print("  2. Install CUDA Toolkit (12.1 or 11.8)")
        print("  3. Reinstall PyTorch with CUDA support")
    else:
        print(f"\n✗ Installation incomplete. {len(failed_packages)} package(s) failed:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\n  Troubleshooting:")
        print("  - See SETUP_GUIDE.md for detailed installation instructions")
        print("  - Make sure you installed PyTorch and MMCV before running pip install -r requirements.txt")
        print("  - Try running: pip install -r requirements.txt again")
        return 1
    
    # Version matrix validation
    print_header("Version Matrix Validation")
    
    recommended = {
        "Python": ("3.9", sys.version.split()[0]),
        "PyTorch": ("2.1.2", check_package("torch", "torch")[1] if check_package("torch", "torch")[0] else "N/A"),
        "MMCV": ("2.1.0", check_package("mmcv", "mmcv")[1] if check_package("mmcv", "mmcv")[0] else "N/A"),
        "MMEngine": ("0.10.5", check_package("mmengine", "mmengine")[1] if check_package("mmengine", "mmengine")[0] else "N/A"),
        "MMPose": ("1.3.2", check_package("mmpose", "mmpose")[1] if check_package("mmpose", "mmpose")[0] else "N/A"),
    }
    
    for pkg, (recommended_ver, actual_ver) in recommended.items():
        if actual_ver.startswith(recommended_ver):
            print(f"✓ {pkg:15s} {actual_ver:15s} (recommended: {recommended_ver})")
        else:
            print(f"⚠ {pkg:15s} {actual_ver:15s} (recommended: {recommended_ver})")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    exit(main())
