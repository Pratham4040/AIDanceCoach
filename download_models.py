#!/usr/bin/env python3
"""
download_models.py
==================
Downloads required pre-trained models from OpenMMLab for the AI Dance Coach pipeline.

Models Downloaded:
  1. RTMO-L (RTMPose) - High-accuracy pose estimation
  2. YOLOX-L - Person detection for pose estimation

The config files are retrieved from the official MMPose/MMDet GitHub repositories
or can be auto-loaded from the model registry.

Run this script ONCE before first use:
  python download_models.py
"""

import os
import sys
from pathlib import Path

def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL to output_path."""
    try:
        import urllib.request
        print(f"  Downloading: {Path(url).name}")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def fetch_config_from_github(repo: str, config_path: str, output_path: str) -> bool:
    """Fetch config file directly from GitHub's model zoo metadata."""
    try:
        import urllib.request
        import json
        
        # Try alternative GitHub raw URL with proper branch
        urls = [
            f"https://raw.githubusercontent.com/{repo}/master/{config_path}",
            f"https://raw.githubusercontent.com/{repo}/dev-1.x/{config_path}",
            f"https://raw.githubusercontent.com/{repo}/main/{config_path}",
        ]
        
        for url in urls:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, output_path)
                print(f"  ✓ Downloaded from {url.split('github.com/')[1].split('/raw')[0]}")
                return True
            except:
                continue
        
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    """Download all required models."""
    print("\n" + "="*80)
    print("  AI Dance Coach - Model Downloader")
    print("="*80 + "\n")
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # CHECKPOINTS (required)
    # ========================================================================
    print("📦 DOWNLOADING MODEL CHECKPOINTS (Required)\n")
    
    checkpoints = {
        "RTMO-L Checkpoint": {
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
            "path": models_dir / "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
        },
        "YOLOX-L Detector": {
            "url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
            "path": models_dir / "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        },
    }
    
    checkpoint_failed = []
    for name, info in checkpoints.items():
        output_path = info["path"]
        
        if output_path.exists():
            print(f"✓ {name}")
            continue
        
        print(f"\n⏳ {name}...")
        if not download_file(info["url"], str(output_path)):
            checkpoint_failed.append(name)
    
    # ========================================================================
    # CONFIG FILES (optional, can be auto-loaded)
    # ========================================================================
    print("\n\n📋 FETCHING CONFIG FILES (Optional - auto-loaded if missing)\n")
    
    configs = {
        "RTMO-L Config": {
            "repo": "open-mmlab/mmpose",
            "path": "projects/rtmo/rtmo-l_16xb16-600e_body7-640x640.py",
            "output": models_dir / "rtmo-l_16xb16-600e_body7-640x640.py",
        },
        "YOLOX-L Config": {
            "repo": "open-mmlab/mmdetection",
            "path": "configs/yolox/yolox_l_8x8_300e_coco.py",
            "output": models_dir / "yolox_l_8x8_300e_coco.py",
        },
    }
    
    config_failed = []
    for name, info in configs.items():
        if info["output"].exists():
            print(f"✓ {name}")
            continue
        
        print(f"\n⏳ {name}...")
        if not fetch_config_from_github(info["repo"], info["path"], str(info["output"])):
            config_failed.append(name)
            print(f"   ℹ️  This is optional - configs can be auto-loaded by MMPose")
    
    # ========================================================================
    # SUMMARY & NEXT STEPS
    # ========================================================================
    print("\n" + "="*80)
    
    if checkpoint_failed:
        print(f"❌ CRITICAL: {len(checkpoint_failed)} checkpoint(s) missing:")
        for name in checkpoint_failed:
            print(f"   - {name}")
        print("\n❌ Checkpoints are REQUIRED for the pipeline to work.")
        return 1
    
    print("✅ All REQUIRED models downloaded!")
    
    if config_failed:
        print(f"\n⚠️  {len(config_failed)} config file(s) failed to download (optional):")
        for name in config_failed:
            print(f"   - {name}")
        print("\n💡 These will be auto-loaded from the MMPose/MMDet registry.")
    
    # ========================================================================
    # USAGE INSTRUCTIONS
    # ========================================================================
    rtmo_ckpt = models_dir / "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth"
    yolox_ckpt = models_dir / "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    rtmo_config = models_dir / "rtmo-l_16xb16-600e_body7-640x640.py"
    yolox_config = models_dir / "yolox_l_8x8_300e_coco.py"
    
    print("\n📝 NEXT STEPS:")
    print("\nOption A: Full paths specified (safest):")
    cmd_a = f'python main.py --url "https://www.youtube.com/watch?v=..." \\\n'
    if rtmo_config.exists() and yolox_config.exists():
        cmd_a += f'    --pose-config "{rtmo_config}" \\\n'
        cmd_a += f'    --pose-ckpt "{rtmo_ckpt}" \\\n'
        cmd_a += f'    --det-config "{yolox_config}" \\\n'
        cmd_a += f'    --det-ckpt "{yolox_ckpt}"'
    else:
        cmd_a += f'    --pose-ckpt "{rtmo_ckpt}" \\\n'
        cmd_a += f'    --det-ckpt "{yolox_ckpt}"'
    print(f"  {cmd_a}")
    
    print("\nOption B: Auto-load models (configs auto-detected):")
    print(f'  python main.py --url "https://www.youtube.com/watch?v=..." \\')
    print(f'      --pose-ckpt "{rtmo_ckpt}" \\')
    print(f'      --det-ckpt "{yolox_ckpt}"')
    
    print("\n" + "="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())

