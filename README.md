# AI Dance Coach 🕺💃

An end-to-end AI system that ingests YouTube dance tutorials, segments them into teachable steps, extracts reference poses, and compares them to a user's live webcam feed to provide **real-time corrective feedback** with color-coded visual guidance.

---

## ⚡ Quick Start for GitHub Users

**New to this project? Start here:**

1. **[FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md)** ← **👈 START HERE** - Step-by-step guide for first-time installers
2. [README.md](README.md) - This file (project overview)
3. [INSTALLATION_STATUS.md](INSTALLATION_STATUS.md) - Detailed troubleshooting

**Already installed?** → Jump to [💻 Usage](#-usage) below

---

## 🎯 Features

- **Automated Video Ingestion**: Download dance tutorials from YouTube using `yt-dlp`
- **Intelligent Segmentation**: Uses Temporal Convolutional Networks (TCN) to segment videos into discrete dance steps
- **Step Image Extraction**: Automatically saves representative frames for each dance step (`step_001.jpg`, `step_002.jpg`, ...)
- **Skeleton Overlay Videos**: Generates MP4 videos with pose skeletons visualized on the reference video
- **Dual Pose Estimation**:
  - **RTMPose/RTMO** (OpenMMLab) for high-accuracy reference pose extraction from tutorials
  - **MediaPipe BlazePose** for fast, lightweight real-time tracking of user movements
- **Temporal Alignment**: Spatial-Weighted Fast Dynamic Time Warping (S-WFDTW) for real-time motion comparison
- **Real-Time Visual Feedback**: Color-coded skeleton overlay (🟢 Green = correct, 🟡 Yellow = minor deviation, 🔴 Red = critical error)
- **Robust UI Error Handling**: Graceful degradation when display windows fail (headless environment support)
- **Hardware Optimization**: ONNX Runtime support for CPU acceleration, TensorRT for GPU edge devices

---

## 🏗️ Architecture

The system operates in **8 phases**:

### **Phase 1: Ingestion & Preprocessing**
Downloads YouTube videos and extracts frames in parallel using `ProcessPoolExecutor` to bypass Python's GIL. Applies grayscale conversion and background subtraction for cleaner pose detection.

### **Phase 2: Reference Pose Extraction**
Uses **OpenMMLab's RTMPose** (specifically RTMO for high accuracy) to extract ground-truth keypoints from the instructor's video. This happens **before** segmentation to ensure high-quality reference data for all frames.

### **Phase 3: Action Segmentation**
Extracts visual features (skeletal bone vectors) and audio features (Mel spectrogram) using `librosa`. Feeds these multi-modal features into a **Temporal Convolutional Network (TCN)** to output segmentation probability curves, identifying discrete dance steps via peak-picking using the poses extracted in Phase 2.

### **Phase 4: Spatial Normalization**
Translates all coordinates (both instructor and user) to a **root origin (hip center)** and scales them based on **torso length**. Converts raw (x, y) coordinates to **angular limb vectors** (RotJoints) for rotation-invariant comparison.

### **Phase 5: Real-Time Edge Tracking**
Uses **MediaPipe BlazePose** for fast tracking of the user's live webcam feed, detecting 33 3D landmarks in real-time.

### **Phase 6: Temporal Alignment & Scoring**
Implements **Spatial-Weighted Fast Dynamic Time Warping (S-WFDTW)** with a sliding window constraint (e.g., 10 frames) and early-termination pruning. Calculates final scores using weighted fusion of:
- **Euclidean Distance** (spatial position accuracy)
- **Cosine Similarity** (angular alignment accuracy)

### **Phase 7: Real-Time UI/UX**
Uses OpenCV to render a dynamic wireframe over the user's live feed. Limbs are color-coded based on S-WFDTW similarity scores to provide immediate visual feedback.

### **Phase 8: Hardware Optimization**
Exports models to **ONNX Runtime** for CPU acceleration and **NVIDIA TensorRT** with FP16 quantization for GPU edge devices.

---

## 🤖 Models: What, Why, and When

This project uses a **dual-model architecture** - different models optimized for different tasks. Understanding why we use each model is crucial to the system design.

### **🎯 Two-Model Strategy**

| Model | Purpose | When | Speed | Accuracy | Size |
|-------|---------|------|-------|----------|------|
| **MediaPipe BlazePose** | Track **your** poses (webcam) | Real-time (Phase 5) | ⚡ Very Fast | Good | Built-in |
| **RTMO-L** (RTMPose) | Extract **instructor** poses | Offline (Phase 3) | 🐢 Slow | Excellent | 171 MB |
| **YOLOX-L** | Detect people for RTMO | Offline (Phase 3) | 🐢 Slow | Excellent | 207 MB |

---

### **1️⃣ MediaPipe BlazePose** (Real-Time User Tracking)

**📍 What:** Google's lightweight pose estimation model optimized for mobile/web  
**📍 When:** Phase 5 - Tracking YOUR movements from webcam in real-time  
**📍 Why:** Fast enough for real-time (30+ FPS on CPU), good enough accuracy  
**📍 Output:** 33 body landmarks (3D coordinates + visibility scores)  
**📍 Location:** `src/pose/tracker.py` → `BlazePoseTracker`

#### **Why MediaPipe for Real-Time?**
- ⚡ **Speed:** Runs at 30-60 FPS on CPU (optimized with TFLite)
- 💻 **Low Resource:** Works on consumer laptops without GPU
- 🎯 **Design:** Specifically built for live video streams
- 📦 **Built-in:** No separate model download needed

#### **Code Example:**
```python
from src.pose.tracker import BlazePoseTracker

tracker = BlazePoseTracker(model_complexity=1)
with tracker:
    pose_result = tracker.process_frame(webcam_frame)
    # pose_result.keypoints → (33, 3) array of landmarks
```

#### **Limitations:**
- ❌ Less accurate than research-grade models
- ❌ Can struggle with complex poses or occlusions
- ✅ But: Good enough for comparing to reference poses

---

### **2️⃣ RTMO-L** (RTMPose One-Stage Model)

**📍 What:** OpenMMLab's state-of-the-art one-stage pose estimation model  
**📍 When:** Phase 3 - Extracting reference poses from instructor's video (offline)  
**📍 Why:** Maximum accuracy for ground-truth reference data  
**📍 Output:** 17 COCO keypoints (body joints) with high confidence  
**📍 Location:** `src/pose/tracker.py` → `RTMPoseTracker`

#### **Why RTMO for Reference Extraction?**
- 🎯 **Accuracy:** State-of-the-art performance on COCO dataset
- 📊 **One-Stage:** Faster than two-stage models (no separate detection)
- 🔬 **Research-Grade:** Better quality reference data = better feedback
- ⏱️ **Offline OK:** Speed doesn't matter, only runs once per tutorial

#### **Technical Details:**
- **Architecture:** CSPDarknet backbone + neck + detection head
- **Input Size:** 640×640 pixels
- **Keypoints:** 17 COCO body keypoints (compatible with YOLO detection)
- **Training:** Pre-trained on COCO, Body7, etc.

#### **Code Example:**
```python
from src.pose.tracker import RTMPoseTracker

tracker = RTMPoseTracker(
    pose_config="models/rtmo-l_16xb16-600e_body7-640x640.py",
    pose_checkpoint="models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
    det_checkpoint="models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    device="cpu"
)
pose_result = tracker.process_frame(instructor_frame)
```

#### **Model Files:**
- **Checkpoint:** `rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth` (171 MB)
- **Config:** `rtmo-l_16xb16-600e_body7-640x640.py` (model architecture)
- **Source:** [OpenMMLab MMPose Model Zoo](https://github.com/open-mmlab/mmpose)

---

### **3️⃣ YOLOX-L** (Person Detection)

**📍 What:** YOLOX Large - Fast and accurate object detector  
**📍 When:** Phase 3 - Detecting people in instructor video before pose extraction  
**📍 Why:** RTMO needs bounding boxes; YOLOX provides high-quality detections  
**📍 Output:** Bounding boxes around detected people  
**📍 Location:** Used internally by `RTMPoseTracker`

#### **Why YOLOX for Detection?**
- 🎯 **Accuracy:** Better person detection than older YOLO versions
- ⚡ **Speed:** Faster than two-stage detectors (Faster R-CNN, etc.)
- 🔗 **Integration:** Native support in MMDetection (pairs with MMPose)
- 📦 **COCO-Trained:** Pre-trained on 80 object classes including 'person'

#### **Technical Details:**
- **Architecture:** Modified YOLOv3 with decoupled head, anchor-free
- **Input Size:** 640×640 pixels
- **Detection:** 80 COCO classes (we only use 'person')
- **NMS:** Built-in non-maximum suppression for overlapping boxes

#### **Model Files:**
- **Checkpoint:** `yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth` (207 MB)
- **Config:** `yolox_l_8x8_300e_coco.py` (detector architecture)
- **Source:** [OpenMMLab MMDetection Model Zoo](https://github.com/open-mmlab/mmdetection)

---

### **🔄 Pipeline Flow: How Models Work Together**

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE PROCESSING (Once per tutorial video)                  │
└─────────────────────────────────────────────────────────────────┘

Tutorial Video (MP4)
     │
     ▼
[YOLOX-L Detector]  ← Detect people in each frame
     │
     ▼
Bounding Boxes (person locations)
     │
     ▼
[RTMO-L Pose Model]  ← Extract poses from detected bounding boxes
     │
     ▼
Reference Pose Database (instructor's keypoints)
     │
     ▼
[Normalize & Store]  ← Hip-centered, torso-scaled, RotJoints


┌─────────────────────────────────────────────────────────────────┐
│  REAL-TIME LOOP (During your dance practice)                   │
└─────────────────────────────────────────────────────────────────┘

Your Webcam Stream
     │
     ▼
[MediaPipe BlazePose]  ← Fast pose detection (30 FPS)
     │
     ▼
Your Pose Keypoints (33 landmarks)
     │
     ▼
[Normalize]  ← Same normalization as instructor
     │
     ▼
[S-WFDTW Alignment]  ← Compare to reference database
     │
     ▼
Similarity Scores (per joint/limb)
     │
     ▼
[OpenCV UI]  ← Color-coded feedback overlay
```

---

### **❓ Why Not Use MediaPipe for Everything?**

**Question:** If MediaPipe is fast and works well, why download 378 MB of models?

**Answer:** Quality of reference data matters!

| Scenario | MediaPipe Only | MediaPipe + RTMO |
|----------|----------------|------------------|
| **Instructor Pose** | 85% accuracy | 95% accuracy |
| **Your Pose** | 85% accuracy | 85% accuracy |
| **Comparison Quality** | ⚠️ Comparing noisy to noisy | ✅ Comparing noisy to accurate |
| **Feedback Quality** | ⚠️ False positives | ✅ Reliable criticism |

**Key Insight:** You can tolerate noise in YOUR poses (it averages out), but the REFERENCE must be as accurate as possible. Using RTMO for reference ensures high-quality ground truth.

---

### **❓ Why Not Use RTMO for Real-Time?**

**Question:** If RTMO is more accurate, why not use it for live tracking?

**Answer:** Speed vs. Accuracy tradeoff

| Aspect | MediaPipe | RTMO + YOLOX |
|--------|-----------|--------------|
| **FPS on CPU** | 30-60 | 2-5 |
| **Latency** | <30 ms | 200-500 ms |
| **User Experience** | ✅ Smooth | ❌ Laggy |
| **GPU Required?** | ❌ No | ⚠️ Recommended |

**Key Insight:** Real-time feedback requires smooth video. A 0.5-second delay would make the system unusable. MediaPipe sacrifices 10% accuracy for 10× speed.

---

### **📦 Model Download**

All models are downloaded automatically via:

```bash
python download_models.py
```

This script fetches RTMO and YOLOX from official OpenMMLab servers. MediaPipe models are included in the `mediapipe` Python package.

**What gets downloaded:**
- ✅ RTMO-L checkpoint (171 MB) - Required
- ✅ YOLOX-L checkpoint (207 MB) - Required  
- ✅ Config files (.py) - Auto-loaded if missing
- ❌ MediaPipe models - Already bundled with pip package

---

### **🎓 Model Performance Comparison**

#### **Accuracy (on COCO val2017):**
| Model | AP (Average Precision) | AR (Average Recall) |
|-------|------------------------|---------------------|
| RTMO-L | 72.4 | 78.1 |
| MediaPipe BlazePose | ~65-70* | ~70-75* |

*Estimated based on community benchmarks (not officially published on COCO)

#### **Speed (on Intel i7, no GPU):**
| Model | Inference Time | FPS |
|-------|---------------|-----|
| MediaPipe | 16-33 ms | 30-60 |
| RTMO-L + YOLOX | 200-500 ms | 2-5 |

**Conclusion:** For this application, using both models gives the best of both worlds - accurate references with smooth real-time feedback.

---

## 📁 Repository Structure

```
AIDanceCoach/
│
├── main.py                          # Pipeline entry point
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
│
├── .github/
│   └── copilot-instructions.md      # Development guidelines and architecture docs
│
├── data/
│   ├── raw/                         # Downloaded videos and extracted frames
│   │   └── .gitkeep
│   ├── extracted_steps/             # Saved step images (step_001.jpg, step_002.jpg, ...)
│   │   └── .gitkeep
│   ├── output_videos/               # Reference skeleton overlay videos (MP4/AVI)
│   │   └── .gitkeep
│   └── rendered_videos/             # Final output videos with side-by-side comparisons
│       └── .gitkeep
│
├── models/                          # Trained TCN models and checkpoints
│   └── .gitkeep
│
├── src/                             # Main source code
│   ├── __init__.py
│   │
│   ├── ingestion/                   # Phase 1: Video download & frame extraction
│   │   ├── __init__.py
│   │   └── video_processor.py       # VideoProcessor class: yt-dlp integration,
│   │                                # parallel frame extraction with ProcessPoolExecutor
│   │
│   ├── segmentation/                # Phase 3: Dance step segmentation
│   │   ├── __init__.py
│   │   └── tcn_segmenter.py         # TCNSegmenter: audio/visual feature extraction,
│   │                                # Temporal Convolutional Network for segmentation
│   │
│   ├── pose/                        # Phase 2, 4-5: Pose estimation & normalization
│   │   ├── __init__.py
│   │   └── tracker.py               # BlazePoseTracker (MediaPipe) for real-time user tracking (Phase 5)
│   │                                # RTMPoseTracker (OpenMMLab) for reference pose extraction (Phase 2)
│   │                                # PoseNormalizer for spatial normalization & RotJoints (Phase 4)
│   │
│   ├── alignment/                   # Phase 6: Temporal alignment & scoring
│   │   ├── __init__.py
│   │   └── dtw_scorer.py            # SWFDTWScorer: Spatial-Weighted Fast DTW,
│   │                                # Euclidean distance, Cosine similarity scoring
│   │
│   └── ui/                          # Phase 7: Real-time visual feedback
│       ├── __init__.py
│       └── feedback_display.py      # FeedbackDisplay: OpenCV rendering loop,
│                                    # color-coded skeleton overlay
│
└── tests/                           # Unit tests (structure mirrors src/)
    ├── __init__.py
    ├── alignment/
    │   └── __init__.py
    ├── ingestion/
    │   └── __init__.py
    ├── pose/
    │   └── __init__.py
    ├── segmentation/
    │   └── __init__.py
    └── ui/
        └── __init__.py
```

### **Module Responsibilities**

| Module | Path | Responsibility |
|--------|------|----------------|
| **VideoProcessor** | `src/ingestion/video_processor.py` | YouTube video download via yt-dlp, parallel frame extraction with background subtraction |
| **TCNSegmenter** | `src/segmentation/tcn_segmenter.py` | Audio/visual feature extraction, TCN-based action segmentation |
| **RTMPoseTracker** | `src/pose/tracker.py` | High-accuracy reference pose extraction using OpenMMLab RTMPose/RTMO |
| **BlazePoseTracker** | `src/pose/tracker.py` | Real-time user pose tracking using MediaPipe BlazePose (33 landmarks) |
| **PoseNormalizer** | `src/pose/tracker.py` | Spatial normalization (hip-centering, torso scaling, RotJoints conversion) |
| **SWFDTWScorer** | `src/alignment/dtw_scorer.py` | Temporal alignment via S-WFDTW, multi-metric scoring (Euclidean + Cosine) |
| **FeedbackDisplay** | `src/ui/feedback_display.py` | OpenCV-based real-time UI with color-coded skeleton rendering |

---

## 🚀 Installation Guide

### **⚠️ IMPORTANT: Why Complex Installation?**

This project uses **OpenMMLab ecosystem + MediaPipe + PyTorch + CUDA**, which requires **careful dependency management**. A slight mismatch between PyTorch, CUDA, and MMCV will cause installation failures or GPU detection issues.

**Recommended:** Follow this guide step-by-step. Don't skip parts!

---

### **Prerequisites**

- **Python 3.9** (recommended) - 3.10/3.11 acceptable, 3.12 not recommended
- **NVIDIA GPU** with CUDA 12.1 or 11.8 (CUDA drivers must be installed)
- **Webcam** for live pose tracking
- **Windows 10/11, macOS, or Linux**
- **5+ GB free disk space**
- **Internet connection** (for downloading models)

### **Check Your System**

Before starting, verify your CUDA version:

```powershell
# Windows PowerShell
nvidia-smi

# Look for line: "CUDA Version: 12.x" or "CUDA Version: 11.x"
```

If `nvidia-smi` doesn't work, install [NVIDIA Drivers](https://www.nvidia.com/Download/driverDetails.aspx).

---

### **🎯 STEP 1: Delete Old Virtual Environment (If You Have One)**

If you already have a `.venv` folder, delete it:

```powershell
# Windows PowerShell
cd C:\MACHINE_LEARNING\DL\AIDanceCoach
Remove-Item -Recurse -Force .venv
```

---

### **🎯 STEP 2: Check Python Version**

```powershell
# Windows PowerShell
py -3.9 --version

# Should output: Python 3.9.x
```

**If Python 3.9 is NOT installed:**
1. Download from [python.org](https://www.python.org/downloads/release/python-3913/)
2. Run installer with ✓ "Add Python to PATH" checked
3. Restart PowerShell and verify

---

### **🎯 STEP 3: Create New Virtual Environment with Python 3.9**

```powershell
# Windows PowerShell - Navigate to project
cd C:\MACHINE_LEARNING\DL\AIDanceCoach

# Create new venv with Python 3.9
py -3.9 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# You should see (.venv) in your prompt
```

---

### **🎯 STEP 4: Upgrade pip, setuptools, wheel**

```powershell
# Make sure you're in the activated venv!
python -m pip install --upgrade pip setuptools wheel

# Should complete without errors
```

---

### **🎯 STEP 5: Install PyTorch 2.1.2 with CUDA Support**

**Check your CUDA version from Step 1 and use the correct command:**

**For CUDA 12.1 or 12.7** (most common):
```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# This will download ~2.5 GB - can take 5-10 minutes
```

**For CUDA 11.8:**
```powershell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch installed correctly:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Should output:
# PyTorch: 2.1.2+cuXXX
# CUDA: True
```

---

### **🎯 STEP 6: Install OpenMIM and MMEngine**

```powershell
# Install OpenMIM package manager
pip install -U openmim

# Install MMEngine (dependency for MMCV)
mim install mmengine==0.10.5

# Should complete without errors
```

---

### **🎯 STEP 7: Install MMCV (Custom CUDA Operators)**

⚠️ **This is critical!** MMCV comes with pre-built wheels matched to your CUDA version.

**For CUDA 12.1 or 12.7:**
```powershell
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# This downloads a pre-built wheel - should be fast (30-60 seconds)
```

**For CUDA 11.8:**
```powershell
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

**Verify MMCV installed:**
```powershell
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"

# Should output: MMCV: 2.1.0
```

---

### **🎯 STEP 8: Fix NumPy (CRITICAL for MMCV)**

⚠️ **MMCV requires NumPy < 2.0** - some packages install NumPy 2.x which breaks MMCV!

```powershell
# Force NumPy 1.x
pip install 'numpy<2.0.0' --force-reinstall --no-deps

# Verify
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Should output NumPy 1.26.x
```

---

### **🎯 STEP 9: Install OpenMMLab High-Level Packages**

```powershell
# Install detection and segmentation frameworks
pip install mmaction2==1.2.0 mmdet==3.3.0 --no-deps

# Note: mmpose will be skipped on Windows due to chumpy build issues
# But you have MediaPipe for pose detection, which is faster anyway!
```

---

### **🎯 STEP 10: Install Remaining Dependencies**

```powershell
# Install video, audio, ML libraries
pip install mediapipe>=0.10.9 opencv-python>=4.8.0 opencv-contrib-python>=4.8.0

pip install yt-dlp>=2024.3.10 librosa>=0.10.1 soundfile>=0.12.0 ffmpeg-python>=0.2.0

pip install onnxruntime-gpu>=1.16.3

pip install scipy>=1.11.0 fastdtw>=0.3.4 tqdm>=4.65.0 pyyaml>=6.0 colorama>=0.4.6

# All should complete successfully
```

---

### **🎯 STEP 11: Verify Complete Installation**

```powershell
# Run verification script
python verify_installation.py

# Should show:
# - All packages with [OK] checkmarks
# - GPU detected
# - GPU computation test passed
# - Any missing packages will show [ERROR]
```

**Expected Output (excerpt):**
```
[OK] torch: 2.1.2+cu121
[OK] mmcv: 2.1.0
[OK] mediapipe: 0.10.32
[OK] PyTorch CUDA: CUDA 12.1, 1 GPU(s): NVIDIA GeForce RTX...
[OK] GPU Computation: GPU computation test passed
```

---

### **✅ Installation Complete!**

If you see `INSTALLATION COMPLETE!` heading, you're ready to use AI Dance Coach.

---

## **❓ Troubleshooting**

### **Issue: "CUDA Available: False"**

**Solution:**
```powershell
# 1. Check nvidia-smi works
nvidia-smi

# 2. Check CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 3. If False, reinstall PyTorch with correct CUDA:
# Run STEP 5 again with correct CUDA version
```

### **Issue: "ImportError: DLL load failed"**

**Solution:** Install Visual C++ Redistributable
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install it
- Restart your terminal

### **Issue: "numpy.\_core" errors**

**Solution:**
```powershell
# NumPy is wrong version
pip install 'numpy<2.0.0' --force-reinstall --no-deps
```

### **Issue: "No module named 'mmpose'"**

**This is OK!** MMPose has Windows build issues. You have:
- ✅ MediaPipe (faster for real-time)
- ✅ MMDet (for detection)
- ✅ MMAction2 (for action segmentation)

See [INSTALLATION_STATUS.md](INSTALLATION_STATUS.md) for workarounds.

### **Issue: Installation takes too long or gets stuck**

**Solution:**
```powershell
# Sometimes pip cache causes issues
pip cache purge

# Try installation again
pip install -r requirements.txt
```

---

## **📝 Installation Complete Checklist**

- [ ] Python 3.9.13 installed and active
- [ ] NVIDIA GPU drivers working (`nvidia-smi` shows GPU)
- [ ] Virtual environment created and activated (.venv)
- [ ] PyTorch 2.1.2 with CUDA support (`torch.cuda.is_available()` = True)
- [ ] MMCV 2.1.0 installed
- [ ] MMEngine 0.10.5 installed
- [ ] MediaPipe installed
- [ ] OpenCV installed
- [ ] All remaining packages installed
- [ ] `verify_installation.py` shows all [OK]
- [ ] GPU computation test passed

---

## **📚 Next Steps**

Once installation is complete:

1. **Download Models** (see Step 4 below)
2. **Run a Test**: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
3. **Start Using**: See **Usage** section below

---

### **Step 4: Download Pretrained Models**

Download the required OpenMMLab models for RTMPose/RTMO:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download RTMO checkpoint (example)
wget -P models/ https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth

# Download detector checkpoint (if needed)
wget -P models/ https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
```

Refer to [OpenMMLab MMPose documentation](https://mmpose.readthedocs.io/) for the latest model URLs and configurations.

---

## 💻 Usage

### **Basic Usage**

Run the pipeline with a YouTube dance tutorial URL:

```bash
python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### **Advanced Options**

```bash
python main.py \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --camera 0 \
    --device cuda \
    --pose-config configs/rtmpose/rtmo-l_8xb32-600e_body7-640x640.py \
    --pose-ckpt models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth \
    --window-size 10 \
    --output-dir data/raw
```

### **Command-Line Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--url` | YouTube URL of the dance tutorial | **Required** |
| `--camera` | Webcam index (0 for default camera) | `0` |
| `--device` | Inference device (`cpu` or `cuda`) | `cpu` |
| `--pose-config` | Path to MMPose config file | Auto-detect |
| `--pose-ckpt` | Path to MMPose checkpoint | Auto-detect |
| `--det-config` | Path to MMDet detector config | Auto-detect |
| `--det-ckpt` | Path to MMDet checkpoint | Auto-detect |
| `--window-size` | S-WFDTW window size (frames) | `10` |
| `--output-dir` | Output directory for processed data | `data/raw` |
| `--extracted-steps-dir` | Directory to save step images (step_001.jpg, ...) | `data/extracted_steps` |
| `--reference-output-videos-dir` | Directory to save reference skeleton overlay videos | `data/output_videos` |

### **Workflow**

1. **Download & Process**: The system downloads the tutorial video and extracts frames
2. **Reference Extraction**: Extracts instructor's poses from the entire video using RTMPose
3. **Action Segmentation**: Identifies discrete dance steps using TCN-based segmentation
4. **Save Outputs**: 
   - Saves one representative frame per step as `step_001.jpg`, `step_002.jpg`, etc.
   - Generates skeleton overlay video of the instructor with pose visualization
5. **Live Tracking**: Opens webcam feed and starts real-time tracking
6. **Feedback Loop**: Displays color-coded skeleton showing how well you're matching the reference

---

## 🔧 Key Dependencies

| Library | Purpose | Phase |
|---------|---------|-------|
| **yt-dlp** | Video download from YouTube | Phase 1 |
| **opencv-python** | Frame extraction, UI rendering | Phase 1, 7 |
| **mmpose/mmdet** | RTMPose reference pose extraction | Phase 2 |
| **librosa** | Audio feature extraction (Mel spectrograms) | Phase 3 |
| **torch** | TCN model training/inference | Phase 3 |
| **mediapipe** | BlazePose real-time user tracking | Phase 5 |
| **numpy/scipy** | Numerical computation, DTW | Phase 6 |
| **onnxruntime** | Accelerated inference | Phase 8 |

---

## 🧪 Testing

Run unit tests:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific module tests
pytest tests/alignment/
pytest tests/pose/
```

---

## 🛣️ Roadmap / Future Enhancements

- [ ] **Multi-person Support**: Track multiple dancers simultaneously
- [ ] **Voice Feedback**: Add audio cues for corrections ("Lift your left arm higher")
- [ ] **Performance Analytics**: Generate progress reports and skill improvement graphs
- [ ] **Mobile App**: Deploy to iOS/Android using TensorFlow Lite
- [ ] **Cloud Processing**: Offload heavy computation to cloud servers
- [ ] **Style Transfer**: Learn and suggest personalized dance style variations
- [ ] **Haptic Feedback**: Integrate with wearable devices for tactile corrections

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or suggestions, please open an issue or contact the maintainers.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for blazing-fast pose tracking
- **OpenMMLab** for RTMPose/RTMO high-accuracy pose estimation
- **yt-dlp** community for robust video downloading
- Research papers on TCN-based action segmentation and DTW variants

---

**Built with ❤️ for dancers and AI enthusiasts**
