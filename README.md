# AI Dance Coach 🕺💃

An end-to-end AI system that ingests YouTube dance tutorials, segments them into teachable steps, extracts reference poses, and compares them to a user's live webcam feed to provide **real-time corrective feedback** with color-coded visual guidance.

---

## 🎯 Features

- **Automated Video Ingestion**: Download dance tutorials from YouTube using `yt-dlp`
- **Intelligent Segmentation**: Uses Temporal Convolutional Networks (TCN) to segment videos into discrete dance steps
- **Dual Pose Estimation**:
  - **RTMPose/RTMO** (OpenMMLab) for high-accuracy reference pose extraction from tutorials
  - **MediaPipe BlazePose** for fast, lightweight real-time tracking of user movements
- **Temporal Alignment**: Spatial-Weighted Fast Dynamic Time Warping (S-WFDTW) for real-time motion comparison
- **Real-Time Visual Feedback**: Color-coded skeleton overlay (🟢 Green = correct, 🟡 Yellow = minor deviation, 🔴 Red = critical error)
- **Hardware Optimization**: ONNX Runtime support for CPU acceleration, TensorRT for GPU edge devices

---

## 🏗️ Architecture

The system operates in **8 phases**:

### **Phase 1: Ingestion & Preprocessing**
Downloads YouTube videos and extracts frames in parallel using `ProcessPoolExecutor` to bypass Python's GIL. Applies grayscale conversion and background subtraction for cleaner pose detection.

### **Phase 2: Action Segmentation**
Extracts visual features (skeletal bone vectors) and audio features (Mel spectrogram) using `librosa`. Feeds these multi-modal features into a **Temporal Convolutional Network (TCN)** to output segmentation probability curves, identifying discrete dance steps via peak-picking.

### **Phase 3: Reference Pose Extraction**
Uses **OpenMMLab's RTMPose** (specifically RTMO for high accuracy) to extract ground-truth keypoints from the instructor's segmented video.

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
│   └── raw/                         # Downloaded videos and extracted frames
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
│   ├── segmentation/                # Phase 2: Dance step segmentation
│   │   ├── __init__.py
│   │   └── tcn_segmenter.py         # TCNSegmenter: audio/visual feature extraction,
│   │                                # Temporal Convolutional Network for segmentation
│   │
│   ├── pose/                        # Phase 3-5: Pose estimation & normalization
│   │   ├── __init__.py
│   │   └── tracker.py               # BlazePoseTracker (MediaPipe) for real-time user tracking
│   │                                # RTMPoseTracker (OpenMMLab) for reference pose extraction
│   │                                # PoseNormalizer for spatial normalization & RotJoints
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

## 🚀 Installation

### **Prerequisites**

- **Python 3.9+** (recommended: 3.10 or 3.11)
- **Webcam** for live pose tracking
- **GPU** (optional but recommended for real-time performance)

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/AIDanceCoach.git
cd AIDanceCoach
```

### **Step 2: Create Virtual Environment**

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes:
- **Core ML/CV**: `numpy`, `scipy`, `opencv-python`
- **Video/Audio**: `yt-dlp`, `librosa`, `soundfile`
- **Pose Estimation**: `mediapipe`, `mmcv`, `mmdet`, `mmpose`
- **Deep Learning**: `torch`, `torchvision`
- **Optimization**: `onnxruntime` (uncomment `onnxruntime-gpu` for GPU support)

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

### **Step 5: Verify Installation**

```bash
python -c "import cv2, mediapipe, torch; print('Setup successful!')"
```

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

### **Workflow**

1. **Download & Process**: The system downloads the tutorial video and segments it into steps
2. **Reference Extraction**: Extracts instructor's poses from each step
3. **Live Tracking**: Opens webcam feed and starts real-time tracking
4. **Feedback Loop**: Displays color-coded skeleton showing how well you're matching the reference

---

## 🔧 Key Dependencies

| Library | Purpose | Phase |
|---------|---------|-------|
| **yt-dlp** | Video download from YouTube | Phase 1 |
| **opencv-python** | Frame extraction, UI rendering | Phase 1, 7 |
| **librosa** | Audio feature extraction (Mel spectrograms) | Phase 2 |
| **torch** | TCN model training/inference | Phase 2 |
| **mmpose/mmdet** | RTMPose reference pose extraction | Phase 3 |
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
