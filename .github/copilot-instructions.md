# AI Dance Coach — Copilot Instructions

## Project Goal
An end-to-end AI system that ingests a YouTube dance tutorial, segments it into teachable steps,
extracts reference poses, and compares them to a user's live webcam feed to provide real-time
corrective feedback.

---

## Architecture Overview

### Phase 1: Ingestion & Preprocessing
- Use **yt-dlp** to programmatically stream/download videos.
- Use **cv2.VideoCapture** with Python's **ProcessPoolExecutor** for parallel frame extraction
  to bypass the GIL.
- Apply grayscale conversion and spatial gradient background subtraction.

### Phase 2: Action Segmentation
- Extract **visual features** (skeletal bone vectors) and **audio features**
  (Mel spectrogram via **librosa**).
- Feed these multi-modal features into a **Temporal Convolutional Network (TCN)** to output
  a segmentation probability curve, using **peak-picking** to identify discrete dance steps.

### Phase 3: Reference Pose Extraction
- Use **OpenMMLab's RTMPose** (specifically **RTMO** for high accuracy) to extract ground-truth
  keypoints from the instructor's segmented video.

### Phase 4: Spatial Normalization
- Translate all coordinates (both instructor and user) to a **root origin (hip center)** and
  scale them based on a stable metric (**torso length**).
- Convert raw (x, y) coordinates to **angular limb vectors** (RotJoints).

### Phase 5: Real-Time Edge Tracking
- Use **MediaPipe BlazePose** for fast, lightweight tracking of the user's live webcam feed
  (detecting 33 3D landmarks).

### Phase 6: Temporal Alignment & Scoring
- Use **Spatial-Weighted Fast Dynamic Time Warping (S-WFDTW)** with a sliding window constraint
  (e.g., 10 frames) and early-termination pruning to align the user's live motion with the
  instructor's reference sequence in real-time.
- Calculate the final score using a **weighted fusion** of:
  - **Euclidean Distance** (spatial position)
  - **Cosine Similarity** (angular alignment)

### Phase 7: Real-Time UI/UX
- Use **OpenCV** (`cv2.line`, `cv2.circle`) to draw a dynamic wireframe over the user's live feed.
- **Color-code limbs dynamically** based on S-WFDTW similarity scores:
  - 🟢 **Green** — correct
  - 🟡 **Yellow** — minor deviation
  - 🔴 **Red** — critical flaw

### Phase 8: Hardware Optimization
- Export models to **ONNX Runtime** for CPU acceleration.
- Use **NVIDIA TensorRT** with **FP16 quantization** for GPU edge devices.

---

## Module Map
| Module | Path | Responsibility |
|---|---|---|
| VideoProcessor | `src/ingestion/video_processor.py` | yt-dlp download, parallel frame extraction |
| TCNSegmenter | `src/segmentation/tcn_segmenter.py` | Audio/visual feature extraction + TCN |
| PoseTracker | `src/pose/tracker.py` | MediaPipe BlazePose + RTMPose wrapper |
| DTWScorer | `src/alignment/dtw_scorer.py` | S-WFDTW, Euclidean distance, Cosine similarity |
| FeedbackDisplay | `src/ui/feedback_display.py` | OpenCV rendering loop with color-coded skeleton |
| main | `main.py` | Pipeline entry point |

---

## Key Libraries
- `yt-dlp` — video download
- `opencv-python` — frame extraction and UI rendering
- `mediapipe` — BlazePose real-time tracking
- `librosa` — Mel spectrogram audio feature extraction
- `numpy`, `scipy` — numerical computation
- `torch` — TCN model
- `onnxruntime` — CPU inference acceleration
- `mmpose` / `mmdet` — RTMPose/RTMO reference pose extraction
