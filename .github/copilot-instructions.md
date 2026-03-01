# AI Dance Coach — Copilot Instructions (Updated March 2026)

## Project Goal
An end-to-end AI system that ingests a YouTube dance tutorial, segments it into teachable steps,
extracts reference poses, and compares them to a user's live webcam feed to provide real-time
corrective feedback with visual guidance.

---

## Actual Architecture Overview

### Phase 1: Ingestion & Preprocessing
- **yt-dlp**: Downloads videos from YouTube or accepts local file paths
- **cv2.VideoCapture + ProcessPoolExecutor**: Streams frames in parallel (bypassing GIL)
- **Optional preprocessing**: Grayscale conversion, MOG2 background subtraction, spatial gradients

### Phase 2: Reference Pose Extraction (Happens Before Segmentation)
**IMPORTANT CORRECTION**: Reference poses are extracted **before** segmentation, using the full video.

- **BlazePoseTracker** (fallback, fast): Lightweight MediaPipe pose detector
- **RTMPoseTracker** (primary, if config/checkpoint provided): OpenMMLab RTMO one-stage detector
  - Wraps mmpose.apis.init_model() and inference_topdown() for high accuracy
  - Uses mmdet for person detection via bounding boxes
  - Returns 17 COCO keypoints per frame
- **Output**: Flat arrays of normalized keypoints (all_poses), used downstream

### Phase 3: Action Segmentation (Happens After Pose Extraction)
- Extract **visual features**: Bone vectors from normalized keypoints
- Extract **audio features** (optional): Mel spectrogram via librosa
- **TCNSegmenter**: Feed features to Temporal Convolutional Network
  - Outputs segmentation probability curve per frame
  - Uses **peak-picking** (scipy.signal.find_peaks) to find step boundaries
  - Returns list of frame indices: [boundary_0, boundary_1, ...]
- **Split reference data**: Use boundaries to group poses into per-step arrays

### Phase 4: Spatial Normalization
- Translate all keypoints to **hip-center (root) origin**
- Scale by **torso length** (distance between shoulders) for scale-invariance
- Convert to **RotJoints**: Angular limb vectors for rotation-invariant comparison

### Phase 5: Real-Time User Tracking
- **BlazePoseTracker** only: Fast (~30 FPS) real-time tracking from webcam
- Returns 33 3D landmarks per frame (MediaPipe format)
- Applied same normalization as reference poses

### Phase 6: Temporal Alignment & Scoring
- **SWFDTWScorer**: Spatial-Weighted Fast Dynamic Time Warping
  - Sliding window constraint (configurable, default 10 frames)
  - Early-termination pruning for speed
  - Weighted fusion: Euclidean distance + Cosine similarity
- **Per-joint scores**: For color-coding limbs

### Phase 7: Real-Time UI/UX
- **FeedbackDisplay**: OpenCV skeleton rendering
  - Uses cv2.line(), cv2.circle() for limbs and joints
  - Color codes by S-WFDTW joint scores
- **New (March 2026)**: Robust error handling for OpenCV display calls

---

## Module Map (Updated)
| Module | Path | Responsibility |
|---|---|---|
| VideoProcessor | `src/ingestion/video_processor.py` | yt-dlp download, parallel frame extraction |
| TCNSegmenter | `src/segmentation/tcn_segmenter.py` | Visual feature extraction + TCN peak-picking |
| PoseTracker | `src/pose/tracker.py` | BlazePoseTracker + RTMPoseTracker (dual models) |
| PoseNormalizer | `src/pose/tracker.py` | Hip-centering, torso scaling, RotJoints |
| SWFDTWScorer | `src/alignment/dtw_scorer.py` | Temporal alignment + per-joint scoring |
| FeedbackDisplay | `src/ui/feedback_display.py` | OpenCV skeleton rendering + UI loop |
| ReferenceVideoRenderer | `src/ui/reference_video_renderer.py` | Utility class (currently not used in main flow) |
| main | `main.py` | Pipeline orchestration + new feature functions |

---

## Key Libraries & Versions (Updated March 2026)

### Computer Vision & Pose Estimation
- `opencv-python` — frame I/O, display, video writing
- `mediapipe` — BlazePose (33 landmarks, 30 FPS on CPU)
- `mmpose` (OpenMMLab) — RTMPose/RTMO (17 keypoints, high accuracy)
- `mmdet` (OpenMMLab) — person detection for RTMPose

### Machine Learning & Signal Processing
- `numpy`, `scipy` — numerical computation, peak-picking
- `torch` — TCN model (segmentation)
- `librosa` — optional audio feature extraction

### Video & Media
- `yt-dlp` — YouTube download
- `tqdm` — progress bars (optional)

### Optional (Future Use)
- `onnxruntime` — CPU inference acceleration (not integrated)
- TensorRT — GPU optimization (not integrated)

---

## New Features (March 2026 Update)

### Goal 1: Extract & Save Dance Step Images
- **Function**: `_save_extracted_steps()` in main.py
- **Behavior**: 
  - One representative JPEG per detected step
  - Saved to `data/extracted_steps/` (customizable via `--extracted-steps-dir`)
  - Naming: `step_001.jpg`, `step_002.jpg`, ...
  - Uses boundary frame as representative
- **Use Case**: Manually verify segmentation accuracy; visual reference for each step

### Goal 2: Save Skeleton Overlay Video (Reference)
- **Function**: `_save_reference_skeleton_overlay()` in main.py
- **Behavior**:
  - Renders RTMPose skeleton on reference video (instructor's poses)
  - Frame-by-frame processing during reference extraction
  - Codec fallback: mp4v (MP4) → MJPG (AVI)
  - Saved to `data/output_videos/` (customizable via `--reference-output-videos-dir`)
  - Output file: `output_reference_skeleton.mp4` or `.avi`
- **Use Case**: Verify that RTMPose accurately captures instructor poses; debug skeleton tracking

### Goal 3: Real-Time UI Crash Hardening (Desktop OpenCV)
- **Scope**: Fixes for Windows/Linux desktop UI mode (cv2.imshow/namedWindow)
- **Changes**:
  - cv2.namedWindow() wrapped in try/except with fallback
  - cv2.imshow() wrapped with error recovery; continues if window closed
  - cv2.waitKey() wrapped; continues despite failures
  - Better logging of frame read failures (ret=False, frame=None)
  - FeedbackDisplay.run() improved with robust error handling
- **Behavior**: UI loop gracefully degrades if display window fails; doesn't crash immediately
- **NOT Supporting**: Streamlit web apps (out of scope; would require st.image() + threading)

---

## Debugging & Troubleshooting

### Real-Time UI Crashing
**Symptom**: Webcam light turns on, script immediately crashes or hangs.

**Checklist**:
1. Verify `cap.read()` returns valid (ret=True, frame not None)
   - Check logs: "No frame received from camera"
   - Try `--camera 1` or `--camera 2` if index 0 doesn't work
2. Verify `cv2.namedWindow()` succeeds
   - Logs: "Failed to create OpenCV display window"
3. Verify `cv2.waitKey(1)` is inside the loop (not before `imshow`)
4. **Not supported**: Streamlit mode (would require `st.image()` + webrtc)
5. Check for dual-window conflicts: only `run_feedback_loop()` in main.py should create windows

### Reference Extraction Failures
**Symptom**: "RTMPoseTracker initialization failed" or skeleton overlay video has 0 frames.

**Checklist**:
1. Verify model checkpoint files exist:
   - `models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth` (171 MB)
   - `models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth` (207 MB)
   - Run `python download_models.py` if missing
2. Verify `--device cuda` if GPU available; fallback to `cpu`
3. Check MMPose/MMDet environment:
   - Run `python -c "import mmpose, mmdet"` to verify installation

### Segmentation Issues
**Symptom**: Wrong number of steps, or no steps detected.

**Checklist**:
1. Verify TCN model checkpoint is loaded (`tcn_segmenter.py`)
2. Check peak-picking thresholds (height, distance) in `segment()` method
3. Verify visual features extracted correctly: bone vectors, not raw keypoints

---

## CLI Parameters (Updated)

```bash
python main.py \
  --url <youtube_url_or_local_path> \
  --camera 0 \
  --device auto \
  --pose-config models/rtmo-l_16xb16-600e_body7-640x640.py \
  --pose-ckpt models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth \
  --det-config models/yolox_l_8x8_300e_coco.py \
  --det-ckpt models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
  --extracted-steps-dir data/extracted_steps \
  --reference-output-videos-dir data/output_videos \
  --save-video \
  --video-output output_annotated.mp4
```

**New Arguments**:
- `--extracted-steps-dir` (default: `data/extracted_steps`) — where to save step images
- `--reference-output-videos-dir` (default: `data/output_videos`) — where to save skeleton overlay MP4

---

## Known Limitations & Future Work

- **Phase 8** (ONNX/TensorRT export): Not yet implemented; requires model export scripts
- **Streamlit support**: Out of scope; would require `st.image()` + threading/webrtc
- **Multi-person tracking**: Currently assumes single person per frame
- **Audio features**: Optional but currently undocumented; TCN trained on visual only
