"""
main.py
========
AI Dance Coach — Pipeline Entry Point

Connects the full processing pipeline:

  1. [Phase 1]  VideoProcessor   — download & extract frames
  2. [Phase 2]  TCNSegmenter     — segment video into dance steps
  3. [Phase 3]  RTMPoseTracker   — extract reference poses per step
  4. [Phase 5]  BlazePoseTracker — track user's live webcam feed
  5. [Phase 6]  SWFDTWScorer     — align & score user vs. reference
  6. [Phase 7]  FeedbackDisplay  — real-time color-coded skeleton UI

Usage
-----
    python main.py --url <youtube_url> [--camera 0] [--device auto]

Arguments
---------
--url           YouTube URL of the dance tutorial to learn from.
--camera        Webcam index (default: 0).
--device        Inference device: "auto" (GPU if available, else CPU), "cpu", or "cuda" (default: "auto").
--pose-config   Path to the MMPose config (RTMPose/RTMO).
--pose-ckpt     Path to the MMPose checkpoint.
--det-config    Path to the MMDet detector config.
--det-ckpt      Path to the MMDet checkpoint.
--window-size   S-WFDTW window size in frames (default: 10).
--output-dir    Root directory for downloaded/processed data (default: data/raw).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import sys
from pathlib import Path
import hashlib

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional runtime dependency
    tqdm = None

from src.ingestion.video_processor import VideoProcessor
from src.segmentation.tcn_segmenter import TCNSegmenter, VisualFeatureExtractor
from src.pose.tracker import BlazePoseTracker, RTMPoseTracker
from src.alignment.dtw_scorer import SWFDTWScorer
from src.ui.feedback_display import FeedbackDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _mark_phase_complete(phase_number: int, phase_name: str, detail: str = "") -> None:
    """Log a standardized phase-completion message.

    Args:
        phase_number: Pipeline phase index (1-8).
        phase_name: Human-readable phase title.
        detail: Optional short detail to append.
    """
    suffix = f" | {detail}" if detail else ""
    logger.info("✅ Phase %d complete — %s%s", phase_number, phase_name, suffix)


def _setup_gpu_optimization(device: str) -> str:
    """Configure GPU for optimal performance and safe memory usage.
    
    Args:
        device: Requested device ("auto", "cpu", or "cuda").
    
    Returns:
        Actual device to use (may fall back to CPU if GPU fails).
    """
    # Normalize device to lowercase
    device = device.lower()

    # Handle auto mode: use GPU if available, else CPU
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.info("Auto device mode: CUDA not available, using CPU")
                device = "cpu"
        except Exception as e:
            logger.warning("Error checking CUDA availability (%s), using CPU", e)
            device = "cpu"
    
    # If device is CPU, no setup needed
    if device == "cpu":
        _mark_phase_complete(8, "Hardware Optimization", "Running on CPU")
        return "cpu"
    
    # Initialize CUDA with optimization
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        
        # Optimize CUDA performance
        torch.cuda.empty_cache()  # Clear any stray allocations
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuning
        torch.backends.cudnn.deterministic = False  # Faster (non-deterministic)
        
        # Set memory optimization
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU initialized: %s (%.1f GB)", gpu_name, gpu_memory)
        
        _mark_phase_complete(8, "Hardware Optimization", "CUDA enabled")
        return "cuda"
    except Exception as e:
        logger.warning("GPU initialization failed (%s), falling back to CPU", e)
        _mark_phase_complete(8, "Hardware Optimization", "Fallback to CPU")
        return "cpu"

def _get_cache_path(video_path: Path, cache_dir: Path) -> Path:
    """Generate a unique cache file path based on video file."""
    video_hash = hashlib.md5(str(video_path.resolve()).encode()).hexdigest()[:8]
    cache_file = cache_dir / f"{video_path.stem}_{video_hash}.npz"
    return cache_file


def _load_cached_reference(cache_path: Path, expected_total_frames: int | None = None) -> tuple[list[np.ndarray], list[int]] | None:
    """Load cached reference features and step boundaries if available.
    
    Args:
        cache_path: Path to the cache file.
        expected_total_frames: Expected total source frame count (for integrity
            validation). If provided and mismatch detected, cache is invalidated.
    
    Returns:
        Tuple of (reference_features, step_boundaries) if valid cache exists and is complete,
        otherwise None.
    """
    if not cache_path.exists():
        return None
    
    try:
        data = np.load(cache_path, allow_pickle=True)
        reference_features = [data[f'ref_{i}'] for i in range(len([k for k in data.files if k.startswith('ref_')]))]
        step_boundaries = data['step_boundaries'].tolist()
        
        if not reference_features:
            logger.warning("Cache integrity check failed: cache contains no reference features.")
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

        cached_total_frames = int(data["meta_total_frames"]) if "meta_total_frames" in data.files else None
        cached_valid_pose_frames = int(data["meta_valid_pose_frames"]) if "meta_valid_pose_frames" in data.files else None
        total_pose_rows = int(sum(ref.shape[0] for ref in reference_features))

        # Invalidate legacy caches that don't include integrity metadata.
        if expected_total_frames is not None and cached_total_frames is None:
            logger.warning(
                "Cache integrity check failed: legacy cache has no metadata. "
                "Discarding cache and reprocessing to ensure consistency."
            )
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

        if (
            expected_total_frames is not None
            and cached_total_frames is not None
            and cached_total_frames != expected_total_frames
        ):
            logger.warning(
                "Cache integrity check failed: expected %d source frames but cache was built from %d. "
                "Discarding cache and reprocessing.",
                expected_total_frames,
                cached_total_frames,
            )
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

        if cached_valid_pose_frames is not None and cached_valid_pose_frames != total_pose_rows:
            logger.warning(
                "Cache integrity check failed: metadata reports %d valid pose frames but cache contains %d rows. "
                "Discarding cache and reprocessing.",
                cached_valid_pose_frames,
                total_pose_rows,
            )
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None
        
        logger.info(
            "Loaded cached reference features from %s (%d pose rows)",
            cache_path.name,
            total_pose_rows,
        )
        return reference_features, step_boundaries
    except Exception as e:
        logger.warning("Failed to load cache (%s), will recompute", e)
        return None


def _save_cached_reference(
    cache_path: Path,
    reference_features: list[np.ndarray],
    step_boundaries: list[int],
    total_frames: int,
    valid_pose_frames: int,
) -> None:
    """Save reference features and step boundaries to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'step_boundaries': np.array(step_boundaries),
        'meta_total_frames': np.array(total_frames, dtype=np.int64),
        'meta_valid_pose_frames': np.array(valid_pose_frames, dtype=np.int64),
    }
    for i, ref in enumerate(reference_features):
        save_dict[f'ref_{i}'] = ref
    
    try:
        np.savez_compressed(cache_path, **save_dict)
        logger.info("Cached reference features to %s", cache_path.name)
    except Exception as e:
        logger.warning("Failed to save cache (%s)", e)


# ---------------------------------------------------------------------------
# Helper functions for Goals 1 & 2
# ---------------------------------------------------------------------------

def _save_extracted_steps(
    video_path: str,
    step_boundaries: list[int],
    output_dir: str,
) -> int:
    """Save one representative frame per detected dance step.

    Args:
        video_path: Path to the reference video.
        step_boundaries: List of frame indices where steps begin.
        output_dir: Directory to save step images (step_1.jpg, step_2.jpg, ...).

    Returns:
        Number of step images saved.
    """
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video for step extraction: %s", video_path)
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Build step ranges: [0...boundary_0), [boundary_0...boundary_1), ..., [boundary_n...total_frames)
    boundaries = [0] + step_boundaries + [total_frames]
    steps_saved = 0
    
    try:
        for step_idx, (start_frame, end_frame) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if start_frame >= end_frame:
                continue
            
            # Use the frame at step boundary (start of step) as representative
            rep_frame_idx = min(start_frame, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, rep_frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                step_filename = output_path / f"step_{step_idx + 1:03d}.jpg"
                cv2.imwrite(str(step_filename), frame)
                logger.debug("Saved step %d (frame %d) to %s", step_idx + 1, rep_frame_idx, step_filename)
                steps_saved += 1
            else:
                logger.warning("Failed to extract frame %d for step %d", rep_frame_idx, step_idx + 1)
    finally:
        cap.release()
    
    logger.info("Saved %d extracted step images to %s", steps_saved, output_path)
    return steps_saved


def _save_reference_skeleton_overlay(
    video_path: str,
    step_boundaries: list[int],
    output_dir: str,
    pose_config: str,
    pose_checkpoint: str,
    det_config: str,
    det_checkpoint: str,
    device: str = "cpu",
) -> bool:
    """Save reference video with RTMPose skeleton overlay as MP4.

    Args:
        video_path: Path to the reference video.
        step_boundaries: List of frame indices where steps begin (for logging).
        output_dir: Directory where output MP4 will be saved.
        pose_config: Path to RTMPose config.
        pose_checkpoint: Path to RTMPose checkpoint.
        det_config: Path to detector config.
        det_checkpoint: Path to detector checkpoint.
        device: Inference device ("cpu" or "cuda").

    Returns:
        True if MP4 was successfully written, False otherwise.
    """
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video for skeleton rendering: %s", video_path)
        return False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize RTMPose tracker
    try:
        tracker = RTMPoseTracker(
            pose_config=pose_config,
            pose_checkpoint=pose_checkpoint,
            det_config=det_config,
            det_checkpoint=det_checkpoint,
            device=device,
        )
    except Exception as e:
        logger.warning("Failed to initialize RTMPoseTracker for skeleton overlay: %s", e)
        cap.release()
        return False
    
    # Try codecs in order of preference
    codec_attempts = [
        ("mp4v", "output_reference_skeleton.mp4"),
        ("MJPG", "output_reference_skeleton.avi"),
    ]
    
    video_writer = None
    output_video_path = None
    
    for codec_name, filename in codec_attempts:
        output_video_path = output_path / filename
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        trial_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        if trial_writer.isOpened():
            video_writer = trial_writer
            logger.info(
                "Skeleton overlay video writer initialized (codec=%s, fps=%d, size=%dx%d)",
                codec_name,
                fps,
                frame_width,
                frame_height,
            )
            break
        trial_writer.release()
    
    if video_writer is None:
        logger.warning("Failed to initialize video writer for skeleton overlay")
        cap.release()
        return False
    
    # Import drawing utility
    from src.ui.feedback_display import _draw_skeleton, RTMPOSE_CONNECTIONS
    
    frames_written = 0
    frame_idx = 0
    
    try:
        if tqdm is not None:
            pbar = tqdm(
                total=total_frames,
                desc="Rendering reference skeleton overlay",
                unit="frame",
                dynamic_ncols=True,
            )
        else:
            pbar = None
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Extract pose
            pose_result = tracker.process_frame(frame, frame_index=frame_idx)
            
            # Draw skeleton (neutral colors for reference)
            if pose_result is not None and pose_result.keypoints is not None:
                output_frame = _draw_skeleton(
                    frame.copy(),
                    pose_result.keypoints[:, :2],
                    joint_scores=None,  # No per-joint scores for reference
                    connections=RTMPOSE_CONNECTIONS,
                    joint_radius=4,
                    bone_thickness=2,
                )
            else:
                output_frame = frame.copy()
            
            # Write to video
            if output_frame.dtype != np.uint8:
                output_frame = np.uint8(np.clip(output_frame, 0, 255))
            video_writer.write(output_frame)
            frames_written += 1
            
            frame_idx += 1
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
    finally:
        cap.release()
        video_writer.release()
    
    if frames_written > 0 and output_video_path and Path(output_video_path).exists():
        size_mb = Path(output_video_path).stat().st_size / (1024 * 1024)
        logger.info(
            "Skeleton overlay video saved: %s (%.2f MB, %d frames)",
            output_video_path,
            size_mb,
            frames_written,
        )
        return True
    else:
        logger.warning("No frames written to skeleton overlay video")
        return False


# ---------------------------------------------------------------------------
# Reference pipeline (run once, offline)
# ---------------------------------------------------------------------------

def build_reference(
    url: str,
    output_dir: str,
    pose_config: str,
    pose_checkpoint: str,
    det_config: str,
    det_checkpoint: str,
    device: str,
    skip_cache: bool = False,
    extracted_steps_dir: str = "data/extracted_steps",
    reference_output_videos_dir: str = "data/output_videos",
) -> tuple[list[np.ndarray], list[int]]:
    """Download the tutorial, segment it, and extract per-step reference poses.

    Args:
        url: YouTube URL of the dance tutorial.
        output_dir: Root directory for downloads and processed data.
        pose_config: Path to the MMPose (RTMPose/RTMO) config file.
        pose_checkpoint: Path to the MMPose checkpoint.
        det_config: Path to the MMDet detector config.
        det_checkpoint: Path to the MMDet detector checkpoint.
        device: Inference device (``"cpu"`` or ``"cuda"``).

    Returns:
        A tuple of:
        - ``reference_features``: List of per-step feature arrays, each of
          shape ``(step_frames, feature_dim)``.
        - ``step_boundaries``: List of frame indices marking step boundaries.
    """
    # Phase 1: download video and stream raw frames directly to the model
    processor = VideoProcessor(output_dir=output_dir)
    logger.info("Downloading tutorial from %s …", url)
    video_path = processor.download(url)
    total_frames = processor.get_frame_count(video_path)
    logger.info("Processing %d source frames directly from video stream", total_frames)
    _mark_phase_complete(1, "Ingestion & Preprocessing", f"{total_frames} source frames ready")
    
    # Check cache before processing poses
    cache_dir = Path(output_dir) / "cache"
    cache_path = _get_cache_path(video_path, cache_dir)
    
    if not skip_cache:
        # Validate cache against current source video frame count.
        cached_result = _load_cached_reference(cache_path, expected_total_frames=total_frames)
        if cached_result is not None:
            return cached_result
    else:
        logger.info("--no-cache: force reprocessing poses")

    # Phase 2: segment into steps
    segmenter = TCNSegmenter()
    visual_extractor = VisualFeatureExtractor()

    # Phase 3: extract reference poses
    use_rtmpose = bool(pose_config and pose_checkpoint)
    if use_rtmpose:
        try:
            ref_tracker: RTMPoseTracker | BlazePoseTracker = RTMPoseTracker(
                pose_config=pose_config,
                pose_checkpoint=pose_checkpoint,
                det_config=det_config,
                det_checkpoint=det_checkpoint,
                device=device,
            )
            logger.info("Using RTMPoseTracker for reference extraction.")
        except Exception as exc:
            logger.warning(
                "RTMPose initialization failed (%s). Falling back to BlazePoseTracker.",
                exc,
            )
            ref_tracker = BlazePoseTracker()
    else:
        logger.info(
            "No RTMPose config/checkpoint provided. Using BlazePoseTracker for reference extraction."
        )
        ref_tracker = BlazePoseTracker()

    visual_features: list[np.ndarray] = []
    all_poses: list[np.ndarray] = []

    tracker_context = ref_tracker if isinstance(ref_tracker, BlazePoseTracker) else nullcontext(ref_tracker)
    with tracker_context as tracker:
        logger.info("Extracting reference poses from %d streamed frames …", total_frames)
        frame_stream = processor.stream_frames(video_path)
        if tqdm is not None:
            frame_stream = tqdm(
                frame_stream,
                total=total_frames,
                desc="Reference pose extraction",
                unit="frame",
                dynamic_ncols=True,
            )

        for frame_index, frame in frame_stream:
            pose_result = tracker.process_frame(frame, frame_index=frame_index)
            if pose_result is None or pose_result.keypoints_normalized is None:
                continue
            kp = pose_result.keypoints_normalized[:, :2]
            bone_vec = visual_extractor.extract(kp)
            visual_features.append(bone_vec)
            all_poses.append(kp.flatten())

    _mark_phase_complete(3, "Reference Pose Extraction", f"{len(all_poses)} valid pose frames")
    _mark_phase_complete(4, "Spatial Normalization", "Normalized keypoints prepared")

    if not visual_features:
        logger.error("No valid frames extracted. Aborting.")
        sys.exit(1)

    visual_array = np.stack(visual_features)  # (T, visual_dim)
    _, step_boundaries = segmenter.segment(visual_array)
    logger.info("Detected %d dance steps.", len(step_boundaries))
    _mark_phase_complete(2, "Action Segmentation", f"{len(step_boundaries)} step boundaries")

    # Split per-step reference feature arrays
    boundaries = [0] + step_boundaries + [len(all_poses)]
    reference_features: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            reference_features.append(np.stack(all_poses[start:end]))

    # Cache the results for next run
    _save_cached_reference(
        cache_path,
        reference_features,
        step_boundaries,
        total_frames=total_frames,
        valid_pose_frames=len(all_poses),
    )

    # Goal 1: Save extracted dance steps as individual images
    logger.info("Saving extracted dance step images...")
    _save_extracted_steps(video_path, step_boundaries, extracted_steps_dir)
    
    # Goal 2: Save skeleton overlay MP4 of reference video
    if pose_config and pose_checkpoint:
        logger.info("Saving reference video with skeleton overlay...")
        _save_reference_skeleton_overlay(
            video_path,
            step_boundaries,
            reference_output_videos_dir,
            pose_config,
            pose_checkpoint,
            det_config,
            det_checkpoint,
            device,
        )
    else:
        logger.info("Skipping skeleton overlay video (no RTMPose config/checkpoint provided)")

    return reference_features, step_boundaries


# ---------------------------------------------------------------------------
# Live feedback loop
# ---------------------------------------------------------------------------

def run_feedback_loop(
    reference_features: list[np.ndarray],
    step_boundaries: list[int],
    camera_index: int,
    window_size: int,
    device: str,
    camera_backend: str = "auto",
    save_video: bool = False,
    video_output_path: str = "output_annotated.mp4",
) -> None:
    """Run the real-time feedback loop against pre-built reference features.

    Args:
        reference_features: Per-step reference pose feature arrays.
        step_boundaries: Frame indices where each step begins.
        camera_index: OpenCV webcam index.
        window_size: S-WFDTW Sakoe-Chiba window size (frames).
        device: Inference device (unused for BlazePose, kept for consistency).
        camera_backend: Camera backend preference: "auto", "default",
            "msmf", or "dshow".
        save_video: If True, save annotated frames to a video file.
        video_output_path: Path where to save the annotated video.
    """
    scorer = SWFDTWScorer(window=window_size)
    display = FeedbackDisplay(camera_index=camera_index)
    _mark_phase_complete(6, "Temporal Alignment & Scoring", "Scoring engine initialized")

    import cv2

    backend_options: dict[str, int | None] = {
        "default": None,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
    }

    if camera_backend == "auto":
        backend_order = ["default", "msmf", "dshow"]
    else:
        backend_order = [camera_backend]

    cap = None
    selected_backend = "unknown"
    for backend_name in backend_order:
        backend_api = backend_options[backend_name]
        trial = (
            cv2.VideoCapture(camera_index)
            if backend_api is None
            else cv2.VideoCapture(camera_index, backend_api)
        )
        if trial.isOpened():
            cap = trial
            selected_backend = backend_name
            break
        trial.release()

    if cap is None:
        raise RuntimeError(
            f"Cannot open camera {camera_index} using backend '{camera_backend}'."
        )

    logger.info(
        "Camera %d opened successfully (backend=%s)",
        camera_index,
        selected_backend,
    )
    _mark_phase_complete(5, "Real-Time Edge Tracking", f"Camera {camera_index} ready")

    # Warmup: verify that at least one frame is readable before entering loop.
    first_frame = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            first_frame = frame
            break
    if first_frame is None:
        cap.release()
        raise RuntimeError(
            f"Camera {camera_index} opened but no frames were received. "
            "Close other apps using the webcam and try another --camera index."
        )

    # Setup video writer if output is requested
    video_writer = None
    if save_video:
        frame_height, frame_width = first_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30  # Default to 30 FPS if invalid
        
        # Try codecs in order of preference
        codec_attempts = [
            ("MJPG", ".avi"),   # MJPEG usually works reliably
            ("mp4v", ".mp4"),   # MP4V for MP4 files
            ("XVID", ".avi"),   # XVID fallback
        ]
        
        for codec_name, suggested_ext in codec_attempts:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            output_file = video_output_path
            if not output_file.lower().endswith(suggested_ext):
                output_file = video_output_path.rsplit(".", 1)[0] + suggested_ext
            
            trial_writer = cv2.VideoWriter(
                output_file, fourcc, fps, (frame_width, frame_height)
            )
            if trial_writer.isOpened():
                video_writer = trial_writer
                video_output_path = output_file
                logger.info(
                    "Video output enabled: writing to %s (codec=%s, fps=%d, size=%dx%d).",
                    output_file,
                    codec_name,
                    fps,
                    frame_width,
                    frame_height,
                )
                break
            trial_writer.release()
        
        if video_writer is None:
            logger.warning(
                "Failed to open video writer; tried multiple codecs. Video output disabled."
            )

    try:
        with BlazePoseTracker() as tracker:
            current_step = 0
            user_buffer: list[np.ndarray] = []
            live_pbar = None
            if tqdm is not None:
                live_pbar = tqdm(
                    desc="Live feedback",
                    unit="frame",
                    dynamic_ncols=True,
                )

            logger.info("Real-time feedback loop started. Press 'q' to quit.")
            logger.info("Opening display window '%s'...", display.window_name)
            
            # Initialize display window with error handling
            try:
                cv2.namedWindow(display.window_name, cv2.WINDOW_NORMAL)
                logger.info("Display window opened successfully.")
            except cv2.error as e:
                logger.error("Failed to create OpenCV display window: %s", e)
                logger.warning("Attempting to continue without display window...")
                # We'll skip imshow calls later if window creation failed
            
            _mark_phase_complete(7, "Real-Time UI/UX", "Display window initialized")
            frame_idx = 0
            frames_written = 0
            window_failed = False

            try:
                while True:
                    if frame_idx == 0 and first_frame is not None:
                        frame = first_frame
                        ret = True
                    else:
                        ret, frame = cap.read()

                    if not ret or frame is None:
                        logger.warning(
                            "No frame received from camera %d at loop frame %d; exiting live loop.",
                            camera_index,
                            frame_idx,
                        )
                        break

                    pose_result = tracker.process_frame(frame, frame_index=frame_idx)

                    overall_score = 0.0
                    joint_scores = None

                    if pose_result is not None and pose_result.keypoints_normalized is not None:
                        kp = pose_result.keypoints_normalized[:, :2].flatten()
                        user_buffer.append(kp)

                        # Keep a rolling window for DTW
                        max_buf = window_size * 2
                        if len(user_buffer) > max_buf:
                            user_buffer = user_buffer[-max_buf:]

                        if current_step < len(reference_features):
                            ref = reference_features[current_step]
                            user_arr = np.stack(user_buffer)
                            overall_score, _ = scorer.score(user_arr, ref)

                            # Per-joint scores for color coding
                            ref_frame = ref[min(len(user_buffer) - 1, len(ref) - 1)]
                            num_joints = pose_result.keypoints_normalized.shape[0]
                            joint_scores = scorer.score_per_joint(kp, ref_frame, num_joints)

                        # Advance step when buffer fills relative to step length
                        step_len = (
                            len(reference_features[current_step])
                            if current_step < len(reference_features)
                            else 1
                        )
                        if len(user_buffer) >= step_len and current_step + 1 < len(reference_features):
                            current_step += 1
                            user_buffer = []
                            logger.info("Advanced to step %d", current_step + 1)

                        display.update(
                            keypoints=pose_result.keypoints,
                            joint_scores=joint_scores,
                            overall_score=overall_score,
                            step_label=f"Step {current_step + 1} / {len(reference_features)}",
                        )

                    # Render and display with robust error handling
                    try:
                        annotated = display.render_frame(frame)
                        
                        # Only attempt imshow if window is available
                        if not window_failed:
                            try:
                                cv2.imshow(display.window_name, annotated)
                            except cv2.error as e:
                                logger.error("cv2.imshow failed: %s. Display window may have been closed.", e)
                                window_failed = True
                                # Continue processing without display
                        
                        # Write to output video if enabled
                        if video_writer is not None:
                            # Ensure frame is uint8 BGR for video writer
                            if annotated.dtype != np.uint8:
                                annotated = np.uint8(np.clip(annotated, 0, 255))
                            
                            write_success = video_writer.write(annotated)
                            if write_success:
                                frames_written += 1
                                if frames_written == 1:
                                    logger.info("First frame written to video successfully")
                                if frames_written % 30 == 0:
                                    logger.debug("Frames written: %d", frames_written)
                            else:
                                logger.warning("Failed to write frame %d to video file", frame_idx)
                    except Exception as e:
                        logger.error("Failed to render or display frame: %s", e, exc_info=True)
                        raise

                    frame_idx += 1

                    if live_pbar is not None:
                        live_pbar.update(1)
                        if frame_idx % 15 == 0:
                            live_pbar.set_postfix(
                                step=f"{current_step + 1}/{len(reference_features)}",
                                score=f"{overall_score:.2f}",
                            )

                    # Check for quit key and handle cv2.waitKey gracefully
                    try:
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord("q"):
                            logger.info("User pressed 'q' to quit.")
                            break
                    except cv2.error as e:
                        logger.warning("cv2.waitKey error (likely display issue): %s", e)
                        # Continue despite waitKey error
            except KeyboardInterrupt:
                logger.info("User interrupted (Ctrl+C)")
            except Exception as e:
                logger.error("Unexpected error in feedback loop: %s", e, exc_info=True)
                raise
            finally:
                if live_pbar is not None:
                    live_pbar.close()
                cap.release()
                if video_writer is not None:
                    logger.info("Closing video writer after %d frames written...", frames_written)
                    video_writer.release()
                    logger.info("Video writer closed.")
                    if Path(video_output_path).exists():
                        size_mb = Path(video_output_path).stat().st_size / (1024 * 1024)
                        logger.info(
                            "Video file created: %s (%.2f MB, %d frames)",
                            video_output_path,
                            size_mb,
                            frames_written,
                        )
                        if frames_written == 0:
                            logger.warning("Warning: No frames were written to the video file!")
                    else:
                        logger.warning("Video file was not created at %s", video_output_path)
                _mark_phase_complete(7, "Real-Time UI/UX", f"Session ended after {frame_idx} frames")
                cv2.destroyAllWindows()
    except Exception as e:
        logger.error("Error initializing live feedback: %s", e, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Interactive camera selection
# ---------------------------------------------------------------------------

def select_camera_interactive() -> int:
    """Interactively probe and select an available camera.

    Probes camera indices 0-3 with default OpenCV backend and presents
    available options to the user.

    Returns:
        Selected camera index.
    """
    import cv2

    print("\n" + "=" * 60)
    print("CAMERA SELECTION")
    print("=" * 60)
    print("\nProbing camera indices 0-3 (this takes ~2 seconds)...\n")

    available = []
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    available.append((i, f"{w}x{h}"))
                    print(f"  [{len(available)}] Camera {i} (resolution: {w}x{h})")
            cap.release()
        except Exception as e:
            pass

    if not available:
        print("\n  [!] No cameras detected on indices 0-3.")
        print("      Try: python main.py --url <url> --camera 1")
        print("      or:  python main.py --url <url> --camera 2")
        sys.exit(1)

    print(f"\nFound {len(available)} camera(s).")
    if len(available) == 1:
        selected_idx = available[0][0]
        print(f"\nUsing camera {selected_idx} (only option).\n")
        return selected_idx

    while True:
        try:
            choice = input(f"Select camera [1-{len(available)}] (default: 1): ").strip()
            if not choice:
                choice = "1"
            choice_num = int(choice)
            if 1 <= choice_num <= len(available):
                selected_idx = available[choice_num - 1][0]
                print(f"\nUsing camera {selected_idx}.\n")
                return selected_idx
            else:
                print(f"Please enter a number between 1 and {len(available)}.")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="ai_dance_coach",
        description="AI Dance Coach — real-time pose feedback from a YouTube tutorial.",
    )
    parser.add_argument("--url", required=True, help="YouTube URL of the dance tutorial.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument(
        "--camera-backend",
        default="auto",
        choices=["auto", "default", "msmf", "dshow"],
        help="Camera backend: auto, default, msmf, or dshow (default: auto).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device: 'auto' (GPU if available, else CPU), 'cpu', or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--pose-config", default="", help="Path to RTMPose/RTMO MMPose config."
    )
    parser.add_argument(
        "--pose-ckpt", default="", help="Path to RTMPose/RTMO checkpoint."
    )
    parser.add_argument(
        "--det-config", default="", help="Path to MMDet detector config."
    )
    parser.add_argument(
        "--det-ckpt", default="", help="Path to MMDet detector checkpoint."
    )
    parser.add_argument(
        "--window-size", type=int, default=10, help="S-WFDTW window size (default: 10)."
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Root directory for data (default: data/raw)."
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Force reprocessing, ignore cached frames and poses."
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save annotated video output during live feedback."
    )
    parser.add_argument(
        "--video-output", default="output_annotated.mp4", help="Path to save annotated video (default: output_annotated.mp4)."
    )
    parser.add_argument(
        "--layout", choices=["side-by-side", "top-bottom"], default="side-by-side",
        help="Display layout for reference + webcam frames (default: side-by-side)."
    )
    parser.add_argument(
        "--skip-render", action="store_true", help="Skip rendering reference video (use cached if available)."
    )
    parser.add_argument(
        "--rendered-output-dir", default="data/rendered_videos",
        help="Directory to save rendered reference frames (default: data/rendered_videos)."
    )
    parser.add_argument(
        "--extracted-steps-dir", default="data/extracted_steps",
        help="Directory to save extracted dance step images (default: data/extracted_steps)."
    )
    parser.add_argument(
        "--reference-output-videos-dir", default="data/output_videos",
        help="Directory to save reference video with skeleton overlay (default: data/output_videos)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Application entry point.

    Args:
        argv: Optional argument list for programmatic invocation.
    """
    args = parse_args(argv)

    # Setup GPU with safe fallback to CPU
    device = _setup_gpu_optimization(args.device)

    reference_features, step_boundaries = build_reference(
        url=args.url,
        output_dir=args.output_dir,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_ckpt,
        det_config=args.det_config,
        det_checkpoint=args.det_ckpt,
        device=device,
        skip_cache=args.no_cache,
        extracted_steps_dir=args.extracted_steps_dir,
        reference_output_videos_dir=args.reference_output_videos_dir,
    )

    logger.info(
        "Reference pipeline complete (%d steps). Starting live feedback on camera %d.",
        len(reference_features),
        args.camera,
    )

    run_feedback_loop(
        reference_features=reference_features,
        step_boundaries=step_boundaries,
        camera_index=args.camera,
        window_size=args.window_size,
        device=args.device,
        camera_backend=args.camera_backend,
        save_video=args.save_video,
        video_output_path=args.video_output,
    )


if __name__ == "__main__":
    main()
