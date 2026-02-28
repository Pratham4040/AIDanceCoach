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
    python main.py --url <youtube_url> [--camera 0] [--device cpu]

Arguments
---------
--url           YouTube URL of the dance tutorial to learn from.
--camera        Webcam index (default: 0).
--device        Inference device: "cpu" or "cuda" (default: "cpu").
--pose-config   Path to the MMPose config (RTMPose/RTMO).
--pose-ckpt     Path to the MMPose checkpoint.
--det-config    Path to the MMDet detector config.
--det-ckpt      Path to the MMDet checkpoint.
--window-size   S-WFDTW window size in frames (default: 10).
--output-dir    Root directory for downloaded/processed data (default: data/raw).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

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
    # Phase 1: download & extract frames
    processor = VideoProcessor(output_dir=output_dir)
    logger.info("Downloading tutorial from %s …", url)
    video_path = processor.download(url)
    logger.info("Extracting frames …")
    frame_paths = processor.extract_frames(video_path)

    # Phase 2: segment into steps
    segmenter = TCNSegmenter()
    visual_extractor = VisualFeatureExtractor()

    # Phase 3: extract reference poses
    ref_tracker = RTMPoseTracker(
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        device=device,
    )

    import cv2

    visual_features: list[np.ndarray] = []
    all_poses: list[np.ndarray] = []

    logger.info("Extracting reference poses from %d frames …", len(frame_paths))
    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            continue
        pose_result = ref_tracker.process_frame(frame, frame_index=i)
        if pose_result is None or pose_result.keypoints_normalized is None:
            continue
        kp = pose_result.keypoints_normalized[:, :2]
        bone_vec = visual_extractor.extract(kp)
        visual_features.append(bone_vec)
        all_poses.append(kp.flatten())

    if not visual_features:
        logger.error("No valid frames extracted. Aborting.")
        sys.exit(1)

    visual_array = np.stack(visual_features)  # (T, visual_dim)
    _, step_boundaries = segmenter.segment(visual_array)
    logger.info("Detected %d dance steps.", len(step_boundaries))

    # Split per-step reference feature arrays
    boundaries = [0] + step_boundaries + [len(all_poses)]
    reference_features: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            reference_features.append(np.stack(all_poses[start:end]))

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
) -> None:
    """Run the real-time feedback loop against pre-built reference features.

    Args:
        reference_features: Per-step reference pose feature arrays.
        step_boundaries: Frame indices where each step begins.
        camera_index: OpenCV webcam index.
        window_size: S-WFDTW Sakoe-Chiba window size (frames).
        device: Inference device (unused for BlazePose, kept for consistency).
    """
    scorer = SWFDTWScorer(window=window_size)
    display = FeedbackDisplay(camera_index=camera_index)

    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    with BlazePoseTracker() as tracker:
        current_step = 0
        user_buffer: list[np.ndarray] = []

        logger.info("Real-time feedback loop started. Press 'q' to quit.")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
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

                annotated = display.render_frame(frame)
                cv2.imshow(display.window_name, annotated)
                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


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
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default: cpu).",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Application entry point.

    Args:
        argv: Optional argument list for programmatic invocation.
    """
    args = parse_args(argv)

    reference_features, step_boundaries = build_reference(
        url=args.url,
        output_dir=args.output_dir,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_ckpt,
        det_config=args.det_config,
        det_checkpoint=args.det_ckpt,
        device=args.device,
    )

    run_feedback_loop(
        reference_features=reference_features,
        step_boundaries=step_boundaries,
        camera_index=args.camera,
        window_size=args.window_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
