"""
src/ui/reference_video_renderer.py
==================================
Reference Video Rendering

Responsibilities:
- Process tutorial video frame-by-frame
- Extract RTMPose skeletons for each frame
- Render skeletons with neutral colors onto frames
- Cache rendered frames to disk for fast retrieval during live feedback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.pose.tracker import RTMPoseTracker
from src.ui.feedback_display import _draw_skeleton, RTMPOSE_CONNECTIONS

logger = logging.getLogger(__name__)


class ReferenceVideoRenderer:
    """Renders RTMPose skeletons on reference tutorial video frames.
    
    Args:
        output_dir: Directory to cache rendered reference frames.
    """

    def __init__(self, output_dir: str | Path = "data/rendered_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rendered_frames: dict[int, dict[int, str]] = {}  # {step_id: {frame_in_step: path}}

    def process_video(
        self,
        video_path: str,
        step_boundaries: list[int],
        video_hash: str,
        device: str = "cuda",
    ) -> dict[int, dict[int, str]]:
        """Process tutorial video and render RTMPose skeletons for all frames.
        
        Args:
            video_path: Path to tutorial video file.
            step_boundaries: Frame indices where each step begins.
            video_hash: Hash of video file for organizing cache.
            device: Device for RTMPose inference ("cuda" or "cpu").
            
        Returns:
            Dictionary mapping step_id → {frame_in_step: path_to_rendered_frame}
        """
        logger.info("Rendering reference video frames with RTMPose skeletons...")
        
        # Create step-specific output directory
        video_output_dir = self.output_dir / video_hash / "reference_frames"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Add sentinel boundary at the end
        boundaries = list(step_boundaries) + [total_frames]
        
        with RTMPoseTracker(device=device) as tracker:
            frame_idx = 0
            step_id = 0
            step_frame_count = 0
            rendered_frames: dict[int, dict[int, str]] = {}
            
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(
                    total=total_frames,
                    desc="Rendering reference frames",
                    unit="frame",
                    dynamic_ncols=True,
                )
            except Exception:
                pbar = None
            
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Determine which step we're in
                while step_id < len(boundaries) - 1 and frame_idx >= boundaries[step_id + 1]:
                    step_id += 1
                    step_frame_count = 0
                
                # Extract and render pose
                pose_result = tracker.process_frame(frame, frame_index=frame_idx)
                
                # Draw skeleton (neutral colors for reference)
                if pose_result is not None and pose_result.keypoints is not None:
                    # Use neutral color mapping (no scores for reference)
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
                
                # Save rendered frame
                step_dir = video_output_dir / f"step_{step_id:02d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                
                frame_path = step_dir / f"frame_{step_frame_count:03d}.png"
                cv2.imwrite(str(frame_path), output_frame)
                
                # Record path
                if step_id not in rendered_frames:
                    rendered_frames[step_id] = {}
                rendered_frames[step_id][step_frame_count] = str(frame_path)
                
                frame_idx += 1
                step_frame_count += 1
                
                if pbar is not None:
                    pbar.update(1)
            
            if pbar is not None:
                pbar.close()
            
            cap.release()
        
        self.rendered_frames = rendered_frames
        logger.info("Rendered %d reference frames across %d steps", frame_idx, len(rendered_frames))
        return rendered_frames

    def get_frame_at_step(self, step_id: int, frame_in_step: int = 0) -> Optional[np.ndarray]:
        """Retrieve rendered reference frame for a given step.
        
        Args:
            step_id: Step index.
            frame_in_step: Frame index within the step (default: first frame).
            
        Returns:
            BGR numpy array, or None if frame doesn't exist.
        """
        if step_id not in self.rendered_frames:
            logger.warning("Step %d not found in rendered frames", step_id)
            return None
        
        step_frames = self.rendered_frames[step_id]
        if frame_in_step not in step_frames:
            # Return first frame of step as fallback
            frame_in_step = min(step_frames.keys())
        
        frame_path = step_frames.get(frame_in_step)
        if frame_path and Path(frame_path).exists():
            return cv2.imread(frame_path)
        
        logger.warning(
            "Rendered frame not found for step=%d, frame_in_step=%d",
            step_id,
            frame_in_step,
        )
        return None
