"""
src/ui/feedback_display.py
===========================
Phase 7: Real-Time UI/UX

Responsibilities:
- Render the user's live webcam feed with an overlaid dynamic wireframe.
- Color-code skeleton limbs based on S-WFDTW per-joint similarity scores:
    Green  (score >= 0.75) — correct alignment
    Yellow (score >= 0.40) — minor deviation
    Red    (score <  0.40) — critical flaw
- Display an overall session score and per-step feedback text.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color constants (BGR)
# ---------------------------------------------------------------------------

COLOR_CORRECT: Tuple[int, int, int] = (0, 255, 0)    # Green
COLOR_MINOR: Tuple[int, int, int] = (0, 255, 255)    # Yellow
COLOR_CRITICAL: Tuple[int, int, int] = (0, 0, 255)   # Red
COLOR_NEUTRAL: Tuple[int, int, int] = (200, 200, 200)  # Grey (no score)


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

# BlazePose 33-landmark connections — each tuple is (joint_a, joint_b)
BLAZEPOSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),
]

# RTMPose 17-point COCO connections (10 bones)
RTMPOSE_CONNECTIONS: List[Tuple[int, int]] = [
    (5, 7), (7, 9),       # Left arm
    (6, 8), (8, 10),      # Right arm
    (5, 11), (6, 12),     # Torso
    (11, 13), (13, 15),   # Left leg
    (12, 14), (14, 16),   # Right leg
]


def _get_bone_joint_map(connections: List[Tuple[int, int]]) -> Dict[int, int]:
    """Dynamically generate bone-to-joint mapping for skeleton topology.
    
    Args:
        connections: List of (joint_a, joint_b) tuples forming the skeleton.
        
    Returns:
        Mapping from bone index to distal joint index for color-coding.
    """
    return {i: conn[1] for i, conn in enumerate(connections)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_color(score: Optional[float]) -> Tuple[int, int, int]:
    """Map a similarity score to a BGR display color.

    Args:
        score: Similarity value in ``[0, 1]``, or ``None`` if unavailable.

    Returns:
        BGR color tuple.
    """
    if score is None:
        return COLOR_NEUTRAL
    if score >= 0.75:
        return COLOR_CORRECT
    if score >= 0.40:
        return COLOR_MINOR
    return COLOR_CRITICAL


def _draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_scores: Optional[np.ndarray],
    connections: List[Tuple[int, int]],
    joint_radius: int = 5,
    bone_thickness: int = 2,
    source: str = "blazepose",
) -> np.ndarray:
    """Draw a wireframe skeleton on *frame*.

    Args:
        frame: BGR image to draw on (modified in-place).
        keypoints: Array of shape ``(num_joints, 2+)`` — pixel coordinates.
        joint_scores: Optional float array of shape ``(num_joints,)`` with
            per-joint similarity scores in ``[0, 1]``.
        connections: List of ``(joint_a, joint_b)`` tuples.
        joint_radius: Radius in pixels for joint circles.
        bone_thickness: Line thickness in pixels for bone segments.
        source: Skeleton source ("blazepose" or "rtmpose") for topology mapping.

    Returns:
        The annotated frame (same object as *frame*).
    """
    num_joints = keypoints.shape[0]
    
    # Get dynamic bone-to-joint mapping for this skeleton topology
    bone_joint_map = _get_bone_joint_map(connections)

    # Draw bones
    for conn_idx, (a, b) in enumerate(connections):
        if a >= num_joints or b >= num_joints:
            continue
        pt_a = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        pt_b = (int(keypoints[b, 0]), int(keypoints[b, 1]))

        distal_joint = bone_joint_map.get(conn_idx, b)
        score = float(joint_scores[distal_joint]) if joint_scores is not None else None
        color = _score_to_color(score)

        cv2.line(frame, pt_a, pt_b, color, bone_thickness, lineType=cv2.LINE_AA)

    # Draw joints
    for j in range(num_joints):
        pt = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        score = float(joint_scores[j]) if joint_scores is not None else None
        color = _score_to_color(score)
        cv2.circle(frame, pt, joint_radius, color, -1, lineType=cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main display class
# ---------------------------------------------------------------------------

class FeedbackDisplay:
    """Real-time feedback rendering loop using OpenCV.

    Opens a webcam (or accepts pre-captured frames) and:
    1. Calls ``pose_callback`` to get the current user pose.
    2. Calls ``score_callback`` to get per-joint similarity scores.
    3. Renders the color-coded skeleton and overall score overlay.

    Args:
        window_name: Title of the OpenCV display window.
        camera_index: Index of the webcam (passed to ``cv2.VideoCapture``).
        connections: Skeleton topology connections.  Defaults to BlazePose.
        joint_radius: Radius (px) for joint circles.
        bone_thickness: Thickness (px) for bone lines.
        score_green_threshold: Minimum score for green (correct) coloring.
        score_yellow_threshold: Minimum score for yellow (minor) coloring.
    """

    def __init__(
        self,
        window_name: str = "AI Dance Coach",
        camera_index: int = 0,
        connections: Optional[List[Tuple[int, int]]] = None,
        joint_radius: int = 5,
        bone_thickness: int = 2,
        score_green_threshold: float = 0.75,
        score_yellow_threshold: float = 0.40,
    ) -> None:
        self.window_name = window_name
        self.camera_index = camera_index
        self.connections = connections or BLAZEPOSE_CONNECTIONS
        self.joint_radius = joint_radius
        self.bone_thickness = bone_thickness
        self.score_green_threshold = score_green_threshold
        self.score_yellow_threshold = score_yellow_threshold

        # Public mutable state — updated by the pipeline each frame
        self.current_keypoints: Optional[np.ndarray] = None
        self.current_joint_scores: Optional[np.ndarray] = None
        self.overall_score: float = 0.0
        self.step_label: str = ""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_frames: int = 0) -> None:
        """Start the real-time feedback loop.

        Reads frames from the webcam, renders the skeleton overlay, and
        displays the window.  Press **q** to quit.

        Args:
            max_frames: If > 0, stop after processing this many frames
                (useful for testing).
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        # Initialize window with error handling
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            logger.debug("Display window '%s' created successfully", self.window_name)
        except cv2.error as e:
            logger.error("Failed to create display window: %s", e)
            cap.release()
            raise RuntimeError(f"Cannot create display window: {e}") from e
        
        frame_count = 0
        window_failed = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(
                        "Failed to grab frame from camera %d (ret=%s, frame=%s)",
                        self.camera_index,
                        ret,
                        "None" if frame is None else "valid",
                    )
                    break

                # Render frame with error handling
                try:
                    annotated = self.render_frame(frame)
                except Exception as e:
                    logger.error("Failed to render frame: %s", e, exc_info=True)
                    break
                
                # Display with error handling
                if not window_failed:
                    try:
                        cv2.imshow(self.window_name, annotated)
                    except cv2.error as e:
                        logger.error("cv2.imshow failed: %s. Continuing without display.", e)
                        window_failed = True

                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    logger.info("Reached max_frames limit (%d frames)", frame_count)
                    break

                # Handle quit key gracefully
                try:
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord("q"):
                        logger.info("User pressed 'q' to quit")
                        break
                except cv2.error as e:
                    logger.warning("cv2.waitKey error: %s", e)
                    # Continue despite waitKey error
        except KeyboardInterrupt:
            logger.info("User interrupted (Ctrl+C)")
        except Exception as e:
            logger.error("Unexpected error in feedback loop: %s", e, exc_info=True)
            raise
        finally:
            logger.debug("Closing camera and display window...")
            cap.release()
            try:
                cv2.destroyAllWindows()
                logger.debug("Display windows closed")
            except cv2.error as e:
                logger.warning("Error closing display windows: %s", e)
            logger.info("Feedback loop finished after %d frames", frame_count)

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Annotate a single frame with skeleton and score overlays.

        Args:
            frame: Input BGR frame.

        Returns:
            Annotated BGR frame.
        """
        output = frame.copy()

        if self.current_keypoints is not None:
            output = _draw_skeleton(
                output,
                self.current_keypoints,
                self.current_joint_scores,
                self.connections,
                joint_radius=self.joint_radius,
                bone_thickness=self.bone_thickness,
            )

        self._draw_score_overlay(output)
        return output

    def update(
        self,
        keypoints: np.ndarray,
        joint_scores: Optional[np.ndarray] = None,
        overall_score: float = 0.0,
        step_label: str = "",
    ) -> None:
        """Update the display state for the next rendered frame.

        Args:
            keypoints: Current user keypoints, shape ``(num_joints, 2+)``.
            joint_scores: Per-joint similarity scores, shape ``(num_joints,)``.
            overall_score: Aggregated session score in ``[0, 1]``.
            step_label: Human-readable label for the current dance step.
        """
        self.current_keypoints = keypoints
        self.current_joint_scores = joint_scores
        self.overall_score = overall_score
        self.step_label = step_label

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _draw_score_overlay(self, frame: np.ndarray) -> None:
        """Draw the score panel and step label onto *frame* in-place.

        Args:
            frame: BGR image to annotate.
        """
        h, w = frame.shape[:2]
        score_pct = int(self.overall_score * 100)

        # Semi-transparent background strip at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Score text
        score_color = _score_to_color(self.overall_score)
        cv2.putText(
            frame,
            f"Score: {score_pct}%",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            score_color,
            2,
            cv2.LINE_AA,
        )

        # Step label
        if self.step_label:
            cv2.putText(
                frame,
                self.step_label,
                (w // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Legend
        legend_items = [
            ("Correct", COLOR_CORRECT),
            ("Minor", COLOR_MINOR),
            ("Critical", COLOR_CRITICAL),
        ]
        for idx, (label, color) in enumerate(legend_items):
            x = w - 130
            y = 30 + idx * 25
            cv2.circle(frame, (x, y - 6), 6, color, -1)
            cv2.putText(
                frame, label, (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
