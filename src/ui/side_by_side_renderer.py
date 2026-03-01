"""
src/ui/side_by_side_renderer.py
==============================
Side-by-Side Display Composition

Responsibilities:
- Composite reference frame (left) + webcam frame (right) into single display
- Overlay metadata (step, score, guidance text)
- Support both side-by-side and top-bottom layouts
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from src.ui.feedback_display import _draw_skeleton, BLAZEPOSE_CONNECTIONS

logger = logging.getLogger(__name__)


class SideBySideRenderer:
    """Composite reference and webcam frames with metadata overlay.
    
    Args:
        layout: Display layout ("side-by-side" or "top-bottom").
        window_name: Title for the display window.
    """

    def __init__(
        self,
        layout: Literal["side-by-side", "top-bottom"] = "side-by-side",
        window_name: str = "AI Dance Coach - Reference vs You",
    ):
        self.layout = layout
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_color = (255, 255, 255)  # White
        self.font_thickness = 1

    def composite_frames(
        self,
        reference_frame: Optional[np.ndarray],
        webcam_frame: np.ndarray,
        webcam_annotated: np.ndarray,
        step_label: str = "Step 1 / 40",
        overall_score: float = 0.0,
    ) -> np.ndarray:
        """Composite reference and webcam frames side-by-side with overlay.
        
        Args:
            reference_frame: Reference video frame with RTMPose skeleton (or None).
            webcam_frame: Original webcam frame.
            webcam_annotated: Webcam frame with BlazePose skeleton and color-coding.
            step_label: Step progress label (e.g., "Step 1 / 40").
            overall_score: Overall alignment score [0, 1].
            
        Returns:
            Composite BGR frame ready for display/writing to video.
        """
        h_ref, w_ref = (480, 640)  # Target dimensions
        
        # Default black reference if not available
        if reference_frame is None:
            ref = np.zeros((h_ref, w_ref, 3), dtype=np.uint8)
            ref_text = "No Reference Frame"
            cv2.putText(
                ref,
                ref_text,
                (w_ref // 2 - 100, h_ref // 2),
                self.font,
                self.font_scale,
                (100, 100, 100),
                self.font_thickness,
            )
        else:
            ref = cv2.resize(reference_frame, (w_ref, h_ref))
        
        # Resize webcam frame
        web = cv2.resize(webcam_annotated, (w_ref, h_ref))
        
        if self.layout == "side-by-side":
            # Horizontal composition (1280 x 480)
            composite = np.hstack([ref, web])
            
            # Divider line in middle
            mid_x = w_ref
            cv2.line(
                composite,
                (mid_x, 0),
                (mid_x, h_ref),
                (100, 100, 100),
                2,
            )
            
            # Center labels (above divider)
            cv2.putText(
                composite,
                "REFERENCE",
                (w_ref // 2 - 50, 30),
                self.font,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                composite,
                "YOU",
                (w_ref + w_ref // 2 - 20, 30),
                self.font,
                0.7,
                (200, 200, 200),
                2,
            )
            
            # Bottom info panel
            info_height = 60
            info_panel = np.zeros((info_height, composite.shape[1], 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)  # Dark gray background
            
            # Step and score text
            step_text = f"Step: {step_label}"
            score_text = f"Score: {overall_score:.2f}"
            
            cv2.putText(
                info_panel,
                step_text,
                (20, 25),
                self.font,
                0.7,
                self.font_color,
                self.font_thickness,
            )
            cv2.putText(
                info_panel,
                score_text,
                (20, 50),
                self.font,
                0.7,
                self.font_color,
                self.font_thickness,
            )
            
            # Append info panel below
            composite = np.vstack([composite, info_panel])
            
        else:  # top-bottom
            # Vertical composition (640 x 1020)
            composite = np.vstack([ref, web])
            
            # Divider line
            mid_y = h_ref
            cv2.line(
                composite,
                (0, mid_y),
                (w_ref, mid_y),
                (100, 100, 100),
                2,
            )
            
            # Side labels
            cv2.putText(
                composite,
                "REFERENCE",
                (w_ref // 2 - 70, 30),
                self.font,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                composite,
                "YOU",
                (w_ref // 2 - 20, h_ref + 30),
                self.font,
                0.7,
                (200, 200, 200),
                2,
            )
            
            # Bottom info panel
            info_height = 40
            info_panel = np.zeros((info_height, w_ref, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)
            
            step_text = f"{step_label} | Score: {overall_score:.2f}"
            cv2.putText(
                info_panel,
                step_text,
                (10, 25),
                self.font,
                0.6,
                self.font_color,
                self.font_thickness,
            )
            
            composite = np.vstack([composite, info_panel])
        
        return composite
