"""
src/pose/tracker.py
====================
Phase 3 & 5: Pose Estimation

Responsibilities:
- Wrap MediaPipe BlazePose for fast, lightweight real-time tracking of the
  user's live webcam feed (33 3D landmarks — Phase 5).
- Wrap OpenMMLab RTMPose / RTMO for high-accuracy reference pose extraction
  from the instructor's segmented video (Phase 3).
- Provide spatial normalization utilities (Phase 4): translate to hip-center
  root, scale by torso length, and convert to angular limb vectors (RotJoints).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PoseResult:
    """Holds keypoints and metadata for a single frame's pose estimate.

    Attributes:
        keypoints: Float array of shape ``(num_joints, 3)`` — each row is
            ``(x, y, visibility/confidence)``.
        keypoints_normalized: Normalized keypoints (hip-centered, torso-scaled)
            with shape ``(num_joints, 3)``.  Populated after
            :meth:`PoseNormalizer.normalize` is called.
        rot_joints: Angular limb vectors (RotJoints) of shape
            ``(num_bones, 2)`` — each row is ``(angle_rad, length)``.
            Populated after :meth:`PoseNormalizer.to_rot_joints` is called.
        frame_index: Source frame index in the video.
        source: ``"blazepose"`` or ``"rtmpose"``.
    """

    keypoints: np.ndarray
    keypoints_normalized: Optional[np.ndarray] = None
    rot_joints: Optional[np.ndarray] = None
    frame_index: int = 0
    source: str = "unknown"


# ---------------------------------------------------------------------------
# Normalization utilities (Phase 4)
# ---------------------------------------------------------------------------

class PoseNormalizer:
    """Spatial normalization for pose keypoints.

    Translates all coordinates to a root origin (hip center) and scales them
    by the torso length (neck-to-hip distance), then optionally converts to
    angular limb vectors (RotJoints).

    Args:
        left_hip_idx: Keypoint index for the left hip.
        right_hip_idx: Keypoint index for the right hip.
        left_shoulder_idx: Keypoint index for the left shoulder.
        right_shoulder_idx: Keypoint index for the right shoulder.
        bone_pairs: Skeleton connectivity for RotJoint computation.
    """

    def __init__(
        self,
        left_hip_idx: int = 23,
        right_hip_idx: int = 24,
        left_shoulder_idx: int = 11,
        right_shoulder_idx: int = 12,
        bone_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.left_hip_idx = left_hip_idx
        self.right_hip_idx = right_hip_idx
        self.left_shoulder_idx = left_shoulder_idx
        self.right_shoulder_idx = right_shoulder_idx
        self.bone_pairs = bone_pairs or [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 23), (12, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
        ]

    def normalize(self, keypoints: np.ndarray) -> np.ndarray:
        """Translate to hip-center root and scale by torso length.

        Args:
            keypoints: Array of shape ``(num_joints, 2)`` or ``(num_joints, 3)``.

        Returns:
            Normalized array of the same shape.
        """
        kp = keypoints.copy()
        xy = kp[:, :2]

        left_hip = xy[self.left_hip_idx]
        right_hip = xy[self.right_hip_idx]
        hip_center = (left_hip + right_hip) / 2.0
        xy -= hip_center

        left_shoulder = xy[self.left_shoulder_idx]
        right_shoulder = xy[self.right_shoulder_idx]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        torso_length = float(np.linalg.norm(shoulder_center))  # distance from hip to shoulder center
        if torso_length > 1e-6:
            xy /= torso_length

        kp[:, :2] = xy
        return kp

    def to_rot_joints(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert (x, y) keypoints to angular limb vectors (RotJoints).

        Args:
            keypoints: Normalized array of shape ``(num_joints, 2+)``.

        Returns:
            Array of shape ``(num_bones, 2)`` where each row is
            ``(angle_radians, bone_length)``.
        """
        rot_joints = []
        xy = keypoints[:, :2]
        for a, b in self.bone_pairs:
            delta = xy[b] - xy[a]
            angle = float(np.arctan2(delta[1], delta[0]))
            length = float(np.linalg.norm(delta))
            rot_joints.append([angle, length])
        return np.array(rot_joints, dtype=np.float32)


# ---------------------------------------------------------------------------
# BlazePose tracker (Phase 5)
# ---------------------------------------------------------------------------

class BlazePoseTracker:
    """Lightweight real-time pose tracker using MediaPipe BlazePose.

    Wraps ``mediapipe.solutions.pose`` to produce :class:`PoseResult` objects
    from BGR frames captured from a webcam.

    Args:
        model_complexity: BlazePose model complexity (0=Lite, 1=Full, 2=Heavy).
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence: Minimum confidence to continue tracking.
        static_image_mode: When ``True``, treat each frame as an independent
            image (slower but more accurate for offline processing).
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )
        self.normalizer = PoseNormalizer()
        logger.info("BlazePoseTracker initialized (complexity=%d)", model_complexity)

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> Optional[PoseResult]:
        """Detect pose landmarks in a single BGR frame.

        Args:
            frame_bgr: Input frame in BGR format (as returned by OpenCV).
            frame_index: Source frame index.

        Returns:
            :class:`PoseResult` with 33 keypoints, or ``None`` if no pose was
            detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return None

        h, w = frame_bgr.shape[:2]
        kps = []
        for lm in results.pose_landmarks.landmark:
            kps.append([lm.x * w, lm.y * h, lm.visibility])
        keypoints = np.array(kps, dtype=np.float32)

        normalized = self.normalizer.normalize(keypoints)
        rot_joints = self.normalizer.to_rot_joints(normalized)

        return PoseResult(
            keypoints=keypoints,
            keypoints_normalized=normalized,
            rot_joints=rot_joints,
            frame_index=frame_index,
            source="blazepose",
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    def __enter__(self) -> "BlazePoseTracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# RTMPose tracker (Phase 3)
# ---------------------------------------------------------------------------

class RTMPoseTracker:
    """High-accuracy reference pose extractor using OpenMMLab RTMPose / RTMO.

    Requires ``mmpose`` and ``mmdet`` to be installed.  This class wraps the
    MMPose inference API and exposes the same :meth:`process_frame` interface
    as :class:`BlazePoseTracker`.

    Args:
        pose_config: Path to the MMPose config file (RTMO or RTMPose).
        pose_checkpoint: Path to the model checkpoint.
        det_config: Path to the MMDet detector config.
        det_checkpoint: Path to the detector checkpoint.
        device: Inference device (``"cpu"`` or ``"cuda:0"`` etc.).
        bbox_thr: Bounding-box detection score threshold.
        kpt_thr: Keypoint confidence threshold.
    """

    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
        det_config: Optional[str] = None,
        det_checkpoint: Optional[str] = None,
        device: str = "cpu",
        bbox_thr: float = 0.3,
        kpt_thr: float = 0.3,
    ) -> None:
        try:
            from mmpose.apis import init_model as init_pose_model, inference_topdown
            from mmdet.apis import init_detector, inference_detector
        except ImportError as exc:
            raise ImportError(
                "mmpose and mmdet are required for RTMPoseTracker. "
                "Install them with: pip install mmpose mmdet"
            ) from exc

        self._init_pose_model = init_pose_model
        self._inference_topdown = inference_topdown

        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        self.normalizer = PoseNormalizer(
            left_hip_idx=11, right_hip_idx=12,
            left_shoulder_idx=5, right_shoulder_idx=6,
        )
        self.bbox_thr = bbox_thr
        self.kpt_thr = kpt_thr

        if det_config and det_checkpoint:
            self.det_model = init_detector(det_config, det_checkpoint, device=device)
            self._inference_detector = inference_detector
        else:
            self.det_model = None
            logger.warning(
                "No detector provided; RTMPoseTracker will use full-image inference."
            )

        logger.info("RTMPoseTracker initialized on device=%s", device)

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> Optional[PoseResult]:
        """Extract reference keypoints from a single BGR frame.

        Args:
            frame_bgr: Input frame in BGR format.
            frame_index: Source frame index.

        Returns:
            :class:`PoseResult` with detected keypoints, or ``None`` if no
            person was detected above the confidence threshold.
        """
        if self.det_model is not None:
            det_result = self._inference_detector(self.det_model, frame_bgr)
            bboxes = det_result.pred_instances.bboxes.cpu().numpy()
            scores = det_result.pred_instances.scores.cpu().numpy()
            bboxes = bboxes[scores > self.bbox_thr]
            if len(bboxes) == 0:
                return None
        else:
            h, w = frame_bgr.shape[:2]
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

        pose_results = self._inference_topdown(self.pose_model, frame_bgr, bboxes)
        if not pose_results:
            return None

        # Take the highest-confidence person
        best = pose_results[0]
        keypoints = best.pred_instances.keypoints[0]  # (num_joints, 2 or 3)

        if keypoints.shape[1] == 2:
            scores_kp = np.ones((keypoints.shape[0], 1), dtype=np.float32)
            keypoints = np.concatenate([keypoints, scores_kp], axis=1)

        normalized = self.normalizer.normalize(keypoints)
        rot_joints = self.normalizer.to_rot_joints(normalized)

        return PoseResult(
            keypoints=keypoints.astype(np.float32),
            keypoints_normalized=normalized,
            rot_joints=rot_joints,
            frame_index=frame_index,
            source="rtmpose",
        )
