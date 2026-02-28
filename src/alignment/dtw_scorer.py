"""
src/alignment/dtw_scorer.py
============================
Phase 6: Temporal Alignment & Scoring

Responsibilities:
- Implement Spatial-Weighted Fast Dynamic Time Warping (S-WFDTW) with a
  sliding window constraint and early-termination pruning.
- Compute per-joint/per-limb similarity scores used for UI color coding.
- Calculate a final composite score via weighted fusion of Euclidean Distance
  (spatial position) and Cosine Similarity (angular alignment).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean distance between two pose feature vectors.

    Args:
        a: 1-D float array representing the first pose (flattened keypoints or
           bone vectors).
        b: 1-D float array of the same length as *a*.

    Returns:
        Scalar Euclidean distance.
    """
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: 1-D float array.
        b: 1-D float array of the same length as *a*.

    Returns:
        Scalar in ``[-1, 1]``.  Returns ``0.0`` for zero vectors.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def combined_distance(
    a: np.ndarray,
    b: np.ndarray,
    euclidean_weight: float = 0.5,
    cosine_weight: float = 0.5,
) -> float:
    """Weighted fusion of Euclidean distance and angular dissimilarity.

    Args:
        a: 1-D float feature vector for frame *t* (user).
        b: 1-D float feature vector for frame *t* (reference).
        euclidean_weight: Weight applied to the normalized Euclidean distance.
        cosine_weight: Weight applied to ``1 - cosine_similarity``.

    Returns:
        Combined distance scalar (lower = better match).
    """
    euc = euclidean_distance(a, b)
    cos_dist = 1.0 - cosine_similarity(a, b)
    return euclidean_weight * euc + cosine_weight * cos_dist


# ---------------------------------------------------------------------------
# S-WFDTW core
# ---------------------------------------------------------------------------

class SWFDTWScorer:
    """Spatial-Weighted Fast Dynamic Time Warping scorer.

    Aligns the user's live motion sequence with the instructor's reference
    sequence using a sliding window constraint (Sakoe-Chiba band) and
    early-termination pruning for real-time performance.

    Args:
        window: Half-width of the Sakoe-Chiba band (in frames).  Limits the
            maximum temporal deviation allowed between the two sequences.
        euclidean_weight: Weight for Euclidean distance in the combined metric.
        cosine_weight: Weight for cosine dissimilarity in the combined metric.
        joint_weights: Optional 1-D array of per-joint spatial weights.  When
            provided, the distance computation weights joints by anatomical
            importance (e.g., hands and feet weighted higher).  Must match the
            feature dimensionality.
        early_termination_factor: If the running cost exceeds
            ``best_path_cost * early_termination_factor``, prune that path.
    """

    def __init__(
        self,
        window: int = 10,
        euclidean_weight: float = 0.5,
        cosine_weight: float = 0.5,
        joint_weights: Optional[np.ndarray] = None,
        early_termination_factor: float = 2.0,
    ) -> None:
        self.window = window
        self.euclidean_weight = euclidean_weight
        self.cosine_weight = cosine_weight
        self.joint_weights = joint_weights
        self.early_termination_factor = early_termination_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        user_seq: np.ndarray,
        ref_seq: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Align *user_seq* to *ref_seq* and return an overall similarity score.

        Args:
            user_seq: Float array of shape ``(T_user, feature_dim)`` — the
                user's live motion features (e.g., flattened normalized
                keypoints or rot-joint vectors).
            ref_seq: Float array of shape ``(T_ref, feature_dim)`` — the
                instructor's reference features.

        Returns:
            A tuple of:
            - ``score``: Float in ``[0, 1]`` where ``1`` = perfect match.
            - ``cost_matrix``: DTW accumulated cost matrix of shape
              ``(T_user, T_ref)`` for downstream analysis.
        """
        dtw_cost, cost_matrix = self._fast_dtw(user_seq, ref_seq)
        # Normalize: map cost to similarity in [0, 1]
        max_possible = float(user_seq.shape[0] + ref_seq.shape[0])
        similarity = float(np.exp(-dtw_cost / (max_possible + 1e-9)))
        return similarity, cost_matrix

    def score_per_joint(
        self,
        user_frame: np.ndarray,
        ref_frame: np.ndarray,
        num_joints: int,
    ) -> np.ndarray:
        """Compute per-joint similarity scores for a single frame pair.

        Used by the UI to determine limb color coding (Green/Yellow/Red).

        Args:
            user_frame: Float array of shape ``(num_joints * 2,)`` or
                ``(num_joints, 2)`` — user's current frame features.
            ref_frame: Float array of the same shape — reference frame.
            num_joints: Number of joints.

        Returns:
            Float array of shape ``(num_joints,)`` with similarity in ``[0, 1]``
            per joint.
        """
        u = user_frame.reshape(num_joints, -1)
        r = ref_frame.reshape(num_joints, -1)
        scores = np.zeros(num_joints, dtype=np.float32)
        for j in range(num_joints):
            dist = combined_distance(
                u[j], r[j],
                euclidean_weight=self.euclidean_weight,
                cosine_weight=self.cosine_weight,
            )
            scores[j] = float(np.exp(-dist))
        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between frames *a* and *b* with optional spatial weighting.

        Args:
            a: 1-D feature vector for the user's frame.
            b: 1-D feature vector for the reference frame.

        Returns:
            Weighted combined distance.
        """
        if self.joint_weights is not None:
            w = self.joint_weights
            a = a * w
            b = b * w
        return combined_distance(a, b, self.euclidean_weight, self.cosine_weight)

    def _fast_dtw(
        self,
        user_seq: np.ndarray,
        ref_seq: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Sakoe-Chiba windowed DTW with early-termination pruning.

        Args:
            user_seq: Array of shape ``(T_user, D)``.
            ref_seq: Array of shape ``(T_ref, D)``.

        Returns:
            Tuple of ``(minimum_cost, accumulated_cost_matrix)``.
        """
        T_u = user_seq.shape[0]
        T_r = ref_seq.shape[0]
        INF = np.inf

        dtw = np.full((T_u, T_r), INF, dtype=np.float64)
        dtw[0, 0] = self._weighted_distance(user_seq[0], ref_seq[0])

        best_cost = INF

        for i in range(T_u):
            j_start = max(0, i - self.window)
            j_end = min(T_r, i + self.window + 1)

            row_min = INF
            for j in range(j_start, j_end):
                cost = self._weighted_distance(user_seq[i], ref_seq[j])

                if i == 0 and j == 0:
                    dtw[i, j] = cost
                elif i == 0:
                    dtw[i, j] = cost + dtw[i, j - 1] if j > 0 and dtw[i, j - 1] < INF else INF
                elif j == 0:
                    dtw[i, j] = cost + dtw[i - 1, j] if i > 0 and dtw[i - 1, j] < INF else INF
                else:
                    prev = INF
                    if dtw[i - 1, j] < INF:
                        prev = min(prev, dtw[i - 1, j])
                    if dtw[i, j - 1] < INF:
                        prev = min(prev, dtw[i, j - 1])
                    if dtw[i - 1, j - 1] < INF:
                        prev = min(prev, dtw[i - 1, j - 1])
                    dtw[i, j] = cost + prev if prev < INF else INF

                # Early-termination pruning
                if dtw[i, j] < row_min:
                    row_min = dtw[i, j]

            if row_min < best_cost:
                best_cost = row_min
            elif row_min > best_cost * self.early_termination_factor:
                logger.debug("Early termination at row %d", i)
                break

        final_cost = dtw[T_u - 1, T_r - 1]
        if final_cost == INF:
            final_cost = best_cost
        return float(final_cost), dtw
