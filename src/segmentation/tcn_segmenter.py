"""
src/segmentation/tcn_segmenter.py
==================================
Phase 2: Action Segmentation

Responsibilities:
- Extract visual features (skeletal bone vectors) from video frames.
- Extract audio features (Mel spectrogram) via librosa.
- Feed multi-modal features into a Temporal Convolutional Network (TCN).
- Output a segmentation probability curve and use peak-picking to identify
  discrete dance steps.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class _DilatedResidualBlock(nn.Module):
    """A single dilated causal residual block used inside the TCN.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output (and residual) channels.
        dilation: Dilation factor for the causal convolution.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability applied after each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self._causal_crop = padding  # crop to maintain causality

    def _causal_pad_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future-context introduced by symmetric padding."""
        return x[:, :, : -self._causal_crop] if self._causal_crop else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self._causal_pad_crop(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._causal_pad_crop(self.conv2(out)))
        out = self.dropout(out)
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for dance step segmentation.

    The model takes a concatenated feature vector (visual + audio) at each
    time step and outputs per-frame segmentation probabilities.

    Args:
        input_dim: Dimensionality of the per-frame feature vector.
        num_channels: Number of feature channels in each TCN block.
        num_layers: Number of dilated residual blocks (dilation doubles each
            layer: 1, 2, 4, …).
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_channels: int = 64,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else num_channels
            layers.append(
                _DilatedResidualBlock(
                    in_ch, num_channels, dilation=2 ** i,
                    kernel_size=kernel_size, dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch, input_dim, time_steps)``.

        Returns:
            Float tensor of shape ``(batch, 1, time_steps)`` containing
            per-frame segmentation probabilities in ``[0, 1]``.
        """
        out = self.network(x)
        return self.sigmoid(self.output_layer(out))


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class AudioFeatureExtractor:
    """Extracts Mel spectrogram features from an audio file or array.

    Args:
        sr: Target sample rate.
        n_mels: Number of Mel filter banks.
        hop_length: Hop length for STFT (controls time resolution).
        n_fft: FFT window size.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract(self, audio_path: str) -> np.ndarray:
        """Compute a log-scaled Mel spectrogram from *audio_path*.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).

        Returns:
            Float32 array of shape ``(n_mels, time_frames)``.
        """
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft,
        )
        log_mel: np.ndarray = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return log_mel


class VisualFeatureExtractor:
    """Extracts skeletal bone-vector features from precomputed keypoints.

    Given per-frame keypoints (from PoseTracker), this class computes bone
    vectors — the directed segment between connected joint pairs — which are
    rotation-invariant with respect to the root.

    Args:
        bone_pairs: List of ``(joint_a_idx, joint_b_idx)`` tuples defining
            the skeleton topology.  Defaults to a subset of BlazePose bones.
    """

    # Default BlazePose bone connections (subset of 33 landmarks)
    _DEFAULT_BONE_PAIRS: List[Tuple[int, int]] = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso sides
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    def __init__(
        self,
        bone_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.bone_pairs = bone_pairs or self._DEFAULT_BONE_PAIRS

    def extract(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute bone vectors for a single frame.

        Args:
            keypoints: Float array of shape ``(num_joints, 2)`` with ``(x, y)``
                coordinates.

        Returns:
            Float array of shape ``(num_bones * 2,)`` — concatenated ``(dx, dy)``
            bone vectors.
        """
        vectors = []
        for a, b in self.bone_pairs:
            vectors.append(keypoints[b] - keypoints[a])
        return np.concatenate(vectors, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level segmenter
# ---------------------------------------------------------------------------

class TCNSegmenter:
    """Segments a dance video into discrete teachable steps.

    The segmenter combines audio and visual features, runs them through the
    TCN, and applies peak-picking on the resulting probability curve to
    identify step boundaries.

    Args:
        model: Pre-instantiated :class:`TCNModel`.  When ``None`` a default
            model is constructed.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        peak_height: Minimum height threshold for ``scipy.signal.find_peaks``.
        peak_distance: Minimum distance (in frames) between consecutive peaks.
    """

    def __init__(
        self,
        model: Optional[TCNModel] = None,
        device: str = "cpu",
        peak_height: float = 0.5,
        peak_distance: int = 15,
    ) -> None:
        self.device = torch.device(device)
        self.model = (model or TCNModel()).to(self.device)
        self.model.eval()
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        self.peak_height = peak_height
        self.peak_distance = peak_distance

    def load_weights(self, checkpoint_path: str) -> None:
        """Load model weights from a PyTorch checkpoint file.

        Args:
            checkpoint_path: Path to the ``.pt`` or ``.pth`` checkpoint.
        """
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        logger.info("Loaded TCN weights from %s", checkpoint_path)

    def segment(
        self,
        visual_features: np.ndarray,
        audio_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Run segmentation on pre-extracted features.

        Args:
            visual_features: Float array of shape ``(time_steps, visual_dim)``.
            audio_features: Optional float array of shape
                ``(audio_dim, audio_time_steps)``.  When provided, it is
                resampled to match *visual_features* time axis and concatenated.

        Returns:
            A tuple of:
            - ``prob_curve``: Float array of shape ``(time_steps,)`` with
              per-frame segmentation probabilities.
            - ``step_boundaries``: List of frame indices where new steps begin.
        """
        T = visual_features.shape[0]

        if audio_features is not None:
            # Resample audio features to match video time resolution
            audio_resampled = self._resample_audio(audio_features, T)
            combined = np.concatenate([visual_features, audio_resampled], axis=1)
        else:
            combined = visual_features

        # Shape: (1, feature_dim, time_steps)
        x = torch.from_numpy(combined.T[np.newaxis]).float().to(self.device)

        with torch.no_grad():
            prob_curve = self.model(x).squeeze().cpu().numpy()  # (T,)

        step_boundaries = self._pick_peaks(prob_curve)
        return prob_curve, step_boundaries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resample_audio(self, audio_features: np.ndarray, target_len: int) -> np.ndarray:
        """Resample *audio_features* along its time axis to *target_len* frames.

        Args:
            audio_features: Array of shape ``(feature_dim, audio_time_steps)``.
            target_len: Desired number of time steps.

        Returns:
            Array of shape ``(target_len, feature_dim)``.
        """
        import scipy.ndimage

        audio_T = audio_features.shape[1]
        if audio_T == target_len:
            return audio_features.T

        zoom_factor = target_len / audio_T
        resampled = scipy.ndimage.zoom(audio_features, (1, zoom_factor), order=1)
        return resampled.T  # (target_len, feature_dim)

    def _pick_peaks(self, prob_curve: np.ndarray) -> List[int]:
        """Identify step boundary frame indices via peak-picking.

        Args:
            prob_curve: 1-D float array of segmentation probabilities.

        Returns:
            Sorted list of frame indices corresponding to detected peaks.
        """
        peaks, _ = find_peaks(
            prob_curve,
            height=self.peak_height,
            distance=self.peak_distance,
        )
        return peaks.tolist()
