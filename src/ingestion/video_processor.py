"""
src/ingestion/video_processor.py
=================================
Phase 1: Ingestion & Preprocessing

Responsibilities:
- Download dance tutorial videos from YouTube using yt-dlp.
- Extract frames in parallel using ProcessPoolExecutor to bypass the GIL.
- Apply grayscale conversion and spatial gradient background subtraction.
"""

from __future__ import annotations

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
import yt_dlp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (module-level so they can be pickled for multiprocessing)
# ---------------------------------------------------------------------------

def _extract_frame_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_dir: str,
    apply_background_subtraction: bool = True,
) -> List[str]:
    """Extract a contiguous range of frames from *video_path* and save them.

    This function is designed to run in a worker process.

    Args:
        video_path: Absolute path to the source video file.
        start_frame: Index of the first frame to extract (inclusive).
        end_frame: Index of the last frame to extract (exclusive).
        output_dir: Directory where extracted frame images are saved.
        apply_background_subtraction: If ``True``, apply MOG2 background
            subtraction and grayscale conversion before saving.

    Returns:
        A list of absolute paths to the saved frame images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2() if apply_background_subtraction else None

    saved_paths: List[str] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if apply_background_subtraction and bg_subtractor is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = bg_subtractor.apply(gray)
            # Compute spatial gradient (Sobel) to highlight motion edges
            grad_x = cv2.Sobel(fg_mask, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(fg_mask, cv2.CV_32F, 0, 1, ksize=3)
            processed = cv2.magnitude(grad_x, grad_y).astype(np.uint8)
        else:
            processed = frame

        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, processed)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VideoProcessor:
    """Downloads and preprocesses dance tutorial videos.

    Attributes:
        output_dir: Root directory where downloaded videos and extracted frames
            are stored.
        max_workers: Number of worker processes used for parallel frame
            extraction.
    """

    def __init__(self, output_dir: str = "data/raw", max_workers: int = 4) -> None:
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, url: str, filename: Optional[str] = None) -> Path:
        """Download a video from *url* using yt-dlp.

        Args:
            url: YouTube (or other yt-dlp-supported) video URL.
            filename: Optional stem for the output file.  When ``None`` the
                video ID is used.

        Returns:
            Path to the downloaded video file.
        """
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        outtmpl = str(video_dir / (filename or "%(id)s")) + ".%(ext)s"
        ydl_opts: dict = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))

        logger.info("Downloaded video to %s", downloaded_path)
        return downloaded_path

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        video_path: str | Path,
        frames_dir: Optional[str | Path] = None,
        apply_background_subtraction: bool = True,
        chunk_size: int = 500,
    ) -> List[str]:
        """Extract all frames from *video_path* in parallel.

        The video is divided into *chunk_size*-frame chunks and each chunk is
        processed by a separate worker process.

        Args:
            video_path: Path to the source video file.
            frames_dir: Directory where frames are saved.  Defaults to
                ``<output_dir>/frames/<video_stem>/``.
            apply_background_subtraction: If ``True``, preprocess frames with
                MOG2 background subtraction and spatial gradient computation.
            chunk_size: Number of frames assigned to each worker process.

        Returns:
            Sorted list of paths to extracted frame images.
        """
        video_path = Path(video_path)
        if frames_dir is None:
            frames_dir = self.output_dir / "frames" / video_path.stem
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine total frame count
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Build work chunks
        chunks: List[Tuple[int, int]] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))

        logger.info(
            "Extracting %d frames from %s using %d workers (%d chunks)",
            total_frames,
            video_path.name,
            self.max_workers,
            len(chunks),
        )

        all_paths: List[str] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _extract_frame_range,
                    str(video_path),
                    start,
                    end,
                    str(frames_dir),
                    apply_background_subtraction,
                ): (start, end)
                for start, end in chunks
            }
            for future in as_completed(futures):
                try:
                    all_paths.extend(future.result())
                except Exception as exc:
                    start, end = futures[future]
                    logger.error("Chunk [%d, %d) failed: %s", start, end, exc)

        all_paths.sort()
        logger.info("Extracted %d frames to %s", len(all_paths), frames_dir)
        return all_paths

    # ------------------------------------------------------------------
    # Generator-based streaming (memory-efficient)
    # ------------------------------------------------------------------

    def stream_frames(
        self, video_path: str | Path, skip: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield ``(frame_index, frame)`` tuples from *video_path*.

        Args:
            video_path: Path to the source video file.
            skip: Yield every *skip*-th frame (default ``1`` yields all frames).

        Yields:
            Tuples of ``(frame_index, BGR frame array)``.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % skip == 0:
                    yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()
