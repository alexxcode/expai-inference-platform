"""Video reading and frame decoding for EXPAI Inference Platform."""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger


class VideoProcessor:
    """Reads a video file and yields frames one at a time."""

    def __init__(self, video_path: str | Path, target_fps: Optional[float] = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self._fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_skip: int = max(1, round(self._fps / target_fps)) if target_fps else 1
        self._frame_index: int = 0

        logger.info(
            f"Video: {self.video_path.name} | {self._width}x{self._height} | "
            f"{self._fps:.1f} fps | {self._total_frames} frames"
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def duration_seconds(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0.0

    def frame_to_timestamp(self, frame_index: int) -> float:
        """Convert frame index to timestamp in seconds."""
        return frame_index / self._fps if self._fps > 0 else 0.0

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_index, timestamp_seconds, frame_bgr) tuples."""
        self._frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self._frame_index % self._frame_skip == 0:
                timestamp = self.frame_to_timestamp(self._frame_index)
                yield self._frame_index, timestamp, frame

            self._frame_index += 1

    def release(self):
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def get_writer(self, output_path: str | Path, fourcc: str = "mp4v") -> cv2.VideoWriter:
        """Create a VideoWriter matching the input video properties."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(
            str(output_path), codec, self._fps, (self._width, self._height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer at {output_path}")
        return writer
