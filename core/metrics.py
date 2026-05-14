"""Base metrics collection shared across all modules."""

import time
from collections import deque
from typing import Optional


class BaseMetrics:
    """Tracks FPS, latency, and per-frame confidence statistics."""

    def __init__(self, fps_window: int = 30):
        self._fps_window = fps_window
        self._frame_times: deque[float] = deque(maxlen=fps_window)
        self._last_frame_time: Optional[float] = None

        self.total_frames: int = 0
        self.total_detections: int = 0
        self.confidence_sum: float = 0.0
        self.inference_latency_ms: float = 0.0

    def tick(self) -> None:
        """Call once per frame to update FPS tracking."""
        now = time.perf_counter()
        if self._last_frame_time is not None:
            self._frame_times.append(now - self._last_frame_time)
        self._last_frame_time = now
        self.total_frames += 1

    def record_inference(self, latency_ms: float, detections: int, confidences: list[float]) -> None:
        """Record inference stats for the current frame."""
        self.inference_latency_ms = latency_ms
        self.total_detections += detections
        self.confidence_sum += sum(confidences)

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_interval = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        if self.total_detections == 0:
            return 0.0
        return self.confidence_sum / self.total_detections

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "fps": round(self.fps, 2),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
        }

    def reset(self) -> None:
        self._frame_times.clear()
        self._last_frame_time = None
        self.total_frames = 0
        self.total_detections = 0
        self.confidence_sum = 0.0
        self.inference_latency_ms = 0.0
