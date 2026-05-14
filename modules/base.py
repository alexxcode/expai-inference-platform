"""Abstract base class that all EXPAI modules must implement."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModule(ABC):
    """Interface that every EXPAI inference module must implement."""

    @abstractmethod
    def load_config(self, config_path: str) -> None:
        """Load module-specific configuration."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray, detections: list) -> dict:
        """Process a single frame with its YOLO detections.

        Args:
            frame: BGR frame from the video.
            detections: List of Detection objects from InferenceEngine.

        Returns:
            dict with keys: annotations, events, metrics
        """

    @abstractmethod
    def get_state(self) -> dict:
        """Return current module state (for API / UI polling)."""

    @abstractmethod
    def generate_report(self) -> dict:
        """Generate the final session report."""

    @abstractmethod
    def reset(self) -> None:
        """Reset module state for a new inference session."""
