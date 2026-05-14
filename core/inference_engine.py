"""YOLO inference engine wrapper using Ultralytics."""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore


class Detection:
    """Represents a single YOLO detection result."""

    __slots__ = ("class_id", "class_name", "confidence", "bbox", "track_id")

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: tuple[int, int, int, int],  # x1, y1, x2, y2
        track_id: Optional[int] = None,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.track_id = track_id

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
            "track_id": self.track_id,
        }


class InferenceEngine:
    """Wraps an Ultralytics YOLO model for frame-by-frame inference."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_names: Optional[dict[int, str]] = None,
        enable_tracking: bool = False,
        tracker: str = "bytetrack",
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.class_names = class_names or {}
        self.enable_tracking = enable_tracking
        self.tracker = tracker

        logger.info(f"Loading YOLO model from {self.model_path} on device={self.device}")
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame. Returns list of Detection objects."""
        if self.enable_tracking:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                tracker=f"{self.tracker}.yaml",
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            class_name = self.class_names.get(class_id, str(class_id))
            track_id = int(box.id[0]) if self.enable_tracking and box.id is not None else None

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                track_id=track_id,
            ))

        return detections

    @classmethod
    def from_module_config(cls, config: dict) -> "InferenceEngine":
        """Instantiate engine from a module YAML config dict."""
        model_cfg = config.get("model", {})
        detection_cfg = config.get("detection", {})
        classes = {int(k): v for k, v in config.get("classes", {}).items()}

        return cls(
            model_path=model_cfg.get("path", "models/model.pt"),
            confidence_threshold=float(model_cfg.get("confidence_threshold", 0.5)),
            iou_threshold=float(model_cfg.get("iou_threshold", 0.45)),
            device=str(model_cfg.get("device", "auto")),
            class_names=classes,
            enable_tracking=bool(detection_cfg.get("enable_tracking", False)),
            tracker=str(detection_cfg.get("tracker", "bytetrack")),
        )
