"""EPP Gloves module — main integration of analyzer, alerts and metrics."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.config import load_module_config, resolve_path, BASE_DIR
from core.inference_engine import Detection
from core.metrics import BaseMetrics
from modules.base import BaseModule
from modules.epp_gloves.analyzer import EPPGlovesAnalyzer
from modules.epp_gloves.alerts import AlertSystem


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class EPPGlovesModule(BaseModule):
    """EPP Gloves module: detects glove compliance from YOLO detections."""

    MODULE_NAME = "epp_gloves"

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: dict = {}
        self.analyzer: Optional[EPPGlovesAnalyzer] = None
        self.alert_system: Optional[AlertSystem] = None
        self.metrics = BaseMetrics()
        self._roi: Optional[list] = None
        self._alert_threshold: int = 15
        self._frame_index: int = 0
        self._timestamp: float = 0.0

        config_path = config_path or DEFAULT_CONFIG_PATH
        self.load_config(str(config_path))

    # ------------------------------------------------------------------ #
    # BaseModule interface                                                 #
    # ------------------------------------------------------------------ #

    def load_config(self, config_path: str) -> None:
        self.config = load_module_config(config_path)
        detection_cfg = self.config.get("detection", {})
        self._alert_threshold = int(detection_cfg.get("alert_frame_threshold", 15))
        self._roi = detection_cfg.get("roi", None)
        self.analyzer = EPPGlovesAnalyzer(alert_frame_threshold=self._alert_threshold)
        self.alert_system = AlertSystem(self.config)
        logger.info(f"EPPGlovesModule config loaded. Alert threshold: {self._alert_threshold} frames")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> dict:
        """Process one frame. Returns annotations, events, metrics."""
        t0 = time.perf_counter()
        self._frame_index = frame_index
        self._timestamp = timestamp

        self.metrics.tick()

        # Filter detections to ROI if configured
        filtered = self._filter_roi(detections) if self._roi else detections
        in_roi = bool(filtered) or self._roi is None

        # Analyze
        analysis = self.analyzer.analyze_frame(filtered, frame_index, timestamp)

        # Alerts
        alert = self.alert_system.evaluate(
            operator_state=analysis["operator_state"],
            consecutive_frames=analysis["consecutive_sin_guante"],
            frame_index=frame_index,
            timestamp=timestamp,
            alert_threshold=self._alert_threshold,
            in_roi=in_roi,
        )

        # Metrics
        latency_ms = (time.perf_counter() - t0) * 1000
        confidences = [d.confidence for d in detections]
        self.metrics.record_inference(latency_ms, len(detections), confidences)

        return {
            "annotations": {
                "detections": [d.to_dict() for d in detections],
                "operator_state": analysis["operator_state"],
                "alert_active": analysis["alert_active"],
                "compliance_pct": self.analyzer.compliance_percentage,
            },
            "events": analysis.get("new_events", []),
            "metrics": self.metrics.to_dict(),
            "alert": alert.to_dict() if alert else None,
        }

    def get_state(self) -> dict:
        return {
            "module": self.MODULE_NAME,
            "frame_index": self._frame_index,
            "timestamp": self._timestamp,
            "operator_state": self._current_operator_state(),
            "alert_active": self.analyzer.alert_active if self.analyzer else False,
            "compliance_percentage": self.analyzer.compliance_percentage if self.analyzer else 100.0,
            "incident_count": self.analyzer.incident_count if self.analyzer else 0,
            "metrics": self.metrics.to_dict(),
        }

    def generate_report(self) -> dict:
        return {
            "module": self.MODULE_NAME,
            "analysis": self.analyzer.get_summary() if self.analyzer else {},
            "alerts": self.alert_system.get_summary() if self.alert_system else {},
            "metrics": self.metrics.to_dict(),
        }

    def reset(self) -> None:
        if self.analyzer:
            self.analyzer.reset()
        if self.alert_system:
            self.alert_system.reset()
        self.metrics.reset()
        self._frame_index = 0
        self._timestamp = 0.0
        logger.info("EPPGlovesModule reset.")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _filter_roi(self, detections: list[Detection]) -> list[Detection]:
        """Keep only detections whose bbox center falls inside the ROI polygon."""
        if not self._roi:
            return detections
        try:
            import cv2
            pts = np.array(self._roi, dtype=np.float32)
            result = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    result.append(det)
            return result
        except Exception:
            return detections

    def _current_operator_state(self) -> str:
        if not self.analyzer or self.analyzer.total_frames_analyzed == 0:
            return "IDLE"
        return "INCUMPLIMIENTO" if self.analyzer.alert_active else "COMPLIANT"
