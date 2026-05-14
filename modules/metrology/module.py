"""Metrology module — bracket OK / NOK inspection."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.config import load_module_config
from core.inference_engine import Detection
from core.metrics import BaseMetrics
from modules.base import BaseModule
from modules.metrology.analyzer import MetrologyAnalyzer
from modules.epp_gloves.alerts import AlertSystem, AlertLevel

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

ALERT_MESSAGES = {
    "CRITICAL": "🚨 PIEZA NO CONFORME — NOK confirmado",
    "WARNING":  "⚠ ADVERTENCIA: Pieza posiblemente NOK",
}


class MetrologyModule(BaseModule):
    """Bracket dimensional inspection — OK / NOK."""

    MODULE_NAME = "metrology"

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: dict = {}
        self.analyzer:     Optional[MetrologyAnalyzer] = None
        self.alert_system: Optional[AlertSystem]       = None
        self.metrics = BaseMetrics()
        self._frame_index: int   = 0
        self._timestamp:   float = 0.0

        self.load_config(str(config_path or DEFAULT_CONFIG_PATH))

    # ── BaseModule ────────────────────────────────────────────────────

    def load_config(self, config_path: str) -> None:
        self.config = load_module_config(config_path)
        det = self.config.get("detection", {})

        self.analyzer = MetrologyAnalyzer(
            nok_classes              = self.config.get("nok_classes", ["bracket_not_ok"]),
            nok_frame_threshold      = int(det.get("nok_frame_threshold",      5)),
            critical_frame_threshold = int(det.get("critical_frame_threshold", 15)),
            no_piece_threshold       = int(det.get("no_piece_threshold",       60)),
            min_incident_duration_s  = float(
                self.config.get("analysis", {}).get("min_incident_duration_seconds", 0.1)
            ),
        )
        self.alert_system = AlertSystem(self.config)
        logger.info(
            f"MetrologyModule loaded. "
            f"NOK threshold: {det.get('nok_frame_threshold', 5)}f | "
            f"Critical: {det.get('critical_frame_threshold', 15)}f"
        )

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int = 0,
        timestamp:   float = 0.0,
    ) -> dict:
        t0 = time.perf_counter()
        self._frame_index = frame_index
        self._timestamp   = timestamp
        self.metrics.tick()

        analysis = self.analyzer.analyze_frame(detections, frame_index, timestamp)

        # ── Alert emission ────────────────────────────────────────────
        alert = None
        if analysis["alert_active"] and self.alert_system:
            level = analysis["alert_level"] or "WARNING"
            msg   = ALERT_MESSAGES.get(level, f"⚠ NOK detectado")
            alert = self.alert_system.emit(
                level=level,
                message=msg,
                frame_index=frame_index,
                timestamp=timestamp,
                details={
                    "verdict":    analysis["verdict"],
                    "class":      analysis["detected_class"],
                    "confidence": analysis["confidence"],
                    "nok_streak": analysis["nok_streak"],
                },
            )

        latency_ms  = (time.perf_counter() - t0) * 1000
        confidences = [d.confidence for d in detections]
        self.metrics.record_inference(latency_ms, len(detections), confidences)

        return {
            "annotations": {
                "detections":     [d.to_dict() for d in detections],
                "verdict":        analysis["verdict"],
                "detected_class": analysis["detected_class"],
                "confidence":     analysis["confidence"],
                "nok_streak":     analysis["nok_streak"],
                "no_piece_streak": analysis["no_piece_streak"],
                "alert_active":   analysis["alert_active"],
                "alert_level":    analysis["alert_level"],
                "ok_rate":        self.analyzer.ok_rate,
                "nok_rate":       self.analyzer.nok_rate,
            },
            "events":  analysis["new_events"],
            "metrics": self.metrics.to_dict(),
            "alert":   alert.to_dict() if alert else None,
        }

    def get_state(self) -> dict:
        return {
            "module":       self.MODULE_NAME,
            "frame_index":  self._frame_index,
            "timestamp":    self._timestamp,
            "verdict":      self.analyzer.current_verdict     if self.analyzer else "NO_PIECE",
            "detected_class": self.analyzer.current_class     if self.analyzer else "",
            "confidence":   self.analyzer.current_confidence  if self.analyzer else 0.0,
            "nok_streak":   self.analyzer.nok_streak          if self.analyzer else 0,
            "alert_active": self.analyzer.alert_active        if self.analyzer else False,
            "alert_level":  self.analyzer.alert_level         if self.analyzer else None,
            "ok_rate":      self.analyzer.ok_rate             if self.analyzer else 0.0,
            "nok_rate":     self.analyzer.nok_rate            if self.analyzer else 0.0,
            "metrics":      self.metrics.to_dict(),
        }

    def generate_report(self) -> dict:
        return {
            "module":   self.MODULE_NAME,
            "analysis": self.analyzer.get_summary() if self.analyzer else {},
            "alerts":   self.alert_system.get_summary() if self.alert_system else {},
            "metrics":  self.metrics.to_dict(),
        }

    def reset(self) -> None:
        if self.analyzer:     self.analyzer.reset()
        if self.alert_system: self.alert_system.reset()
        self.metrics.reset()
        self._frame_index = 0
        self._timestamp   = 0.0
        logger.info("MetrologyModule reset.")
