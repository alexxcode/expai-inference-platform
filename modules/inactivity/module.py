"""Inactivity module — integrates analyzer, alerts and metrics."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.config import load_module_config, BASE_DIR
from core.inference_engine import Detection
from core.metrics import BaseMetrics
from modules.base import BaseModule
from modules.inactivity.analyzer import InactivityAnalyzer, STATE_COLORS
from modules.epp_gloves.alerts import AlertSystem, AlertLevel

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class InactivityModule(BaseModule):
    """Detects operator inactivity from YOLO state classifications."""

    MODULE_NAME = "inactivity"

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: dict = {}
        self.analyzer:     Optional[InactivityAnalyzer] = None
        self.alert_system: Optional[AlertSystem]        = None
        self.metrics = BaseMetrics()
        self._frame_index: int   = 0
        self._timestamp:   float = 0.0

        self.load_config(str(config_path or DEFAULT_CONFIG_PATH))

    # ------------------------------------------------------------------ #
    # BaseModule                                                           #
    # ------------------------------------------------------------------ #

    def load_config(self, config_path: str) -> None:
        self.config = load_module_config(config_path)

        det = self.config.get("detection", {})
        ana = self.config.get("analysis",  {})

        alert_threshold    = int(det.get("alert_frame_threshold",    30))
        critical_threshold = int(det.get("critical_frame_threshold", 150))
        min_duration       = float(ana.get("min_incident_duration_seconds", 1.0))

        productive = set(self.config.get("productive_states", ["estado_activo", "estado_preparacion", "estado_descarga"]))
        inactive   = set(self.config.get("inactive_states",  ["estado_inactivo"]))

        self.analyzer = InactivityAnalyzer(
            alert_frame_threshold=alert_threshold,
            critical_frame_threshold=critical_threshold,
            min_incident_duration_s=min_duration,
            productive_states=productive,
            inactive_states=inactive,
        )
        self.alert_system = AlertSystem(self.config)
        logger.info(
            f"InactivityModule loaded. "
            f"Alert: {alert_threshold}f | Critical: {critical_threshold}f"
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

        # Alert evaluation
        alert = None
        if self.alert_system and analysis["alert_active"]:
            streak = analysis["inactive_streak"]
            level  = analysis["alert_level"]
            msg = (
                f"Operario inactivo durante {streak} frames consecutivos"
                if level == "WARNING"
                else f"Inactividad prolongada: {streak} frames (crítico)"
            )
            alert = self.alert_system.emit(
                level=level,
                message=msg,
                frame_index=frame_index,
                timestamp=timestamp,
                details={"streak": streak, "state": analysis["current_state"]},
            )
        elif analysis["new_transitions"] and self.alert_system:
            for t in analysis["new_transitions"]:
                if t.get("is_inactivity_start"):
                    self.alert_system.emit(
                        AlertLevel.INFO,
                        f"Inicio de inactividad detectado (track {t['track_id']})",
                        frame_index, timestamp,
                        {"from_state": t["from_state"]},
                    )
                elif t.get("is_inactivity_end"):
                    self.alert_system.emit(
                        AlertLevel.INFO,
                        f"Fin de inactividad (track {t['track_id']}) → {t['to_state']}",
                        frame_index, timestamp,
                        {"duration_approx": "see incidents"},
                    )

        latency_ms  = (time.perf_counter() - t0) * 1000
        confidences = [d.confidence for d in detections]
        self.metrics.record_inference(latency_ms, len(detections), confidences)

        return {
            "annotations": {
                "detections":    [d.to_dict() for d in detections],
                "current_state": analysis["current_state"],
                "alert_active":  analysis["alert_active"],
                "alert_level":   analysis["alert_level"],
                "activity_rate": self.analyzer.activity_rate,
                "inactive_streak": analysis["inactive_streak"],
            },
            "events":  analysis["new_transitions"],
            "metrics": self.metrics.to_dict(),
            "alert":   alert.to_dict() if alert else None,
        }

    def get_state(self) -> dict:
        return {
            "module":          self.MODULE_NAME,
            "frame_index":     self._frame_index,
            "timestamp":       self._timestamp,
            "current_state":   self._current_state(),
            "alert_active":    self.analyzer.alert_active if self.analyzer else False,
            "alert_level":     self.analyzer.alert_level  if self.analyzer else None,
            "activity_rate":   self.analyzer.activity_rate if self.analyzer else 100.0,
            "inactivity_rate": self.analyzer.inactivity_rate if self.analyzer else 0.0,
            "incident_count":  self.analyzer.incident_count if self.analyzer else 0,
            "metrics":         self.metrics.to_dict(),
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
        logger.info("InactivityModule reset.")

    # ------------------------------------------------------------------ #

    def _current_state(self) -> str:
        if not self.analyzer or self.analyzer.total_frames_analyzed == 0:
            return "IDLE"
        return self.analyzer._dominant_state(self.analyzer._track_states) \
               if self.analyzer._track_states else "no_detection"
