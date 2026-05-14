"""Mixed module — unified EPP + operator state monitoring."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.config import load_module_config
from core.inference_engine import Detection
from core.metrics import BaseMetrics
from modules.base import BaseModule
from modules.mixed.analyzer import MixedAnalyzer, RiskLevel, CLASS_COLORS_MIXED
from modules.epp_gloves.alerts import AlertSystem, AlertLevel

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

ALERT_MESSAGES = {
    "CRITICAL": "🚨 RIESGO CRÍTICO: Operario activo sin protección de manos",
    "WARNING":  "⚠ ADVERTENCIA: Manos desprotegidas detectadas en estación",
    "INFO":     "ℹ Operario en transición prolongada",
}


class MixedModule(BaseModule):
    """Unified EPP + operator state module for matricería monitoring."""

    MODULE_NAME = "mixed"

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: dict = {}
        self.analyzer:     Optional[MixedAnalyzer] = None
        self.alert_system: Optional[AlertSystem]   = None
        self.metrics = BaseMetrics()
        self._frame_index: int   = 0
        self._timestamp:   float = 0.0

        self.load_config(str(config_path or DEFAULT_CONFIG_PATH))

    # ── BaseModule ────────────────────────────────────────────────────

    def load_config(self, config_path: str) -> None:
        self.config = load_module_config(config_path)
        det = self.config.get("detection", {})

        self.analyzer = MixedAnalyzer(
            critical_frame_threshold  = int(det.get("critical_frame_threshold",  10)),
            warning_frame_threshold   = int(det.get("warning_frame_threshold",   25)),
            transition_idle_threshold = int(det.get("transition_idle_threshold", 75)),
            min_incident_duration_s   = float(
                self.config.get("analysis", {}).get("min_incident_duration_seconds", 0.5)
            ),
        )
        self.alert_system = AlertSystem(self.config)
        logger.info(
            f"MixedModule loaded. "
            f"Critical: {det.get('critical_frame_threshold', 10)}f | "
            f"Warning: {det.get('warning_frame_threshold', 25)}f"
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
        risk  = analysis["risk_level"]
        if analysis["alert_active"] and self.alert_system:
            msg = ALERT_MESSAGES.get(analysis["alert_level"] or risk, f"Riesgo: {risk}")
            alert = self.alert_system.emit(
                level=analysis["alert_level"] or risk,
                message=msg,
                frame_index=frame_index,
                timestamp=timestamp,
                details={
                    "operator_state": analysis["operator_state"],
                    "hand_state":     analysis["hand_state"],
                    "risk_streak":    analysis["risk_streak"],
                },
            )
        # Emit INFO events for EPP/transition events
        for evt in analysis["new_events"]:
            if evt["event_type"] in ("EPP_VIOLATION", "EPP_RESTORED", "TRANSITION_IDLE") \
               and self.alert_system:
                lvl = AlertLevel.WARNING if evt["event_type"] == "EPP_VIOLATION" \
                      else AlertLevel.INFO
                self.alert_system.emit(
                    level=lvl,
                    message=f"{evt['event_type']}: {evt['operator_state']} / {evt['hand_state']}",
                    frame_index=frame_index,
                    timestamp=timestamp,
                )

        latency_ms  = (time.perf_counter() - t0) * 1000
        confidences = [d.confidence for d in detections]
        self.metrics.record_inference(latency_ms, len(detections), confidences)

        return {
            "annotations": {
                "detections":       [d.to_dict() for d in detections],
                "operator_state":   analysis["operator_state"],
                "hand_state":       analysis["hand_state"],
                "risk_level":       risk,
                "alert_active":     analysis["alert_active"],
                "alert_level":      analysis["alert_level"],
                "epp_compliance":   self.analyzer.epp_compliance_rate,
                "activity_rate":    self.analyzer.activity_rate,
                "risk_streak":      analysis["risk_streak"],
                "transition_streak": analysis["transition_streak"],
            },
            "events":  analysis["new_events"],
            "metrics": self.metrics.to_dict(),
            "alert":   alert.to_dict() if alert else None,
        }

    def get_state(self) -> dict:
        return {
            "module":          self.MODULE_NAME,
            "frame_index":     self._frame_index,
            "timestamp":       self._timestamp,
            "operator_state":  self.analyzer.current_operator if self.analyzer else "no_operator",
            "hand_state":      self.analyzer.current_hand     if self.analyzer else "no_hand",
            "risk_level":      self.analyzer.current_risk.value if self.analyzer else "OK",
            "alert_active":    self.analyzer.alert_active     if self.analyzer else False,
            "alert_level":     self.analyzer.alert_level      if self.analyzer else None,
            "epp_compliance":  self.analyzer.epp_compliance_rate if self.analyzer else 100.0,
            "activity_rate":   self.analyzer.activity_rate    if self.analyzer else 0.0,
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
        logger.info("MixedModule reset.")
