"""Procesado module — piece processing counter (pieza_ok / pieza_sin_proceso)."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.config import load_module_config
from core.inference_engine import Detection
from core.metrics import BaseMetrics
from modules.base import BaseModule
from modules.procesado.analyzer import ProcesadoAnalyzer
from modules.epp_gloves.alerts import AlertSystem

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ProcesadoModule(BaseModule):
    """Piece processing counter module."""

    MODULE_NAME = "procesado"

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: dict = {}
        self.analyzer:     Optional[ProcesadoAnalyzer] = None
        self.alert_system: Optional[AlertSystem]       = None
        self.metrics = BaseMetrics()
        self._frame_index: int   = 0
        self._timestamp:   float = 0.0
        self.load_config(str(config_path or DEFAULT_CONFIG_PATH))

    def load_config(self, config_path: str) -> None:
        self.config = load_module_config(config_path)
        det = self.config.get("detection", {})
        ana = self.config.get("analysis", {})

        self.analyzer = ProcesadoAnalyzer(
            processed_classes       = self.config.get("processed_classes", ["pieza_ok"]),
            pending_classes         = self.config.get("pending_classes",   ["pieza_sin_proceso"]),
            stall_frame_threshold   = int(det.get("stall_frame_threshold",   180)),
            min_track_frames        = int(det.get("min_track_frames",        3)),
            backlog_count_threshold = int(det.get("backlog_count_threshold", 3)),
            backlog_frame_threshold = int(det.get("backlog_frame_threshold", 60)),
            rate_window_seconds     = float(ana.get("rate_window_seconds",   60)),
        )
        self.alert_system = AlertSystem(self.config)
        logger.info(f"ProcesadoModule loaded.")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> dict:
        t0 = time.perf_counter()
        self._frame_index = frame_index
        self._timestamp   = timestamp
        self.metrics.tick()

        analysis = self.analyzer.analyze_frame(detections, frame_index, timestamp)

        alert = None
        if analysis["alert_active"] and self.alert_system:
            if analysis["backlog_active"]:
                msg = f"⚠ Acumulación — {analysis['pending_count']} piezas sin procesar"
            else:
                msg = f"⚠ Producción parada — {analysis['total_processed']} piezas procesadas"
            alert = self.alert_system.emit(
                level="WARNING",
                message=msg,
                frame_index=frame_index,
                timestamp=timestamp,
                details={
                    "total_processed": analysis["total_processed"],
                    "pending_count":   analysis["pending_count"],
                },
            )

        latency_ms  = (time.perf_counter() - t0) * 1000
        confidences = [d.confidence for d in detections]
        self.metrics.record_inference(latency_ms, len(detections), confidences)

        return {
            "annotations": {
                "detections":      [d.to_dict() for d in detections],
                "total_processed": analysis["total_processed"],
                "pending_count":   analysis["pending_count"],
                "pieces_per_min":  analysis["pieces_per_min"],
                "stall_active":    analysis["stall_active"],
                "backlog_active":  analysis["backlog_active"],
                "alert_active":    analysis["alert_active"],
                "alert_level":     analysis["alert_level"],
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
            "total_processed": self.analyzer.total_processed  if self.analyzer else 0,
            "pending_count":   self.analyzer.pending_count    if self.analyzer else 0,
            "stall_active":    self.analyzer._stall_active    if self.analyzer else False,
            "backlog_active":  self.analyzer._backlog_active  if self.analyzer else False,
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
        logger.info("ProcesadoModule reset.")
