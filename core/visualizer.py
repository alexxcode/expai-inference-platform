"""Rendering of bounding boxes, labels and overlays on frames."""

from typing import Optional

import cv2
import numpy as np

from core.inference_engine import Detection


# Color palette by class name (BGR) — works across all modules
CLASS_NAME_COLORS: dict[str, tuple[int, int, int]] = {
    # EPP Gloves
    "con_guante":         (0, 200, 0),
    "sin_guante":         (0, 0, 220),
    # Inactivity
    "estado_activo":      (0, 200, 200),
    "estado_preparacion": (0, 200, 0),
    "estado_inactivo":    (0, 0, 220),
    "estado_descarga":    (200, 0, 200),
}

# Fallback by class_id
CLASS_ID_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),
    1: (0, 0, 220),
    2: (0, 200, 200),
    3: (200, 0, 200),
}
DEFAULT_COLOR = (180, 180, 0)

# Alert banner colors per module state
ALERT_COLORS = {
    "WARNING":  (0, 100, 200),
    "CRITICAL": (0, 0, 200),
}


class Visualizer:
    """Draws detections and status overlays onto video frames."""

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        show_confidence: bool = True,
        show_track_id: bool = True,
    ):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id

    def _color(self, det: Detection) -> tuple[int, int, int]:
        if det.class_name in CLASS_NAME_COLORS:
            return CLASS_NAME_COLORS[det.class_name]
        return CLASS_ID_COLORS.get(det.class_id, DEFAULT_COLOR)

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels for each detection."""
        for det in detections:
            color = self._color(det)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            parts = [det.class_name]
            if self.show_confidence:
                parts.append(f"{det.confidence:.2f}")
            if self.show_track_id and det.track_id is not None:
                parts.append(f"#{det.track_id}")
            label = " ".join(parts)

            (lw, lh), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            label_y = max(y1 - 4, lh + baseline)
            cv2.rectangle(frame, (x1, label_y - lh - baseline), (x1 + lw, label_y + baseline), color, -1)
            cv2.putText(frame, label, (x1, label_y - baseline), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame

    def draw_status_overlay(
        self,
        frame: np.ndarray,
        operator_state: str,
        metric_value: float,      # compliance % for EPP, activity % for inactivity
        metric_label: str = "Métrica",
        frame_index: int = 0,
        timestamp: float = 0.0,
        alert_active: bool = False,
        alert_level: Optional[str] = None,
        alert_text: str = "⚠ ALERTA",
    ) -> np.ndarray:
        """Draw a HUD overlay with operator state and KPIs."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # State color
        productive = {"COMPLIANT", "estado_activo", "estado_preparacion", "estado_descarga"}
        state_color = (0, 200, 0) if operator_state in productive else (0, 0, 220)
        cv2.putText(frame, f"Estado: {operator_state}", (10, 22), self.font, 0.65, state_color, 2)
        cv2.putText(frame, f"{metric_label}: {metric_value:.1f}%", (10, 50), self.font, 0.6, (200, 200, 200), 1)

        ts_str = f"T={timestamp:.2f}s  F={frame_index}"
        cv2.putText(frame, ts_str, (w - 220, 22), self.font, 0.55, (180, 180, 180), 1)

        if alert_active:
            banner_color = ALERT_COLORS.get(alert_level or "WARNING", (0, 0, 180))
            cv2.rectangle(frame, (0, h - 50), (w, h), banner_color, -1)
            cv2.putText(frame, alert_text, (w // 2 - 200, h - 15), self.font, 0.9, (255, 255, 255), 2)

        return frame

    def draw_roi(self, frame: np.ndarray, roi_points: list[list[int]], color: tuple = (0, 255, 255)) -> np.ndarray:
        pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        return frame
