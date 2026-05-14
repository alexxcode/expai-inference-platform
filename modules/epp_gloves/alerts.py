"""Alert system for EPP Gloves module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class AlertLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert:
    def __init__(
        self,
        level: str,
        message: str,
        frame_index: int,
        timestamp: float,
        details: Optional[dict] = None,
    ):
        self.level = level
        self.message = message
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.details = details or {}
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "message": self.message,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "details": self.details,
            "created_at": self.created_at,
        }


class AlertSystem:
    """Generates, stores, and dispatches alerts from EPP analysis."""

    def __init__(self, config: dict):
        alerts_cfg = config.get("alerts", {})
        self.enabled: bool = alerts_cfg.get("enabled", True)
        self.allowed_levels: set[str] = set(alerts_cfg.get("levels", ["INFO", "WARNING", "CRITICAL"]))
        self._outputs: list[dict] = alerts_cfg.get("output", [])
        self.history: list[Alert] = []

        # Setup log file output
        for output in self._outputs:
            if output.get("type") == "log" and output.get("path"):
                log_path = Path(output["path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                logger.add(str(log_path), filter=lambda r: "EPP_ALERT" in r["extra"], level="INFO")

    def emit(
        self,
        level: str,
        message: str,
        frame_index: int,
        timestamp: float,
        details: Optional[dict] = None,
    ) -> Optional[Alert]:
        if not self.enabled or level not in self.allowed_levels:
            return None

        alert = Alert(level=level, message=message, frame_index=frame_index, timestamp=timestamp, details=details)
        self.history.append(alert)

        # Log output
        log_fn = {
            AlertLevel.INFO: logger.bind(EPP_ALERT=True).info,
            AlertLevel.WARNING: logger.bind(EPP_ALERT=True).warning,
            AlertLevel.CRITICAL: logger.bind(EPP_ALERT=True).critical,
        }.get(level, logger.info)
        log_fn(f"[EPP] {level} | F={frame_index} T={timestamp:.2f}s | {message}")

        # Webhook (fire and forget if configured)
        for output in self._outputs:
            if output.get("type") == "webhook" and output.get("url"):
                self._send_webhook(output["url"], alert)

        return alert

    def evaluate(
        self,
        operator_state: str,
        consecutive_frames: int,
        frame_index: int,
        timestamp: float,
        alert_threshold: int,
        in_roi: bool = True,
    ) -> Optional[Alert]:
        """Evaluate current state and emit the appropriate alert if needed."""
        if not self.enabled:
            return None

        if operator_state == "NO_DETECTION":
            return None

        if operator_state == "INCUMPLIMIENTO":
            if not in_roi:
                return None

            if consecutive_frames == 1:
                return self.emit(AlertLevel.INFO, "Transición detectada: sin_guante", frame_index, timestamp,
                                 {"consecutive_frames": consecutive_frames})
            elif consecutive_frames >= alert_threshold * 2:
                return self.emit(AlertLevel.CRITICAL, f"Sin guantes en zona peligrosa ({consecutive_frames} frames)",
                                 frame_index, timestamp, {"consecutive_frames": consecutive_frames})
            elif consecutive_frames >= alert_threshold:
                return self.emit(AlertLevel.WARNING, f"Operario sin guantes > {alert_threshold} frames consecutivos",
                                 frame_index, timestamp, {"consecutive_frames": consecutive_frames})

        return None

    def _send_webhook(self, url: str, alert: Alert) -> None:
        try:
            import urllib.request
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                logger.debug(f"Webhook sent to {url}: {resp.status}")
        except Exception as exc:
            logger.warning(f"Webhook failed to {url}: {exc}")

    def get_summary(self) -> dict:
        by_level: dict[str, int] = {}
        for a in self.history:
            by_level[a.level] = by_level.get(a.level, 0) + 1
        return {
            "total_alerts": len(self.history),
            "by_level": by_level,
            "alerts": [a.to_dict() for a in self.history],
        }

    def reset(self) -> None:
        self.history.clear()
