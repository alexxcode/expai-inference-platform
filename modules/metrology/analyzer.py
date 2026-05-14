"""Metrology analyzer — bracket OK / NOK inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.inference_engine import Detection


@dataclass
class InspectionResult:
    frame_index: int
    timestamp: float
    verdict: str          # "OK" | "NOK" | "NO_PIECE"
    detected_class: str
    confidence: float
    alert_level: Optional[str] = None   # None | "WARNING" | "CRITICAL"


@dataclass
class InspectionIncident:
    start_frame: int
    start_timestamp: float
    end_frame: Optional[int] = None
    end_timestamp: Optional[float] = None
    peak_confidence: float = 0.0
    detected_class: str = "bracket_not_ok"

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_timestamp is not None:
            return round(self.end_timestamp - self.start_timestamp, 3)
        return None


class MetrologyAnalyzer:
    """
    Frame-by-frame inspection analyzer.
    Classifies each frame as OK / NOK / NO_PIECE and
    manages streak counters + incident recording.
    """

    NOK_VERDICT   = "NOK"
    OK_VERDICT    = "OK"
    NOPIECE_VERDICT = "NO_PIECE"

    def __init__(
        self,
        nok_classes: list[str],
        nok_frame_threshold: int = 5,
        critical_frame_threshold: int = 15,
        no_piece_threshold: int = 60,
        min_incident_duration_s: float = 0.1,
    ):
        self.nok_classes             = set(nok_classes)
        self.nok_frame_threshold     = nok_frame_threshold
        self.critical_frame_threshold = critical_frame_threshold
        self.no_piece_threshold      = no_piece_threshold
        self.min_incident_duration_s = min_incident_duration_s

        # State
        self.current_verdict: str            = self.NOPIECE_VERDICT
        self.current_class: str              = ""
        self.current_confidence: float       = 0.0
        self.nok_streak: int                 = 0
        self.no_piece_streak: int            = 0
        self.alert_active: bool              = False
        self.alert_level: Optional[str]      = None

        # Stats
        self.total_frames: int               = 0
        self.ok_frames: int                  = 0
        self.nok_frames: int                 = 0
        self.no_piece_frames: int            = 0
        self.incidents: list[InspectionIncident] = []
        self._current_incident: Optional[InspectionIncident] = None
        self._new_events: list[dict]         = []

    # ── Public API ─────────────────────────────────────────────────────

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
    ) -> dict:
        self._new_events = []
        self.total_frames += 1

        verdict, cls, conf = self._classify(detections)
        self.current_verdict    = verdict
        self.current_class      = cls
        self.current_confidence = conf

        self._update_streaks(verdict, frame_index, timestamp)

        return {
            "verdict":       verdict,
            "detected_class": cls,
            "confidence":    round(conf, 3),
            "nok_streak":    self.nok_streak,
            "no_piece_streak": self.no_piece_streak,
            "alert_active":  self.alert_active,
            "alert_level":   self.alert_level,
            "new_events":    self._new_events,
        }

    def get_summary(self) -> dict:
        closed = [i for i in self.incidents if i.end_frame is not None]
        durations = [i.duration_seconds for i in closed if i.duration_seconds is not None]
        ok_rate = round(self.ok_frames / max(self.total_frames, 1) * 100, 1)
        nok_rate = round(self.nok_frames / max(self.total_frames, 1) * 100, 1)
        return {
            "total_frames_analyzed": self.total_frames,
            "ok_frames":    self.ok_frames,
            "nok_frames":   self.nok_frames,
            "no_piece_frames": self.no_piece_frames,
            "ok_rate":      ok_rate,
            "nok_rate":     nok_rate,
            "incident_count": len(closed),
            "avg_incident_duration_s": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "max_incident_duration_s": round(max(durations), 2) if durations else 0.0,
            "incidents": [
                {
                    "start_frame": i.start_frame,
                    "start_timestamp": round(i.start_timestamp, 3),
                    "end_frame": i.end_frame,
                    "duration_seconds": i.duration_seconds,
                    "peak_confidence": round(i.peak_confidence, 3),
                    "class": i.detected_class,
                }
                for i in closed
            ],
        }

    def reset(self) -> None:
        self.current_verdict = self.NOPIECE_VERDICT
        self.current_class   = ""
        self.current_confidence = 0.0
        self.nok_streak = 0
        self.no_piece_streak = 0
        self.alert_active = False
        self.alert_level  = None
        self.total_frames = 0
        self.ok_frames = 0
        self.nok_frames = 0
        self.no_piece_frames = 0
        self.incidents = []
        self._current_incident = None
        self._new_events = []

    # ── Internal ───────────────────────────────────────────────────────

    def _classify(
        self, detections: list[Detection]
    ) -> tuple[str, str, float]:
        """Return (verdict, class_name, confidence) from detections."""
        if not detections:
            return self.NOPIECE_VERDICT, "", 0.0

        # Best detection by confidence
        best = max(detections, key=lambda d: d.confidence)
        cls  = best.class_name

        if cls in self.nok_classes:
            return self.NOK_VERDICT, cls, best.confidence
        else:
            return self.OK_VERDICT, cls, best.confidence

    def _update_streaks(
        self, verdict: str, frame_index: int, timestamp: float
    ) -> None:
        if verdict == self.NOK_VERDICT:
            self.nok_frames += 1
            self.no_piece_streak = 0
            self.nok_streak += 1
            self._handle_nok_streak(frame_index, timestamp)

        elif verdict == self.OK_VERDICT:
            self.ok_frames += 1
            self.no_piece_streak = 0
            if self.nok_streak > 0:
                self._close_incident(frame_index, timestamp)
            self.nok_streak = 0
            self.alert_active = False
            self.alert_level  = None

        else:  # NO_PIECE
            self.no_piece_frames += 1
            self.no_piece_streak += 1
            if self.nok_streak > 0:
                self._close_incident(frame_index, timestamp)
            self.nok_streak  = 0
            self.alert_active = False
            self.alert_level  = None

    def _handle_nok_streak(self, frame_index: int, timestamp: float) -> None:
        # Open incident on first NOK frame
        if self.nok_streak == 1:
            self._current_incident = InspectionIncident(
                start_frame=frame_index,
                start_timestamp=timestamp,
                detected_class=self.current_class,
            )
            self._new_events.append({
                "event_type": "NOK_DETECTED",
                "frame":      frame_index,
                "timestamp":  timestamp,
                "class":      self.current_class,
                "confidence": self.current_confidence,
            })

        # Update peak confidence
        if self._current_incident:
            if self.current_confidence > self._current_incident.peak_confidence:
                self._current_incident.peak_confidence = self.current_confidence

        # Escalate alert level
        if self.nok_streak >= self.critical_frame_threshold:
            if self.alert_level != "CRITICAL":
                self.alert_level  = "CRITICAL"
                self.alert_active = True
                self._new_events.append({
                    "event_type": "CRITICAL_NOK",
                    "frame":      frame_index,
                    "timestamp":  timestamp,
                    "nok_streak": self.nok_streak,
                })
        elif self.nok_streak >= self.nok_frame_threshold:
            if not self.alert_active:
                self.alert_active = True
                self.alert_level  = "WARNING"
                self._new_events.append({
                    "event_type": "NOK_ALERT",
                    "frame":      frame_index,
                    "timestamp":  timestamp,
                    "nok_streak": self.nok_streak,
                })

    def _close_incident(self, frame_index: int, timestamp: float) -> None:
        if self._current_incident is None:
            return
        inc = self._current_incident
        inc.end_frame     = frame_index
        inc.end_timestamp = timestamp
        if (inc.duration_seconds or 0) >= self.min_incident_duration_s:
            self.incidents.append(inc)
            self._new_events.append({
                "event_type":       "NOK_RESOLVED",
                "frame":            frame_index,
                "timestamp":        timestamp,
                "duration_seconds": inc.duration_seconds,
            })
        self._current_incident = None

    @property
    def ok_rate(self) -> float:
        return round(self.ok_frames / max(self.total_frames, 1) * 100, 1)

    @property
    def nok_rate(self) -> float:
        return round(self.nok_frames / max(self.total_frames, 1) * 100, 1)
