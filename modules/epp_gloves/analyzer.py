"""EPP Gloves analyzer: tracks operator state and detects glove events."""

from dataclasses import dataclass, field
from typing import Optional

from core.inference_engine import Detection


CLASS_CON_GUANTE = "con_guante"
CLASS_SIN_GUANTE = "sin_guante"


@dataclass
class GloveEvent:
    event_type: str          # "GLOVE_REMOVED" | "GLOVE_PUT_ON"
    frame_index: int
    timestamp: float
    track_id: Optional[int]
    from_state: str
    to_state: str

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "track_id": self.track_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
        }


@dataclass
class IncidentRecord:
    start_frame: int
    start_timestamp: float
    end_frame: Optional[int] = None
    end_timestamp: Optional[float] = None
    track_id: Optional[int] = None

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is not None:
            return round(self.end_timestamp - self.start_timestamp, 3)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "start_timestamp": round(self.start_timestamp, 3),
            "end_frame": self.end_frame,
            "end_timestamp": round(self.end_timestamp, 3) if self.end_timestamp else None,
            "duration_seconds": self.duration_seconds,
            "track_id": self.track_id,
        }


class EPPGlovesAnalyzer:
    """Tracks per-operator glove state and detects transitions between frames."""

    def __init__(self, alert_frame_threshold: int = 15):
        self.alert_frame_threshold = alert_frame_threshold

        # Per-track state: track_id -> last known state
        self._track_states: dict[int, str] = {}
        # How many consecutive frames each track has been sin_guante
        self._sin_guante_streak: dict[int, int] = {}

        self.events: list[GloveEvent] = []
        self.incidents: list[IncidentRecord] = []
        self._open_incidents: dict[int, IncidentRecord] = {}  # track_id -> open incident

        # Frame-level counters
        self.frames_with_gloves: int = 0
        self.frames_without_gloves: int = 0
        self.total_frames_analyzed: int = 0

        # Alert state
        self.alert_active: bool = False
        self._alert_track_ids: set[int] = set()

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
    ) -> dict:
        """
        Analyze detections for a single frame.

        Returns:
            dict with: operator_state, alert_active, new_events, consecutive_sin_guante
        """
        self.total_frames_analyzed += 1
        new_events: list[GloveEvent] = []

        # Group detections by track_id (or use a synthetic id=0 for no-tracking mode)
        track_detections: dict[int, str] = {}
        for det in detections:
            tid = det.track_id if det.track_id is not None else 0
            # If multiple detections for same track, prefer sin_guante (safety first)
            if tid not in track_detections or det.class_name == CLASS_SIN_GUANTE:
                track_detections[tid] = det.class_name

        # Evaluate each tracked entity
        active_sin_guante_tracks: set[int] = set()
        active_con_guante_tracks: set[int] = set()

        for tid, current_class in track_detections.items():
            prev_class = self._track_states.get(tid)

            # Transition detection
            if prev_class is not None and prev_class != current_class:
                event_type = "GLOVE_REMOVED" if current_class == CLASS_SIN_GUANTE else "GLOVE_PUT_ON"
                event = GloveEvent(
                    event_type=event_type,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    track_id=tid,
                    from_state=prev_class,
                    to_state=current_class,
                )
                self.events.append(event)
                new_events.append(event)

            self._track_states[tid] = current_class

            if current_class == CLASS_SIN_GUANTE:
                active_sin_guante_tracks.add(tid)
                self._sin_guante_streak[tid] = self._sin_guante_streak.get(tid, 0) + 1

                # Open incident if not already open
                if tid not in self._open_incidents:
                    incident = IncidentRecord(
                        start_frame=frame_index,
                        start_timestamp=timestamp,
                        track_id=tid,
                    )
                    self._open_incidents[tid] = incident

            else:
                active_con_guante_tracks.add(tid)
                self._sin_guante_streak[tid] = 0

                # Close incident if open
                if tid in self._open_incidents:
                    inc = self._open_incidents.pop(tid)
                    inc.end_frame = frame_index
                    inc.end_timestamp = timestamp
                    self.incidents.append(inc)

        # Close incidents for tracks no longer detected
        disappeared = set(self._open_incidents.keys()) - set(track_detections.keys())
        for tid in disappeared:
            inc = self._open_incidents.pop(tid)
            inc.end_frame = frame_index
            inc.end_timestamp = timestamp
            self.incidents.append(inc)
            self._sin_guante_streak[tid] = 0

        # Operator state (frame-level)
        has_sin_guante = bool(active_sin_guante_tracks)
        has_con_guante = bool(active_con_guante_tracks)

        if not track_detections:
            operator_state = "NO_DETECTION"
        elif has_sin_guante:
            operator_state = "INCUMPLIMIENTO"
            self.frames_without_gloves += 1
        else:
            operator_state = "COMPLIANT"
            self.frames_with_gloves += 1

        # Alert logic: any track exceeds threshold
        max_streak = max(self._sin_guante_streak.values()) if self._sin_guante_streak else 0
        self.alert_active = max_streak >= self.alert_frame_threshold

        return {
            "operator_state": operator_state,
            "alert_active": self.alert_active,
            "new_events": [e.to_dict() for e in new_events],
            "consecutive_sin_guante": max_streak,
            "active_tracks": {
                "sin_guante": list(active_sin_guante_tracks),
                "con_guante": list(active_con_guante_tracks),
            },
        }

    @property
    def compliance_percentage(self) -> float:
        analyzed = self.frames_with_gloves + self.frames_without_gloves
        if analyzed == 0:
            return 100.0
        return round((self.frames_with_gloves / analyzed) * 100, 2)

    @property
    def incident_count(self) -> int:
        return len(self.incidents)

    @property
    def avg_incident_duration(self) -> float:
        closed = [i for i in self.incidents if i.end_timestamp is not None]
        if not closed:
            return 0.0
        return round(sum(i.duration_seconds for i in closed) / len(closed), 3)

    def get_summary(self) -> dict:
        return {
            "compliance_percentage": self.compliance_percentage,
            "frames_with_gloves": self.frames_with_gloves,
            "frames_without_gloves": self.frames_without_gloves,
            "total_frames_analyzed": self.total_frames_analyzed,
            "incident_count": self.incident_count,
            "avg_incident_duration_seconds": self.avg_incident_duration,
            "events": [e.to_dict() for e in self.events],
            "incidents": [i.to_dict() for i in self.incidents],
        }

    def reset(self) -> None:
        self._track_states.clear()
        self._sin_guante_streak.clear()
        self.events.clear()
        self.incidents.clear()
        self._open_incidents.clear()
        self.frames_with_gloves = 0
        self.frames_without_gloves = 0
        self.total_frames_analyzed = 0
        self.alert_active = False
        self._alert_track_ids.clear()
