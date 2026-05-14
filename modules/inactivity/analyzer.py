"""Inactivity analyzer: tracks operator state transitions and inactivity incidents."""

from dataclasses import dataclass, field
from typing import Optional

from core.inference_engine import Detection


# State classification
INACTIVE_STATES  = {"estado_inactivo"}
PRODUCTIVE_STATES = {"estado_activo", "estado_preparacion", "estado_descarga"}
ALL_STATES = INACTIVE_STATES | PRODUCTIVE_STATES

# State display colors (for visualizer hints)
STATE_COLORS = {
    "estado_inactivo":    (0,   0,   220),   # red
    "estado_preparacion": (0,   200, 0),     # green
    "estado_activo":      (0,   200, 200),   # yellow-ish
    "estado_descarga":    (200, 0,   200),   # purple
}


@dataclass
class StateTransition:
    from_state: str
    to_state:   str
    frame_index: int
    timestamp:   float
    track_id:    Optional[int]

    def to_dict(self) -> dict:
        return {
            "event_type":   "STATE_TRANSITION",
            "from_state":   self.from_state,
            "to_state":     self.to_state,
            "frame_index":  self.frame_index,
            "timestamp":    round(self.timestamp, 3),
            "track_id":     self.track_id,
            "is_inactivity_start": self.to_state in INACTIVE_STATES,
            "is_inactivity_end":   self.from_state in INACTIVE_STATES and self.to_state not in INACTIVE_STATES,
        }


@dataclass
class InactivityIncident:
    start_frame:     int
    start_timestamp: float
    track_id:        Optional[int] = None
    end_frame:       Optional[int] = None
    end_timestamp:   Optional[float] = None
    # State right before and after inactivity
    prev_state:      Optional[str] = None
    next_state:      Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is not None:
            return round(self.end_timestamp - self.start_timestamp, 3)
        return 0.0

    @property
    def is_closed(self) -> bool:
        return self.end_timestamp is not None

    def to_dict(self) -> dict:
        return {
            "start_frame":      self.start_frame,
            "start_timestamp":  round(self.start_timestamp, 3),
            "end_frame":        self.end_frame,
            "end_timestamp":    round(self.end_timestamp, 3) if self.end_timestamp else None,
            "duration_seconds": self.duration_seconds,
            "track_id":         self.track_id,
            "prev_state":       self.prev_state,
            "next_state":       self.next_state,
        }


class InactivityAnalyzer:
    """
    Tracks per-operator state and detects inactivity incidents.

    States:
      estado_inactivo    → operator idle       → triggers alert
      estado_preparacion → operator preparing  → productive
      estado_activo      → operator working    → productive
      estado_descarga     → operator unloading → productive
    """

    def __init__(
        self,
        alert_frame_threshold:    int   = 30,
        critical_frame_threshold: int   = 150,
        min_incident_duration_s:  float = 1.0,
        productive_states:        set   = None,
        inactive_states:          set   = None,
        fps:                      float = 25.0,
    ):
        self.alert_frame_threshold    = alert_frame_threshold
        self.critical_frame_threshold = critical_frame_threshold
        self.min_incident_duration_s  = min_incident_duration_s
        self.productive_states        = productive_states or PRODUCTIVE_STATES
        self.inactive_states          = inactive_states  or INACTIVE_STATES
        self.fps                      = fps

        # Per-track state
        self._track_states:         dict[int, str]                  = {}
        self._inactive_streak:      dict[int, int]                  = {}
        self._open_incidents:       dict[int, InactivityIncident]   = {}

        # History
        self.transitions:  list[StateTransition]    = []
        self.incidents:    list[InactivityIncident] = []

        # Frame counters per state
        self._state_frame_counts: dict[str, int] = {s: 0 for s in ALL_STATES}
        self._state_frame_counts["no_detection"] = 0
        self.total_frames_analyzed = 0

        # Alert state
        self.alert_active    = False
        self.alert_level     = None   # None | "WARNING" | "CRITICAL"
        self.max_streak      = 0

    # ------------------------------------------------------------------ #

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp:   float,
    ) -> dict:
        """
        Analyze one frame.
        Returns dict: current_state, alert_active, alert_level, new_transitions, inactive_streak
        """
        self.total_frames_analyzed += 1
        new_transitions: list[StateTransition] = []

        # Build track → best_state map
        # Priority if multiple detections for same track: inactive > others (safety)
        track_states: dict[int, str] = {}
        for det in detections:
            tid   = det.track_id if det.track_id is not None else 0
            state = det.class_name
            if tid not in track_states:
                track_states[tid] = state
            else:
                # inactive takes priority
                if state in self.inactive_states:
                    track_states[tid] = state

        # Process each tracked entity
        for tid, current_state in track_states.items():
            prev_state = self._track_states.get(tid)

            # Detect transition
            if prev_state is not None and prev_state != current_state:
                t = StateTransition(
                    from_state=prev_state,
                    to_state=current_state,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    track_id=tid,
                )
                self.transitions.append(t)
                new_transitions.append(t)

            self._track_states[tid] = current_state

            if current_state in self.inactive_states:
                self._inactive_streak[tid] = self._inactive_streak.get(tid, 0) + 1
                # Open incident if needed
                if tid not in self._open_incidents:
                    self._open_incidents[tid] = InactivityIncident(
                        start_frame=frame_index,
                        start_timestamp=timestamp,
                        track_id=tid,
                        prev_state=prev_state,
                    )
            else:
                # Close incident if open
                if tid in self._open_incidents:
                    inc = self._open_incidents.pop(tid)
                    inc.end_frame     = frame_index
                    inc.end_timestamp = timestamp
                    inc.next_state    = current_state
                    if inc.duration_seconds >= self.min_incident_duration_s:
                        self.incidents.append(inc)
                self._inactive_streak[tid] = 0

        # Close incidents for disappeared tracks
        disappeared = set(self._open_incidents) - set(track_states)
        for tid in disappeared:
            inc = self._open_incidents.pop(tid)
            inc.end_frame     = frame_index
            inc.end_timestamp = timestamp
            if inc.duration_seconds >= self.min_incident_duration_s:
                self.incidents.append(inc)
            self._inactive_streak[tid] = 0

        # Frame-level dominant state
        current_state_frame = self._dominant_state(track_states)
        self._state_frame_counts[current_state_frame] = \
            self._state_frame_counts.get(current_state_frame, 0) + 1

        # Alert logic
        self.max_streak = max(self._inactive_streak.values()) if self._inactive_streak else 0
        if self.max_streak >= self.critical_frame_threshold:
            self.alert_active = True
            self.alert_level  = "CRITICAL"
        elif self.max_streak >= self.alert_frame_threshold:
            self.alert_active = True
            self.alert_level  = "WARNING"
        else:
            self.alert_active = False
            self.alert_level  = None

        return {
            "current_state":     current_state_frame,
            "alert_active":      self.alert_active,
            "alert_level":       self.alert_level,
            "new_transitions":   [t.to_dict() for t in new_transitions],
            "inactive_streak":   self.max_streak,
            "active_tracks":     track_states,
        }

    # ------------------------------------------------------------------ #
    # KPIs                                                                 #
    # ------------------------------------------------------------------ #

    @property
    def total_productive_frames(self) -> int:
        return sum(self._state_frame_counts.get(s, 0) for s in self.productive_states)

    @property
    def total_inactive_frames(self) -> int:
        return sum(self._state_frame_counts.get(s, 0) for s in self.inactive_states)

    @property
    def activity_rate(self) -> float:
        """% of frames with a productive state."""
        total = self.total_productive_frames + self.total_inactive_frames
        if total == 0:
            return 100.0
        return round(self.total_productive_frames / total * 100, 2)

    @property
    def inactivity_rate(self) -> float:
        return round(100.0 - self.activity_rate, 2)

    @property
    def incident_count(self) -> int:
        return len(self.incidents)

    @property
    def avg_incident_duration(self) -> float:
        closed = [i for i in self.incidents if i.is_closed]
        if not closed:
            return 0.0
        return round(sum(i.duration_seconds for i in closed) / len(closed), 3)

    @property
    def longest_incident_duration(self) -> float:
        closed = [i for i in self.incidents if i.is_closed]
        return round(max((i.duration_seconds for i in closed), default=0.0), 3)

    def state_breakdown(self) -> dict[str, float]:
        """% of analyzed frames per state."""
        total = self.total_frames_analyzed or 1
        return {
            state: round(count / total * 100, 2)
            for state, count in self._state_frame_counts.items()
            if count > 0
        }

    def get_summary(self) -> dict:
        return {
            "activity_rate":              self.activity_rate,
            "inactivity_rate":            self.inactivity_rate,
            "total_productive_frames":    self.total_productive_frames,
            "total_inactive_frames":      self.total_inactive_frames,
            "total_frames_analyzed":      self.total_frames_analyzed,
            "incident_count":             self.incident_count,
            "avg_incident_duration_s":    self.avg_incident_duration,
            "longest_incident_duration_s":self.longest_incident_duration,
            "state_breakdown_pct":        self.state_breakdown(),
            "transitions":                [t.to_dict() for t in self.transitions],
            "incidents":                  [i.to_dict() for i in self.incidents],
        }

    def reset(self) -> None:
        self._track_states.clear()
        self._inactive_streak.clear()
        self._open_incidents.clear()
        self.transitions.clear()
        self.incidents.clear()
        self._state_frame_counts = {s: 0 for s in ALL_STATES}
        self._state_frame_counts["no_detection"] = 0
        self.total_frames_analyzed = 0
        self.alert_active = False
        self.alert_level  = None
        self.max_streak   = 0

    # ------------------------------------------------------------------ #

    @staticmethod
    def _dominant_state(track_states: dict[int, str]) -> str:
        """Return the most critical state across all tracks in this frame."""
        if not track_states:
            return "no_detection"
        states = list(track_states.values())
        # Priority: inactivo > activo > preparacion > descarga
        priority = ["estado_inactivo", "estado_activo", "estado_preparacion", "estado_descarga"]
        for s in priority:
            if s in states:
                return s
        return states[0]
