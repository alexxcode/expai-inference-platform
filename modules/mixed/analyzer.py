"""
Mixed module analyzer — unified EPP + operator state monitoring.

Model classes:
  0  mano_con_guante      hand with glove
  1  mano_sin_guante      hand without glove
  2  operario_activo      executing active fold cycle
  3  operario_transicion  in station, not in active cycle

Risk is evaluated by crossing the detected operator state with hand protection state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.inference_engine import Detection


# ── Class sets ────────────────────────────────────────────────────────
HAND_PROTECTED   = {"mano_con_guante"}
HAND_UNPROTECTED = {"mano_sin_guante"}
HAND_CLASSES     = HAND_PROTECTED | HAND_UNPROTECTED

OPERATOR_ACTIVE     = {"operario_activo"}
OPERATOR_TRANSITION = {"operario_transicion"}
OPERATOR_CLASSES    = OPERATOR_ACTIVE | OPERATOR_TRANSITION

ALL_CLASSES = HAND_CLASSES | OPERATOR_CLASSES


class RiskLevel(str, Enum):
    OK       = "OK"
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# Risk matrix: (operator_state, hand_state) → RiskLevel
RISK_MATRIX: dict[tuple[str, str], RiskLevel] = {
    ("operario_activo",      "mano_sin_guante"): RiskLevel.CRITICAL,
    ("operario_activo",      "mano_con_guante"): RiskLevel.OK,
    ("operario_activo",      "no_hand"):         RiskLevel.WARNING,
    ("operario_transicion",  "mano_sin_guante"): RiskLevel.WARNING,
    ("operario_transicion",  "mano_con_guante"): RiskLevel.OK,
    ("operario_transicion",  "no_hand"):         RiskLevel.OK,
    ("no_operator",          "mano_sin_guante"): RiskLevel.INFO,
    ("no_operator",          "mano_con_guante"): RiskLevel.OK,
    ("no_operator",          "no_hand"):         RiskLevel.OK,
}

# BGR colors for visualizer
CLASS_COLORS_MIXED = {
    "mano_con_guante":      (0,   200, 0),     # green
    "mano_sin_guante":      (0,   0,   220),   # red
    "operario_activo":      (0,   200, 200),   # cyan
    "operario_transicion":  (0,   165, 255),   # orange
}


@dataclass
class MixedEvent:
    event_type:    str           # RISK_CHANGE | EPP_VIOLATION | TRANSITION_IDLE | CYCLE_START | CYCLE_END
    risk_level:    str
    operator_state: str
    hand_state:    str
    frame_index:   int
    timestamp:     float
    track_id:      Optional[int] = None
    details:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":     self.event_type,
            "risk_level":     self.risk_level,
            "operator_state": self.operator_state,
            "hand_state":     self.hand_state,
            "frame_index":    self.frame_index,
            "timestamp":      round(self.timestamp, 3),
            "track_id":       self.track_id,
            "details":        self.details,
        }


@dataclass
class MixedIncident:
    incident_type:   str           # EPP_VIOLATION | TRANSITION_IDLE
    risk_level:      str
    operator_state:  str
    hand_state:      str
    start_frame:     int
    start_timestamp: float
    end_frame:       Optional[int]   = None
    end_timestamp:   Optional[float] = None
    track_id:        Optional[int]   = None

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is not None:
            return round(self.end_timestamp - self.start_timestamp, 3)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "incident_type":   self.incident_type,
            "risk_level":      self.risk_level,
            "operator_state":  self.operator_state,
            "hand_state":      self.hand_state,
            "start_frame":     self.start_frame,
            "start_timestamp": round(self.start_timestamp, 3),
            "end_frame":       self.end_frame,
            "end_timestamp":   round(self.end_timestamp, 3) if self.end_timestamp else None,
            "duration_seconds": self.duration_seconds,
            "track_id":        self.track_id,
        }


class MixedAnalyzer:
    """
    Unified EPP + operator state analyzer.

    Per frame it determines:
      - operator_state: operario_activo | operario_transicion | no_operator
      - hand_state:     mano_con_guante | mano_sin_guante | no_hand
      - risk_level:     OK | INFO | WARNING | CRITICAL

    Incidents are opened when risk > OK and closed when risk returns to OK.
    """

    def __init__(
        self,
        critical_frame_threshold:  int   = 10,
        warning_frame_threshold:   int   = 25,
        transition_idle_threshold: int   = 75,
        min_incident_duration_s:   float = 0.5,
    ):
        self.critical_frame_threshold  = critical_frame_threshold
        self.warning_frame_threshold   = warning_frame_threshold
        self.transition_idle_threshold = transition_idle_threshold
        self.min_incident_duration_s   = min_incident_duration_s

        # Current risk streak counters  {risk_level: frames}
        self._risk_streak:         dict[str, int] = {}
        self._transition_streak:   int = 0   # consecutive frames of operario_transicion

        # Previous frame state (for change detection)
        self._prev_operator: str = "no_operator"
        self._prev_hand:     str = "no_hand"
        self._prev_risk:     RiskLevel = RiskLevel.OK

        # Open incidents  {(operator_state, incident_type): MixedIncident}
        self._open_incidents: dict[str, MixedIncident] = {}

        self.events:    list[MixedEvent]    = []
        self.incidents: list[MixedIncident] = []

        # Frame counters per state combination
        self._frame_counts: dict[str, int] = {
            "ok": 0, "info": 0, "warning": 0, "critical": 0, "no_detection": 0,
        }
        self._operator_frame_counts: dict[str, int] = {
            "operario_activo": 0, "operario_transicion": 0, "no_operator": 0,
        }

        self.total_frames_analyzed = 0
        self.current_risk:     RiskLevel = RiskLevel.OK
        self.current_operator: str       = "no_operator"
        self.current_hand:     str       = "no_hand"
        self.alert_active:     bool      = False
        self.alert_level:      Optional[str] = None
        self.max_risk_streak:  int       = 0

    # ─────────────────────────────────────────────────────────────────

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp:   float,
    ) -> dict:
        self.total_frames_analyzed += 1
        new_events: list[MixedEvent] = []

        # ── 1. Classify detections ──────────────────────────────────
        operator_state = self._get_operator_state(detections)
        hand_state     = self._get_hand_state(detections)
        risk           = RISK_MATRIX.get((operator_state, hand_state), RiskLevel.OK)

        self.current_operator = operator_state
        self.current_hand     = hand_state
        self.current_risk     = risk

        # ── 2. Update frame counters ─────────────────────────────────
        self._frame_counts[risk.value.lower()] = self._frame_counts.get(risk.value.lower(), 0) + 1
        self._operator_frame_counts[operator_state] = \
            self._operator_frame_counts.get(operator_state, 0) + 1

        if operator_state in OPERATOR_TRANSITION:
            self._transition_streak += 1
        else:
            self._transition_streak = 0

        # ── 3. Risk streak ───────────────────────────────────────────
        if risk != RiskLevel.OK:
            self._risk_streak[risk.value] = self._risk_streak.get(risk.value, 0) + 1
        else:
            self._risk_streak.clear()

        self.max_risk_streak = max(self._risk_streak.values()) if self._risk_streak else 0

        # ── 4. State change events ───────────────────────────────────
        risk_changed = risk != self._prev_risk
        if risk_changed and risk != RiskLevel.OK:
            evt = MixedEvent(
                event_type="RISK_CHANGE",
                risk_level=risk.value,
                operator_state=operator_state,
                hand_state=hand_state,
                frame_index=frame_index,
                timestamp=timestamp,
                details={"from_risk": self._prev_risk.value},
            )
            self.events.append(evt)
            new_events.append(evt)

        # EPP violation event (first frame of violation)
        if hand_state in HAND_UNPROTECTED and self._prev_hand not in HAND_UNPROTECTED:
            evt = MixedEvent(
                event_type="EPP_VIOLATION",
                risk_level=risk.value,
                operator_state=operator_state,
                hand_state=hand_state,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            self.events.append(evt)
            new_events.append(evt)

        # EPP restored event
        if hand_state in HAND_PROTECTED and self._prev_hand in HAND_UNPROTECTED:
            evt = MixedEvent(
                event_type="EPP_RESTORED",
                risk_level=RiskLevel.OK.value,
                operator_state=operator_state,
                hand_state=hand_state,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            self.events.append(evt)
            new_events.append(evt)

        # Transition idle event
        if self._transition_streak == self.transition_idle_threshold:
            evt = MixedEvent(
                event_type="TRANSITION_IDLE",
                risk_level=RiskLevel.INFO.value,
                operator_state=operator_state,
                hand_state=hand_state,
                frame_index=frame_index,
                timestamp=timestamp,
                details={"streak": self._transition_streak},
            )
            self.events.append(evt)
            new_events.append(evt)

        # ── 5. Incident management ───────────────────────────────────
        self._manage_incidents(risk, operator_state, hand_state, frame_index, timestamp)

        # ── 6. Alert state ───────────────────────────────────────────
        if risk == RiskLevel.CRITICAL and self.max_risk_streak >= self.critical_frame_threshold:
            self.alert_active = True
            self.alert_level  = "CRITICAL"
        elif risk == RiskLevel.WARNING and self.max_risk_streak >= self.warning_frame_threshold:
            self.alert_active = True
            self.alert_level  = "WARNING"
        elif risk == RiskLevel.INFO:
            self.alert_active = False
            self.alert_level  = "INFO"
        else:
            self.alert_active = False
            self.alert_level  = None

        # ── 7. Save previous state ───────────────────────────────────
        self._prev_operator = operator_state
        self._prev_hand     = hand_state
        self._prev_risk     = risk

        return {
            "operator_state":    operator_state,
            "hand_state":        hand_state,
            "risk_level":        risk.value,
            "alert_active":      self.alert_active,
            "alert_level":       self.alert_level,
            "risk_streak":       self.max_risk_streak,
            "transition_streak": self._transition_streak,
            "new_events":        [e.to_dict() for e in new_events],
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_operator_state(self, detections: list[Detection]) -> str:
        """Dominant operator state across all detections."""
        states = [d.class_name for d in detections if d.class_name in OPERATOR_CLASSES]
        if not states:
            return "no_operator"
        # active > transition
        return "operario_activo" if "operario_activo" in states else "operario_transicion"

    def _get_hand_state(self, detections: list[Detection]) -> str:
        """Worst-case hand state (unprotected takes priority)."""
        hands = [d.class_name for d in detections if d.class_name in HAND_CLASSES]
        if not hands:
            return "no_hand"
        return "mano_sin_guante" if "mano_sin_guante" in hands else "mano_con_guante"

    def _manage_incidents(
        self,
        risk: RiskLevel,
        operator_state: str,
        hand_state: str,
        frame_index: int,
        timestamp: float,
    ) -> None:
        incident_type = None
        if hand_state in HAND_UNPROTECTED:
            incident_type = "EPP_VIOLATION"
        elif self._transition_streak >= self.transition_idle_threshold:
            incident_type = "TRANSITION_IDLE"

        if incident_type:
            key = incident_type
            if key not in self._open_incidents:
                self._open_incidents[key] = MixedIncident(
                    incident_type=incident_type,
                    risk_level=risk.value,
                    operator_state=operator_state,
                    hand_state=hand_state,
                    start_frame=frame_index,
                    start_timestamp=timestamp,
                )
        else:
            # Close all open incidents
            for key, inc in list(self._open_incidents.items()):
                inc.end_frame     = frame_index
                inc.end_timestamp = timestamp
                if inc.duration_seconds >= self.min_incident_duration_s:
                    self.incidents.append(inc)
                del self._open_incidents[key]

    def _close_all_incidents(self, frame_index: int, timestamp: float) -> None:
        for inc in self._open_incidents.values():
            inc.end_frame     = frame_index
            inc.end_timestamp = timestamp
            if inc.duration_seconds >= self.min_incident_duration_s:
                self.incidents.append(inc)
        self._open_incidents.clear()

    # ── KPIs ─────────────────────────────────────────────────────────

    @property
    def epp_compliance_rate(self) -> float:
        """% frames with protected or no hands (not unprotected)."""
        unprotected = self._frame_counts.get("critical", 0) + self._frame_counts.get("warning", 0)
        total = self.total_frames_analyzed or 1
        return round((1 - unprotected / total) * 100, 2)

    @property
    def activity_rate(self) -> float:
        active = self._operator_frame_counts.get("operario_activo", 0)
        total  = self.total_frames_analyzed or 1
        return round(active / total * 100, 2)

    @property
    def critical_incident_count(self) -> int:
        return sum(1 for i in self.incidents if i.risk_level == "CRITICAL")

    @property
    def warning_incident_count(self) -> int:
        return sum(1 for i in self.incidents if i.risk_level == "WARNING")

    @property
    def epp_violations(self) -> list[MixedIncident]:
        return [i for i in self.incidents if i.incident_type == "EPP_VIOLATION"]

    @property
    def transition_idle_incidents(self) -> list[MixedIncident]:
        return [i for i in self.incidents if i.incident_type == "TRANSITION_IDLE"]

    def get_summary(self) -> dict:
        epp_v = self.epp_violations
        tran  = self.transition_idle_incidents
        return {
            "epp_compliance_rate":      self.epp_compliance_rate,
            "activity_rate":            self.activity_rate,
            "total_frames_analyzed":    self.total_frames_analyzed,
            "operator_frame_breakdown": {
                k: round(v / max(self.total_frames_analyzed, 1) * 100, 2)
                for k, v in self._operator_frame_counts.items()
            },
            "risk_frame_breakdown": {
                k: round(v / max(self.total_frames_analyzed, 1) * 100, 2)
                for k, v in self._frame_counts.items() if v > 0
            },
            "incidents": {
                "total":        len(self.incidents),
                "critical":     self.critical_incident_count,
                "warning":      self.warning_incident_count,
                "epp_violations": {
                    "count":    len(epp_v),
                    "avg_duration_s": round(
                        sum(i.duration_seconds for i in epp_v) / len(epp_v), 3
                    ) if epp_v else 0,
                    "max_duration_s": round(
                        max((i.duration_seconds for i in epp_v), default=0), 3
                    ),
                    "list": [i.to_dict() for i in epp_v],
                },
                "transition_idle": {
                    "count":    len(tran),
                    "avg_duration_s": round(
                        sum(i.duration_seconds for i in tran) / len(tran), 3
                    ) if tran else 0,
                    "list": [i.to_dict() for i in tran],
                },
            },
            "events": [e.to_dict() for e in self.events],
        }

    def reset(self) -> None:
        self._risk_streak.clear()
        self._transition_streak = 0
        self._prev_operator = "no_operator"
        self._prev_hand     = "no_hand"
        self._prev_risk     = RiskLevel.OK
        self._open_incidents.clear()
        self.events.clear()
        self.incidents.clear()
        self._frame_counts = {k: 0 for k in self._frame_counts}
        self._operator_frame_counts = {k: 0 for k in self._operator_frame_counts}
        self.total_frames_analyzed = 0
        self.current_risk = RiskLevel.OK
        self.alert_active = False
        self.alert_level  = None
        self.max_risk_streak = 0
