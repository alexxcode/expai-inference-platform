"""Kitting analyzer — counts completed kits using track ID deduplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.inference_engine import Detection


@dataclass
class KitEvent:
    event_type: str        # KIT_COMPLETADO | PRODUCCION_PAUSADA | PRODUCCION_REANUDADA
    frame: int
    timestamp: float
    track_id: Optional[int] = None
    total_kits: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":  self.event_type,
            "frame":       self.frame,
            "timestamp":   round(self.timestamp, 3),
            "track_id":    self.track_id,
            "total_kits":  self.total_kits,
            **self.details,
        }


class KittingAnalyzer:
    """
    Frame-by-frame kitting counter.

    Counting logic:
      - Tracks every detection with ByteTrack
      - When a 'kit_listo' track ID is seen for the first time → +1 kit
      - Minimum track_frames required to avoid transient detections
      - Tracks disappearing after being counted are NOT re-counted
    """

    def __init__(
        self,
        kit_done_classes: list[str],
        kit_wip_classes: list[str],
        stall_frame_threshold: int = 300,
        min_track_frames: int = 3,
        rate_window_seconds: float = 60.0,
    ):
        self.kit_done_classes   = set(kit_done_classes)
        self.kit_wip_classes    = set(kit_wip_classes)
        self.stall_frame_threshold = stall_frame_threshold
        self.min_track_frames   = min_track_frames
        self.rate_window_seconds = rate_window_seconds

        # Counting state
        self._counted_track_ids: set[int] = set()    # track IDs already counted
        self._track_frames: dict[int, int] = {}      # track_id → frames seen
        self._track_class: dict[int, str]  = {}      # track_id → last class

        # KPI state
        self.total_kits: int        = 0
        self.kits_in_progress: int  = 0
        self.operator_present: bool = False
        self._frames_since_kit: int = 0
        self._stall_active: bool    = False
        self._new_events: list[dict] = []

        # Rate calculation — timestamps of completed kits in rolling window
        self._completion_times: deque[float] = deque()

        # Summary stats
        self.total_frames: int = 0
        self._kit_timestamps: list[float] = []   # all completion timestamps

    # ── Public API ─────────────────────────────────────────────────────

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
    ) -> dict:
        self._new_events = []
        self.total_frames += 1
        self._frames_since_kit += 1

        # Separate detections by category
        done_dets     = [d for d in detections if d.class_name in self.kit_done_classes]
        wip_dets      = [d for d in detections if d.class_name in self.kit_wip_classes]
        operator_dets = [d for d in detections if d.class_name == "operario"]

        self.operator_present = len(operator_dets) > 0
        self.kits_in_progress = len(wip_dets)

        # ── Count completed kits via track IDs ─────────────────────────
        for det in done_dets:
            tid = det.track_id
            if tid is None:
                # No tracking — count each unique high-conf detection once per burst
                continue

            # Accumulate frames per track
            self._track_frames[tid]  = self._track_frames.get(tid, 0) + 1
            self._track_class[tid]   = det.class_name

            # Count once track is confirmed (min_track_frames) and not yet counted
            if (
                self._track_frames[tid] >= self.min_track_frames
                and tid not in self._counted_track_ids
            ):
                self._counted_track_ids.add(tid)
                self.total_kits += 1
                self._frames_since_kit = 0
                self._kit_timestamps.append(timestamp)
                self._completion_times.append(timestamp)
                self._new_events.append(KitEvent(
                    event_type="KIT_COMPLETADO",
                    frame=frame_index,
                    timestamp=timestamp,
                    track_id=tid,
                    total_kits=self.total_kits,
                ).to_dict())

                # Reanudar si estaba parado
                if self._stall_active:
                    self._stall_active = False
                    self._new_events.append(KitEvent(
                        event_type="PRODUCCION_REANUDADA",
                        frame=frame_index,
                        timestamp=timestamp,
                        total_kits=self.total_kits,
                    ).to_dict())

        # ── Stall detection ────────────────────────────────────────────
        if (
            self._frames_since_kit >= self.stall_frame_threshold
            and not self._stall_active
            and self.total_kits > 0          # only alert after first kit
        ):
            self._stall_active = True
            self._new_events.append(KitEvent(
                event_type="PRODUCCION_PAUSADA",
                frame=frame_index,
                timestamp=timestamp,
                total_kits=self.total_kits,
                details={"frames_sin_kit": self._frames_since_kit},
            ).to_dict())

        # ── Purge old entries from rate window ─────────────────────────
        cutoff = timestamp - self.rate_window_seconds
        while self._completion_times and self._completion_times[0] < cutoff:
            self._completion_times.popleft()

        return {
            "total_kits":      self.total_kits,
            "kits_in_progress": self.kits_in_progress,
            "kits_per_min":    self._kits_per_min(),
            "operator_present": self.operator_present,
            "stall_active":    self._stall_active,
            "alert_active":    self._stall_active,
            "alert_level":     "WARNING" if self._stall_active else None,
            "new_events":      self._new_events,
        }

    def get_summary(self) -> dict:
        duration_s = self._kit_timestamps[-1] - self._kit_timestamps[0] \
                     if len(self._kit_timestamps) >= 2 else 0.0
        kph = round(self.total_kits / (duration_s / 3600), 1) if duration_s > 0 else 0.0
        return {
            "total_kits_completed":  self.total_kits,
            "kits_per_hour":         kph,
            "total_frames_analyzed": self.total_frames,
            "completion_timestamps": [round(t, 2) for t in self._kit_timestamps],
        }

    def reset(self) -> None:
        self._counted_track_ids.clear()
        self._track_frames.clear()
        self._track_class.clear()
        self.total_kits        = 0
        self.kits_in_progress  = 0
        self.operator_present  = False
        self._frames_since_kit = 0
        self._stall_active     = False
        self._new_events       = []
        self._completion_times.clear()
        self.total_frames      = 0
        self._kit_timestamps   = []

    # ── Helpers ────────────────────────────────────────────────────────

    def _kits_per_min(self) -> float:
        if len(self._completion_times) < 2:
            return 0.0
        span = self._completion_times[-1] - self._completion_times[0]
        if span <= 0:
            return 0.0
        return round(len(self._completion_times) / (span / 60), 2)
