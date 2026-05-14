"""Procesado analyzer — counts processed pieces via track ID deduplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.inference_engine import Detection


@dataclass
class ProcesadoEvent:
    event_type: str        # PIEZA_PROCESADA | ACUMULACION_DETECTADA | PRODUCCION_RETOMADA | STALL_DETECTADO
    frame: int
    timestamp: float
    track_id: Optional[int] = None
    total_processed: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":       self.event_type,
            "frame":            self.frame,
            "timestamp":        round(self.timestamp, 3),
            "track_id":         self.track_id,
            "total_processed":  self.total_processed,
            **self.details,
        }


class ProcesadoAnalyzer:
    """
    Frame-by-frame piece processing counter.

    Logic:
      - pieza_ok  track appears for min_track_frames → +1 processed (deduplicated)
      - pieza_sin_proceso count tracked as pending backlog
      - Stall alert: no new pieza_ok for stall_frame_threshold frames
      - Backlog alert: pending >= backlog_count_threshold for backlog_frame_threshold frames
    """

    def __init__(
        self,
        processed_classes: list[str],
        pending_classes: list[str],
        stall_frame_threshold: int = 180,
        min_track_frames: int = 3,
        backlog_count_threshold: int = 3,
        backlog_frame_threshold: int = 60,
        rate_window_seconds: float = 60.0,
    ):
        self.processed_classes      = set(processed_classes)
        self.pending_classes        = set(pending_classes)
        self.stall_frame_threshold  = stall_frame_threshold
        self.min_track_frames       = min_track_frames
        self.backlog_count_threshold  = backlog_count_threshold
        self.backlog_frame_threshold  = backlog_frame_threshold
        self.rate_window_seconds    = rate_window_seconds

        # Counting state
        self._counted_track_ids: set[int]  = set()
        self._track_frames: dict[int, int] = {}

        # KPI state
        self.total_processed: int   = 0
        self.pending_count: int     = 0
        self._frames_since_ok: int  = 0
        self._backlog_frames: int   = 0
        self._stall_active: bool    = False
        self._backlog_active: bool  = False
        self._new_events: list[dict] = []

        # Rate calculation
        self._completion_times: deque[float] = deque()

        # Summary
        self.total_frames: int           = 0
        self._processed_timestamps: list[float] = []

    # ── Public API ────────────────────────────────────────────────────

    def analyze_frame(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
    ) -> dict:
        self._new_events = []
        self.total_frames += 1
        self._frames_since_ok += 1

        ok_dets      = [d for d in detections if d.class_name in self.processed_classes]
        pending_dets = [d for d in detections if d.class_name in self.pending_classes]

        self.pending_count = len(pending_dets)

        # ── Count processed pieces via track IDs ───────────────────────
        for det in ok_dets:
            tid = det.track_id
            if tid is None:
                continue
            self._track_frames[tid] = self._track_frames.get(tid, 0) + 1
            if (
                self._track_frames[tid] >= self.min_track_frames
                and tid not in self._counted_track_ids
            ):
                self._counted_track_ids.add(tid)
                self.total_processed += 1
                self._frames_since_ok = 0
                self._processed_timestamps.append(timestamp)
                self._completion_times.append(timestamp)
                self._new_events.append(ProcesadoEvent(
                    event_type="PIEZA_PROCESADA",
                    frame=frame_index,
                    timestamp=timestamp,
                    track_id=tid,
                    total_processed=self.total_processed,
                ).to_dict())

                # Resume after stall
                if self._stall_active:
                    self._stall_active = False
                    self._new_events.append(ProcesadoEvent(
                        event_type="PRODUCCION_RETOMADA",
                        frame=frame_index,
                        timestamp=timestamp,
                        total_processed=self.total_processed,
                    ).to_dict())

        # ── Stall detection ────────────────────────────────────────────
        if (
            self._frames_since_ok >= self.stall_frame_threshold
            and not self._stall_active
            and self.total_processed > 0
        ):
            self._stall_active = True
            self._new_events.append(ProcesadoEvent(
                event_type="STALL_DETECTADO",
                frame=frame_index,
                timestamp=timestamp,
                total_processed=self.total_processed,
                details={"frames_sin_ok": self._frames_since_ok},
            ).to_dict())

        # ── Backlog detection ──────────────────────────────────────────
        if self.pending_count >= self.backlog_count_threshold:
            self._backlog_frames += 1
        else:
            self._backlog_frames = 0
            if self._backlog_active:
                self._backlog_active = False

        if (
            self._backlog_frames >= self.backlog_frame_threshold
            and not self._backlog_active
        ):
            self._backlog_active = True
            self._new_events.append(ProcesadoEvent(
                event_type="ACUMULACION_DETECTADA",
                frame=frame_index,
                timestamp=timestamp,
                total_processed=self.total_processed,
                details={"pending_count": self.pending_count},
            ).to_dict())

        # ── Rate window purge ──────────────────────────────────────────
        cutoff = timestamp - self.rate_window_seconds
        while self._completion_times and self._completion_times[0] < cutoff:
            self._completion_times.popleft()

        alert_active = self._stall_active or self._backlog_active
        alert_level  = "WARNING" if alert_active else None

        return {
            "total_processed":   self.total_processed,
            "pending_count":     self.pending_count,
            "pieces_per_min":    self._pieces_per_min(),
            "stall_active":      self._stall_active,
            "backlog_active":    self._backlog_active,
            "alert_active":      alert_active,
            "alert_level":       alert_level,
            "new_events":        self._new_events,
        }

    def get_summary(self) -> dict:
        duration_s = self._processed_timestamps[-1] - self._processed_timestamps[0] \
                     if len(self._processed_timestamps) >= 2 else 0.0
        pph = round(self.total_processed / (duration_s / 3600), 1) if duration_s > 0 else 0.0
        return {
            "total_pieces_processed": self.total_processed,
            "pieces_per_hour":        pph,
            "total_frames_analyzed":  self.total_frames,
            "processed_timestamps":   [round(t, 2) for t in self._processed_timestamps],
        }

    def reset(self) -> None:
        self._counted_track_ids.clear()
        self._track_frames.clear()
        self.total_processed    = 0
        self.pending_count      = 0
        self._frames_since_ok   = 0
        self._backlog_frames    = 0
        self._stall_active      = False
        self._backlog_active    = False
        self._new_events        = []
        self._completion_times.clear()
        self.total_frames       = 0
        self._processed_timestamps = []

    # ── Helpers ───────────────────────────────────────────────────────

    def _pieces_per_min(self) -> float:
        if len(self._completion_times) < 2:
            return 0.0
        span = self._completion_times[-1] - self._completion_times[0]
        if span <= 0:
            return 0.0
        return round(len(self._completion_times) / (span / 60), 2)
