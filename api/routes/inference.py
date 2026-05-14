"""Inference control endpoints — supports epp_gloves, inactivity and mixed modules."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from api.schemas import InferenceStartRequest, JobStatus
from core.config import BASE_DIR, resolve_path
from core.inference_engine import InferenceEngine
from core.video_processor import VideoProcessor
from core.visualizer import Visualizer
from modules.epp_gloves.module import EPPGlovesModule
from modules.inactivity.module import InactivityModule
from modules.mixed.module import MixedModule
from modules.metrology.module import MetrologyModule
from modules.kitting.module import KittingModule
from modules.procesado.module import ProcesadoModule

router = APIRouter(prefix="/inference", tags=["inference"])

AVAILABLE_MODULES = {"epp_gloves", "inactivity", "mixed", "metrology", "kitting", "procesado"}

_jobs: dict[str, dict] = {}
_sse_queues: dict[str, asyncio.Queue] = {}


def _get_job_or_404(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


def _load_module(module_name: str):
    if module_name == "epp_gloves":
        return EPPGlovesModule()
    elif module_name == "inactivity":
        return InactivityModule()
    elif module_name == "mixed":
        return MixedModule()
    elif module_name == "metrology":
        return MetrologyModule()
    elif module_name == "kitting":
        return KittingModule()
    elif module_name == "procesado":
        return ProcesadoModule()
    raise ValueError(f"Unknown module: {module_name}")


@router.post("/start", response_model=JobStatus, summary="Start an inference job")
async def start_inference(request: InferenceStartRequest, background_tasks: BackgroundTasks):
    if request.module not in AVAILABLE_MODULES:
        raise HTTPException(status_code=400, detail=f"Module '{request.module}' not available. Use: {AVAILABLE_MODULES}")

    job_id     = str(uuid.uuid4())[:8]
    video_path = resolve_path(request.video_path)
    model_path = resolve_path(request.model_path)

    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model not found: {model_path}")

    _jobs[job_id] = {
        "job_id":           job_id,
        "module":           request.module,
        "status":           "pending",
        "progress":         0.0,
        "current_frame":    0,
        "total_frames":     0,
        "video_path":       str(video_path),
        "model_path":       str(model_path),
        "config_overrides": request.config_overrides or {},
        "report":           None,
        "annotated_video_path": None,
        "started_at":       time.time(),
    }
    _sse_queues[job_id] = asyncio.Queue(maxsize=200)
    background_tasks.add_task(_run_inference, job_id)
    logger.info(f"Job {job_id} queued: module={request.module}")
    return JobStatus(job_id=job_id, module=request.module, status="pending")


@router.get("/{job_id}/status", response_model=JobStatus)
async def get_status(job_id: str):
    job = _get_job_or_404(job_id)
    return JobStatus(**{k: job[k] for k in JobStatus.model_fields if k in job})


@router.get("/{job_id}/stream", summary="SSE stream of real-time detections")
async def stream_results(job_id: str):
    _get_job_or_404(job_id)
    queue = _sse_queues.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="No stream for this job")

    async def event_generator() -> AsyncIterator[str]:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in ("completed", "failed"):
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{job_id}/stop")
async def stop_inference(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] == "running":
        job["status"] = "stopping"
    return {"job_id": job_id, "status": job["status"]}


# ────────────────────────────────────────────────────────────────────── #
# Background task                                                         #
# ────────────────────────────────────────────────────────────────────── #

async def _run_inference(job_id: str) -> None:
    job   = _jobs[job_id]
    queue = _sse_queues[job_id]
    job["status"] = "running"

    try:
        module = _load_module(job["module"])
        if job["config_overrides"]:
            _apply_overrides(module.config, job["config_overrides"])
        module.config["model"]["path"] = job["model_path"]

        engine     = InferenceEngine.from_module_config(module.config)
        visualizer = Visualizer()
        video      = VideoProcessor(job["video_path"])
        job["total_frames"] = video.total_frames

        out_cfg = module.config.get("output", {})
        writer  = None
        if out_cfg.get("annotated_video", True):
            out_dir = resolve_path(out_cfg.get("annotated_video_path", "outputs/annotated_videos/"))
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = out_dir / f"{job_id}_annotated.mp4"
            writer = video.get_writer(annotated_path)
            job["annotated_video_path"] = str(annotated_path)

        loop = asyncio.get_event_loop()

        with video:
            for frame_idx, timestamp, frame in video.frames():
                if job.get("status") == "stopping":
                    break

                detections = await loop.run_in_executor(None, engine.predict, frame)
                result     = module.process_frame(frame, detections, frame_idx, timestamp)
                ann        = result["annotations"]

                visualizer.draw_detections(frame, detections)
                _mod = job["module"]
                visualizer.draw_status_overlay(
                    frame,
                    operator_state=ann.get("verdict") or ann.get("current_state") or ann.get("operator_state", ""),
                    metric_value=ann.get("total_processed") or ann.get("total_kits") or ann.get("ok_rate") or ann.get("activity_rate") or ann.get("compliance_pct") or ann.get("epp_compliance") or 0.0,
                    metric_label="Conformes %" if _mod == "metrology" else "Kits completados" if _mod == "kitting" else "Piezas procesadas" if _mod == "procesado" else "Actividad" if _mod == "inactivity" else "Cumplimiento EPP",
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    alert_active=ann.get("alert_active", False),
                    alert_level=ann.get("alert_level") or ann.get("risk_level"),
                    alert_text=_alert_text(_mod, ann),
                )

                if writer:
                    writer.write(frame)

                job["current_frame"] = frame_idx
                job["progress"]      = round(frame_idx / max(video.total_frames, 1), 4)

                sse = {
                    "frame":         frame_idx,
                    "timestamp":     round(timestamp, 3),
                    "operator_state": ann.get("current_state") or ann.get("operator_state", ""),
                    "alert_active":  ann.get("alert_active", False),
                    "alert_level":   ann.get("alert_level"),
                    # EPP
                    "compliance_pct": ann.get("compliance_pct"),
                    # Inactivity
                    "activity_rate":  ann.get("activity_rate"),
                    "inactive_streak": ann.get("inactive_streak"),
                    # Mixed
                    "hand_state":     ann.get("hand_state"),
                    "risk_level":     ann.get("risk_level"),
                    "epp_compliance": ann.get("epp_compliance"),
                    "risk_streak":    ann.get("risk_streak"),
                    # Kitting
                    "total_kits":     ann.get("total_kits"),
                    "kits_in_progress": ann.get("kits_in_progress"),
                    "kits_per_min":   ann.get("kits_per_min"),
                    "operator_present": ann.get("operator_present"),
                    "stall_active":   ann.get("stall_active"),
                    # Metrology
                    "verdict":        ann.get("verdict"),
                    "detected_class": ann.get("detected_class"),
                    "confidence":     ann.get("confidence"),
                    "nok_streak":     ann.get("nok_streak"),
                    "ok_rate":        ann.get("ok_rate"),
                    "nok_rate":       ann.get("nok_rate"),
                    # Procesado
                    "total_processed":    ann.get("total_processed"),
                    "pending_count":      ann.get("pending_count"),
                    "pieces_per_min":     ann.get("pieces_per_min"),
                    "backlog_active":     ann.get("backlog_active"),
                    "events":        result.get("events", []),
                    "status":        "running",
                }
                try:
                    queue.put_nowait(sse)
                except asyncio.QueueFull:
                    pass

        if writer:
            writer.release()

        report = module.generate_report()
        _save_report(job_id, report, out_cfg)
        job["report"] = report
        job["status"] = "completed"
        job["progress"] = 1.0
        logger.info(f"Job {job_id} completed.")
        await queue.put({"status": "completed", "report": report})

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        job["status"]  = "failed"
        job["message"] = str(exc)
        try:
            await queue.put({"status": "failed", "error": str(exc)})
        except Exception:
            pass


def _alert_text(module: str, ann: dict) -> str:
    if module == "inactivity":
        level = ann.get("alert_level", "WARNING")
        streak = ann.get("inactive_streak", 0)
        return f"⚠ INACTIVIDAD DETECTADA ({streak} frames)" if level == "WARNING" \
               else f"🚨 INACTIVIDAD CRÍTICA ({streak} frames)"
    if module == "mixed":
        risk = ann.get("risk_level", "OK")
        hand = ann.get("hand_state", "")
        if risk == "CRITICAL":
            return f"🚨 RIESGO CRÍTICO: Operario activo sin protección"
        if risk == "WARNING":
            return f"⚠ ADVERTENCIA: {hand}"
        return ""
    if module == "metrology":
        verdict = ann.get("verdict", "")
        cls     = ann.get("detected_class", "")
        streak  = ann.get("nok_streak", 0)
        level   = ann.get("alert_level", "WARNING")
        if level == "CRITICAL":
            return f"🚨 PIEZA NO CONFORME — {cls} ({streak} frames)"
        return f"⚠ POSIBLE NOK — {cls}"
    if module == "procesado":
        pending = ann.get("pending_count", 0)
        if ann.get("backlog_active"):
            return f"⚠ ACUMULACIÓN — {pending} piezas sin procesar"
        total = ann.get("total_processed", 0)
        return f"⚠ PRODUCCIÓN PARADA — {total} piezas procesadas"
    return "⚠ ALERTA: SIN GUANTES"


def _apply_overrides(config: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(config.get(k), dict):
            _apply_overrides(config[k], v)
        else:
            config[k] = v


def _save_report(job_id: str, report: dict, out_cfg: dict) -> None:
    import json as _json
    report_dir = resolve_path(out_cfg.get("report_path", "outputs/reports/"))
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"{job_id}_report.json", "w", encoding="utf-8") as f:
        _json.dump(report, f, indent=2, ensure_ascii=False)
