"""Results and report retrieval endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/results", tags=["results"])


def _get_job_or_404(job_id: str) -> dict:
    from api.routes.inference import _jobs
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@router.get("/{job_id}", summary="Get final report for a completed job")
async def get_results(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Job is not yet completed")
    return job.get("report", {})


@router.get("/{job_id}/video", summary="Download the annotated video")
async def get_annotated_video(job_id: str):
    job = _get_job_or_404(job_id)
    video_path = job.get("annotated_video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Annotated video not available")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}_annotated.mp4")


@router.get("/{job_id}/incidents", summary="List detected incidents")
async def get_incidents(job_id: str):
    job = _get_job_or_404(job_id)
    report = job.get("report") or {}
    analysis = report.get("analysis", {})
    return {
        "job_id": job_id,
        "incident_count": analysis.get("incident_count", 0),
        "incidents": analysis.get("incidents", []),
    }
