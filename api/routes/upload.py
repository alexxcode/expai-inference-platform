"""File upload endpoints for models and videos."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger

from core.config import BASE_DIR

router = APIRouter(prefix="/upload", tags=["upload"])

MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MODEL_EXTS = {".pt", ".onnx"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_MODEL_SIZE = 2 * 1024 ** 3   # 2 GB
MAX_VIDEO_SIZE = 10 * 1024 ** 3  # 10 GB


def _safe_filename(name: str) -> str:
    """Strip any path separators from the filename."""
    return Path(name).name


@router.post("/model", summary="Upload a YOLO .pt model file")
async def upload_model(file: UploadFile = File(...)):
    filename = _safe_filename(file.filename or "model.pt")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXTS:
        raise HTTPException(status_code=400, detail=f"Extension not allowed: {ext}. Use {ALLOWED_MODEL_EXTS}")

    dest = MODELS_DIR / filename
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            size += len(chunk)
            if size > MAX_MODEL_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Model file too large (max 2 GB)")
            f.write(chunk)

    logger.info(f"Model uploaded: {filename} ({size / 1024 / 1024:.1f} MB)")
    return {"path": f"models/{filename}", "filename": filename, "size_mb": round(size / 1024 / 1024, 2)}


@router.post("/video", summary="Upload a video file")
async def upload_video(file: UploadFile = File(...)):
    filename = _safe_filename(file.filename or "video.mp4")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"Extension not allowed: {ext}. Use {ALLOWED_VIDEO_EXTS}")

    dest = VIDEOS_DIR / filename
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video file too large (max 10 GB)")
            f.write(chunk)

    logger.info(f"Video uploaded: {filename} ({size / 1024 / 1024:.1f} MB)")
    return {"path": f"videos/{filename}", "filename": filename, "size_mb": round(size / 1024 / 1024, 2)}


@router.get("/models", summary="List available models")
async def list_models():
    files = sorted(MODELS_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.onnx"))
    return [{"filename": f.name, "path": f"models/{f.name}", "size_mb": round(f.stat().st_size / 1024 / 1024, 2)} for f in files]


@router.get("/videos", summary="List available videos")
async def list_videos():
    exts = ALLOWED_VIDEO_EXTS
    files = [f for f in sorted(VIDEOS_DIR.iterdir()) if f.suffix.lower() in exts]
    return [{"filename": f.name, "path": f"videos/{f.name}", "size_mb": round(f.stat().st_size / 1024 / 1024, 2)} for f in files]


@router.delete("/model/{filename}", summary="Delete a model file")
async def delete_model(filename: str):
    path = MODELS_DIR / _safe_filename(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    path.unlink()
    return {"deleted": filename}


@router.delete("/video/{filename}", summary="Delete a video file")
async def delete_video(filename: str):
    path = VIDEOS_DIR / _safe_filename(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    path.unlink()
    return {"deleted": filename}
