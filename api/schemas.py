"""Pydantic schemas for the EXPAI Inference API."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class InferenceStartRequest(BaseModel):
    module: str = Field(..., description="Module name, e.g. 'epp_gloves'")
    video_path: str = Field(..., description="Path to input video (relative to project root)")
    model_path: str = Field(..., description="Path to YOLO .pt model file")
    config_overrides: Optional[dict[str, Any]] = Field(default=None, description="Override module config keys")


class JobStatus(BaseModel):
    job_id: str
    module: str
    status: str   # pending | running | completed | failed
    progress: float = 0.0   # 0.0 - 1.0
    current_frame: int = 0
    total_frames: int = 0
    message: Optional[str] = None


class InferenceStopRequest(BaseModel):
    job_id: str


class ModuleInfo(BaseModel):
    name: str
    status: str   # active | inactive
    description: str
