"""FastAPI application entry point for EXPAI Inference Platform."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.routes import inference, inspection, modules, results, upload
from core.config import global_config, BASE_DIR

app = FastAPI(
    title="EXPAI Inference Platform",
    description="Real-time YOLO inference platform for industrial use cases (EUREKA 21028)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router)
app.include_router(inspection.router)
app.include_router(modules.router)
app.include_router(results.router)
app.include_router(upload.router)

# Serve UI static files
UI_DIR = BASE_DIR / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


@app.get("/", include_in_schema=False)
async def root():
    index = UI_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "docs": "/docs"}


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "env": global_config.env}


@app.on_event("startup")
async def startup():
    logger.info(f"EXPAI Inference API starting | env={global_config.env} | port={global_config.api_port}")
