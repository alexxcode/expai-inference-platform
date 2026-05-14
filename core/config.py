"""Global configuration loader for EXPAI Inference Platform."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from loguru import logger


BASE_DIR = Path(__file__).resolve().parent.parent


class GlobalConfig(BaseModel):
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=BASE_DIR)
    cuda_visible_devices: str = Field(default="0")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)


def load_global_config() -> GlobalConfig:
    """Load global config from environment variables."""
    return GlobalConfig(
        env=os.getenv("EXPAI_ENV", "development"),
        log_level=os.getenv("EXPAI_LOG_LEVEL", "INFO"),
        data_dir=Path(os.getenv("EXPAI_DATA_DIR", str(BASE_DIR))),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_workers=int(os.getenv("API_WORKERS", "1")),
    )


def load_module_config(config_path: str | Path) -> dict[str, Any]:
    """Load a module YAML config file and return it as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Module config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded module config from {config_path}")
    return config


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """Resolve a path relative to BASE_DIR if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or BASE_DIR) / p


# Singleton
global_config: GlobalConfig = load_global_config()
