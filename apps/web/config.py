"""Configuration helpers for the DeepSynth web application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class JobManagerConfig:
    """Runtime options for the background job manager."""

    state_dir: Path = Path(
        os.getenv("DEEPSYNTH_UI_STATE_DIR", _BASE_DIR / "state")
    )
    max_background_threads: int = int(os.getenv("DEEPSYNTH_UI_MAX_THREADS", "4"))
    progress_poll_interval: float = float(
        os.getenv("DEEPSYNTH_UI_PROGRESS_INTERVAL", "2.0")
    )


class WebConfig:
    """Default configuration for the Flask application."""

    SECRET_KEY: str = os.getenv(
        "DEEPSYNTH_WEB_SECRET_KEY",
        os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
    )
    JOB_MANAGER: JobManagerConfig = JobManagerConfig()


__all__ = ["JobManagerConfig", "WebConfig"]
