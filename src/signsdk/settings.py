"""Settings — minimal runtime config for the repo-free ONNX inference path.

Only ``device`` is needed at inference time; model artifacts are resolved/downloaded
by :mod:`signsdk.hf` (env ``SIGNSDK_HF_REPO`` / ``SIGNSDK_HOME``). Override via
``SIGNSDK_DEVICE`` (``cuda`` | ``cpu``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    device: str = field(default_factory=lambda: os.environ.get("SIGNSDK_DEVICE", "cuda"))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()


#: process-wide default, built from the environment
DEFAULT = Settings.from_env()
