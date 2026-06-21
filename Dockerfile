# syntax=docker/dockerfile:1
#
# Single inference image for the PJM models (signsdk + serving/app.py).
# The SAME image runs on CPU and GPU — signsdk autodetects the device at runtime
# (settings.py defaults to "cuda"; gpu.py probes nvidia-smi and falls back to CPU
# when there is no GPU / not enough VRAM, and ONNX Runtime falls back to the CPU
# provider when no CUDA provider is available). So nothing device-specific is baked in.
#
# Build context: the pjm-models repo (this directory IS the `signsdk` package).

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Model graphs + HF artifacts cache to a volume (see compose) so they survive
    # restarts and aren't re-downloaded from HuggingFace on every container start.
    SIGNSDK_HOME=/cache/signsdk \
    HF_HOME=/cache/huggingface

WORKDIR /app

# System libs: ffmpeg (video remux), libGL/glib (opencv + mediapipe runtime deps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install signsdk with the GPU + serving extras. The override drops the plain
# `onnxruntime` that rtmlib pins (it would clobber onnxruntime-gpu); the GPU wheel
# still provides the `import onnxruntime` namespace, so CPU also works. This is the
# mechanism documented in pjm-models/README.md.
RUN echo "onnxruntime ; sys_platform == 'never'" > /tmp/ort-override.txt \
    && uv venv /app/.venv \
    && VIRTUAL_ENV=/app/.venv uv pip install --override /tmp/ort-override.txt ".[gpu,serve]"

ENV PATH="/app/.venv/bin:$PATH"

# serving/app.py exposes `app` (FastAPI). signsdk auto-loads a .env from the cwd.
WORKDIR /app/serving
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
