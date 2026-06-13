"""FastAPI serving example for the PJM sign-language → English models (signsdk).

Quick start:
    uv venv && uv pip install "signsdk[gpu]"      # or signsdk[cpu] / plain signsdk
    cp .env.example .env                          # tune device / memory
    uv run uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /            health + default model
    GET  /models      list the 16 models (filter ?backend=unisign|spamo)
    POST /translate   upload a video, get the decoded text
                      form fields: model, num_beams, max_new_tokens

Memory: models are loaded via signsdk.load() — an LRU cache that frees the GPU when it
evicts. Each SpaMo model is ~11.5 GB / ~21 GB peak, so on a 24 GB card keep
SIGNSDK_MAX_RESIDENT=1 (see .env.example). If a model won't fit, signsdk falls back to CPU.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import signsdk
from signsdk import GenerationConfig, SignTranslatorError

app = FastAPI(title="PJM sign-language translation", version="1.0")


@app.get("/")
def health():
    return {"status": "ok", "default_model": signsdk.DEFAULT_MODEL}


@app.get("/models")
def models(backend: Optional[str] = None):
    """List available models. Filter with ?backend=unisign or ?backend=spamo."""
    return {"default": signsdk.DEFAULT_MODEL, "models": signsdk.describe_models(backend)}


@app.post("/translate")
async def translate(
    video: UploadFile = File(..., description="sign-language video clip (mp4)"),
    model: str = Form(signsdk.DEFAULT_MODEL),
    num_beams: int = Form(4),
    max_new_tokens: int = Form(64),
):
    """Translate one uploaded sign-language video to text."""
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        shutil.copyfileobj(video.file, tmp)
        tmp.close()
        gen = GenerationConfig(num_beams=num_beams, max_new_tokens=max_new_tokens)
        try:
            result = signsdk.load(model).predict(tmp.name, generation=gen)
        except SignTranslatorError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "text": result.text,
            "model": result.model,
            "lang": result.lang,
            "timings": result.timings,
        }
    finally:
        Path(tmp.name).unlink(missing_ok=True)
