# Serving the PJM models (FastAPI)

A minimal FastAPI server over `signsdk` (this repository): upload a
sign-language video, get English text. Models are pulled from this HF repo on first use.

## Run

```bash
uv venv
uv pip install "signsdk[gpu,serve] @ git+https://github.com/crocosign/pjm-models"    # GPU + FastAPI; drop `gpu` for CPU-only
cp .env.example .env                   # optional: tune device + memory
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

**GPU note:** `rtmlib` pins CPU `onnxruntime`, so add this to your `pyproject.toml` to keep
the GPU wheel:
```toml
[tool.uv]
override-dependencies = ["onnxruntime ; sys_platform == 'never'"]
```

## Use

```bash
# list models
curl localhost:8000/models

# translate (default model = unisign-pjm-how2sign-ms)
curl -F video=@clip.mp4 -F model=spamo-pjm-phoenix-ms localhost:8000/translate
```

## Memory (`.env`)
Each SpaMo model is ~11.5 GB on GPU (~21 GB peak). On a 24 GB card set
`SIGNSDK_MAX_RESIDENT=1`: `signsdk.load()` then frees the resident model before loading a
new one. If the GPU is too full, it falls back to CPU instead of OOMing. See `.env.example`.
