# signsdk

Repo-free **inference** SDK for PJM (Polish Sign Language → English) translation.
One call turns a raw `.mp4` of signing into text. Pure pip + ONNX Runtime — no model
repos, no PyTorch. Model graphs and assets are pulled from the HuggingFace repo
[`croco-corp/pjm-models`](https://huggingface.co/croco-corp/pjm-models) on first use.

Two model families (16 checkpoints):
- **Uni-Sign** — pose-based (RTMPose keypoints → mT5).
- **SpaMo / Mod-SpaMo** — CLIP + VideoMAE (+ hand-CLIP + MediaPipe for `mod-`) → Flan-T5-XL.

## Install

```bash
uv sync          # or: pip install .
```
Requires Python ≥ 3.10 (CPU `onnxruntime` by default).

**GPU** (CUDA 12): depend on the `gpu` extra **and** add a uv override so rtmlib's CPU
`onnxruntime` pin doesn't clobber the GPU wheel — in your *project's* `pyproject.toml`:
```toml
dependencies = ["signsdk[gpu]", ...]

[tool.uv]
override-dependencies = ["onnxruntime ; sys_platform == 'never'"]
```
`uv sync` then installs `onnxruntime-gpu` + the CUDA 12 / cuDNN 9 libs, and
`SIGNSDK_DEVICE=cuda` runs on GPU (`=cpu` to force CPU). Verified: a SpaMo translate
drops from minutes (CPU) to ~20 s incl. model load.

## Use

```python
import signsdk

# pulls the model from HF on first call, then translates
text = signsdk.from_pretrained("unisign-pjm-how2sign-ms").predict("clip.mp4").text
print(text)

signsdk.list_models()   # all 16 registered names
```

**Servers:** use `signsdk.load(name)` instead of caching handles yourself — it keeps an
LRU set of `SIGNSDK_MAX_RESIDENT` models resident (env, default 2) and **frees the GPU**
when evicting. Each SpaMo model is ~11.5 GB on GPU, so on a 24 GB card set
`SIGNSDK_MAX_RESIDENT=1` (or 2 if mixing one SpaMo + Uni-Sign).

### Config — `.env` is auto-loaded
Drop a `.env` in your working dir (see `.env.example`); `signsdk` reads it on import
(real env vars still win):
- `SIGNSDK_DEVICE=cuda|cpu` — runtime device (needs `onnxruntime-gpu` for `cuda`)
- `SIGNSDK_HOME` — model cache dir (default `~/.cache/signsdk`)
- `SIGNSDK_HF_REPO` — artifact repo (default `croco-corp/pjm-models`)

## Notes
- `from_pretrained(name)` raises a clear error if a model isn't published on HF yet.
- This is the **inference** package only; ONNX export / quantization / publishing lives in
  the separate `signsdk-build` repo.
