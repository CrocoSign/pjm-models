# pjm-models (signsdk)

Inference package for CrocoSign, our research on translating Polish Sign Language (PJM) video into English text. It wraps 16 trained checkpoints (Uni-Sign and SpaMo families) behind one Python call, runs them with ONNX Runtime (the package does not depend on PyTorch, see `pyproject.toml`) and downloads model graphs from [croco-corp/pjm-models](https://huggingface.co/croco-corp/pjm-models) on first use. A small FastAPI server is included.

This is a research prototype, not a replacement for an interpreter. Project site: https://crocosign.github.io

## Where this fits

All repositories are under [github.com/crocosign](https://github.com/crocosign):

CrocoSign (data preparation) -> SpaMo-PJM / Uni-Sign-PJM (training) -> **pjm-models** (inference) -> croco-sign-backend / croco-sign-frontend (app).

The models were trained on clips from the Open Repository of the PJM Corpus (University of Warsaw), prepared by us. [croco-corp/pjm-data](https://huggingface.co/datasets/croco-corp/pjm-data) publishes the features, splits, translations and videos of the filtered set; [croco-corp/pjm-segments](https://huggingface.co/datasets/croco-corp/pjm-segments) holds the unfiltered segments as WebDataset shards. The corpus is licensed for non-commercial use only, so treat the models the same way.

## Requirements

- Python 3.10 or newer, [uv](https://docs.astral.sh/uv/)
- Disk: the Uni-Sign example below downloads about 3.5 GB including dependencies
- Optional GPU (CUDA 12). A SpaMo model takes about 11.5 GB of GPU memory once loaded and about 21 GB at peak, so it needs a 24 GB card

## Quick start

`signsdk` is not on PyPI; install it from this repository. From a clean clone (put your own PJM recording in place of `clip.mp4`):

```bash
git clone https://github.com/crocosign/pjm-models
cd pjm-models
uv sync
uv run python -c "import signsdk; print(signsdk.from_pretrained('unisign-pjm-openasl-ms').predict('clip.mp4').text)"
```

## Python API

```python
import signsdk

signsdk.list_models()                  # the 16 registered names
signsdk.describe_models("unisign")     # name, backend, language, description, split, source, bleu4

model = signsdk.from_pretrained("unisign-pjm-openasl-ms")
result = model.predict("clip.mp4")     # TranslationResult
print(result.text, result.timings)
model.predict_batch(["a.mp4", "b.mp4"])  # one model load

signsdk.translate("clip.mp4")          # one-shot, default model unisign-pjm-how2sign-ms
```

Decoding is deterministic by default (beam search, no sampling). Override with `signsdk.GenerationConfig(num_beams=..., max_new_tokens=...)` passed as `generation=`. Defaults in the SDK are 4 beams and 100 new tokens.

In a long-running process use `signsdk.load(name)` instead of keeping handles yourself. It keeps an LRU set of resident models and frees GPU memory on eviction.

## Models

Names follow `<family>-pjm-<transfer source>-<split>`. All output English.

- `unisign-pjm-{cslnews,how2sign,openasl,scratch}-{ms,si}`: pose-based. RTMPose keypoints, graph encoders, mT5.
- `spamo-pjm-{phoenix,scratch}-{ms,si}`: CLIP ViT-L/14 + VideoMAE-large, Flan-T5-XL.
- `mod-spamo-pjm-{phoenix,scratch}-{ms,si}`: our SpaMo variant, adds hand-crop ViT and MediaPipe keypoint streams.

"Transfer source" is the sign language the model was trained on before PJM. "scratch" means trained from scratch.

### Test results (BLEU-4)

Single run per cell, test set. The numbers come from our training repositories ([SpaMo-PJM](https://github.com/crocosign/SpaMo-PJM), [Uni-Sign-PJM](https://github.com/crocosign/Uni-Sign-PJM)) at beam 5; the SDK defaults differ (4 beams, 100 new tokens), so results from this package can differ slightly. **MS** = multi-signer: test signers also appear in training. **SI** = signer-independent: test signers are never seen in training.

| Model | Transfer source | MS | SI |
| --- | --- | --- | --- |
| Uni-Sign | How2Sign (ASL) | 7.61 | 4.77 |
| Uni-Sign | OpenASL (ASL) | 7.38 | 4.79 |
| Uni-Sign | CSL-News (Chinese) | 2.42 | 1.43 |
| Uni-Sign | scratch | 1.62 | 0.72 |
| SpaMo | PHOENIX-2014T (German) | 5.54 | 0.53 |
| SpaMo | scratch | 2.03 | 0.93 |
| Mod-SpaMo | PHOENIX-2014T | 5.06 | 0.47 |
| Mod-SpaMo | scratch | 1.56 | 1.17 |

On BLEURT (the metric we trust most) Uni-Sign with OpenASL transfer is best: 39.9 on MS, 35.7 on SI. Reference sentences are DeepL translations of the Polish transcripts, so scores are against machine-translated references. Full tables and discussion are in our thesis (PJATK, in Polish).

### Speed

Measured on a 6.6 s clip, beam 4, 64 new tokens, model already loaded, RTX 3090 and Ryzen 9 9900X:

| Family | GPU | CPU |
| --- | --- | --- |
| Uni-Sign | about 1.6 s | about 11 s |
| SpaMo | about 13 s | about 110 s |

SpaMo is limited to 512 frames, so longer clips are cut.

## GPU

To use `signsdk` with the GPU from your own project, depend on the repository with the `gpu` extra and add a uv override to your `pyproject.toml`, so rtmlib's CPU `onnxruntime` pin does not replace the GPU wheel:

```toml
dependencies = ["signsdk[gpu] @ git+https://github.com/crocosign/pjm-models"]

[tool.uv]
override-dependencies = ["onnxruntime ; sys_platform == 'never'"]
```

Then set `SIGNSDK_DEVICE=cuda` (or `cpu` to force CPU).

## Configuration

Environment variables (a `.env` in the working directory is loaded on import, real environment variables win; see `.env.example`):

- `SIGNSDK_DEVICE`: `cuda` or `cpu` (default `cuda`; falls back to CPU when no GPU or not enough free memory)
- `SIGNSDK_HOME`: model cache directory (default `~/.cache/signsdk`)
- `SIGNSDK_HF_REPO`: artifact repository (default `croco-corp/pjm-models`)
- `SIGNSDK_MAX_RESIDENT`: models kept loaded by `signsdk.load` (default 2; use 1 on a 24 GB card with SpaMo)

## Serving

`serving/app.py` is a FastAPI app over the SDK. It has no CLI; the endpoints are `GET /` (health), `GET /models` and `POST /translate` (multipart `video`, optional `model`, `num_beams`, `max_new_tokens`; defaults 4 and 64).

```bash
uv sync --extra serve
cd serving
uv run uvicorn app:app --host 0.0.0.0 --port 8000

curl localhost:8000/models
curl -F video=@clip.mp4 -F model=spamo-pjm-phoenix-ms localhost:8000/translate
```

The `Dockerfile` builds one image (GPU and serve extras) that runs on both GPU and CPU. [croco-sign-backend](https://github.com/crocosign/croco-sign-backend) calls this service through `CROCO_INFERENCE_ENDPOINT`.

## Caveats

- Results are for research. On signers never seen in training (SI) every model does poorly; the best reaches 4.79 BLEU-4.
- Most of the quality comes from the starting checkpoint, not the architecture. The best models started from American Sign Language checkpoints.
- This repository is inference only. ONNX export, quantisation and publishing to Hugging Face are done in a separate build repository (`signsdk-build`), which is private. The ONNX export code is not public yet, so the ONNX files cannot be reproduced from public sources.

## License

Our code is under the MIT licence (see `LICENSE`). The training data (PJM Corpus, University of Warsaw) is non-commercial only. The models build on Uni-Sign and SpaMo; their upstream terms apply to those parts.

## Authors

Wojciech Sieńko and Rostyslav Kovalenko, PJATK engineering thesis (field: Artificial Intelligence), supervised by dr Albert Śledzianowski, MBA. We build on [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) (Li et al., ICLR 2025) and [SpaMo](https://github.com/eddie-euijun-hwang/SpaMo) (Hwang et al., NAACL 2025).
