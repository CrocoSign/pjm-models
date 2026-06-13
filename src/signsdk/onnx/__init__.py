"""In-process ONNX runtimes for the SDK (repo-free inference).

  unisign_runtime.py -- UniSignRuntime: pose (rtmlib) -> projector -> mT5 encoder
                        -> beam decoder, all on onnxruntime.
  spamo_runtime.py   -- SpaMoRuntime (dual/quad): YOLO crop + CLIP s2 + VideoMAE
                        (+ hand-CLIP + MediaPipe) -> projector -> T5 + beam.

Model artifacts (ONNX graphs + assets) are pulled from HuggingFace on demand by
``signsdk.hf``. Glue helpers live in ``onnx/glue/``.
"""
