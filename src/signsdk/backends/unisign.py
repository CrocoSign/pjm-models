from __future__ import annotations
import os
from typing import List
from ..base import SignTranslator
from ..config import GenerationConfig
from ..registry import ModelSpec
from ..settings import Settings, DEFAULT
from ..hf import resolve_unisign

class OnnxUniSignBackend(SignTranslator):
    def __init__(self, spec: ModelSpec, *, settings: Settings = DEFAULT, **_):
        super().__init__(name=spec.name, lang=spec.lang)
        self.spec = spec; self.s = settings; self._rt = None
    def _runtime(self):
        if self._rt is None:
            from ..onnx.unisign_runtime import UniSignRuntime
            from ..gpu import pick_device
            need = int(os.environ.get("SIGNSDK_UNISIGN_GPU_MIB", "5000"))
            dev = pick_device(self.s.device, need)
            if dev != self.s.device:
                print(f"[signsdk] GPU free < {need} MiB; running {self.spec.name} on CPU", flush=True)
            p = resolve_unisign(self.spec.name)
            self._rt = UniSignRuntime(p["model_dir"], device=dev,
                assets=p["assets"], max_length=int(self.spec.extra.get("max_length", 256)))
        return self._rt
    def _infer(self, video_paths: List[str], gen: GenerationConfig) -> List[dict]:
        rt = self._runtime()
        return [{"text": rt.translate(v, num_beams=int(gen.num_beams),
                 max_new_tokens=int(gen.max_new_tokens), seed=int(gen.seed))}
                for v in video_paths]
    def release(self):
        if self._rt is not None:
            try:
                self._rt.close()
            except Exception:
                pass
        self._rt = None
