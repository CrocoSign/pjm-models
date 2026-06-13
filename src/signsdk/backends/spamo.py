from __future__ import annotations
import os
from ..base import SignTranslator
from ..config import GenerationConfig
from ..registry import ModelSpec
from ..settings import Settings, DEFAULT
from ..hf import resolve_spamo

class OnnxSpaMoBackend(SignTranslator):
    def __init__(self, spec: ModelSpec, *, settings: Settings = DEFAULT, **_):
        super().__init__(name=spec.name, lang=spec.lang)
        self.spec = spec; self.s = settings
        self.kind = spec.extra.get("kind", "dual"); self._rt = None
    def _runtime(self):
        if self._rt is None:
            from ..onnx.spamo_runtime import SpaMoRuntime
            from ..gpu import pick_device
            need = int(os.environ.get("SIGNSDK_SPAMO_GPU_MIB", "22000"))
            dev = pick_device(self.s.device, need)
            if dev != self.s.device:
                print(f"[signsdk] GPU free < {need} MiB; running {self.spec.name} on CPU "
                      f"(set SIGNSDK_SPAMO_GPU_MIB to tune)", flush=True)
            p = resolve_spamo(self.spec.name, self.kind)
            self._rt = SpaMoRuntime(p["model_dir"], shared=p["shared"],
                assets=p["assets"], device=dev)
        return self._rt
    def _infer(self, video_paths, gen: GenerationConfig):
        rt = self._runtime()
        return [{"text": rt.translate(v, num_beams=int(gen.num_beams),
                 max_new_tokens=int(gen.max_new_tokens))} for v in video_paths]
    def release(self):
        if self._rt is not None:
            try:
                self._rt.close()
            except Exception:
                pass
        self._rt = None
