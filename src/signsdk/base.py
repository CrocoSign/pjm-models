"""Core abstractions: the SignTranslator interface.

Patterns at work here:
  * Strategy / Template Method -- ``SignTranslator`` fixes the public pipeline
    (validate -> run -> wrap into TranslationResult); concrete backends supply
    only the ``_infer`` step.
  * Adapter (in subclasses) -- each backend adapts a model's bespoke
    load/preprocess/generate code to this one interface.

Inference backends run the ONNX runtimes in-process and pull model artifacts
from HuggingFace on demand. The SDK core therefore needs no heavy dependencies.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .config import GenerationConfig
from .result import TranslationResult

PathLike = Union[str, "os.PathLike[str]"]


class SignTranslatorError(RuntimeError):
    pass


class SignTranslator(ABC):
    """Common interface for every sign-language translation backend."""

    #: registry name of the loaded model, e.g. "unisign-pjm-si"
    name: str
    #: output language of the loaded checkpoint
    lang: Optional[str] = None

    def __init__(self, name: str, lang: Optional[str] = None):
        self.name = name
        self.lang = lang

    # -- public API (Template Method) -------------------------------------
    def predict(
        self,
        video: PathLike,
        *,
        generation: Optional[GenerationConfig] = None,
    ) -> TranslationResult:
        """Translate a single sign-language video file to text."""
        return self.predict_batch([video], generation=generation)[0]

    def predict_batch(
        self,
        videos: Sequence[PathLike],
        *,
        generation: Optional[GenerationConfig] = None,
    ) -> List[TranslationResult]:
        """Translate several videos with a single model load (efficient batch)."""
        gen = generation or GenerationConfig()
        paths = [self._validate_input(v) for v in videos]
        t0 = time.time()
        raw = self._infer(paths, gen)  # backend-specific (Strategy)
        elapsed = time.time() - t0
        results = []
        for item in raw:
            results.append(
                TranslationResult(
                    text=item["text"],
                    model=self.name,
                    lang=self.lang,
                    key=item.get("key"),
                    timings={"total_s": round(elapsed, 3), **item.get("timings", {})},
                )
            )
        return results

    # -- to be implemented by concrete backends --------------------------
    @abstractmethod
    def _infer(self, video_paths: List[str], gen: GenerationConfig) -> List[dict]:
        """Run the model on validated input paths; return [{text, key?, timings?}]."""

    # -- resource management ---------------------------------------------
    def release(self) -> None:
        """Drop any loaded model/runtime, freeing its GPU/CPU memory. The next
        ``predict`` lazily reloads. Safe to call repeatedly; no-op if not loaded."""

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _validate_input(video: PathLike) -> str:
        p = Path(video).expanduser()
        if not p.exists():
            raise SignTranslatorError(f"input video not found: {p}")
        return str(p.resolve())
