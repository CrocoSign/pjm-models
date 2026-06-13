"""signsdk -- unified inference SDK for PJM sign-language translation models.

Facade over heterogeneous research backends (Uni-Sign, SpaMo). One API:

    >>> import signsdk
    >>> signsdk.translate("clip.mp4")                      # default model
    >>> m = signsdk.from_pretrained("unisign-pjm-si")
    >>> m.predict("clip.mp4").text
    >>> m.predict_batch(["a.mp4", "b.mp4"])                # one model load

Backends run the ONNX runtimes in-process and pull model artifacts from
HuggingFace on demand, so the package is repo-free and deployable.
"""
from __future__ import annotations

# Auto-load a .env (SIGNSDK_DEVICE, SIGNSDK_HOME, SIGNSDK_HF_REPO, ...) before any
# Settings is built. override=False so real environment variables still win.
from dotenv import load_dotenv

load_dotenv(override=False)

import gc
import os
from collections import OrderedDict
from typing import Optional

from .config import GenerationConfig
from .result import TranslationResult
from .base import SignTranslator, SignTranslatorError
from .registry import ModelSpec, from_pretrained, list_models, register

__all__ = [
    "translate",
    "load",
    "from_pretrained",
    "list_models",
    "describe_models",
    "register",
    "GenerationConfig",
    "TranslationResult",
    "SignTranslator",
    "ModelSpec",
    "SignTranslatorError",
]


#: LRU set of resident models, bounded by SIGNSDK_MAX_RESIDENT
_RESIDENT: "OrderedDict[tuple, SignTranslator]" = OrderedDict()


def load(name: str, **overrides) -> SignTranslator:
    """Like :func:`from_pretrained` but with an LRU resident cache, so GPU/CPU memory
    stays bounded. Keeps at most ``SIGNSDK_MAX_RESIDENT`` models loaded (env var,
    default 2); evicting one calls ``.release()`` to free its memory (each SpaMo model
    is ~11.5 GB on the GPU). Use this in a server instead of caching handles yourself.
    """
    key = (name, tuple(sorted(overrides.items())))
    if key in _RESIDENT:
        _RESIDENT.move_to_end(key)
        return _RESIDENT[key]
    model = from_pretrained(name, **overrides)
    _RESIDENT[key] = model
    cap = max(1, int(os.environ.get("SIGNSDK_MAX_RESIDENT", "2")))
    while len(_RESIDENT) > cap:
        _, victim = _RESIDENT.popitem(last=False)
        victim.release()
    gc.collect()
    return model


def describe_models(backend: Optional[str] = None):
    """Return the registry menu as a list of dicts (name, backend, lang, ...).

    Filter with ``backend="unisign"`` or ``backend="spamo"``.
    """
    from .registry import REGISTRY
    out = []
    for name in list_models():
        s = REGISTRY[name]
        if backend and s.backend != backend:
            continue
        out.append({
            "name": s.name,
            "backend": s.backend,
            "lang": s.lang,
            "description": s.description,
            **s.extra,
        })
    return out

DEFAULT_MODEL = "unisign-pjm-how2sign-ms"  # best reported BLEU4 (7.61)


def translate(
    video,
    model: str = DEFAULT_MODEL,
    *,
    generation: Optional[GenerationConfig] = None,
    **gen_kwargs,
) -> TranslationResult:
    """One-shot convenience: load `model` and translate a single `video`.

    Extra keyword args (e.g. ``num_beams=5``) build a GenerationConfig.
    For repeated calls, prefer ``from_pretrained(model)`` to reuse the handle.
    """
    if generation is None and gen_kwargs:
        generation = GenerationConfig(**gen_kwargs)
    return from_pretrained(model).predict(video, generation=generation)
