"""Model registry + factory (ONNX inference only).

``from_pretrained(name)`` looks up a :class:`ModelSpec` and constructs the in-process
ONNX backend. Artifacts are pulled from HF on demand by :mod:`signsdk.hf`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import SignTranslator, SignTranslatorError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str          # "unisign" | "spamo"
    lang: Optional[str] = None
    description: str = ""
    extra: dict = field(default_factory=dict)   # kind (spamo), bleu4, source, split, max_length


REGISTRY: Dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    REGISTRY[spec.name] = spec


def list_models() -> List[str]:
    return sorted(REGISTRY)


def from_pretrained(name: str, *, settings=None, **overrides) -> SignTranslator:
    """Construct the ONNX backend for a model. ``settings`` defaults to the env default."""
    if name not in REGISTRY:
        raise SignTranslatorError(
            f"unknown model {name!r}. Known: {', '.join(list_models()) or '(none)'}")
    from .settings import DEFAULT
    settings = settings or DEFAULT
    spec = REGISTRY[name]
    engine = overrides.pop("engine", "onnx")
    if engine != "onnx":
        raise SignTranslatorError("inference package supports engine='onnx' only")
    if spec.backend == "unisign":
        from .backends.unisign import OnnxUniSignBackend
        return OnnxUniSignBackend(spec, settings=settings, **overrides)
    if spec.backend == "spamo":
        from .backends.spamo import OnnxSpaMoBackend
        return OnnxSpaMoBackend(spec, settings=settings, **overrides)
    raise SignTranslatorError(f"no backend registered for {spec.backend!r}")


# (name-suffix, source-label, split, bleu4) -- all PJM->English, pose-based.
_UNISIGN = [
    ("cslnews-ms",  "CSL-News", "ms", 5.16), ("cslnews-si",  "CSL-News", "si", 2.41),
    ("how2sign-ms", "How2Sign", "ms", 8.61), ("how2sign-si", "How2Sign", "si", 5.91),
    ("openasl-ms",  "OpenASL",  "ms", 8.53), ("openasl-si",  "OpenASL",  "si", 6.09),
    ("scratch-ms",  "scratch",  "ms", 1.82), ("scratch-si",  "scratch",  "si", 0.85),
]

# SpaMo=dual (CLIP ViT-L/14 + VideoMAE-large); Mod-SpaMo=quad (+ hand-ViT + MediaPipe).
# (name, kind, source-label, split, bleu4)
_SPAMO = [
    ("spamo-pjm-phoenix-ms",     "dual", "PHOENIX", "ms", 5.54),
    ("spamo-pjm-scratch-ms",     "dual", "scratch", "ms", 2.03),
    ("spamo-pjm-scratch-si",     "dual", "scratch", "si", 0.93),
    ("mod-spamo-pjm-phoenix-ms", "quad", "PHOENIX", "ms", 5.06),
    ("mod-spamo-pjm-scratch-ms", "quad", "scratch", "ms", 1.56),
    ("mod-spamo-pjm-scratch-si", "quad", "scratch", "si", 1.17),
]


def _register_defaults() -> None:
    for suffix, source, split, bleu4 in _UNISIGN:
        register(ModelSpec(
            name=f"unisign-pjm-{suffix}", backend="unisign", lang="English",
            description=f"Uni-Sign pose-based PJM->English, {source} pretrain, "
                        f"{split} split (BLEU4 {bleu4}).",
            extra={"split": split, "source": source, "bleu4": bleu4}))
    for name, kind, source, split, bleu4 in _SPAMO:
        label = "Mod-SpaMo" if kind == "quad" else "SpaMo"
        streams = "CLIP ViT-L/14 + VideoMAE-large" + (" + hand-ViT + MediaPipe" if kind == "quad" else "")
        register(ModelSpec(
            name=name, backend="spamo", lang="English",
            description=f"{label} ({streams}) -> FlanT5-XL, PJM->English, "
                        f"{source} pretrain, {split} split (BLEU4 {bleu4}).",
            extra={"kind": kind, "source": source, "split": split, "bleu4": bleu4}))


_register_defaults()
