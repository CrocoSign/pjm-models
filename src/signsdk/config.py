"""Generation configuration — an immutable value object.

Rather than scattering beam/token/sampling kwargs across the call sites, the SDK
assembles them into one frozen dataclass that is passed down to whichever backend
runs. (Lightweight "Builder"/parameter-object pattern.)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters shared by every backend.

    Defaults are deterministic on purpose: beam search, no sampling, fixed seed.
    SpaMo's repo defaults to ``do_sample=True`` (non-reproducible); the SDK
    overrides that to make outputs stable and comparable. Opt back into sampling
    explicitly if you want it.
    """

    max_new_tokens: int = 100
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
