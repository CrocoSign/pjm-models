"""Result value object returned by every backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranslationResult:
    """The decoded translation plus provenance/metadata.

    ``lang`` is a property of the loaded checkpoint (Uni-Sign PJM->English,
    SpaMo Phoenix->German, etc.), never hard-coded by the SDK.
    """

    text: str
    model: str
    lang: Optional[str] = None
    key: Optional[str] = None
    timings: dict = field(default_factory=dict)

    def __str__(self) -> str:  # so print(result) shows the translation
        return self.text
