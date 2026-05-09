"""Fast bulk tokenizer + token counter for OpenAI BPE encodings.

Wraps `toklab._native.Tokenizer` (PyO3 + tiktoken-rs) with a small Python
shim so callers get keyword-only `budget=` on `fits` and `truncate_to`,
matching the prevailing Python style for keyword arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from typing import Final

from toklab._native import (
    Tokenizer as _NativeTokenizer,
)
from toklab._native import (
    ToklabError,
)


def _read_version() -> str:
    try:
        return metadata.version("toklab")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: Final[str] = _read_version()

__all__ = [
    "Tokenizer",
    "ToklabError",
    "__version__",
]


class Tokenizer:
    """A tokenizer for one specific OpenAI BPE encoding."""

    def __init__(self, _inner: _NativeTokenizer) -> None:
        self._inner = _inner

    @classmethod
    def for_model(cls, model: str) -> Tokenizer:
        """Construct from an OpenAI model name, e.g. ``"gpt-4"`` or ``"gpt-4o"``."""
        return cls(_NativeTokenizer.for_model(model))

    @classmethod
    def for_encoding(cls, name: str) -> Tokenizer:
        """Construct from an encoding name (``"cl100k_base"`` or ``"o200k_base"``)."""
        return cls(_NativeTokenizer.for_encoding(name))

    @property
    def encoding_name(self) -> str:
        """The underlying encoding (``cl100k_base`` or ``o200k_base``)."""
        return self._inner.encoding_name

    def count(self, text: str) -> int:
        """Count BPE tokens in `text`."""
        return int(self._inner.count(text))

    def count_many(self, texts: Sequence[str], *, parallel: bool = False) -> list[int]:
        """Count tokens for each input. With ``parallel=True``, use rayon."""
        return list(self._inner.count_many(list(texts), parallel))

    def encode(self, text: str) -> list[int]:
        """Encode `text` to BPE token IDs."""
        return list(self._inner.encode(text))

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode BPE token IDs back to a string."""
        return str(self._inner.decode(list(tokens)))

    def fits(self, text: str, *, budget: int) -> bool:
        """`True` iff `text` encodes to ``<= budget`` BPE tokens."""
        return bool(self._inner.fits(text, budget=budget))

    def truncate_to(self, text: str, *, budget: int) -> str:
        """Return `text` truncated to its first ``budget`` BPE tokens."""
        return str(self._inner.truncate_to(text, budget=budget))

    def __repr__(self) -> str:
        return repr(self._inner)
