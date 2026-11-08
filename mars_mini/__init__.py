"""Mars research package.

Data-generation and scoring utilities have no PyTorch dependency.
The original Transformer model is loaded only when requested.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MarsConfig, MarsMini

__all__ = ["MarsConfig", "MarsMini"]


def __getattr__(name: str):
    if name in __all__:
        from .model import MarsConfig, MarsMini
        return {"MarsConfig": MarsConfig, "MarsMini": MarsMini}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
