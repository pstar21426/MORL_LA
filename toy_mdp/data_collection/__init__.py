"""Offline dataset collection for Appendix B LA."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CollectConfig",
    "collect_dataset",
    "collect_epsilon_grid",
    "collect_episode",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import collect

        return getattr(collect, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
