"""Shared helpers for MORL_LA (DP behavioral policy, ...)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DPSolution",
    "GreedyPolicy",
    "action_optimality_frequency",
    "solve_la_dp",
    "value_iteration",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dp

        return getattr(dp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
