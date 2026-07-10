"""Strategy registry for the live trading engine."""
from __future__ import annotations

from types import ModuleType

from . import dm42, r50d

STRATEGIES: dict[str, ModuleType] = {
    r50d.NAME: r50d,
    dm42.NAME: dm42,
}


def get_strategy(name: str) -> ModuleType:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown strategy {name!r}; known: {known}") from exc


def list_strategies() -> list[str]:
    return sorted(STRATEGIES)
