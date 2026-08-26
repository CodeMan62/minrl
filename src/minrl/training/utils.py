from __future__ import annotations
from typing import Any, Callable, Dict
from minrl.types import Batch

# name -> update(model, optimizer, data, **hparams) -> metrics
UPDATES: Dict[str, Callable[..., Dict[str, float]]] = {}

def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """``@register("grpo")`` on an update fn → ``Algorithm("grpo")`` finds it."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        UPDATES[name] = fn
        return fn

    return decorator


def get_update(name: str) -> Callable[..., Dict[str, float]]:
    try:
        return UPDATES[name]
    except KeyError as e:
        known = ", ".join(sorted(UPDATES)) or "(none)"
        raise KeyError(
            f"unknown algorithm {name!r}; registered: {known}"
        ) from e

def batch_data(batch: Any) -> Any:
    """``Batch`` → rollouts or examples; leave plain lists unchanged."""
    if isinstance(batch, Batch):
        if batch.rollouts is not None:
            return batch.rollouts
        if batch.examples is not None:
            return batch.examples
    return batch
