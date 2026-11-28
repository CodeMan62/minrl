"""Small helpers shared across minrl."""
from __future__ import annotations

import importlib
from typing import Any


def load_path(path: str) -> Any:
    """``"pkg.module.Name"`` -> the ``Name`` object, imported from ``pkg.module``."""
    module, _, name = path.rpartition(".")
    if not module:
        raise ValueError(f"{path!r} is not a dotted path like 'pkg.module.Name'")
    return getattr(importlib.import_module(module), name)
