"""Core modules for STOnco framework.

Lazy imports keep `python -m stonco.core.xxx` and utility scripts from eagerly
loading training and inference stacks that are unrelated to the requested
submodule.
"""

from __future__ import annotations

import importlib
from typing import Any


_SUBMODULES = {
    'models',
    'train',
    'train_hpo',
    'infer',
    'batch_infer',
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = sorted(_SUBMODULES)
