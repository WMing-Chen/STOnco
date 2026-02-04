"""Utility modules for STOnco framework

This package contains utility functions and tools including:
- Data preparation and preprocessing
- Model evaluation and visualization
- Plotting and analysis tools

注意：这里使用惰性导入（lazy import），避免在 `python -m stonco.utils.xxx` 时
因为包级别提前导入同名模块而触发 runpy 的 RuntimeWarning。
"""

from __future__ import annotations

import importlib
from typing import Any


_SUBMODULES = {
    'utils',
    'prepare_data',
    'preprocessing',
    'evaluate_models',
    'visualize_prediction',
    'plot_accuracy_bars',
    'plot_loco_per_slide',
    'plot_roc',
    'extract_best_config',
    'export_spot_embeddings',
    'visualize_umap_tsne',
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = sorted(_SUBMODULES)
