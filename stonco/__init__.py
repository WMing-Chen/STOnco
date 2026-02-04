"""STOnco: Spatial Transcriptomics Oncology Analysis

A PyTorch Geometric-based framework for spatial transcriptomics analysis
with dual-domain adversarial learning capabilities.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "STOnco Team"

import importlib
from typing import Any


_EXPORTS: dict[str, str] = {
    'models': 'stonco.core.models',
    'train': 'stonco.core.train',
    'infer': 'stonco.core.infer',
    'batch_infer': 'stonco.core.batch_infer',
    'train_hpo': 'stonco.core.train_hpo',
    'utils': 'stonco.utils.utils',
    'prepare_data': 'stonco.utils.prepare_data',
    'preprocessing': 'stonco.utils.preprocessing',
    'evaluate_models': 'stonco.utils.evaluate_models',
    'visualize_prediction': 'stonco.utils.visualize_prediction',
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return importlib.import_module(target)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
