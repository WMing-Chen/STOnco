"""STRIDE: Spatial Transcriptomics Tumor/Non-tumor Classification

A PyTorch Geometric-based framework for spatial transcriptomics analysis
with dual-domain adversarial learning capabilities.
"""

__version__ = "1.0.0"
__author__ = "STRIDE Team"

# Core modules
from .core import models, train, infer, batch_infer, train_hpo
from .utils import utils, prepare_data, preprocessing, evaluate_models, visualize_prediction

__all__ = [
    "models",
    "train", 
    "infer",
    "batch_infer",
    "train_hpo",
    "utils",
    "prepare_data",
    "preprocessing", 
    "evaluate_models",
    "visualize_prediction"
]