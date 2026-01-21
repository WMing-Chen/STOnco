# STOnco: Spatial Transcriptomics Tumor Region Identification with Dual-Domain Adversarial Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

STOnco is a PyTorch Geometric-based framework for tumor/non-tumor binary classification in 10x Visium spatial transcriptomics data. It features dual-domain adversarial learning to improve model generalization across different cancer types and experimental batches.

<img width="1468" height="406" alt="image" src="https://github.com/user-attachments/assets/4b2f0acf-65d9-4a81-b52d-dc33e0460e36" />


## Key Features

- **Dual-Domain Adversarial Learning**: Reduces dependency on cancer-specific signals and batch effects
- **Multiple GNN Architectures**: Support for GATv2, GCN, and GraphSAGE models
- **Complete Pipeline**: From data preparation to training, inference, and visualization
- **Hyperparameter Optimization**: Built-in HPO with multi-stage pipeline
- **Batch Processing**: Efficient inference on multiple samples
- **Rich Visualization**: Comprehensive plotting and analysis tools

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install STOnco

```bash
pip install -e .
```

## Quick Start

### 1. Data Preparation

```bash
python -m stonco.utils.prepare_data build-train-npz \
    --train_dir /path/to/visium/data/ST_train_datasets \
    --out_npz ./processed_data/train_data.npz \
    --xy_cols row col \
    --label_col true_label

python -m stonco.utils.prepare_data build-val-npz \
    --val_dir /path/to/visium/data/ST_validation_datasets \
    --out_dir ./processed_data/val_npz \
    --xy_cols row col \
    --label_col true_label
```

### 2. Model Training

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts \
    --model gatv2
```

Training outputs include `loss_components.csv`, `train_loss.svg`, and `train_val_metrics.svg` in `--artifacts_dir`. Use `--val_sample_dir` to include external validation NPZs in validation metrics.
Disable early stopping:

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts \
    --model gatv2 \
    --early_patience 0
```

### 3. Inference

```bash
# Single sample inference
python -m stonco.core.infer \
    --npz sample.npz \
    --artifacts_dir ./artifacts \
    --out_csv predictions.csv

# Batch inference
python -m stonco.core.batch_infer \
    --npz_glob "./test_samples/*.npz" \
    --artifacts_dir ./artifacts \
    --out_csv ./predictions/batch_predictions.csv
```

### 4. Visualization

```bash
python -m stonco.utils.visualize_prediction \
    --npz sample.npz \
    --artifacts_dir ./artifacts \
    --out_svg visualization.svg
```

## Model Architecture

STOnco employs a unified `STOnco_Classifier` architecture with:

- **GNN Backbone**: Configurable graph neural network (GATv2/GCN/GraphSAGE)
- **Task Head**: Binary classification for tumor/non-tumor prediction
- **Domain Heads**: Optional adversarial heads for cancer type and batch-level domain adaptation

### Dual-Domain Adversarial Learning

```
Total_Loss = Task_Loss + λ₁ × Cancer_Domain_Loss + λ₂ × Batch_Domain_Loss
```

- **Cancer Domain Adversarial**: Reduces cancer-type-specific bias
- **Batch Domain Adversarial**: Mitigates batch effects between slides

## Project Structure

```
STOnco/
├── stonco/
│   ├── core/                 # Core training and inference modules
│   │   ├── models.py         # Model architectures
│   │   ├── train.py          # Training script
│   │   ├── train_hpo.py      # Hyperparameter optimization
│   │   ├── infer.py          # Single sample inference
│   │   └── batch_infer.py    # Batch inference
│   └── utils/                # Utility functions
│       ├── prepare_data.py   # Data preprocessing
│       ├── evaluate_models.py # Model evaluation
│       ├── visualize_prediction.py # Visualization
│       └── ...
├── examples/                 # Example scripts and tutorials
├── synthetic_test/           # Synthetic data for testing
├── docs/                     # Documentation
└── requirements.txt
```

## Advanced Usage

### Hyperparameter Optimization

```bash
python -m stonco.core.train_hpo \
    --train_npz train_data.npz \
    --artifacts_dir ./hpo_results \
    --tune all \
    --n_trials 100
```

### Cross-Cancer Evaluation (LOCO)

```bash
python -m stonco.core.train \
    --train_npz train_data.npz \
    --artifacts_dir ./loco_results \
    --leave_one_cancer_out
```

### K-Fold Batch Inference

```bash
# Run batch inference across all kfold outputs (fold_1..fold_10)
for i in {1..10}; do
  python -m stonco.core.batch_infer \
    --npz_glob './processed_data/val_npz/*.npz' \
    --artifacts_dir "./kfold_val/fold_${i}/" \
    --out_csv "./kfold_val/fold_${i}/batch_preds.csv" \
    --num_threads 4 --num_workers 0
done
```

### Model Evaluation

```bash
python -m stonco.utils.evaluate_models \
    --model GATv2=./predictions/batch_predictions.csv \
    --out_dir ./evaluation \
    --plot
```

## Input Data Format

STOnco expects NPZ files with the following structure:

**Training NPZ:**
- `features`: Gene expression matrix (n_spots × n_genes)
- `labels`: Binary labels (n_spots,)
- `coordinates`: Spatial coordinates (n_spots × 2)
- `slide_ids`: Slide identifiers (n_spots,)
- `cancer_types`: Cancer type labels (n_spots,)

**Single Sample NPZ:**
- `features`: Gene expression matrix
- `coordinates`: Spatial coordinates
- `slide_id`: Single slide identifier

## Examples

Check out the `examples/` directory for:
- `generate_synthetic_data.py`: Create synthetic data for testing
- `*_gatv2.py`: GATv2-specific implementations
- Complete end-to-end workflows


## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Inspired by advances in spatial transcriptomics analysis
- Thanks to all contributors and the open-source community
