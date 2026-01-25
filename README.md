# STOnco: Spatial Transcriptomics Tumor Region Identification with Dual-Domain Adversarial Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

STOnco is a PyTorch Geometric-based framework for tumor/non-tumor binary classification in 10x Visium spatial transcriptomics data. It features dual-domain adversarial learning to improve model generalization across different cancer types and experimental batches.

<img width="1468" height="406" alt="image" src="https://github.com/user-attachments/assets/4b2f0acf-65d9-4a81-b52d-dc33e0460e36" />


## Key Features

- **Dual-Domain Adversarial Learning**: Reduces dependency on cancer-specific signals and batch effects
- **Spot-Level Domain Heads + GRL Scheduling**: Domain adversarial learning at the spot level with decoupled loss weight and GRL strength
- **Multiple GNN Architectures**: Support for GATv2, GCN, and GraphSAGE models
- **Complete Pipeline**: From data preparation to training, inference, and visualization
- **Hyperparameter Optimization**: Built-in HPO with multi-stage pipeline
- **Batch Processing**: Efficient inference on multiple samples
- **Rich Visualization**: Comprehensive plotting and analysis tools
- **Latent Export + UMAP/t-SNE**: Export 64-d spot embeddings and visualize with UMAP and t-SNE

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

Training outputs include `loss_components.csv`, `train_loss.svg`, and `train_val_metrics.svg` in `--artifacts_dir`. Use `--val_sample_dir` to include external validation NPZs in validation metrics. The `meta.json` now records `train_ids`, `val_ids`, and `metrics` for reproducibility. You can control validation splitting with `--val_ratio` (default 0.2) or disable stratification via `--no_stratify_by_cancer`.
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

Batch inference also supports predicting internal validation slides + external NPZs together:

```bash
python -m stonco.core.batch_infer \
    --train_npz ./processed_data/train_data.npz \
    --external_val_dir ./processed_data/val_npz \
    --artifacts_dir ./artifacts \
    --out_csv ./predictions/batch_predictions.csv
```
Batch inference writes spot-level predictions to `out_csv` and also emits a slide-level summary CSV (`batch_preds_summary.csv`) in the same directory.

KFold batch inference (per fold):

```bash
for i in {1..10}; do
  python -m stonco.core.batch_infer \
    --npz_glob "./processed_data/val_npz/*.npz" \
    --artifacts_dir "./kfold_val/fold_${i}/" \
    --out_csv "./kfold_val/fold_${i}/batch_predictions.csv" \
    --num_threads 4 \
    --num_workers 0
done
```

KFold batch inference with internal validation + external validation together:

```bash
for i in {1..10}; do
  python -m stonco.core.batch_infer \
    --train_npz ./processed_data/train_data.npz \
    --external_val_dir ./processed_data/val_npz \
    --artifacts_dir "./kfold_val/fold_${i}/" \
    --out_csv "./kfold_val/fold_${i}/batch_predictions.csv" \
    --num_threads 4 \
    --num_workers 0
done
```

### 4. Visualization

```bash
python -m stonco.utils.visualize_prediction \
    --npz sample.npz \
    --artifacts_dir ./artifacts \
    --out_svg visualization.svg
```

### 5. Export Spot Embeddings (z64) + UMAP/t-SNE

Export 64-d spot embeddings from the trained model:

```bash
python -m stonco.utils.export_spot_embeddings \
    --artifacts_dir ./artifacts \
    --npz_glob "./processed_data/val_npz/*.npz" \
    --out_csv ./artifacts/spot_embeddings_val_npz.csv
```

Visualize the exported embeddings with UMAP + t-SNE (requires `umap-learn`, included in `requirements.txt`):

```bash
python -m stonco.utils.visualize_umap_tsne \
    --embeddings_csv ./artifacts/spot_embeddings_val_npz.csv \
    --out_dir ./artifacts/embedding_plots \
    --max_points 50000 \
    --seed 42
```

## Model Architecture

STOnco employs a unified `STOnco_Classifier` architecture with:

- **GNN Backbone**: Configurable graph neural network (GATv2/GCN/GraphSAGE)
- **Task Head**: Spot-level tumor/non-tumor prediction with a fixed MLP head `[256, 128, 64, 1]`
- **Domain Heads**: Optional adversarial heads (spot-level) for cancer type and batch domain adaptation

### Dual-Domain Adversarial Learning

```
Total_Loss = Task_Loss + λ_slide × Batch_Domain_Loss + λ_cancer × Cancer_Domain_Loss
```

In `train.py`:

- `--lambda_slide` / `--lambda_cancer` are **loss weights (alpha)**.
- GRL strength uses a DANN-style schedule (fixed) with:
  - `--grl_beta_slide_target`, `--grl_beta_cancer_target` (target beta; default `1.0/0.5`)
  - `--grl_beta_gamma` (schedule steepness; default `10`)
- Domain labels are read from `data/cancer_sample_labels.csv`: `cancer_type` and `Batch_id` (`Batch_id` falls back to `slide_id` if missing). Domain class counts are inferred per run/fold.

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
│       ├── export_spot_embeddings.py # Export z64 embeddings
│       ├── visualize_umap_tsne.py # UMAP + t-SNE visualization
│       └── ...
├── examples/                 # Example scripts and tutorials
├── synthetic_data/           # (Generated) synthetic data for testing
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
`train_hpo.py` supports the same cancer-stratified split controls as `train.py`, including `--val_ratio` (default 0.2) and `--no_stratify_by_cancer`.

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

**Training NPZ (multi-slide, from `prepare_data build-train-npz`):**
- `Xs`: list/array of per-slide gene expression matrices `(n_spots_i, n_genes)`
- `ys`: list/array of per-slide binary labels `(n_spots_i,)` (0/1)
- `xys`: list/array of per-slide spatial coordinates `(n_spots_i, 2)`
- `slide_ids`: per-slide identifiers (strings)
- `gene_names`: gene names (length `n_genes`)
- optional `barcodes`: per-slide barcodes

**Single-slide NPZ (from `prepare_data build-single-npz` / `build-val-npz`):**
- `X`: gene expression matrix `(n_spots, n_genes)`
- `xy`: spatial coordinates `(n_spots, 2)`
- `gene_names`: gene names
- `sample_id`: slide/sample id (optional but recommended)
- optional `y`: binary labels, `barcodes`

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
