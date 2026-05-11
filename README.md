# STOnco: Spatial Transcriptomics Tumor Region Identification via Pan-Cancer Continuous Wasserstein Barycenter Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

STOnco is a graph neural network framework for tumor region identification in spatial transcriptomics data. It uses Continuous Wasserstein Barycenter Learning to learn a shared pan-cancer latent barycenter, enabling cancer-specific spot-level latent distributions to form consistent representations in a common latent space and improving generalization across cancer types and experimental batches.

The Continuous Wasserstein Barycenter (CWB) module is used only during training. Inference remains a standard and lightweight prediction path:

```text
input graph -> GNN encoder -> spot latent h -> classifier head -> tumor probability
```

<img width="1468" height="406" alt="image" src="https://github.com/user-attachments/assets/4b2f0acf-65d9-4a81-b52d-dc33e0460e36" />


## Key Features

- **Continuous Wasserstein Barycenter Learning**: Learns a shared pan-cancer latent barycenter that regularizes cancer-specific spot-level latent distributions into a common representation space.
- **Prior-Generated Continuous Support**: Parameterizes the shared barycenter distribution with a training-only generator `b = G_psi(z)`, where `z` is sampled from a normal or uniform prior.
- **Wasserstein-Style Distribution Losses**: Supports `sliced_wasserstein`, `sinkhorn_divergence`, and fixed-kernel `mmd` for prior-generator barycenter learning.
- **Spot-Level Tumor Region Classification**: Predicts tumor/non-tumor labels for each spatial spot from gene-expression graphs.
- **Flexible GNN Backbones**: Supports GATv2, GCN, and GraphSAGE models.
- **Optional Dual-Domain Adversarial Learning**: Adds batch-domain and cancer-domain adversarial heads when domain confusion is desired as an extra regularizer.
- **Optional Image Feature Fusion**: Supports per-spot image features with independent preprocessing and optional PCA.
- **Embedding Export + UMAP/t-SNE**: Exports spot-level latent embeddings and visualizes them with UMAP and t-SNE.

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

Each slide directory must contain exactly 3 CSV files (case-insensitive suffix matching):
- `*exp.csv`
- `*coordinates.csv`
- `*image_features.csv` (Barcode + 2048 image feature columns; column names/order must be consistent across slides)

If a spot is missing from `*image_features.csv`, its image vector is filled with zeros and `img_mask=0` is recorded (with a warning). Non-finite image values (NaN/inf) will raise an error during data preparation.

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

### 2. Baseline Training

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts_baseline \
    --model gatv2
```

### 3. Training with Continuous Wasserstein Barycenter Learning

The recommended CWB configuration uses `wb_support_mode=prior_generator` and `wb_loss_type=sliced_wasserstein`. This learns a prior-generated shared support distribution and regularizes cancer-specific spot-level latent distributions toward the learned pan-cancer barycenter.

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts_cwb \
    --model gatv2 \
    --use_wb_align 1 \
    --wb_support_mode prior_generator \
    --wb_loss_type sliced_wasserstein \
    --lambda_wb 0.15 \
    --wb_warmup_epochs 10 \
    --wb_ramp_epochs 20 \
    --wb_spots_per_graph 512 \
    --wb_support_size 256 \
    --wb_sw_num_projections 64
```

For a debiased Sinkhorn divergence variant, keep the same command and replace the WB-specific options with:

```bash
--wb_support_mode prior_generator \
--wb_loss_type sinkhorn_divergence \
--lambda_wb 0.15 \
--wb_epsilon 0.1 \
--wb_sinkhorn_iters 50 \
--wb_spots_per_graph 128 \
--wb_support_size 256
```

The prior generator is training-only. At inference time, STOnco uses only the trained encoder and classifier head.

Training outputs include `loss_components.csv`, `train_loss.svg`, and `train_val_metrics.svg` in `--artifacts_dir`. Use `--val_sample_dir` to include external validation NPZs in validation metrics. The `meta.json` now records `train_ids`, `val_ids`, and `metrics` for reproducibility. You can control validation splitting with `--val_ratio` (default 0.2) or disable stratification via `--no_stratify_by_cancer`.
You can also adjust the classifier/domain-head widths via `--clf_hidden` (comma-separated positive integers, variable depth) and `--dom_hidden` (one hidden layer for both domain heads); values are saved into `meta.json` to keep inference consistent.
If you want the final saved `model.pt` to be the last epoch (instead of best-by-validation), use `--save_last` (only overwrites `model.pt` when training actually reaches `--epochs`; you can disable early stopping with `--early_patience 0`). When enabled, the best checkpoint is also saved as `model_best.pt`, and `meta.json` includes `last_epoch/last_metrics/saved_checkpoint`.
Disable early stopping:

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts \
    --model gatv2 \
    --early_patience 0
```

### 4. Optional Image Feature Fusion

```bash
python -m stonco.core.train \
    --train_npz ./processed_data/train_data.npz \
    --artifacts_dir ./artifacts_image \
    --model gatv2 \
    --use_image_features 1 \
    --img_use_pca 1 \
    --img_pca_dim 256
```

When `--use_image_features 1`, training also saves the image preprocessor artifacts to `--artifacts_dir`:
`img_feature_names.txt`, `img_scaler.joblib`, and (if `--img_use_pca 1`) `img_pca.joblib`.
If `--img_use_pca 1`, training requires the number of valid image spots (where `img_mask==1`) to be at least `--img_pca_dim`.

### 5. Inference

```bash
# Single sample inference
python -m stonco.core.infer \
    --npz sample.npz \
    --artifacts_dir ./artifacts_cwb \
    --out_csv predictions.csv

# Batch inference
python -m stonco.core.batch_infer \
    --npz_glob "./test_samples/*.npz" \
    --artifacts_dir ./artifacts_cwb \
    --out_csv ./predictions/batch_predictions.csv
```

Batch inference also supports predicting internal validation slides + external NPZs together:

```bash
python -m stonco.core.batch_infer \
    --train_npz ./processed_data/train_data.npz \
    --external_val_dir ./processed_data/val_npz \
    --artifacts_dir ./artifacts_cwb \
    --out_csv ./predictions/batch_predictions.csv
```
Batch inference writes spot-level predictions to `out_csv` and also emits a slide-level summary CSV (`batch_preds_summary.csv`) in the same directory.

Note: if the model was trained with `use_image_features=1`, inference will use image features according to `meta.json:cfg`.
If the input NPZ contains `X_img/img_mask/img_feature_names`, they must match `artifacts_dir/img_feature_names.txt` (otherwise an error is raised).
If the input NPZ does not include image keys, inference falls back to `X_img=0` and `img_mask=0` (runs as gene-only for that input, but the model still expects the fused input dimension).

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

### 6. Visualization

```bash
python -m stonco.utils.visualize_prediction \
    --npz sample.npz \
    --artifacts_dir ./artifacts_cwb \
    --out_svg visualization.svg
```

### 7. Export Spot Embeddings (h / z_clf) + UMAP/t-SNE

Export 64-d spot embeddings from the trained model:

```bash
python -m stonco.utils.export_spot_embeddings \
    --artifacts_dir ./artifacts_cwb \
    --npz_glob "./processed_data/val_npz/*.npz" \
    --out_csv ./artifacts_cwb/spot_embeddings_val_npz.csv
```

Visualize the exported embeddings with UMAP + t-SNE (requires `umap-learn`, included in `requirements.txt`):

```bash
python -m stonco.utils.visualize_umap_tsne \
    --embeddings_csv ./artifacts_cwb/spot_embeddings_val_npz.csv \
    --out_dir ./artifacts_cwb/embedding_plots \
    --max_points 50000 \
    --seed 42
```

## Model Architecture

STOnco consists of three main training-time components:

1. A GNN encoder that maps each spatial spot to a latent representation `h`.
2. A tumor/non-tumor classifier head that predicts spot-level tumor probability from `h`.
3. A Continuous Wasserstein Barycenter Learning module that learns a shared pan-cancer latent barycenter and regularizes cancer-specific latent distributions toward it.

### Continuous Wasserstein Barycenter Learning

For cancer type `k`, let `P_k^h` denote the empirical distribution of spot-level GNN latents. STOnco learns a shared support distribution:

```text
z ~ p(z)
b = G_psi(z)
Q_psi = Law(G_psi(z))
```

and minimizes a barycenter regularization term:

```text
L_CWB = sum_k D(P_k^h, Q_psi)
```

where `D` can be `sliced_wasserstein`, `sinkhorn_divergence`, or fixed-kernel `mmd`. This is a distribution-level constraint: it does not force one-to-one spot matching across cancer types. Instead, it encourages different cancer-specific spot-level latent distributions to share a common pan-cancer geometry while preserving the supervised tumor/non-tumor objective.

The overall training objective is:

```text
L_total = L_task
        + optional domain adversarial losses
        + optional MMD losses
        + lambda_wb(t) * L_CWB
```

`lambda_wb(t)` follows a warmup and ramp schedule controlled by `--wb_warmup_epochs`, `--wb_ramp_epochs`, and `--lambda_wb`.

### Support Modes

STOnco supports two barycenter support modes:

| `wb_support_mode` | Support definition | Main use |
|---|---|---|
| `prior_generator` | `b = G_psi(z)` from random prior samples | Recommended CWB mode. Learns an independent continuous pan-cancer support distribution. |
| `generated_support` | `b = T_phi(h)` from spot latents | Legacy generated-support mode with spot-wise transformed support points. |

In `prior_generator` mode, pointwise anchor loss is disabled and logged as `wb_anchor=0`, because generated support samples are independent of individual spots.

### WB Loss Choices

| `wb_loss_type` | Supported support modes | Description |
|---|---|---|
| `sliced_wasserstein` | `prior_generator`, `generated_support` | Efficient Wasserstein-style distribution matching through random 1D projections. Recommended default for CWB experiments. |
| `sinkhorn_divergence` | `prior_generator`, `generated_support` | Debiased entropy-regularized OT distance. More expensive, but closer to standard optimal transport. |
| `mmd` | `prior_generator` only | Fixed-kernel MMD barycenter ablation. |
| `euclidean_pairwise` | `generated_support` only | Lightweight legacy generated-support surrogate. |
| `dual_potential` | `generated_support` only | Neural dual-potential WB surrogate. |

### Optional Dual-Domain Adversarial Learning

STOnco can optionally add spot-level adversarial heads for batch and cancer domains. These losses are controlled by:

- `--use_domain_adv_slide`, `--lambda_slide`
- `--use_domain_adv_cancer`, `--lambda_cancer`
- `--grl_beta_mode` and the corresponding GRL target, delay, and warmup arguments

Domain labels are read from `data/cancer_sample_labels.csv`: `cancer_type` and `Batch_id`. If `Batch_id` is missing, it falls back to `slide_id`. Domain class counts are inferred for each run or fold.

## Training Artifacts

Standard training outputs include:

- `model.pt`: final or best model, depending on `--save_last` and early stopping.
- `model_best.pt`: best validation checkpoint when available.
- `meta.json`: full training configuration, train/validation slide IDs, metrics, and checkpoint metadata.
- `genes_hvg.txt`, `scaler.joblib`, `pca.joblib`: gene preprocessing artifacts.
- `train_loss.svg`, `train_val_loss.svg`, `train_val_metrics.svg`, `lr.svg`: training curves.
- `loss_components.csv`: epoch-level loss components and validation metrics.

When `--use_wb_align 1` is enabled, STOnco additionally saves:

- `wb_support_map_last.pt`: trained support map or prior generator state.
- `wb_potentials_last.pt`: WB module state, kept for a consistent artifact layout.
- `wb_config.json`: WB/CWB configuration.
- `wb_train_loss.svg`: WB loss and diagnostic curves.
- `wb_support_diagnostics.svg`: prior-support diagnostics when `wb_support_mode=prior_generator`.

`loss_components.csv` includes WB/CWB metrics when available, such as:

- `avg_wb_loss`
- `avg_wb_sliced_wasserstein`
- `avg_wb_sinkhorn`
- `avg_wb_mmd`
- `avg_wb_active_cancers`
- `avg_wb_active_spots`
- `wb_lambda`

If `--wb_eval_loss 1` is enabled, validation-side WB diagnostics are also recorded, for example `val_avg_wb_loss`, `val_avg_wb_sliced_wasserstein`, `val_avg_wb_sinkhorn`, and `val_avg_wb_mmd`.

## Project Structure

```
STOnco/
├── stonco/
│   ├── core/
│   │   ├── models.py         # GNN encoder, classifier, and domain heads
│   │   ├── train.py          # Training CLI and CWB/WB integration
│   │   ├── wb_potentials.py  # Support generators and WB/CWB losses
│   │   ├── train_hpo.py      # Hyperparameter optimization
│   │   ├── infer.py          # Single-sample inference
│   │   ├── batch_infer.py    # Batch inference
│   │   └── sampler.py        # Training samplers
│   └── utils/
│       ├── prepare_data.py
│       ├── evaluate_models.py
│       ├── visualize_prediction.py
│       ├── export_spot_embeddings.py
│       ├── visualize_umap_tsne.py
│       ├── evaluate_embedding_mixing.py
│       └── ...
├── docs/
├── examples/
├── tests/
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
- optional image keys (required if training with `--use_image_features 1`):
  - `X_imgs`: list/array of per-slide image feature matrices `(n_spots_i, 2048)` (float32)
  - `img_masks`: list/array of per-slide image masks `(n_spots_i,)` (uint8 0/1)
  - `img_feature_names`: image feature names (length 2048; must match across slides and between train/infer)

**Single-slide NPZ (from `prepare_data build-single-npz` / `build-val-npz`):**
- `X`: gene expression matrix `(n_spots, n_genes)`
- `xy`: spatial coordinates `(n_spots, 2)`
- `gene_names`: gene names
- `sample_id`: slide/sample id (optional but recommended)
- optional `y`: binary labels, `barcodes`
- optional image keys:
  - `X_img`: image feature matrix `(n_spots, 2048)` (float32)
  - `img_mask`: image mask `(n_spots,)` (uint8 0/1)
  - `img_feature_names`: image feature names (length 2048; must match training)

## Examples

Check out the `examples/` directory for:
- `generate_synthetic_data.py`: Create synthetic data for testing
- `*_gatv2.py`: GATv2-specific implementations
- Complete end-to-end workflows

## References

- `docs/PLAN_prior_generator_barycenter_STOnco.md`
- `docs/PLAN_sliced_wasserstein_WB_STOnco.md`
- `docs/PLAN_sinkhorn_divergence_WB_STOnco.md`
- `docs/Tutorial.md`


## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Inspired by advances in spatial transcriptomics, optimal transport, and pan-cancer representation learning
- Thanks to all contributors and the open-source community
