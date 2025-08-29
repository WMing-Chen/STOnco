# STRIDE: Spatial Transcriptomics Tumor Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

STRIDE is a PyTorch Geometric-based framework for tumor/non-tumor binary classification in 10x Visium spatial transcriptomics data. It features dual-domain adversarial learning to improve model generalization across different cancer types and experimental batches.

## 🚀 Key Features

- **Dual-Domain Adversarial Learning**: Reduces dependency on cancer-specific signals and batch effects
- **Multiple GNN Architectures**: Support for GATv2, GCN, and GraphSAGE models
- **Complete Pipeline**: From data preparation to training, inference, and visualization
- **Hyperparameter Optimization**: Built-in HPO with multi-stage pipeline
- **Batch Processing**: Efficient inference on multiple samples
- **Rich Visualization**: Comprehensive plotting and analysis tools

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install STRIDE

```bash
pip install -e .
```

## 🏃‍♂️ Quick Start

### 1. Data Preparation

```bash
python -m stride.utils.prepare_data \
    --input_dir /path/to/visium/data \
    --output_dir ./processed_data \
    --gene_list_path genes.txt
```

### 2. Model Training

```bash
python -m stride.core.train \
    --train_data ./processed_data/train_data.npz \
    --val_dir ./processed_data/val_npz \
    --artifacts_dir ./artifacts \
    --model_type gatv2
```

### 3. Inference

```bash
# Single sample inference
python -m stride.core.infer \
    --model_path ./artifacts/model.pt \
    --input_data sample.npz \
    --output_path predictions.csv

# Batch inference
python -m stride.core.batch_infer \
    --model_path ./artifacts/model.pt \
    --input_dir ./test_samples \
    --output_dir ./predictions
```

### 4. Visualization

```bash
python -m stride.utils.visualize_prediction \
    --prediction_file predictions.csv \
    --output_path visualization.svg
```

## 📊 Model Architecture

STRIDE employs a unified `STRIDE_Classifier` architecture with:

- **GNN Backbone**: Configurable graph neural network (GATv2/GCN/GraphSAGE)
- **Task Head**: Binary classification for tumor/non-tumor prediction
- **Domain Heads**: Optional adversarial heads for cancer type and slide-level domain adaptation

### Dual-Domain Adversarial Learning

```
Total_Loss = Task_Loss + λ₁ × Cancer_Domain_Loss + λ₂ × Slide_Domain_Loss
```

- **Cancer Domain Adversarial**: Reduces cancer-type-specific bias
- **Slide Domain Adversarial**: Mitigates batch effects between slides

## 📁 Project Structure

```
STRIDE/
├── stride/
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

## 🔧 Advanced Usage

### Hyperparameter Optimization

```bash
python -m stride.core.train_hpo \
    --train_data train_data.npz \
    --val_dir val_npz \
    --artifacts_dir ./hpo_results \
    --n_trials 100
```

### Cross-Cancer Evaluation (LOCO)

```bash
python -m stride.core.train \
    --train_data train_data.npz \
    --val_dir val_npz \
    --artifacts_dir ./loco_results \
    --eval_mode loco
```

### Model Evaluation

```bash
python -m stride.utils.evaluate_models \
    --predictions_dir ./predictions \
    --output_file evaluation_results.csv
```

## 📈 Input Data Format

STRIDE expects NPZ files with the following structure:

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

## 🎯 Examples

Check out the `examples/` directory for:
- `generate_synthetic_data.py`: Create synthetic data for testing
- `*_gatv2.py`: GATv2-specific implementations
- Complete end-to-end workflows

## 📚 Documentation

For detailed documentation, please refer to:
- [Dual-Domain Adversarial Learning](docs/Dual-Domain_Adversarial_Learning.md)
- [API Reference](docs/api_reference.md)
- [Training Guide](docs/training_guide.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions and support, please open an issue on GitHub or contact the development team.

## 🙏 Acknowledgments

- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Inspired by advances in spatial transcriptomics analysis
- Thanks to all contributors and the open-source community