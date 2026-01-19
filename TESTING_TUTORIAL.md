# STOnco 测试使用教程

本教程基于您的测试环境（`/root/Project/STOnco_tests`）提供完整的命令示例，可直接复制使用。

## 环境要求

- Python 3.8+
- CUDA 支持（推荐）
- 内存：至少 16GB RAM（推荐 32GB+）

## 1. 安装配置

```bash
cd /root/Project/STOnco

# 安装包（可编辑模式）
pip install -e .

# 验证安装
python -c "import stonco; print('✓ STOnco 安装成功')"
```

## 2. 数据准备

**建议目录结构**：将处理后的数据保存在对应数据集目录下，便于管理多个数据集。

### 2.1 训练数据准备

假设您的 Visium 数据在 `/root/Project/STOnco_tests/data/data_3326genes/ST_train_datasets` 目录中，每个 slide 子目录包含 `*_exp.csv` 和 `*_coordinates.csv`：

```bash
# 准备训练数据（输出到数据集目录下）
python -m stonco.utils.prepare_data build-train-npz \
    --train_dir /root/Project/STOnco_tests/data/data_3326genes/ST_train_datasets \
    --out_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --xy_cols row col \
    --label_col true_label

# 输出：/root/Project/STOnco_tests/data/data_3326genes/train_data.npz
```

**参数说明：**
- `build-train-npz`: 扫描目录中所有slide子目录，构建训练数据
- `--train_dir`: 包含slide子目录的父目录（注意：指向实际的slides目录，如 ST_train_datasets）
- `--out_npz`: 输出NPZ文件路径（建议保存到对应数据集目录）
- `--xy_cols`: 坐标CSV中的x/y列名（默认为"row col"）
- `--label_col`: 标签列名（默认为"true_label"，值应为0/1）

 **💡 提示**  ：
- 处理其他数据集时，只需修改 `data_3326genes/ST_train_datasets` 为对应路径
- 注意区分：`--train_dir` 指向包含slide子目录的目录，`--out_npz` 是输出的NPZ文件

### 2.2 验证数据准备

```bash
# 准备验证数据（生成独立的NPZ文件）
python -m stonco.utils.prepare_data build-val-npz \
    --val_dir /root/Project/STOnco_tests/data/validation_data \
    --out_dir /root/Project/STOnco_tests/data/validation_data/val_npz \
    --xy_cols row col \
    --label_col true_label
```

**参数说明：**
- `build-val-npz`: 为每个验证slide生成独立NPZ文件
- `--val_dir`: 包含验证slide子目录的目录
- `--out_dir`: 输出NPZ文件的目录（建议保存到对应数据集目录）

### 2.3 单样本准备（可选）

如果需要为单个slide准备数据：

```bash
python -m stonco.utils.prepare_data build-single-npz \
    --exp_csv /root/Project/STOnco_tests/data/slide_001/slide_001_exp.csv \
    --coord_csv /root/Project/STOnco_tests/data/slide_001/slide_001_coordinates.csv \
    --out_npz /root/Project/STOnco_tests/processed_data/slide_001.npz \
    --xy_cols row col \
    --sample_id slide_001
```

## 3. 模型训练

**建议**：将每个模型的训练结果保存到独立的实验目录，便于管理多个实验。

### 3.1 基础训练（带双域对抗学习）

```bash
# 创建实验目录（示例：test_260115）
# 您可以自定义目录名，如 test_260115_exp1, test_260115_exp2 等
mkdir -p /root/Project/STOnco_tests/test_260115/artifacts

# 训练模型
# 注意：验证数据划分由代码自动处理（根据--stratify_by_cancer或--kfold_cancer参数）
python -m stonco.core.train \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --model gatv2 \
    --use_domain_adv_slide 1 \
    --use_domain_adv_cancer 1 \
    --epochs 80 \
    --early_patience 20 \
    --batch_size_graphs 2 \
    --hidden 128 \
    --heads 4 \
    --lr 0.001
```

**关键参数说明：**
- `--use_domain_adv_slide 1`: 启用切片级别的域对抗学习（减少batch效应）
- `--use_domain_adv_cancer 1`: 启用癌种类的域对抗学习（减少癌种偏差）
- `--epochs`: 训练轮数
- `--early_patience`: 早停等待轮数
- `--hidden`: 隐藏层维度
- `--heads`: GATv2 注意力头数

 **💡 提示**  ：
- 修改 `/root/Project/STOnco_tests/test_260115` 为您想要的实验目录名
- 同一数据集可以运行多个实验，只需更改实验目录即可
- 最佳模型自动保存在 `artifacts_dir/model.pt`

### 3.2 K折训练模式（K-fold by Cancer）

基于癌种进行K折交叉验证，每个fold从不同癌种各选1个样本作为验证集，训练K个独立的模型。

- 运行5折交叉验证
- 结果会保存在：/root/Project/STOnco_tests/test_260115/kfold_val/
python -m stonco.core.train \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --kfold_cancer 10 \
    --split_seed 2026 \
    --use_domain_adv_slide 1 \
    --use_domain_adv_cancer 1
```

**关键参数说明：**
- `--kfold_cancer 5`: 指定折数（K），默认每个癌种选1个样本作为验证
- `--split_seed 42`: 随机种子，保证结果可复现
- 结果保存在 `{artifacts_dir_parent}/kfold_val/`，包含：
  - `fold_1/`, `fold_2/`, ..., `fold_5/`: 每个fold的独立模型和结果
  - `kfold_summary.csv`: 所有fold的指标汇总

 **💡 提示**  ：
- K折模式下，`artifacts_dir` 仅作为基准参考点，实际结果保存在其同级目录的 `kfold_val/` 中
- 如需调整验证集大小，可在每个fold中手动修改代码中的划分逻辑

### 3.3 跨癌种评估（LOCO）

- 注意：LOCO模式下，验证集自动从训练数据中划分（每个癌种留一作为验证）
python -m stonco.core.train \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --leave_one_cancer_out \
    --use_domain_adv_slide 1 \
    --use_domain_adv_cancer 1
```
**💡 提示**  ：
- LOCO模式下，`artifacts_dir` 仅作为基准参考点，实际结果保存在其同级目录的 `loco_val/` 中


## 4. 模型推理

推理时，请使用对应的实验目录中的模型。

### 4.1 单样本推理

```bash
# 创建预测结果目录（在当前实验目录下）
mkdir -p /root/Project/STOnco_tests/test_260115/predictions

python -m stonco.core.infer \
    --npz /root/Project/STOnco_tests/data/test_slide.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --out_csv /root/Project/STOnco_tests/test_260115/predictions/test_slide_predictions.csv \
    --explain_method ig
```

**关键参数说明：**
- `--npz`: 输入的NPZ文件路径（必需）
- `--artifacts_dir`: 包含 model.pt 和预处理器产物的目录（必需）
- `--out_csv`: 输出CSV文件路径（默认：preds.csv）
- `--index`: 如果NPZ包含多个样本，指定索引（默认为0）
- `--gene_attr_out`: 基因重要性输出CSV路径（可选）

### 4.2 批量推理

```bash
python -m stonco.core.batch_infer \
    --npz_glob "/root/Project/STOnco_tests/data/validation_data/npz/*.npz" \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --out_csv /root/Project/STOnco_tests/test_260115/predictions/batch_predictions.csv \
    --gene_attr_out_dir /root/Project/STOnco_tests/test_260115/predictions/gene_attr \
    --explain_method ig
```

**关键参数说明：**
- `--npz_glob`: 必需参数，使用glob模式匹配多个NPZ文件（示例中使用引号包裹，防止shell扩展）
- `--artifacts_dir`: 包含 model.pt 和预处理器产物的目录
- `--out_csv`: 输出CSV文件的路径（包含所有样本的预测结果）
- `--gene_attr_out_dir`: 基因重要性结果保存目录（可选）
- `--no_plot`: 可添加此参数禁用自动生成准确率柱状图

## 5. 模型评估

```bash
# 评估预测结果（在当前实验目录下）
python -m stonco.utils.evaluate_models \
    --predictions_dir /root/Project/STOnco_tests/test_260115/predictions \
    --output_file /root/Project/STOnco_tests/test_260115/evaluation_results.csv

# 可视化结果
mkdir -p /root/Project/STOnco_tests/test_260115/visualizations
python -m stonco.utils.visualize_prediction \
    --prediction_file /root/Project/STOnco_tests/test_260115/predictions/slide_001_predictions.csv \
    --output_path /root/Project/STOnco_tests/test_260115/visualizations/slide_001.svg
```

## 6. 超参数优化（可选）

```bash
# 创建HPO实验目录
mkdir -p /root/Project/STOnco_tests/test_260115_hpo

# 注意：train_hpo 使用 --tune 参数指定优化阶段
python -m stonco.core.train_hpo \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115_hpo \
    --tune all \
    --n_trials 50 \
    --n_jobs 4
```

## 7. 使用 Console Scripts（便捷方式）

setup.py 中已配置 console scripts，可直接使用：

```bash
# 训练（等同于 python -m stonco.core.train）
stonco-train \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts

# 推理
stonco-infer \
    --model_path /root/Project/STOnco_tests/test_260115/artifacts/model.pt \
    --input_data /root/Project/STOnco_tests/data/test_slide.npz \
    --output_path /root/Project/STOnco_tests/test_260115/predictions/test_slide_predictions.csv

# 数据准备
stonco-prepare build-train-npz \
    --train_dir /root/Project/STOnco_tests/data/data_3326genes/ST_train_datasets \
    --out_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz
```

## 8. 灵活的目录结构建议

**推荐组织方式**：数据集和实验分离，便于管理多个数据集和实验。

```
STOnco_tests/
├── data/
│   ├── data_3326genes/          # 数据集1（3326个基因）
│   │   ├── ST_train_datasets/   # 训练数据slides
│   │   │   ├── OV12/
│   │   │   ├── OSCC10/
│   │   │   ├── BRCA13/
│   │   │   └── ...              # 更多slide子目录
│   │   ├── train_data.npz       # 处理后的训练数据
│   │   └── validation/
│   │       └── ST_validation_datasets/
│   │           └── ...          # 验证slide子目录
│   │
│   ├── data_5000genes/          # 数据集2（5000个基因）
│   │   ├── ST_train_datasets/
│   │   │   └── ...
│   │   ├── train_data.npz
│   │   └── validation/
│   │       └── ST_validation_datasets/
│   │           └── ...
│   │
│   └── test_slides/             # 独立测试数据
│       ├── test_slide_001.npz
│       └── test_slide_002.npz
│
└── test_260115/                 # 实验1：260115（可自定义名称）
    ├── artifacts/               # 训练模型和配置
    │   ├── model.pt
    │   └── config.json
    ├── predictions/             # 推理结果
    │   ├── slide_001_predictions.csv
    │   └── slide_002_predictions.csv
    ├── visualizations/          # 可视化图片
    │   └── slide_001.svg
    └── evaluation_results.csv   # 评估结果

# 实验2：不同参数
└── test_260115_exp2/            # 同一数据集，不同超参数
    ├── artifacts/
    ├── predictions/
    └── ...

# 实验3：LOCO评估
└── test_260115_loco/
    └── ...
```

**优势**：
- ✅ 同一数据集可被多个实验复用
- ✅ 实验结果独立存储，互不影响
- ✅ 便于比较不同模型的效果
- ✅ 易于扩展和管理多个数据集

## 9. 常见问题

### 9.1 ModuleNotFoundError
确保已安装包：
```bash
cd /root/Project/STOnco && pip install -e .
```

### 9.2 内存不足
- 减小 `--batch_size_graphs`（默认 2）
- 减小 `--hidden` 维度
- 减小 `--n_hvg` 基因数量

### 9.3 训练时间长
- 使用 GPU：检查 `torch.cuda.is_available()`
- 减少 `--epochs`
- 使用更小的模型：`--model sage` 或 `--model gcn`

## 10. 重要说明

1. **始终使用 `-m` 参数**：`python -m stonco.core.train` 而不是 `python train.py`
2. **目录需预先创建**：输出目录不会自动创建，请使用 `mkdir -p` 创建
3. **数据格式**：每个slide子目录需包含 `*_exp.csv` 和 `*_coordinates.csv`
4. **模型保存**：最佳模型自动保存在 `artifacts_dir/model.pt`
5. **基因统一性**：`prepare_data` 会自动提取并统一所有slide的基因

## 11. 快速测试命令（直接复制运行）

复制以下命令快速开始测试，**只需修改目录名即可适配不同数据集和实验**：

```bash
# ===================================================================
# 第1步：准备数据（仅需运行一次）
# 输出保存在数据集目录下，可被多个实验复用
# ===================================================================
python -m stonco.utils.prepare_data build-train-npz \
    --train_dir /root/Project/STOnco_tests/data/data_3326genes/ST_train_datasets \
    --out_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --xy_cols row col \
    --label_col true_label

# 准备验证数据（如果验证数据也在子目录中）
python -m stonco.utils.prepare_data build-val-npz \
    --val_dir /root/Project/STOnco_tests/data/validation_data/ST_validation_datasets \
    --out_dir /root/Project/STOnco_tests/data/validation_data/npz \
    --xy_cols row col \
    --label_col true_label

# ===================================================================
# 第2步：创建实验目录（自定义名称）
# 示例：test_260115 - 可以改为 test_260115_exp1, test_260115_exp2 等
# ===================================================================
mkdir -p /root/Project/STOnco_tests/test_260115/artifacts

# ===================================================================
# 第3步：训练（小型测试）
# 注意：验证集自动从训练数据中划分（通过--stratify_by_cancer参数）
# ===================================================================
python -m stonco.core.train \
    --train_npz /root/Project/STOnco_tests/data/data_3326genes/train_data.npz \
    --artifacts_dir /root/Project/STOnco_tests/test_260115/artifacts \
    --stratify_by_cancer \
    --epochs 10 \
    --batch_size_graphs 1 \
    --early_patience 5

# 训练完成！模型保存在：/root/Project/STOnco_tests/test_260115/artifacts/model.pt
```

**如何使用不同数据集？**

只需替换两处路径（示例：切换到 data_5000genes）：
```bash
# 将 data_3326genes/ST_train_datasets 替换为 data_5000genes/ST_train_datasets
# 旧：/root/Project/STOnco_tests/data/data_3326genes/ST_train_datasets
# 新：/root/Project/STOnco_tests/data/data_5000genes/ST_train_datasets
```

**如何运行多个实验？**

创建不同的实验目录（数据集不变）：
```bash
# 实验1
mkdir -p /root/Project/STOnco_tests/test_260115_exp1/artifacts

# 实验2（不同超参数）
mkdir -p /root/Project/STOnco_tests/test_260115_exp2/artifacts

# 实验3（LOCO评估）
mkdir -p /root/Project/STOnco_tests/test_260115_loco/artifacts
```

## 相关文档

- [详细教程](./docs/Tutorial.md) - 中文完整教程
- [API 文档](./docs/API.md) - API 参考
- [配置说明](./docs/Configuration.md) - 参数详解
