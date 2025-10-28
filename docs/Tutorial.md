# STOnco: Visium 空间转录组 GNN 分类与域自适应

本项目基于 PyTorch Geometric，面向 10x Visium 空间转录组的肿瘤/非肿瘤二分类任务，提供从数据准备 → 训练（含双域自适应）→ 多阶段超参数优化（HPO）→ 推理与可视化的完整工具链。核心模型采用统一的 STOnco_Classifier，支持多种 GNN 主干（GATv2 / GraphSAGE / GCN）、拉普拉斯位置编码（LapPE）与可选的双域对抗（癌种域 + 切片域）。

---

## 1. 目录结构与核心脚本

- prepare_data.py：从 CSV（exp + coordinates）构建训练/单样本/验证 NPZ
- preprocessing.py：预处理器与图构建（HVG 选择、标准化、可选 PCA，KNN 图 + 高斯边权，LapPE）
- models.py：STRIDE_Classifier（GNN 主干 + 分类头 + 可选双域对抗头）
- train.py：训练与验证（单次、癌种分层、k-fold 按癌种、LOCO 留一癌种评估）
- train_hpo.py：三阶段 HPO 流水线（含多种子复评/重打分）
- infer.py：单切片 NPZ 推理（输出预测 CSV；可选保存基因重要性 CSV）
- batch_infer.py：批量推理，支持多进程预处理；可选保存每张切片的基因重要性 CSV
- visualize_prediction.py：单切片真实/预测空间分布可视化（SVG）
- Dual-Domain Adversarial Learning.md：双域自适应设计说明
- utils.py：模型与元信息存取
- synthetic_data/：示例合成/模拟数据（可选）

---

## 2. 环境依赖与安装

- Python 3.12（已测）
- PyTorch（示例：2.6.0+cu124）
- PyTorch Geometric 2.6.1（以及 pyg-lib、torch-scatter、torch-sparse、torch-cluster）
- 其他：scikit-learn、scipy、pandas、joblib、matplotlib

安装示例（请按你本机 PyTorch/CUDA 版本替换链接与版本号）：

```bash
pip install scikit-learn pandas scipy joblib matplotlib
pip install --default-timeout=600 \
  torch_scatter torch_sparse torch_cluster pyg-lib \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric==2.6.1
```

## 快速开始

一键式示例流程：从原始 CSV 准备数据 → 训练（示例使用双域对抗默认配置）→ 单/批量推理 → 可视化。

```bash
# 0) 路径约定（请按需修改）
DATA_ROOT=/path/to/dataset
ARTIFACTS=/path/to/artifacts
VAL_NPZ_DIR=$DATA_ROOT/val_npz
PRED_CSV=/path/to/preds.csv

# 1) 数据准备
# 1.1 训练集（多切片）NPZ
python STRIDE/prepare_data.py build-train-npz \
  --train_dir $DATA_ROOT/ST_train_datasets \
  --out_npz $DATA_ROOT/train_data.npz \
  --xy_cols row col \
  --label_col true_label

# 1.2 验证集目录转为一组单切片 NPZ
python STRIDE/prepare_data.py build-val-npz \
  --val_dir $DATA_ROOT/ST_validation_datasets \
  --out_dir $VAL_NPZ_DIR \
  --xy_cols row col \
  --label_col true_label

# 2) 训练（双域对抗：默认启用；未显式给出 lambda_* 时回退到 0.3）
python STRIDE/train.py \
  --train_npz $DATA_ROOT/train_data.npz \
  --artifacts_dir $ARTIFACTS \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 1 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2

# 3) 单切片推理（对验证集中的任意一张）
# 先任选一个 NPZ（例如第一张）
SAMPLE_NPZ=$(ls $VAL_NPZ_DIR/*.npz | head -n 1)
python STRIDE/infer.py \
  --npz "$SAMPLE_NPZ" \
  --artifacts_dir $ARTIFACTS \
  --out_csv $PRED_CSV \
  --num_threads 4
# 将在与 out_csv 同目录生成 per_slide_gene_saliency_{...}.csv（默认解释开启）

# 4) 批量推理（对整个验证集目录）
python STRIDE/batch_infer.py \
  --npz_glob "$VAL_NPZ_DIR/*.npz" \
  --artifacts_dir $ARTIFACTS \
  --out_csv ${PRED_CSV%.*}_batch.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0
# 将在 out_csv 同目录下创建 gene_attr/ 并为每张切片保存 per_slide_gene_saliency_{sample_id}.csv

# 5) 可视化（单张或批量）
# 5.1 单张：若单 NPZ 含 y，可直接绘图并显示准确率
python STRIDE/visualize_prediction.py \
  --npz "$SAMPLE_NPZ" \
  --artifacts_dir $ARTIFACTS \
  --out_svg ${PRED_CSV%.*}_vis.svg

# 5.2 使用训练 NPZ 的最后一张（常作为验证）进行可视化
python STRIDE/visualize_prediction.py \
  --train_npz $DATA_ROOT/train_data.npz \
  --artifacts_dir $ARTIFACTS \
  --slide_idx -1 \
  --out_svg ${PRED_CSV%.*}_vis_val.svg
```

默认参数要点：
- prepare_data：`--xy_cols row col`、`--label_col true_label` 为默认；build-single-npz 若不传 `--sample_id`，将用坐标文件上一级目录名。
- 训练（train.py）：
  - 设备：默认自动检测 CUDA；线程：`--num_threads` 不传则由 PyTorch 默认；DataLoader `--num_workers` 默认 0。
  - 双域对抗：旧总开关为 `--disable_domain_adv`（不传则启用对抗）；细粒度开关 `--use_domain_adv_slide/--use_domain_adv_cancer` 可单独控制。
  - 损失权重：未显式给出 `--lambda_slide/--lambda_cancer` 时回退到 `domain_lambda=0.3`。
  - 解释性：默认开启，`--explain_method ig`，`--ig_steps 50`；训练结束会保存 `per_gene_saliency.csv` 到 artifacts_dir。
- 推理（infer.py/batch_infer.py）：
  - 解释性：默认开启，仅保存 CSV（`gene, attr`）；单张通过 `--gene_attr_out` 指定路径；批量用 `--gene_attr_out_dir` 指定目录（默认为 `out_csv` 同目录下的 `gene_attr/`）。
  - 阈值：`--threshold 0.5`；输出 CSV 包含 `p_tumor` 与 `pred_label`（批量）。
- 可视化：从 `artifacts_dir/meta.json` 读取 cfg 保证图构建一致；`--threshold` 默认 0.5。

---

## 3. 输入数据格式

项目训练/推理均基于 NPZ 文件：

- 训练 NPZ（由 prepare_data.py 生成，包含多张切片）：
  - Xs：长度 N_slide 的列表，每项形状 (n_spots_i, n_genes)
  - ys：同上，每项形状 (n_spots_i,)，取值 {0,1}
  - xys：同上，每项形状 (n_spots_i, 2)
  - slide_ids：长度 N_slide 的列表/数组（字符串或整数）
  - gene_names：长度 n_genes 的列表

- 单切片 NPZ（推理或验证单张）：
  - X：形状 (n_spots, n_genes)
  - xy：形状 (n_spots, 2)
  - gene_names：长度 n_genes 的列表
  - barcodes：该切片的条形码（字符串数组）
  - sample_id：样本/切片 ID（字符串）
  - 可选 y：形状 (n_spots,)，取值 {0,1}

从 CSV 构建 NPZ 的约定（见 prepare_data.py）：
- exp.csv：首列必须为 Barcode，其余列为基因（列名即基因名）；数值列会转为浮点，缺失置 0。
- coordinates.csv：必须包含 Barcode 列（大小写不敏感匹配），以及指定的 x/y 列（默认 row、col）。
- 构建时按 Barcode 交集对齐，保持坐标文件的顺序；基因顺序保持与 exp.csv 一致。

---

## 4. 数据准备（prepare_data.py）

具体命令：
python prepare_data.py build-train-npz --train_dir data/data_SC_3331genes/ST_train_datasets  --out_npz data/data_SC_3331genes/train_data.npz 
python prepare_data.py build-val-npz --val_dir data/data_SC_3331genes/ST_validation_datasets --out_dir data/data_SC_3331genes/val_npz/

常用子命令：

- 扫描训练目录构建多切片训练集 NPZ：
```bash
python STRIDE/prepare_data.py build-train-npz \
  --train_dir /path/to/train_slides \
  --out_npz /path/to/train_data.npz \
  --xy_cols row col \
  --label_col true_label
```

- 由一对 CSV 构建单样本 NPZ（可用于推理或验证单张）：
```bash
python STRIDE/prepare_data.py build-single-npz \
  --exp_csv /path/slide_exp.csv \
  --coord_csv /path/slide_coordinates.csv \
  --out_npz /path/slide.npz \
  --xy_cols row col \
  --label_col true_label \
  --sample_id SLIDE_ID
```

- 为验证集目录中的每张切片各自生成一个 NPZ：
```bash
python STRIDE/prepare_data.py build-val-npz \
  --val_dir /path/to/val_slides \
  --out_dir /path/to/val_npz_dir \
  --xy_cols row col \
  --label_col true_label
```

### 4.1 输入目录结构示例

- 训练目录（用于 build-train-npz）：
```
/train_slides/
  SlideA/
    xxx_coordinates.csv
    xxx_exp.csv
  SlideB/
    yyy_coordinates.csv
    yyy_exp.csv
  ...
```
- 验证目录（用于 build-val-npz）：
```
/val_slides/
  Slide1/
    *_coordinates.csv
    *_exp.csv
  Slide2/
    *_coordinates.csv
    *_exp.csv
  ...
```
注意：每个切片子目录中脚本会“自动检测”文件，优先选择以 `coordinates.csv` 和 `exp.csv` 结尾的文件名；文件名前缀可以任意。

### 4.2 CSV 字段要求与校验

- 表达矩阵（exp.csv）：
  - 第 1 列必须为 `Barcode`（区分大小写），其余列为基因（列名即基因名）。
  - 所有基因表达值会转换为浮点，无法解析的值记为 0.0。
- 坐标（coordinates.csv）：
  - 必须包含 `Barcode` 列（大小写不敏感匹配），以及指定的 x/y 列（默认 `row`、`col`）。
  - x/y 列会强制转为数值，并丢弃坐标缺失的行。
  - 训练时要求存在标签列（默认 `true_label`）。标签可以是：
    - 数值型：会被强制为二值（非 0 视为 1）。
    - 字符型：支持映射 `tumor`→1、`normal`→0、`mal`→1、`nmal`→0；未知值会提示 Warning 并映射为 0。

### 4.3 条形码对齐与顺序

- 以两个 CSV 的 `Barcode` 交集为准做对齐；若交集为空会报错。
- 对齐后严格“按坐标文件的顺序”排列 spots；表达矩阵按该顺序重建。

### 4.4 基因集合与顺序策略

- 训练 NPZ（多切片）：
  - 取所有切片的“基因并集”作为最终基因列表，按第一次出现的顺序确定统一顺序；
  - 对某切片缺失的基因列使用 0 填充。
- 单切片 NPZ：
  - 保留 exp.csv 中的原始基因顺序，不做并集扩展。

### 4.5 输出文件字段与类型

- 训练 NPZ（build-train-npz）：
  - `Xs`: List[np.ndarray]，每项 (n_spots_i, n_genes) float32
  - `ys`: List[np.ndarray]，每项 (n_spots_i,) int64（二值 0/1）
  - `xys`: List[np.ndarray]，每项 (n_spots_i, 2) float32
  - `slide_ids`: List[str]，切片/子目录名
  - `gene_names`: List[str]，并集后的基因列表
  - `barcodes`: List[np.ndarray]，每项为该切片的条形码（dtype='U64'）
- 单切片 NPZ（build-single-npz）：
  - `X` (n_spots, n_genes) float32，`xy` (n_spots,2) float32
  - `gene_names` List[str]，`barcodes` (n_spots,) U64
  - `sample_id` str（默认从坐标文件所在文件夹名推断）
  - 可选 `y` (n_spots,) int64
- 验证集 NPZ（build-val-npz）：
  - 为每个子目录分别生成一个单切片 NPZ，文件名为 `<子目录名>.npz`。

### 4.6 常用参数说明补充

- `--xy_cols X_COL Y_COL`：指定坐标文件中的 x/y 列名（默认 `row col`）。
- `--label_col COL`：指定标签列名（训练必需，默认 `true_label`）。
- `--sample_id`（仅 build-single-npz）：不传则自动使用坐标文件上一级目录名。

### 4.7 错误处理与提示

- build-train-npz：
  - 若某切片缺少 `coordinates.csv` 或 `exp.csv`，或标签列缺失，会报错并中止（训练需要完整标签）。
  - 若 `Barcode` 交集为空，报错并中止。
- build-val-npz：
  - 若某子目录文件缺失，会 Warning 并跳过，不中止整个流程。

---

## 5. 训练（train.py）

最小示例：
```bash
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --epochs 100 \
  --early_patience 30 \
  --batch_size_graphs 2
```

主要参数（部分）：
- 基本：
  - --train_npz 必需；--artifacts_dir 默认为 artifacts
  - --epochs，--early_patience，--batch_size_graphs
  - --device cpu/cuda（不指定自动检测,默认cuda）
  - --num_threads 控制 PyTorch 线程数（CPU 时限制核心占用）
  - --num_workers DataLoader 进程数（验证集内部自动≤2）
- 模型与结构：
  - --model {gatv2,sage,gcn}（默认 gatv2），--heads（仅 gatv2）
  - --hidden，--num_layers，--dropout
  - LapPE：--lap_pe_dim（>0 启用，默认 16）、--concat_lap_pe {0,1}、--lap_pe_use_gaussian {0,1}
- 预处理：
  - --use_pca {0,1}（可选 PCA；当前版本默认关闭）
  - --n_hvg N 或 'all'（默认 'all' 表示使用所有基因）
- 优化器：--lr，--weight_decay
- 域自适应（双域，对抗式）：
  - 旧总开关：--disable_domain_adv（关闭域对抗；否则默认启用）
  - 新细粒度：--use_domain_adv_slide {0,1}，--use_domain_adv_cancer {0,1}
  - 损失权重：--lambda_slide，--lambda_cancer（未指定时回退到 domain_lambda=0.3）
- 划分/验证：
  - --stratify_by_cancer 按癌种分层（每癌种随机 1 张验证）
  - --kfold_cancer K 基于癌种的 K 组组合评估，产物位于 artifacts_dir 的同级目录 kfold_val/fold_{i}/，并写出 kfold_val/kfold_summary.csv
  - --leave_one_cancer_out LOCO 留一癌种评估（每个癌种单独训练与验证）；产物位于 artifacts_dir 的父目录下 loco_eval/{CancerType}/
  - --split_seed 随机种子；--split_test_only 仅打印划分统计不训练
- 其他：
  - --out_csv 验证集预测 CSV（未指定时默认写到 artifacts_dir 同级 predictions/val_preds.csv）
  - --config_json 从 JSON 加载一组超参（支持扁平或 {"cfg": {...}} 格式）
  - 解释性输出（默认开启，可用 --no_explain 关闭）：--explain_saliency/--no_explain，--explain_method {ig,saliency}（默认 ig），--ig_steps（默认 50）；若开启，将在训练结束后基于最佳模型对验证图计算整体基因重要性并保存 CSV（默认 artifacts_dir/per_gene_saliency.csv）

训练产物（artifacts_dir）：
- 预处理：genes_hvg.txt，scaler.joblib，pca.joblib（若启用）
- 模型：model.pt；元信息：meta.json（含 cfg 与 best_epoch）
- 可视化：train_metrics.svg（1×3：Loss / AUROC / AUPRC）

### 5.1 产物路径与内容补充

- 单次训练（默认模式）：
  - 模型与预处理均保存到指定的 artifacts_dir；meta.json 中的 `cfg` 为实际训练用到的完整配置，可被推理脚本复用。
  - 若提供 --out_csv，则会将验证集（或最后一张切片）在相同预处理/图构建设置下的预测结果写出到该路径；未显式提供时，默认写到 artifacts_dir 的同级目录 `predictions/val_preds.csv`（会自动创建目录）。
- 按癌种 KFold（--kfold_cancer）：
  - 统一在 artifacts_dir 的同级目录创建 `kfold_val/`；每一折的产物位于 `kfold_val/fold_{i}/`：
    - 预处理器产物（与单次训练一致）
    - 最优模型 `model.pt` 与 `meta.json`（包含本折的 train/val 划分与最佳指标）
  - 汇总表：`kfold_val/kfold_summary.csv`，包含每折的 `fold, best_epoch, auroc, auprc, accuracy, macro_f1, n_train, n_val`，并在日志中打印均值概览。
- LOCO 留一癌种（--leave_one_cancer_out）：
  - 在 artifacts_dir 的父目录创建 `loco_eval/`，并为每个癌种建立子目录 `loco_eval/{CancerType}/`，内部包含最优 `model.pt`、`meta.json` 与该癌种的划分信息。
  - 汇总表：`loco_eval/loco_summary.csv`，字段与 KFold 类似（含 per-cancer 的最佳指标与样本统计）。

> 详细实现参见 <mcfile name="train.py" path="/root/Project/Spotonco/train.py"></mcfile>

### 5.2 常见命令行示例：双域、KFold、LOCO

- 双域对抗（癌种域 + 切片域）的典型组合：
常用（双域）：
python train.py --train_npz data/data_SC_3331genes/train_data.npz --artifacts_dir test_train0822_SC3331genes/artifacts_dir/ --use_domain_adv_slide 1 --use_domain_adv_cancer 1

```bash
# 仅切片域对抗（Slide-only），常用：lambda_slide=0.3
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_slide_only \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 0 \
  --lambda_slide 0.3 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

# 仅癌种域对抗（Cancer-only），常用：lambda_cancer=0.3
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_cancer_only \
  --use_domain_adv_slide 0 --use_domain_adv_cancer 1 \
  --lambda_cancer 0.3 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

# 双域同时启用（Dual），示例：slide 0.2 + cancer 0.1
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_dual \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 1 \
  --lambda_slide 0.2 --lambda_cancer 0.1 \
  --epochs 100 --early_patience 30 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

```
提示：若未显式传入 `--lambda_slide/--lambda_cancer`，二者会回退到旧字段 `domain_lambda=0.3`（详见 meta.json 中的 cfg）。

- 基于癌种的 KFold（随机生成 K 组“每癌种 1 张验证”组合）：

```bash
# 训练并保存每折产物到 artifacts_dir 的同级目录 kfold_val/
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --kfold_cancer 5 --split_seed 42 \
  --epochs 60 --early_patience 20 --num_workers 10 --device cuda

# 仅查看划分，不训练
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --kfold_cancer 5 --split_seed 42 --split_test_only
```
产物：`../kfold_val/fold_{i}/` 模型与预处理，和 `../kfold_val/kfold_summary.csv` 汇总。

- LOCO 留一癌种评估（逐癌种训练/验证）：

```bash
python STRIDE/train.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --leave_one_cancer_out \
  --epochs 60 --early_patience 20 --num_workers 0 --device cuda
```
产物：`../loco_eval/{CancerType}/` 每癌种一个子目录，含该设定下的最优 `model.pt` 与 `meta.json`，并在 `../loco_eval/loco_summary.csv` 汇总。

---

## 6. 多阶段超参数优化（train_hpo.py）

HPO 已独立到 <mcfile name="train_hpo.py" path="/root/Project/Spotonco/train_hpo.py"></mcfile>，提供统一三阶段流水线与可选多种子复评：
- 阶段：stage1（优化训练稳定性/学习率等）→ stage2（结构/正则）→ stage3（位置编码）
- 复评（重打分）：对指定阶段 Top-K 以多随机种子复训，按 mean_accuracy 重排
- 产物：按阶段保存 trial 结果，最终合并最优配置为 tuning/best_config.json

常用参数（节选）：
- --tune {all,stage1,stage2,stage3}
- --n_trials 每阶段 trial 数
- --rescore_topk K，多种子复评的 Top-K
- --rescore_stages 需要复评的阶段列表（逗号分隔）
- --seeds 多种子列表（逗号分隔）
- 其余训练相关参数（如 --epochs、--early_patience、--model、--lap_pe_dim 等）与 train.py 保持一致

示例：
```bash
# 全流程 + 复评（示例）
python STRIDE/train_hpo.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --tune all \
  --n_trials 30 \
  --rescore_topk 3 \
  --rescore_stages 1 \
  --seeds 42,2023,2024 \
  --epochs 100 --early_patience 30 --num_workers 10 --device cuda

# 仅搜索 stage2
python STRIDE/train_hpo.py \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --tune stage2 \
  --n_trials 24 \
  --device cuda
```

更多细节：<mcfile name="hyperparameter_optimization_plan.md" path="/root/Project/Spotonco/hyperparameter_optimization_plan.md"></mcfile> <mcfile name="train_hpo_refactor_plan.md" path="/root/Project/Spotonco/train_hpo_refactor_plan.md"></mcfile>

---

## 7. 推理与批量推理

单切片推理（infer.py）：
```bash
python STRIDE/infer.py \
  --npz /path/to/slide.npz \
  --artifacts_dir /path/to/artifacts \
  --out_csv /path/to/preds.csv \
  --num_threads 4
```
- 从 `artifacts_dir/meta.json` 读取训练时的 cfg，以保证图构建（KNN、LapPE、是否拼接等）与训练一致。
- 支持两类输入：
  - 单切片 NPZ：包含 X, xy, gene_names（以及可选 barcodes、y、sample_id）。
  - 数据集 NPZ：包含 Xs/xys（多张切片），可通过 `--index` 选择第几张（默认 0）。
- 输出 CSV 列：`spot_idx, x, y, p_tumor`；默认写到 `preds.csv`，或按 `--out_csv` 指定的路径保存（自动创建目录）。
- 解释性输出（默认开启，可用 `--no_explain` 关闭）：
  - `--explain_saliency` 默认开启；`--no_explain` 关闭解释性计算
  - `--explain_method {ig,saliency}`（默认 ig），`--ig_steps`（默认 50）
  - `--gene_attr_out` 指定保存路径；未指定时默认与 `out_csv` 同目录，命名为 `per_slide_gene_saliency_{single|idxK}.csv`
  - 输出仅保存 CSV（两列：`gene, attr`），不再生成 NPZ

批量推理（batch_infer.py）：
```bash
python STRIDE/batch_infer.py \
  --npz_glob '/path/to/infer_*.npz' \
  --artifacts_dir /path/to/artifacts \
  --out_csv /path/to/batch_preds.csv \
  --num_threads 4 --num_workers 0
```
- 读取 `artifacts_dir` 下的预处理器与 `meta.json` 的 cfg，对匹配到的每个 NPZ 执行“读取→预处理→图构建→推理”。
- 性能参数：
  - `--num_threads` 控制 CPU 线程；`--num_workers` 控制 DataLoader 进程数（>0 时在子进程并行读取/预处理/构图）。
- 二值阈值：`--threshold`（默认 0.5），用于将 `p_tumor` 转成 `pred_label`。
- 输出 CSV 列定义（按行拼接所有切片的结果）：
  - `sample_id`（来自单 NPZ；若缺失为 `unknown`）
  - `Barcode`（若 NPZ 含条形码，否则自动回填为 `spot_{i}`）
  - `spot_idx, x, y`（空间位置与索引）
  - `p_tumor`（肿瘤概率，sigmoid 后）
  - `pred_label`（按阈值得到的二值预测）
  - `y_true`（若 NPZ 含标签则输出，否则为 None）
  - `threshold`（写入用于推断的阈值，便于复现）
- 解释性输出（默认开启，可用 `--no_explain` 关闭）：
  - `--explain_saliency` 默认开启；`--no_explain` 关闭
  - `--explain_method {ig,saliency}`，`--ig_steps`（默认 50）
  - `--gene_attr_out_dir` 指定输出目录（未指定时默认写到 `out_csv` 同目录下的 `gene_attr/`）
  - 每张切片保存一份 `per_slide_gene_saliency_{sample_id|文件名}.csv`（两列：`gene, attr`），仅 CSV
- 可视化：默认会自动生成“各切片准确率柱状图”（png），关闭请加 `--no_plot`；输出路径可用 `--plot_out` 自定义，标题可用 `--model_name` 指定（默认从 cfg.model 推断）。

> 详细实现参见 <mcfile name="infer.py" path="/root/Project/Spotonco/infer.py"></mcfile> 与 <mcfile name="batch_infer.py" path="/root/Project/Spotonco/batch_infer.py"></mcfile>

### 7.1 与 evaluate_models/plot_accuracy_bars 的简单联动示例

目标：对比两个（或多个）不同训练产物在同一批量推理集上的表现，并输出汇总指标与可视化。

步骤 1）分别对每个模型产物运行批量推理，得到各自的预测 CSV：
```bash
# 模型 A（例如 GATv2）
python STRIDE/batch_infer.py \
  --npz_glob '/path/to/val_npz_dir/*.npz' \
  --artifacts_dir /path/to/artifacts_gatv2 \
  --out_csv /path/to/preds_gatv2.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0 --model_name GATv2

python batch_infer.py --npz_glob 'data/data_SC_3331genes/val_npz/*.npz' --artifacts_dir test_train0822_SC3331genes/gatv2/artifacts_dir/ --out_csv test_train0822_SC3331genes/gatv2/predictions/batch_pred.csv --num_workers 10 --model_name GATv2

# 模型 B（例如 GraphSAGE）
python STRIDE/batch_infer.py \
  --npz_glob '/path/to/val_npz_dir/*.npz' \
  --artifacts_dir /path/to/artifacts_sage \
  --out_csv /path/to/preds_sage.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0 --model_name SAGE
```

步骤 2）用 evaluate_models 汇总评估并可选绘图：
```bash
python STRIDE/evaluate_models.py \
  --model GATv2=/path/to/preds_gatv2.csv \
  --model SAGE=/path/to/preds_sage.csv \
  --out_dir /path/to/compare \
  --threshold 0.5 \
  --plot --plot_metric accuracy --plot_formats svg,png --plot_dpi 180
```
输出：在 `/path/to/compare/` 目录生成 `gatv2_metrics.csv`、`sage_metrics.csv` 两个指标表，并导出图表：
- comparison_overall_bars（Overall 三子图：accuracy/balanced_accuracy/precision）
- per_slide_bars_accuracy（按切片分组柱状图）
- heatmap_accuracy（按切片热力图）
- distribution_accuracy（箱线图/小提琴图）

步骤 3）（可选）对单个模型绘制“按切片准确率柱状图”：
```bash
python STRIDE/plot_accuracy_bars.py \
  --pred_csv /path/to/preds_gatv2.csv \
  --model_name GATv2 \
  --out_path /path/to/compare/gatv2_per_slide_acc.png \
  --threshold 0.5
```

---

## 8. 可视化（visualize_prediction.py）

对单张或批量切片绘制真实/预测在空间坐标下的散点图，并在标题显示准确率（若存在 y 或可从验证目录解析）。脚本会读取 artifacts_dir/meta.json 的 cfg，确保与训练时的预处理和图构建一致。

- 使用训练 NPZ 中的某张（例如最后一张作为验证）：
```bash
python STRIDE/visualize_prediction.py \
  --train_npz /path/to/...
```

- 使用单切片 NPZ（若 npz 含 y 可直接计算准确率；若不含，可结合验证目录自动加载真实标签）：
```bash
python STRIDE/visualize_prediction.py \
  --npz /path/to/slide.npz \
  --artifacts_dir /path/to/artifacts \
  --val_root /path/to/ST_validation_datasets \
  --xy_cols row col --label_col true_label \
  --out_svg /path/to/vis_slide.svg
```

- 批量处理一组单切片 NPZ（为每张切片导出一个 SVG 到 out_dir）：
```bash
python STRIDE/visualize_prediction.py \
  --npz_glob '/path/to/val_npz/*.npz' \
  --artifacts_dir /path/to/artifacts \
  --out_dir /path/to/visualizations \
  --threshold 0.5
```

参数说明：
- 输入模式（三选一，必须且仅能选择一个）：`--train_npz` | `--npz` | `--npz_glob`
- 基本：`--slide_idx`（用于 train_npz，默认 -1）、`--threshold`（默认 0.5）、`--artifacts_dir`
- 输出：单文件用 `--out_svg`；批量模式用 `--out_dir`
- 可选的真实标签解析（当单切片 npz 不含 y 时）：
  - `--val_root` 指向验证集根目录（每个切片一个子目录，内含 *coordinates.csv 且有标签列）
  - `--xy_cols X_COL Y_COL`（默认 row col）和 `--label_col`（默认 true_label）用于指定坐标与标签列名

可视化将生成两列子图：左侧为 Ground Truth（若存在，否则显示 N/A），右侧为 Prediction（阈值由 --threshold 控制），并在图标题显示准确率和样本数。

---

## 9. 合成/模拟数据（可选）

可用 generate_synthetic_data.py 生成示例数据，快速端到端验证：
```bash
python STRIDE/generate_synthetic_data.py --output_dir Spotonco/synthetic_data --n_genes 2000 --seed 42
python STRIDE/train.py --train_npz Spotonco/synthetic_data/train_data.npz --artifacts_dir Spotonco/artifacts --epochs 2 --early_patience 2 --batch_size_graphs 1
python STRIDE/infer.py --npz Spotonco/synthetic_data/infer_slide_1.npz --artifacts_dir Spotonco/artifacts --out_csv Spotonco/synthetic_data/preds_infer_slide_1.csv
python STRIDE/visualize_prediction.py --train_npz Spotonco/synthetic_data/train_data.npz --artifacts_dir Spotonco/artifacts --slide_idx -1 --out_svg Spotonco/synthetic_data/vis_val_slide.svg
```

---

若需了解模型内部与双域自适应实现，请参考：
- <mcfile name="models.py" path="/root/Project/Spotonco/models.py"></mcfile>
- <mcfile name="Dual-Domain Adversarial Learning.md" path="/root/Project/Spotonco/Dual-Domain Adversarial Learning.md"></mcfile>
- 训练入口与参数：<mcfile name="train.py" path="/root/Project/Spotonco/train.py"></mcfile>
