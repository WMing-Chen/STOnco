# STOnco: Visium 空间转录组 GNN 分类与域自适应

本项目基于 PyTorch Geometric，面向 10x Visium 空间转录组的肿瘤/非肿瘤二分类任务，提供从数据准备 → 训练（含双域自适应）→ 多阶段超参数优化（HPO）→ 推理与可视化的完整工具链。核心模型采用统一的 STOnco_Classifier，支持多种 GNN 主干（GATv2 / GraphSAGE / GCN）、拉普拉斯位置编码（LapPE）、可选的双域对抗（癌种域 + 批次域）、MMD 对齐，以及 generated-support Wasserstein barycenter（WB）多癌种 latent 对齐。

---

## 1. 目录结构与核心脚本

- prepare_data.py：从 CSV（exp + coordinates）构建训练/单样本/验证 NPZ
- preprocessing.py：预处理器与图构建（HVG 选择、标准化、可选 PCA，KNN 图 + 高斯边权，LapPE）
- models.py：STOnco_Classifier（GNN 主干 + 分类头 + 可选双域对抗头）
- train.py：训练与验证（单次、癌种分层、k-fold 按癌种、LOCO 留一癌种评估）
- train_hpo.py：三阶段 HPO 流水线（含多种子复评/重打分）
- infer.py：单切片 NPZ 推理（输出预测 CSV；可选保存基因重要性 CSV）
- batch_infer.py：批量推理，支持多进程预处理；可选保存每张切片的基因重要性 CSV
- visualize_prediction.py：单切片真实/预测空间分布可视化（SVG）
- export_spot_embeddings.py：导出每个 spot 的 embedding，支持 `h`（默认）或 `z_clf`（分类头 latent）；`z64` 作为旧兼容别名保留
- visualize_umap_tsne.py：对导出的 `h_*` / `z_clf_*` embedding 做 UMAP + t-SNE 可视化（SVG）；兼容读取历史 `z64_*`
- evaluate_embedding_mixing.py：在 embedding / UMAP / t-SNE 空间上计算 `iLISI` / `cLISI`，输出混合评估汇总 CSV，并可选导出每个 spot 的局部 LISI
- run_embedding_analysis_pipeline.py：一键串联 embedding 导出、UMAP/t-SNE 绘图与 LISI 量化
- analyze_spot_embedding_domains.py：对导出的 `spot_embeddings_*.csv` 计算域均值/方差统计，并生成热图、箱线图、KDE 与二维散点诊断图
- scripts/plot_loco_umap_tsne.sh：批量绘制 LOCO 各癌种 fold 的 UMAP/t-SNE，可选遍历 epoch checkpoints，默认突出留出癌种为三角形
- scripts/eval_visualize_loco.sh：汇总 LOCO per-slide 指标、绘图，并可选生成 best/worst slide 空间预测图
- Dual-Domain Adversarial Learning.md：双域自适应设计说明
- utils.py：模型与元信息存取
- synthetic_data/：示例合成/模拟数据（可选）

---

## 1.1 核心模块与关键函数速览

- `stonco/utils/prepare_data.py`
  - `_read_expression`：读取 `*_exp.csv`，校验 Barcode 列并将基因表达转为数值矩阵。
  - `_read_coords`：读取 `*_coordinates.csv`，解析坐标与标签列（数值/字符标签均可）。
  - `build_train_npz`：遍历训练目录，统一基因集合并生成 `train_data.npz`。
  - `build_single_npz`：将单张切片 CSV 生成独立 NPZ（含 `sample_id`）。
  - `build_val_npz`：为验证目录下每个切片生成独立 NPZ。
- `stonco/utils/preprocessing.py`
  - `Preprocessor.fit/transform`：HVG 选择、标准化、可选 PCA。
  - `GraphBuilder.build_knn`：基于坐标构建 KNN 图与高斯边权。
  - `GraphBuilder.lap_pe`：生成 Laplacian 位置编码并可拼接到节点特征。
- `stonco/core/train.py`
  - `prepare_graphs`：加载训练 NPZ，构图并进行癌种分层/划分。
  - `train_and_validate`：核心训练循环，支持双域对抗、MMD/WB 对齐与指标评估。
  - `run_single_training` / `run_kfold_training` / `run_loco_training`：三种训练模式入口。
- `stonco/core/wb_potentials.py`
  - `GeneratedSupportMap`：训练阶段生成 `b=T_phi(h)` 的 identity-initialized residual map。
  - `GeneratedSupportWBLoss`：generated-support WB loss helper，支持 `euclidean_pairwise`、`dual_potential` 与 `sinkhorn_divergence`。
- `stonco/core/infer.py`
  - `InferenceEngine`：封装预处理、构图、预测与基因重要性计算。
- `stonco/core/batch_infer.py`
  - `SlideNPZDataset`：批量加载 NPZ 并在 DataLoader 中并行预处理。

---

## 2. 环境依赖与安装

- Python 3.12（已测）
- PyTorch（示例：2.6.0+cu124）
- PyTorch Geometric 2.6.1（以及 pyg-lib、torch-scatter、torch-sparse、torch-cluster）
- 其他：scikit-learn、scipy、pandas、joblib、matplotlib、umap-learn（UMAP 可视化）

安装示例（请按你本机 PyTorch/CUDA 版本替换链接与版本号）：

```bash
pip install scikit-learn pandas scipy joblib matplotlib umap-learn
pip install --default-timeout=600 \
  torch_scatter torch_sparse torch_cluster pyg-lib \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric==2.6.1
```

## 快速开始

一键式示例流程：从原始 CSV 准备数据 → 训练（示例使用双域对抗默认配置）→ 单/批量推理 → 可视化 →（可选）导出 embedding、做 UMAP/t-SNE，并进一步做 mixing/LISI 量化。

```bash
# 0) 路径约定（请按需修改）
DATA_ROOT=/path/to/dataset
ARTIFACTS=/path/to/artifacts
VAL_NPZ_DIR=$DATA_ROOT/val_npz
PRED_CSV=/path/to/preds.csv

# 1) 数据准备
# 注：每个切片子目录需包含 3 个 CSV：`*_exp.csv`、`*_coordinates.csv`、`*_image_features.csv`（`Barcode` + 2048 维特征）。
# 若某些 spot 在 image_features.csv 中缺失，会自动填充为 0，并记录 `img_mask=0`（同时打印 warning）。
# 1.1 训练集（多切片）NPZ
python -m stonco.utils.prepare_data build-train-npz \
  --train_dir $DATA_ROOT/ST_train_datasets \
  --out_npz $DATA_ROOT/train_data.npz \
  --xy_cols row col \
  --label_col true_label

# 1.2 验证集目录转为一组单切片 NPZ
python -m stonco.utils.prepare_data build-val-npz \
  --val_dir $DATA_ROOT/ST_validation_datasets \
  --out_dir $VAL_NPZ_DIR \
  --xy_cols row col \
  --label_col true_label

# 2) 训练（示例显式启用双域对抗；默认仅启用 GNN + 分类头）
python -m stonco.core.train \
  --train_npz $DATA_ROOT/train_data.npz \
  --artifacts_dir $ARTIFACTS \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 1 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2 \
  --save_loss_components 1
# （可选）启用图像特征早期融合（方案1）：在训练命令后追加
#   --use_image_features 1 --img_use_pca 1 --img_pca_dim 256
# （可选）启用学习率调度：默认 `--lr_scheduler none`
#   --lr_scheduler warmup_cosine --lr_warmup_epochs 10 --min_lr_ratio 0.01
# （可选）使用 plateau：建议关闭早停或显著增大 early_patience
#   --lr_scheduler plateau --plateau_metric val_accuracy --plateau_factor 0.5 --plateau_patience 10 --early_patience 0

# 3) 单切片推理（对验证集中的任意一张）
# 先任选一个 NPZ（例如第一张）
SAMPLE_NPZ=$(ls $VAL_NPZ_DIR/*.npz | head -n 1)
python -m stonco.core.infer \
  --npz "$SAMPLE_NPZ" \
  --artifacts_dir $ARTIFACTS \
  --out_csv $PRED_CSV \
  --num_threads 4
# 将在与 out_csv 同目录生成 per_slide_gene_saliency_{...}.csv（默认解释开启）

# 4) 批量推理（对整个验证集目录）
python -m stonco.core.batch_infer \
  --npz_glob "$VAL_NPZ_DIR/*.npz" \
  --artifacts_dir $ARTIFACTS \
  --out_csv ${PRED_CSV%.*}_batch.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0
# 将在 out_csv 同目录下创建 gene_attr/ 并为每张切片保存 per_slide_gene_saliency_{sample_id}.npz 与 .csv

# 5) 可视化（单张或批量）
# 5.1 单张：若单 NPZ 含 y，可直接绘图并显示准确率
python -m stonco.utils.visualize_prediction \
  --npz "$SAMPLE_NPZ" \
  --artifacts_dir $ARTIFACTS \
  --out_svg ${PRED_CSV%.*}_vis.svg

# 5.2 使用训练 NPZ 的最后一张（常作为验证）进行可视化
python -m stonco.utils.visualize_prediction \
  --train_npz $DATA_ROOT/train_data.npz \
  --artifacts_dir $ARTIFACTS \
  --slide_idx -1 \
  --out_svg ${PRED_CSV%.*}_vis_val.svg

# 6) 导出 spot embedding（默认 `h`）并做 UMAP + t-SNE（可选）
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir $ARTIFACTS \
  --npz_glob "$VAL_NPZ_DIR/*.npz" \
  --out_csv $ARTIFACTS/spot_embeddings_val_npz.csv \
  --embed_source h
python -m stonco.utils.visualize_umap_tsne \
  --embeddings_csv $ARTIFACTS/spot_embeddings_val_npz.csv \
  --out_dir $ARTIFACTS/embedding_plots \
  --max_points 50000 \
  --seed 42 \
  --color_cols sample_id tumor_label batch_id \
  --embed_source h

# 7) 计算 embedding / UMAP 空间上的 mixing 指标（可选）
python -m stonco.utils.evaluate_embedding_mixing \
  --embeddings_csv $ARTIFACTS/spot_embeddings_val_npz.csv \
  --out_csv $ARTIFACTS/embedding_plots/mixing_metrics_h.csv \
  --embed_source h \
  --spaces embedding umap \
  --group_cols sample_id batch_id tumor_label \
  --group_roles sample_id:integration batch_id:integration tumor_label:conservation \
  --k_values 15 30 50

# 8) 或者直接一条命令串联导出 + 可视化 + LISI 评估（可选）
python -m stonco.utils.run_embedding_analysis_pipeline \
  --artifacts_dir $ARTIFACTS \
  --out_dir $ARTIFACTS/embedding_analysis \
  --npz_glob "$VAL_NPZ_DIR/*.npz" \
  --embed_source h \
  --group_cols sample_id batch_id tumor_label \
  --group_roles sample_id:integration batch_id:integration tumor_label:conservation \
  --metric_spaces embedding umap \
  --k_values 15 30 50 \
  --color_cols sample_id batch_id tumor_label
```

默认参数要点：
- `visualize_umap_tsne.py` 默认仍会生成按 `tumor_label`、`batch_id`、`cancer_type` 上色的图；若传 `--color_cols ...`，则改为按指定列生成
- `visualize_umap_tsne.py` / `run_embedding_analysis_pipeline.py` 可选 `--highlight_col` + `--highlight_values`，把指定 cancer/sample/batch 的 spot 画成三角形；该功能只影响 SVG 点形状，不影响 embedding、coords 或 LISI 结果
- `evaluate_embedding_mixing.py` 默认只评估 `embedding` 空间；若要把 UMAP / t-SNE 作为辅助指标，需显式传 `--spaces embedding umap` 或 `--spaces embedding umap tsne`
- `run_embedding_analysis_pipeline.py` 默认会先导出 embedding，再保存降维坐标 CSV，最后输出 LISI 汇总 CSV；默认指标空间是 `embedding umap`
- prepare_data：`--xy_cols row col`、`--label_col true_label` 为默认；build-single-npz 若不传 `--sample_id`，将用坐标文件上一级目录名。
- 训练（train.py）：
  - 设备：默认自动检测 CUDA；线程：`--num_threads` 不传则由 PyTorch 默认；DataLoader `--num_workers` 默认 0。
  - 学习率：`--lr` / `--weight_decay` 保留为全局默认；可用 `--gnn_lr`、`--clf_lr`、`--dom_lr`、`--wb_support_lr` 以及对应 `*_weight_decay` 分别覆盖 GNN、分类头、域对抗 head、WB support map
  - 学习率调度：`--lr_scheduler` 默认 `none`；主 optimizer 使用 PyTorch param groups，scheduler 采用统一节奏、分组 base lr
    - `none`：各参数组固定使用自己的 base lr
    - `linear`：按总 step 线性衰减到各自 `base_lr * min_lr_ratio`
    - `cosine`：按总 step 余弦衰减到各自 `base_lr * min_lr_ratio`
    - `warmup_cosine`：各组先 warmup 到自己的 base lr，再做 cosine decay
    - `plateau`：按验证指标停滞自动降 lr，各组使用自己的 `base_lr * min_lr_ratio` 作为下限
  - scheduler 相关默认值：
    - `--lr_warmup_epochs 10`
    - `--min_lr_ratio 0.01`
    - `--plateau_metric val_accuracy`
    - `--plateau_factor 0.5`
    - `--plateau_patience 10`
    - `--plateau_threshold 1e-4`
    - `--plateau_cooldown 0`
  - `plateau_metric` 可选：
    - `val_accuracy`
    - `val_avg_total_loss`
    - `val_macro_f1`
    - `val_auroc`
    - `val_auprc`
  - 若使用 `--lr_scheduler plateau`：
    - 推荐关闭早停：`--early_patience 0`
    - 若不关闭，需显著增大 `--early_patience`，否则 plateau 刚降 lr 训练就可能被 early stopping 提前终止
  - 双域对抗：细粒度开关 `--use_domain_adv_slide/--use_domain_adv_cancer` 可单独控制；默认两者均关闭（slide 对应 batch 域）。如需启用，显式传入 `--use_domain_adv_slide 1` 和/或 `--use_domain_adv_cancer 1`。
  - alpha（域 loss 权重）：`--lambda_slide/--lambda_cancer`（默认 **1.0/1.0**）。
  - beta（GRL 对抗强度）：由 `--grl_beta_mode` 控制（默认 `dann`）
    - `dann`：支持两路独立 `delay_epochs`；delay 前 `beta=0`，之后在剩余训练过程上重新归一化并走完整 DANN 曲线。核心参数：`--grl_beta_slide_target/--grl_beta_cancer_target/--grl_beta_gamma`（默认 **1.0/0.5/10**）以及 `--grl_beta_slide_delay_epochs/--grl_beta_cancer_delay_epochs`（默认 **1/3**）
    - `constant`：全程恒定 beta=`*_target`（忽略 `--grl_beta_gamma` 与 4 个 `*_epochs`）
    - `linear`：delay 前 `beta=0`，delay 后线性升到 `*_target`；默认 slide 为 **delay=1, warmup=8**，cancer 为 **delay=3, warmup=12**
  - Wasserstein barycenter 对齐（可选）：
    - `--use_wb_align 1` 启用 generated-support WB；推荐与 `--use_mmd 0` 搭配作为 MMD 替代实验
    - `--wb_loss_type euclidean_pairwise` 是低计算量默认起点；`dual_potential` 使用 neural dual surrogate；`sinkhorn_divergence` 使用 batch 内 debiased Sinkhorn divergence，更接近标准 entropy-regularized OT，但计算量更高
    - WB 仍让分类 head 使用 `h`，训练阶段额外学习 `b=T_phi(h)` 并通过 anchor 把对齐效果传回 `h`
    - 推荐配合 `--sampler_mode cancer_balanced --sampler_k_cancers 4 --sampler_m_per_cancer 2/3`
  - 外部验证：`--val_sample_dir` 指定外部验证 NPZ 目录（单切片），验证指标会合并计算。
  - Loss 组件：`--save_loss_components 1`（默认开启）会保存 `loss_components.csv` 到 artifacts_dir。
  - 解释性：默认开启，`--explain_method ig`，`--ig_steps 50`；训练结束会保存 `per_gene_saliency.csv` 到 artifacts_dir。
- 推理（infer.py/batch_infer.py）：
  - 解释性：默认开启；单张保存 CSV（`gene, attr`），可用 `--gene_attr_out` 指定路径；批量保存每张切片的 `.npz` 与 `.csv`，用 `--gene_attr_out_dir` 指定目录（默认为 `out_csv` 同目录下的 `gene_attr/`）。
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
  - （可选，图像特征）X_imgs/img_masks/img_feature_names：
    - X_imgs：长度 N_slide 的列表，每项 (n_spots_i, 2048) float32
    - img_masks：长度 N_slide 的列表，每项 (n_spots_i,) uint8，取值 {0,1}
    - img_feature_names：长度 2048 的列表（训练/推理必须一致）

- 单切片 NPZ（推理或验证单张）：
  - X：形状 (n_spots, n_genes)
  - xy：形状 (n_spots, 2)
  - gene_names：长度 n_genes 的列表
  - barcodes：该切片的条形码（字符串数组）
  - sample_id：样本/切片 ID（字符串）
  - 可选 y：形状 (n_spots,)，取值 {0,1}
  - （可选，图像特征）X_img/img_mask/img_feature_names：
    - X_img：(n_spots, 2048) float32
    - img_mask：(n_spots,) uint8，取值 {0,1}
    - img_feature_names：长度 2048 的列表（与训练产物中的 img_feature_names.txt 一致）

说明：
- 本次更新将图像特征作为“早期融合”节点特征的一部分：`x = [Xp_gene, Xp_img, img_mask, (LapPE)]`。
- 是否启用由训练时 `meta.json:cfg.use_image_features` 决定；推理/批推理/可视化/导出 embedding 会自动读取 cfg 保持一致。

从 CSV 构建 NPZ 的约定（见 prepare_data.py）：
- exp.csv：首列必须为 Barcode，其余列为基因（列名即基因名）；数值列会转为浮点，缺失置 0。
- coordinates.csv：必须包含 Barcode 列（大小写不敏感匹配），以及指定的 x/y 列（默认 row、col）。
- image_features.csv：必须包含 Barcode 列，且除 Barcode 外恰好 2048 个特征列；特征值需为有限数（NaN/inf 会报错），列名/顺序需在全体切片一致。
- 构建时按 Barcode 对齐并保持坐标文件的顺序：gene 用 `coord ∩ exp`；image 对 `barcodes_used` 做 reindex，缺失则填 0 并记录 `img_mask=0`。

---

## 4. 数据准备（prepare_data.py）

具体命令：
python -m stonco.utils.prepare_data build-train-npz --train_dir data/data_SC_3331genes/ST_train_datasets --out_npz data/data_SC_3331genes/train_data.npz
python -m stonco.utils.prepare_data build-val-npz --val_dir data/data_SC_3331genes/ST_validation_datasets --out_dir data/data_SC_3331genes/val_npz/

常用子命令：

- 扫描训练目录构建多切片训练集 NPZ：
```bash
python -m stonco.utils.prepare_data build-train-npz \
  --train_dir /path/to/train_slides \
  --out_npz /path/to/train_data.npz \
  --xy_cols row col \
  --label_col true_label
```

- 由一对 CSV 构建单样本 NPZ（可用于推理或验证单张）：
```bash
python -m stonco.utils.prepare_data build-single-npz \
  --exp_csv /path/slide_exp.csv \
  --coord_csv /path/slide_coordinates.csv \
  --out_npz /path/slide.npz \
  --xy_cols row col \
  --label_col true_label \
  --sample_id SLIDE_ID
```

- 为验证集目录中的每张切片各自生成一个 NPZ：
```bash
python -m stonco.utils.prepare_data build-val-npz \
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
    xxx_image_features.csv
  SlideB/
    yyy_coordinates.csv
    yyy_exp.csv
    yyy_image_features.csv
  ...
```
- 验证目录（用于 build-val-npz）：
```
/val_slides/
  Slide1/
    *_coordinates.csv
    *_exp.csv
    *_image_features.csv
  Slide2/
    *_coordinates.csv
    *_exp.csv
    *_image_features.csv
  ...
```
注意：每个切片子目录中脚本会按后缀严格检测 `*coordinates.csv` / `*exp.csv` / `*image_features.csv`（大小写不敏感）。每一类必须且只能匹配到 1 个文件，否则会报错（build-train-npz）或 Warning 跳过该切片（build-val-npz）。

### 4.2 CSV 字段要求与校验

- 表达矩阵（exp.csv）：
  - 第 1 列必须为 `Barcode`（大小写不敏感，但必须是第 1 列），其余列为基因（列名即基因名）。
  - 所有基因表达值会转换为浮点，无法解析的值记为 0.0。
- 坐标（coordinates.csv）：
  - 必须包含 `Barcode` 列（大小写不敏感匹配），以及指定的 x/y 列（默认 `row`、`col`）。
  - x/y 列会强制转为数值，并丢弃坐标缺失的行。
  - 训练时要求存在标签列（默认 `true_label`）。标签可以是：
    - 数值型：会被强制为二值（非 0 视为 1）。
    - 字符型：支持映射 `tumor`→1、`normal`→0、`mal`→1、`nmal`→0；未知值会提示 Warning 并映射为 0。
- 图像特征（image_features.csv）：
  - 必须包含 `Barcode` 列。
  - 除 `Barcode` 外必须有且仅有 2048 个特征列；特征列名/顺序必须在所有切片之间一致（训练/推理也必须一致）。
  - `Barcode` 不允许重复；特征值不允许出现 NaN/inf（数据准备阶段会直接报错）。
  - 对齐时允许部分 spot 缺失图像特征：缺失行会填充为 0，并写入 `img_mask=0`（同时打印 warning）。

### 4.3 条形码对齐与顺序

- gene（exp/coordinates）：以两个 CSV 的 `Barcode` 交集为准做对齐；若交集为空会报错。
- 对齐后严格“按坐标文件的顺序”排列 spots；表达矩阵按该顺序重建。
- image（image_features）：对 `barcodes_used` 做 reindex/左连接：
  - 缺失的 barcode：该行图像向量填 0，`img_mask=0`
  - 匹配到的 barcode：写入 2048 维向量，`img_mask=1`

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
  - 图像特征（新增）：
    - `X_imgs`: List[np.ndarray]，每项 (n_spots_i, 2048) float32
    - `img_masks`: List[np.ndarray]，每项 (n_spots_i,) uint8（0/1）
    - `img_feature_names`: List[str]，长度 2048（全体切片一致，存 1 份）
- 单切片 NPZ（build-single-npz）：
  - `X` (n_spots, n_genes) float32，`xy` (n_spots,2) float32
  - `gene_names` List[str]，`barcodes` (n_spots,) U64
  - `sample_id` str（默认从坐标文件所在文件夹名推断）
  - 可选 `y` (n_spots,) int64
  - 图像特征（新增）：
    - `X_img` (n_spots, 2048) float32
    - `img_mask` (n_spots,) uint8（0/1）
    - `img_feature_names` List[str]，长度 2048
- 验证集 NPZ（build-val-npz）：
  - 为每个子目录分别生成一个单切片 NPZ，文件名为 `<子目录名>.npz`。

### 4.6 常用参数说明补充

- `--xy_cols X_COL Y_COL`：指定坐标文件中的 x/y 列名（默认 `row col`）。
- `--label_col COL`：指定标签列名（训练必需，默认 `true_label`）。
- `--sample_id`（仅 build-single-npz）：不传则自动使用坐标文件上一级目录名。

### 4.7 错误处理与提示

- build-train-npz：
  - 每个切片子目录必须且只能匹配到 1 个 `*coordinates.csv`、1 个 `*exp.csv`、1 个 `*image_features.csv`；否则会报错并中止。
  - 若标签列缺失，会报错并中止（训练需要完整标签）。
  - 若 `Barcode` 交集为空，报错并中止。
  - `image_features.csv` 若出现 `Barcode` 重复、特征列数≠2048、特征值出现 NaN/inf、或跨切片特征列名/顺序不一致，会报错并中止。
- build-val-npz：
  - 若某子目录缺少上述任一类文件，或文件匹配不唯一，会 Warning 并跳过该切片，不中止整个流程。

---

## 5. 训练（train.py）

最小示例：
```bash
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --val_sample_dir /path/to/val_npz_dir \
  --epochs 100 \
  --batch_size_graphs 2
```

主要参数（部分）：
- 基本：
  - --train_npz 必需；--artifacts_dir 默认为 artifacts
  - --epochs，--early_patience（默认 0，<=0 表示关闭早停），--batch_size_graphs
  - --device cpu/cuda（不指定自动检测,默认cuda）
  - --num_threads 控制 PyTorch 线程数（CPU 时限制核心占用）
  - --num_workers DataLoader 进程数（验证集内部自动≤2）
- 模型与结构：
  - --model {gatv2,sage,gcn}（默认 gatv2），--heads（仅 gatv2）
  - --GNN_hidden，--num_layers，--dropout
  - Dropout：`--dropout` 保留为全局默认 dropout（默认 0.3）；可用 `--gnn_dropout`、`--clf_dropout`、`--dom_dropout` 分别覆盖 GNN backbone、分类头、域对抗 head。若模块级参数不传，则回退使用 `--dropout`
    - `--gnn_dropout` 作用于 GNN 每层输出后的 dropout；当 `--model gatv2` 时也传给 `GATv2Conv(..., dropout=...)`
    - `--clf_dropout` 作用于分类头每个隐藏层后的 dropout；分类头不再使用历史写死的 0.1
    - `--dom_dropout` 作用于 slide/cancer 两个域对抗 head 的 `Linear -> ReLU -> Dropout -> Linear` 中间 dropout
  - `--GNN_hidden` 支持单个整数或逗号分隔列表，默认 `256,128,64`
  - 例：
    - `--GNN_hidden 128` 且 `--num_layers 3` -> 实际为 `[128,128,128]`
    - `--GNN_hidden 128,45,64` -> 实际为 3 层，逐层 hidden 为 `[128,45,64]`
  - 若 `--GNN_hidden` 是列表，默认以列表长度作为有效 `num_layers`
  - 若显式传入的 `--num_layers` 与 `len(GNN_hidden)` 不一致：直接报错
  - 旧参数 `--hidden` 仅保留兼容，不要与 `--GNN_hidden` 同时传入；同传会直接报错
  - 当 `--model gatv2` 时，每层实际输出维度为 `GNN_hidden[i] * heads`
  - 当 `--model gcn/sage` 时，每层实际输出维度为 `GNN_hidden[i]`
  - 分类头：--clf_hidden（默认 256,128,64；支持任意长度的正整数列表，最后一维即分类头 latent 维度）
  - 例：
    - `--clf_hidden 256,128,64` -> 3 层隐藏层，latent 维度为 64
    - `--clf_hidden 128,64` -> 2 层隐藏层，latent 维度为 64
    - `--clf_hidden 96` -> 1 层隐藏层，latent 维度为 96
    - `--clf_hidden 256,128,32` -> 3 层隐藏层，latent 维度为 32
  - 训练产物会在 `meta.json:cfg` 中保存：
    - `clf_hidden`：分类头各隐藏层宽度列表
    - `clf_latent_dim`：分类头最后一层隐藏表示维度，即 `clf_hidden[-1]`
  - 域头：--dom_hidden（默认 64；作用于 batch/cancer 两个域头）
  - LapPE：--lap_pe_dim（>0 启用，默认 0）、--concat_lap_pe {0,1}（默认 0，需显式传 1 才会拼接）、--lap_pe_use_gaussian {0,1}
- 预处理：
  - --use_pca {0,1}（可选 PCA；默认关闭）
  - --n_hvg N 或 'all'（默认 'all' 表示使用所有基因）
  - 图像特征早期融合（方案1，可选）：
    - --use_image_features {0,1}（默认 0，保持 gene-only 行为；启用后节点输入为 `gene → image → img_mask → (LapPE)`）
    - --img_use_pca {0,1}（默认 1，对 2048 维图像特征做 PCA 降维）
    - --img_pca_dim（默认 256，仅当 --img_use_pca 1 生效；若有效 spot 数（img_mask==1）小于该值会直接报错）
- 训练 sampler / 子图：
  - `--sampler_mode {random,cancer_balanced,cancer_balanced_subgraph}`（默认 `random`）
  - `--sampler_k_cancers`：每个 batch 优先采样的癌种数 `K`
  - `--sampler_m_per_cancer`：每个癌种优先采样的父切片数 `M`
  - `--sampler_enforce_distinct_batch {0,1}`：同癌种内尽量约束不同 `bat_dom`（默认 1）
  - `--subgraph_mode {off,static,online}`（默认 `off`）
  - `--subgraph_target_spots`：静态子图目标 spot 数，默认 1000
  - `--subgraph_min_spots`：静态子图最小 spot 数，默认 300
  - `cancer_balanced_subgraph` 是兼容别名：内部会归一化为 `sampler_mode=cancer_balanced + subgraph_mode=static`
  - `subgraph_mode=online` 目前保留接口但尚未实现；当前应使用 `static`
  - 当 `batch_size_graphs < 4` 或有效训练图数不足时，会自动回退到普通随机 shuffle
  - 若未显式给 `K/M`，会按 `batch_size_graphs` 自动推断：
    - `4 -> K=2, M=2`
    - `6 -> K=3, M=2`
    - `8 -> K=4, M=2`
    - `>=12` 且可被 3 整除时优先用 `M=3`
- 优化器与分模块学习率：
  - `--lr`、`--weight_decay`：全局默认学习率与权重衰减，旧命令保持兼容
  - 模块级 lr 覆盖：`--gnn_lr`、`--clf_lr`、`--dom_lr`、`--wb_support_lr`；不传时分别回退到 `--lr`
  - 模块级 weight decay 覆盖：`--gnn_weight_decay`、`--clf_weight_decay`、`--dom_weight_decay`、`--wb_support_weight_decay`；不传时分别回退到 `--weight_decay`
  - 主 optimizer 使用 PyTorch 原生 param groups：`gnn`、`clf`、`dom`、`wb_support`、`wb_main`；其中 `wb_main` 目前主要对应 WB 的 `state_direction`，跟随 `wb_support_lr/wb_support_weight_decay`
  - WB potential 使用独立 optimizer：`--wb_potential_lr` 默认 `1e-4`，`--wb_potential_weight_decay` 默认 `0.0`；第一版不跟随主 scheduler
- 学习率调度：
  - `--lr_scheduler {none,linear,cosine,warmup_cosine,plateau}`（默认 `none`）
  - `--lr_warmup_epochs`：仅 `warmup_cosine` 使用，默认 10
  - `--min_lr_ratio`：最小学习率比例，默认 0.01
  - `--plateau_metric {val_accuracy,val_avg_total_loss,val_macro_f1,val_auroc,val_auprc}`（默认 `val_accuracy`）
  - `--plateau_factor`：触发后各主参数组 `lr = lr * factor`，默认 0.5
  - `--plateau_patience`：默认 10
  - `--plateau_threshold`：默认 `1e-4`
  - `--plateau_cooldown`：默认 0
  - 语义：
    - `linear`：各参数组从自己的 base lr 线性衰减到 `base_lr * min_lr_ratio`
    - `cosine`：各参数组从自己的 base lr 余弦衰减到 `base_lr * min_lr_ratio`
    - `warmup_cosine`：各参数组先 warmup 到自己的 base lr，后续按同一 cosine 节奏衰减
    - `plateau`：每轮验证后按监控指标调用 `ReduceLROnPlateau.step(metric)`，各参数组使用自己的 `base_lr * min_lr_ratio` 作为 `min_lr`
  - `plateau` 与 early stopping 可能相互竞争，推荐优先关闭早停使用；若监控指标为 `NaN`，该轮会跳过 `plateau.step`
- 域自适应（双域，对抗式）：
  - 细粒度：--use_domain_adv_slide {0,1}（batch 域，默认 0），--use_domain_adv_cancer {0,1}（默认 0）
  - alpha（域 loss 权重）：--lambda_slide（batch 域），--lambda_cancer（cancer 域）（默认 1.0/1.0）
  - beta（GRL 对抗强度）：--grl_beta_mode {dann,constant,linear}（默认 dann）
    - dann：delay-aware DANN schedule
      - --grl_beta_slide_target（默认 1.0）
      - --grl_beta_cancer_target（默认 0.5）
      - --grl_beta_gamma（默认 10）
      - --grl_beta_slide_delay_epochs（默认 1）
      - --grl_beta_cancer_delay_epochs（默认 3）
      - 行为：delay 前 `beta=0`，delay 后在剩余训练过程上重新归一化并走完整 DANN 曲线
    - constant：全程恒定 beta=`*_target`（忽略 --grl_beta_gamma 和 4 个 `*_epochs`）
    - linear：两路独立的 delay + warm-up
      - --grl_beta_slide_delay_epochs（默认 1）
      - --grl_beta_slide_warmup_epochs（默认 8）
      - --grl_beta_cancer_delay_epochs（默认 3）
      - --grl_beta_cancer_warmup_epochs（默认 12）
      - 行为：delay 前 `beta=0`，delay 后线性升到 `*_target`
- Wasserstein barycenter 对齐（可选，作用于 GNN latent `h`）：
  - `--use_wb_align {0,1}`：启用 generated-support WB 对齐，默认 0
  - `--wb_loss_type {euclidean_pairwise,dual_potential,sinkhorn_divergence}`：WB loss 类型；默认 `euclidean_pairwise`
    - `euclidean_pairwise`：用 generated support 上的两两欧氏距离 / energy-distance surrogate 加 potential critic，计算量较低
    - `dual_potential`：source-side `f_k(h)` + support-side `g_k(b)` 的 dual-potential WB objective，计算量较高
    - `sinkhorn_divergence`：对每个 active cancer 的 `h` 分布与 generated support `b=T_phi(h)` 计算 debiased Sinkhorn divergence；不训练跨 batch potential bank，更接近标准 entropy-regularized OT，但比 `euclidean_pairwise` 更耗时
  - `--wb_support_mode generated_support`：第一版仅支持 `b=T_phi(h)`；不使用 pooled support、memory bank 或独立 finite atoms
  - `--lambda_wb`、`--wb_warmup_epochs`、`--wb_ramp_epochs`：WB 总权重与 warmup/ramp 调度
  - `--wb_anchor_weight`：`h` 与 `b` 的 anchor 权重；anchor 在训练循环中单独加一次，避免 generated support 与分类表征脱耦
  - `--wb_support_hidden`、`--wb_support_dropout`：`GeneratedSupportMap` 结构参数；若不显式传 `--wb_support_dropout`，默认跟随全局 `--dropout`，显式传入时覆盖该值
  - `--wb_potential_hidden`、`--wb_potential_lr`、`--wb_potential_weight_decay`、`--wb_pot_every_n_steps`：potential 网络、独立 optimizer 与交替更新频率；WB potential MLP 当前不使用 dropout
  - `--wb_spots_per_graph`：每张 graph/slide 最多抽取多少 spot 参与 WB，默认 64
  - `--wb_spots_per_cancer`：每个 cancer 的二级 spot cap，默认 0 表示关闭
  - `--wb_support_size`：`dual_potential` 中从 selected `b` 再抽取的 support-side 点数，默认 128
  - `--wb_min_cancers`、`--wb_min_spots`：计算 WB 的最少 active cancer 数和每癌种最少 spot 数；不足时该 batch 跳过 WB
  - `--wb_regularizer {l2,entropy}`、`--wb_epsilon`：`dual_potential` 中表示 regularizer 设置；`sinkhorn_divergence` 中 `--wb_regularizer` 会被忽略，`--wb_epsilon` 作为 Sinkhorn entropy epsilon
  - `--wb_sinkhorn_iters`：`sinkhorn_divergence` 的 log-domain Sinkhorn 迭代次数，默认 50
  - `--wb_label_balanced_sampling {0,1}`：可选 label-balanced WB 内部抽样，默认关闭
  - `--wb_state_direction {0,1}`、`--wb_state_direction_weight`：可选 tumor-normal shared state direction 约束，默认关闭
  - `--wb_eval_loss {0,1}`：验证阶段是否计算 WB diagnostics，默认关闭；开启会增加验证耗时
  - `--best_metric {val_macro_f1,val_auprc,val_accuracy,val_auroc,val_avg_total_loss}`：最佳 checkpoint 选择指标，默认 `val_macro_f1`
  - 实现细节：
    - 训练推理路径不变：`GNNBackbone -> h -> ClassifierHead`
    - WB 训练阶段额外使用 `GeneratedSupportMap` 得到 `b=T_phi(h)`；`euclidean_pairwise`/`dual_potential` 会交替更新 potentials 与主模型，`sinkhorn_divergence` 则直接在主模型步骤中计算 Sinkhorn loss
    - `sinkhorn_divergence` 不使用 neural potentials；Sinkhorn 内部的 scaling/dual variables 是当前 batch 的数值求解变量，不作为跨 batch 参数保存
    - 若 `use_domain_adv_cancer=0` 但 `use_wb_align=1`，训练仍会构建 `n_domains_cancer` 供 potential bank 使用
    - 如果同时开启 `use_mmd=1` 和 `use_wb_align=1`，代码会打印 warning；主实验建议二者互斥
- 划分/验证：
  - --stratify_by_cancer 按癌种分层（默认启用，按比例划分且每癌种保底 1 张，n=1 仅训练）
  - --no_stratify_by_cancer 关闭分层，使用最后 1 张作为验证
  - --val_ratio 验证集比例（默认 0.2）
  - --kfold_cancer K 基于癌种的 K 组组合评估（按比例分配验证集并随机组合），产物位于 artifacts_dir 的同级目录 kfold_val/fold_{i}/，并写出 kfold_val/kfold_summary.csv
  - --leave_one_cancer_out LOCO 留一癌种评估（每个癌种单独训练与验证）；产物位于 artifacts_dir 的父目录下 loco_eval/{CancerType}/
  - --split_seed 随机种子；--split_test_only 仅打印划分统计不训练
- 其他：
  - --config_json 从 JSON 加载一组超参（支持扁平或 {"cfg": {...}} 格式）
  - --val_sample_dir 外部验证 NPZ 目录（单切片），验证指标与内部验证合并计算
  - --save_loss_components 0/1（默认 1）：保存 Loss 组件曲线 CSV 到 artifacts_dir/loss_components.csv
  - --save_train_curves 0/1（默认 1）：保存 `lr.svg`、`train_loss.svg`、`train_val_loss.svg` 与 `train_val_metrics.svg`
  - --save_last 0/1（默认 1；也兼容不带值的 `--save_last`）：若训练实际跑满 `--epochs`，则用最后一个 epoch 覆盖 `artifacts_dir/model.pt`；同时额外保存最优模型到 `artifacts_dir/model_best.pt`。默认 `--early_patience 0`，因此默认会跑满；若显式开启早停且提前结束，则 `model.pt` 回退为最优模型
  - --save_epoch_checkpoints：额外保存指定 epoch 的模型快照，例如 `--epochs 300 --save_epoch_checkpoints 100 200` 会在训练结束后额外写出 `epoch_checkpoints/epoch_100/` 与 `epoch_checkpoints/epoch_200/`；每个 `epoch_XXX/` 会保存与常规训练目录一致的核心文件布局（`model.pt`、`meta.json`、预处理器文件，以及按该轮截断的曲线/`loss_components.csv`）；若训练因早停未跑到某个请求轮次，会在日志中提示未保存
  - 解释性输出（默认开启，可用 --no_explain 关闭）：--explain_saliency/--no_explain，--explain_method {ig,saliency}（默认 ig），--ig_steps（默认 50）；若开启，将在训练结束后基于 `artifacts_dir/model.pt` 对应的模型计算总体基因重要性并保存 CSV（默认 artifacts_dir/per_gene_saliency.csv）

训练产物（artifacts_dir）：
- 预处理：genes_hvg.txt，scaler.joblib，pca.joblib（若启用）
- 图像预处理（当 --use_image_features 1）：img_feature_names.txt，img_scaler.joblib，img_pca.joblib（若 --img_use_pca 1）
- 模型：`model.pt`（默认保存最后一轮；若 `--save_last 0` 则保存最优模型；若显式开启早停且提前结束，则回退为最优模型）；`model_best.pt`（默认保存最优模型，启用 save_last 时写出）
- WB 训练 artifacts（当 `--use_wb_align 1`）：`wb_support_map_last.pt`、`wb_potentials_last.pt`、`wb_config.json`
  - `sinkhorn_divergence` 不训练跨 batch neural potential bank，但仍保留 `wb_potentials_last.pt` 文件名用于兼容 WB artifact 布局；其中主要记录 WB module 的可保存状态
- 额外轮次快照：若设置 `--save_epoch_checkpoints`，会额外生成自包含的 `epoch_checkpoints/epoch_XXX/` 目录，包含 `model.pt`、`meta.json`、预处理器文件，以及按该轮截断的训练曲线与 `loss_components.csv`（若对应开关启用）
- 元信息：meta.json（含 cfg、best_epoch、train_ids、val_ids、metrics；并追加 last_epoch、last_metrics、completed_full_epochs、saved_checkpoint、saved_epoch_checkpoints、requested_save_epoch_checkpoints）
- GNN 主干配置：新训练产物以 `cfg.GNN_hidden` 为准；旧产物若仅含 `cfg.hidden` 仍可兼容加载
- 可视化：
  - `lr.svg`：单独保存学习率曲线；当前主曲线沿用兼容字段 `lr`，即 `gnn` 参数组 lr
  - `train_loss.svg`：`3x3`，包含训练损失组件和 `train_accuracy`
  - `train_val_loss.svg`：`3x3`，包含训练/验证损失对照和 `accuracy`
  - `train_val_metrics.svg`：`2x2`，包含 `val_accuracy/val_macro_f1/val_auroc/val_auprc`
  - `wb_train_loss.svg`：启用 WB 时额外保存，包含 `avg_total_loss`、`avg_task_loss`、`avg_wb_loss`、`avg_wb_potential_loss`、`avg_wb_dual_obj`、`avg_wb_euclid_pairwise`、`avg_wb_sinkhorn`、`avg_wb_anchor`、`wb_lambda`、active cancers/spots 等 WB diagnostics；图中子图会按当前 loss 类型和可用曲线自动布局
- Loss 组件：`loss_components.csv` 保留 `lr` 列，并新增 `lr_gnn`、`lr_clf`、`lr_dom`、`lr_wb_support`、`lr_wb_potential`；继续保存训练/验证损失及 `val_*` 指标；启用 WB 时额外包含 `avg_wb_loss`、`avg_wb_potential_loss`、`avg_wb_dual_obj`、`avg_wb_euclid_pairwise`、`avg_wb_sinkhorn`、`avg_wb_anchor`、`avg_wb_active_cancers`、`avg_wb_active_spots`、`wb_lambda` 等列；若 `--wb_eval_loss 1`，还会记录对应 validation WB diagnostics，例如 `val_avg_wb_loss` 与 `val_avg_wb_sinkhorn`
- 命名更新：此前文档中的 `Var_risk/var_risk` 已统一更名为 `batch_loss_variance`，含义不变，仍表示每个 epoch 内 mini-batch 总损失的方差。

### 5.1 产物路径与内容补充

- 单次训练（默认模式）：
  - 模型与预处理均保存到指定的 artifacts_dir；meta.json 中的 `cfg` 为实际训练用到的完整配置，可被推理脚本复用。
- 按癌种 KFold（--kfold_cancer）：
  - 统一在 artifacts_dir 的同级目录创建 `kfold_val/`；每一折的产物位于 `kfold_val/fold_{i}/`：
    - 预处理器产物（与单次训练一致）
    - `model.pt`（默认保存最后一轮；若 `--save_last 0` 则保存最优模型；若显式开启早停且提前结束，则回退为最优模型）与 `meta.json`（包含本折的 train/val 划分与 best/last 指标等）
    - `model_best.pt`（默认保存最优模型，启用 save_last 时写出）
  - 汇总表：`kfold_val/kfold_summary.csv`，包含每折的 `fold, best_epoch, auroc, auprc, accuracy, macro_f1, n_train, n_val`，并在日志中打印均值概览。
- LOCO 留一癌种（--leave_one_cancer_out）：
  - 在 artifacts_dir 的父目录创建 `loco_eval/`，并为每个癌种建立子目录 `loco_eval/{CancerType}/`，内部包含 `model.pt`、`meta.json` 与该癌种的划分信息（默认跑满 epochs 并用最后一轮覆盖 `model.pt`，同时保存最优到 `model_best.pt`；若 `--save_last 0` 则 `model.pt` 保存最优模型）。
  - 汇总表：`loco_eval/loco_summary.csv`，字段与 KFold 类似（含 per-cancer 的最佳指标与样本统计）。

> 详细实现参见 `stonco/core/train.py`

### 5.2 常见命令行示例：双域、WB、KFold、LOCO

- 双域对抗（癌种域 + 批次域）的典型组合：
```bash
# 仅批次域对抗（Batch-only），示例：alpha(lambda_slide)=0.3
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_slide_only \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 0 \
  --lambda_slide 0.3 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --GNN_hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

# 仅癌种域对抗（Cancer-only），示例：alpha(lambda_cancer)=0.3
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_cancer_only \
  --use_domain_adv_slide 0 --use_domain_adv_cancer 1 \
  --lambda_cancer 0.3 \
  --epochs 80 --early_patience 20 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --GNN_hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

# 双域同时启用（Dual），示例：batch 0.2 + cancer 0.1
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_dual \
  --use_domain_adv_slide 1 --use_domain_adv_cancer 1 \
  --lambda_slide 0.2 --lambda_cancer 0.1 \
  --epochs 100 --early_patience 30 --batch_size_graphs 2 \
  --model gatv2 --heads 4 --GNN_hidden 128 --num_layers 3 --dropout 0.3 \
  --device cuda

```

启用图像特征早期融合（方案1）时，在上述任意训练命令后追加：
- `--use_image_features 1 --img_use_pca 1 --img_pca_dim 256`
提示：
- 若不显式传 `--GNN_hidden`，当前默认主干宽度为 `256,128,64`
- 若想保留旧版“所有层都是 128”的写法，应改为 `--GNN_hidden 128 --num_layers 3`
- 若想指定逐层不同宽度，可写为 `--GNN_hidden 128,45,64`；此时可不再单独传 `--num_layers`
- 若想拆分不同模块的 dropout，可在全局 `--dropout` 之外追加模块级覆盖，例如 `--dropout 0.3 --gnn_dropout 0.2 --clf_dropout 0.4 --dom_dropout 0.1`；未覆盖的模块继续使用 `--dropout`
- LapPE 默认不计算也不拼接；若需要使用 LapPE，应同时传入正数 `--lap_pe_dim` 和 `--concat_lap_pe 1`，例如 `--lap_pe_dim 16 --concat_lap_pe 1`
- 若想拆分不同模块的学习率，可在全局 `--lr` / `--weight_decay` 之外追加模块级覆盖，例如 `--lr 1e-3 --gnn_lr 5e-4 --clf_lr 1e-3 --dom_lr 2e-4 --wb_support_lr 7e-4`；scheduler 会保持这些组间 base lr 比例
- 域标签从 `data/cancer_sample_labels.csv` 读取：`cancer_type`（cancer 域）与 `Batch_id`（batch 域）；`Batch_id` 缺失时回退为 `slide_id`；训练时按当前 fold/train 出现的类别动态映射到连续索引（K 动态）。
- `--lambda_*` 是 alpha（域 loss 权重），beta（GRL 对抗强度）由 `--grl_beta_mode` 控制：`dann` 为带 delay 的 DANN schedule，`constant` 为全程恒定，`linear` 为显式 delay + 线性 warm-up。
- 域 loss 以 spot-level 计算（对所有 spot 做全局 mean）；域 CE 默认启用 graph-frequency 的 sqrt 反频率 class weight，并做 `clamp(0.5, 5.0)` + mean-normalize 稳定化。

- 对齐项只开 WB、关闭 MMD 与 cancer GRL 的推荐起点：

```bash
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_wb_only \
  --epochs 100 \
  --early_patience 0 \
  --batch_size_graphs 12 \
  --model gatv2 \
  --heads 4 \
  --GNN_hidden 128,32,16 \
  --num_layers 3 \
  --dropout 0.3 \
  --clf_hidden 256,128,64 \
  --use_pca 0 \
  --use_image_features 0 \
  --n_hvg all \
  --lap_pe_dim 16 \
  --concat_lap_pe 1 \
  --use_domain_adv_slide 1 \
  --use_domain_adv_cancer 0 \
  --lambda_slide 0.1 \
  --lambda_cancer 0 \
  --grl_beta_mode constant \
  --grl_beta_slide_target 4.0 \
  --use_mmd 0 \
  --use_wb_align 1 \
  --wb_loss_type euclidean_pairwise \
  --wb_support_mode generated_support \
  --lambda_wb 0.005 \
  --wb_warmup_epochs 10 \
  --wb_ramp_epochs 20 \
  --wb_anchor_weight 0.5 \
  --wb_spots_per_graph 64 \
  --wb_spots_per_cancer 0 \
  --wb_support_size 128 \
  --wb_min_cancers 2 \
  --wb_min_spots 2 \
  --wb_label_balanced_sampling 0 \
  --wb_state_direction 0 \
  --wb_eval_loss 0 \
  --best_metric val_macro_f1 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 3 \
  --sampler_enforce_distinct_batch 1 \
  --subgraph_mode off \
  --save_train_curves 1 \
  --save_loss_components 1
```

`dual_potential` 版本更接近正规 dual-potential WB objective，但计算量更高。可在上面命令基础上改为：

```bash
--wb_loss_type dual_potential \
--lambda_wb 0.001 \
--wb_spots_per_graph 32 \
--wb_support_size 64 \
--wb_regularizer l2 \
--wb_epsilon 0.1
```

`sinkhorn_divergence` 版本使用 batch 内 debiased Sinkhorn divergence，可作为更接近标准 OT 的 WB 主损失。可在上面命令基础上改为：

```bash
--wb_loss_type sinkhorn_divergence \
--lambda_wb 0.15 \
--wb_epsilon 0.1 \
--wb_sinkhorn_iters 50 \
--wb_spots_per_graph 128 \
--wb_support_size 256 \
--wb_anchor_weight 1
```

WB 运行建议：

- WB 主实验推荐 `--use_mmd 0`，避免与 MMD 同时约束导致归因不清。
- 若保留 cancer GRL，建议把 `--lambda_cancer` 降到 `0.05-0.1`；如果下游任务性能下降，可先关闭 cancer GRL。
- `euclidean_pairwise` 可以先用于快速筛参；`dual_potential` 建议在较小 `wb_spots_per_graph` / `wb_support_size` 下验证稳定后再放大；`sinkhorn_divergence` 优先调 `--lambda_wb`、`--wb_epsilon`、`--wb_sinkhorn_iters`、`--wb_support_size` 与 `--wb_spots_per_graph`。

- 基于癌种的 KFold（随机生成 K 组“每癌种 1 张验证”组合）：

```bash
# 训练并保存每折产物到 artifacts_dir 的同级目录 kfold_val/
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --kfold_cancer 5 --split_seed 42 \
  --epochs 60 --early_patience 20 --num_workers 10 --device cuda

# 仅查看划分，不训练
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --kfold_cancer 5 --split_seed 42 --split_test_only
```
产物：`../kfold_val/fold_{i}/` 模型与预处理，和 `../kfold_val/kfold_summary.csv` 汇总。

KFold 推理循环示例（对每一折分别批量推理）：
```bash
# 一次跑所有fold（示例：fold_1..fold_10）
for i in {1..10}; do
  python -m stonco.core.batch_infer \
    --npz_glob '/path/to/val_npz/*.npz' \
    --artifacts_dir "/path/to/kfold_val/fold_${i}/" \
    --out_csv "/path/to/kfold_val/fold_${i}/batch_preds.csv" \
    --num_threads 4 --num_workers 0
done
```

KFold 推理（内部验证集 + 外部验证集一起预测）：
```bash
# 一次跑所有fold（示例：fold_1..fold_10）
for i in {1..10}; do
  python -m stonco.core.batch_infer \
    --train_npz /path/to/train_data.npz \
    --external_val_dir /path/to/val_npz \
    --artifacts_dir "/path/to/kfold_val/fold_${i}/" \
    --out_csv "/path/to/kfold_val/fold_${i}/batch_preds.csv" \
    --num_threads 4 --num_workers 0
done
```

- LOCO 留一癌种评估（逐癌种训练/验证）：

```bash
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --leave_one_cancer_out \
  --epochs 60 --early_patience 20 --num_workers 0 --device cuda
```
产物：`../loco_eval/{CancerType}/` 每癌种一个子目录，含该设定下的最优 `model.pt` 与 `meta.json`，并在 `../loco_eval/loco_summary.csv` 汇总。

### 5.3 Sampler 双模式与子图训练

`train.py` 当前已支持两层控制：

- `sampler_mode=random`：默认随机打乱训练图
- `sampler_mode=cancer_balanced`：在训练 batch 内尽量同时覆盖多个 `cancer_dom`、多个父切片，并在可行时约束不同 `bat_dom`
- `subgraph_mode=off|static|online`：控制训练集是否展开为子图；验证集始终保持整图

实际行为：

- 训练集先按父切片完成 train/val split，再决定是否对子图展开，避免 train/val 泄漏
- `subgraph_mode=static` 时，只对训练集父图切子图；验证与测试仍使用整图
- 静态子图会为每个子图生成唯一 `slide_id`，同时保留 `parent_slide_id` 与 `subgraph_id`
- 同一 batch 的 sampler 去重优先基于 `parent_slide_id`，因此子图模式下会尽量避免同一 batch 重复同一个父切片
- 若训练图 spot 数小于 `max(subgraph_target_spots, subgraph_min_spots)`，则该图会以单个 `__sg000` 全图子图形式保留
- `subgraph_mode=online` 当前会直接报 `NotImplementedError`

推荐示例：

```bash
# 一图一切片 + cancer-balanced sampler
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_sampler \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --sampler_enforce_distinct_batch 1 \
  --epochs 80 --early_patience 20

# 静态子图训练 + cancer-balanced sampler
python -m stonco.core.train \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts_sampler_subgraph \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced_subgraph \
  --subgraph_target_spots 1000 \
  --subgraph_min_spots 300 \
  --epochs 80 --early_patience 20
```

日志与运行期提示：

- 启用 `static` 子图模式时，训练开始前会打印父图数到训练图数的展开统计
- 启用 `cancer_balanced` 时，会打印 sampler 预览摘要：`avg_unique_cancers`、`avg_unique_batches`、`avg_unique_slides`、`avg_unique_parents`
- 验证 loader 不使用该 sampler，仍保持 `shuffle=False`

---

## 6. 多阶段超参数优化（train_hpo.py）

HPO 已独立到 `stonco/core/train_hpo.py`，提供统一三阶段流水线与可选多种子复评：
- 阶段：stage1（优化训练稳定性/学习率等）→ stage2（结构/正则）→ stage3（位置编码）
- 复评（重打分）：对指定阶段 Top-K 以多随机种子复训，按 mean_accuracy 重排
- 产物：按阶段保存 trial 结果，最终合并最优配置为 tuning/best_config.json
- 说明：当前 `train_hpo.py` 仍保持旧风格的标量 `hidden` 搜索；列表式 `GNN_hidden` 目前用于手动训练/推理配置，不在本轮 HPO 搜索范围内

常用参数（节选）：
- --tune {all,stage1,stage2,stage3}
- --n_trials 每阶段 trial 数
- --rescore_topk K，多种子复评的 Top-K
- --rescore_stages 需要复评的阶段列表（逗号分隔）
- --seeds 多种子列表（逗号分隔）
- 其余训练相关参数（如 --epochs、--early_patience、--model、--lap_pe_dim 等）基本与 train.py 保持一致；但本轮新增的 `linear` 模式与 4 个 `*_delay_epochs/*_warmup_epochs` 参数目前只在 `train.py` 中实现，`train_hpo.py` 仍停留在旧版 `dann/constant` 接口。
  - 包含划分参数：`--val_ratio`（默认 0.2）与 `--no_stratify_by_cancer`

示例：
```bash
# 全流程 + 复评（示例）
python -m stonco.core.train_hpo \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --tune all \
  --n_trials 30 \
  --rescore_topk 3 \
  --rescore_stages 1 \
  --seeds 42,2023,2024 \
  --epochs 100 --early_patience 30 --num_workers 10 --device cuda

# 仅搜索 stage2
python -m stonco.core.train_hpo \
  --train_npz /path/to/train_data.npz \
  --artifacts_dir /path/to/artifacts \
  --tune stage2 \
  --n_trials 24 \
  --device cuda
```

更多细节：`docs/hyperparameter_optimization_plan.md` 与 `docs/train_hpo_refactor_plan.md`

---

## 7. 推理与批量推理

单切片推理（infer.py）：
```bash
python -m stonco.core.infer \
  --npz /path/to/slide.npz \
  --artifacts_dir /path/to/artifacts \
  --out_csv /path/to/preds.csv \
  --num_threads 4
```
- 从 `artifacts_dir/meta.json` 读取训练时的 cfg，以保证图构建（KNN、LapPE、是否拼接等）与训练一致。
- 支持两类输入：
  - 单切片 NPZ：包含 X, xy, gene_names（以及可选 barcodes、y、sample_id）。
  - 数据集 NPZ：包含 Xs/xys（多张切片），可通过 `--index` 选择第几张（默认 0）。
- 若训练 cfg 启用了图像特征（`meta.json:cfg.use_image_features=1`）：
  - 推理会加载 `artifacts_dir` 下的 `img_scaler.joblib` +（可选）`img_pca.joblib`，并按 `x=[Xp_gene, Xp_img, img_mask, (LapPE)]` 构图。
  - 输入 NPZ 若包含 `X_img/img_mask`（或数据集 NPZ 的 `X_imgs/img_masks`），则必须同时包含 `img_feature_names`，并与 `artifacts_dir/img_feature_names.txt` 完全一致；否则直接报错。
  - 输入 NPZ 不含 image keys 时会自动兜底为 `X_img=0, img_mask=0`（可运行，但等价于该输入“无图像信息”）。
- 输出 CSV 列：`spot_idx, x, y, p_tumor`；默认写到 `preds.csv`，或按 `--out_csv` 指定的路径保存（自动创建目录）。
- 解释性输出（默认开启，可用 `--no_explain` 关闭）：
  - `--explain_saliency` 默认开启；`--no_explain` 关闭解释性计算
  - `--explain_method {ig,saliency}`（默认 ig），`--ig_steps`（默认 50）
  - `--gene_attr_out` 指定保存路径；未指定时默认与 `out_csv` 同目录，命名为 `per_slide_gene_saliency_{single|idxK}.csv`
  - 输出保存 CSV（两列：`gene, attr`）

批量推理（batch_infer.py）：
```bash
python -m stonco.core.batch_infer \
  --npz_glob '/path/to/infer_*.npz' \
  --artifacts_dir /path/to/artifacts \
  --out_csv /path/to/batch_preds.csv \
  --num_threads 4 --num_workers 0
```
- 也支持“内部验证集 + 外部验证集”一起预测：
```bash
python -m stonco.core.batch_infer \
  --train_npz /path/to/train_data.npz \
  --external_val_dir /path/to/val_npz_dir \
  --artifacts_dir /path/to/artifacts \
  --out_csv /path/to/batch_preds.csv
```
- 读取 `artifacts_dir` 下的预处理器与 `meta.json` 的 cfg，对匹配到的每个 NPZ 执行“读取→预处理→图构建→推理”。
- 若 `meta.json:cfg.use_image_features=1`，批量推理会对每个 NPZ 按同样规则读取/兜底 image keys，并强校验 `img_feature_names` 一致性（不一致直接报错）。
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
- 样本级汇总：会在 `out_csv` 同目录生成 `batch_preds_summary.csv`，列为 `sample_id, source, n_spots, threshold, accuracy, auroc, auprc, macro_f1`（若无 `y_true` 则为 NaN）。
- 解释性输出（默认开启，可用 `--no_explain` 关闭）：
  - `--explain_saliency` 默认开启；`--no_explain` 关闭
  - `--explain_method {ig,saliency}`，`--ig_steps`（默认 50）
  - `--gene_attr_out_dir` 指定输出目录（未指定时默认写到 `out_csv` 同目录下的 `gene_attr/`）
  - 每张切片保存一份 `per_slide_gene_saliency_{sample_id|文件名}.npz` 与 `.csv`（两列：`gene, attr`）
- 可视化：默认会自动生成“各切片准确率柱状图”（png），关闭请加 `--no_plot`；输出路径可用 `--plot_out` 自定义，标题可用 `--model_name` 指定（默认从 cfg.model 推断）。

> 详细实现参见 `stonco/core/infer.py` 与 `stonco/core/batch_infer.py`

### 7.1 与 evaluate_models/plot_accuracy_bars 的简单联动示例

目标：对比两个（或多个）不同训练产物在同一批量推理集上的表现，并输出汇总指标与可视化。

步骤 1）分别对每个模型产物运行批量推理，得到各自的预测 CSV：
```bash
# 模型 A（例如 GATv2）
python -m stonco.core.batch_infer \
  --npz_glob '/path/to/val_npz_dir/*.npz' \
  --artifacts_dir /path/to/artifacts_gatv2 \
  --out_csv /path/to/preds_gatv2.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0 --model_name GATv2

python -m stonco.core.batch_infer --npz_glob 'data/data_SC_3331genes/val_npz/*.npz' --artifacts_dir test_train0822_SC3331genes/gatv2/artifacts_dir/ --out_csv test_train0822_SC3331genes/gatv2/predictions/batch_pred.csv --num_workers 10 --model_name GATv2

# 模型 B（例如 GraphSAGE）
python -m stonco.core.batch_infer \
  --npz_glob '/path/to/val_npz_dir/*.npz' \
  --artifacts_dir /path/to/artifacts_sage \
  --out_csv /path/to/preds_sage.csv \
  --threshold 0.5 --num_threads 4 --num_workers 0 --model_name SAGE
```

步骤 2）用 evaluate_models 汇总评估并可选绘图：
```bash
python -m stonco.utils.evaluate_models \
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
python -m stonco.utils.plot_accuracy_bars \
  --pred_csv /path/to/preds_gatv2.csv \
  --model_name GATv2 \
  --out_path /path/to/compare/gatv2_per_slide_acc.png \
  --threshold 0.5
```

---

## 8. 可视化（visualize_prediction.py）

对单张或批量切片绘制真实/预测在空间坐标下的散点图，并在标题显示准确率（若存在 y 或可从验证目录解析）。脚本会读取 artifacts_dir/meta.json 的 cfg，确保与训练时的预处理和图构建一致。
若 `meta.json:cfg.use_image_features=1`，可视化脚本也会按同样规则读取/兜底 `X_img/img_mask`（并强校验 `img_feature_names` 与训练一致），以保证模型输入维度一致。

- 使用训练 NPZ 中的某张（例如最后一张作为验证）：
```bash
python -m stonco.utils.visualize_prediction \
  --train_npz /path/to/...
```

- 使用单切片 NPZ（若 npz 含 y 可直接计算准确率；若不含，可结合验证目录自动加载真实标签）：
```bash
python -m stonco.utils.visualize_prediction \
  --npz /path/to/slide.npz \
  --artifacts_dir /path/to/artifacts \
  --val_root /path/to/ST_validation_datasets \
  --xy_cols row col --label_col true_label \
  --out_svg /path/to/vis_slide.svg
```

- 批量处理一组单切片 NPZ（为每张切片导出一个 SVG 到 out_dir）：
```bash
python -m stonco.utils.visualize_prediction \
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

## 9. Embedding 导出、UMAP/t-SNE 与 Mixing/LISI 评估

本章用于 embedding 分析与去批次/跨域整合评估，覆盖常规 embedding pipeline 和 LOCO 专用可视化脚本：

- `export_spot_embeddings.py`：导出每个 spot 的 embedding CSV
- `visualize_umap_tsne.py`：计算 UMAP / t-SNE 并输出 SVG；可选保存降维坐标 CSV
- `evaluate_embedding_mixing.py`：在 embedding / UMAP / t-SNE 空间上计算 `iLISI` / `cLISI`
- `run_embedding_analysis_pipeline.py`：一条命令串联前面三步
- `analyze_spot_embedding_domains.py`：针对已导出的 `spot_embeddings_*.csv` 做域统计、局部维度分布和二维散点诊断
- `scripts/plot_loco_umap_tsne.sh`：批量绘制 LOCO 各癌种 fold 的 UMAP/t-SNE，默认高亮当前留出癌种
- `scripts/eval_visualize_loco.sh`：汇总 LOCO per-slide 指标并生成辅助图

推荐解读口径：

- embedding-space LISI 是主指标，应作为正式比较依据
- UMAP-space / t-SNE-space LISI 是辅助指标，主要用于把图上的“看起来混不混”转成数字
- 对希望被混合的标签列使用 `iLISI`（integration）
- 对希望被保留的标签列使用 `cLISI`（conservation）

常见场景：

- 单癌种去批次：重点看 `sample_id`、`batch_id` 的 `iLISI` 是否提升，同时监控 `tumor_label` 的 `cLISI` 不要异常升高
- 多癌种联合训练：`cancer_type` 不应写死，若目标是跨癌种整合则用 `iLISI`，若目标是保留癌种差异则用 `cLISI`
- 默认导出 `h`：即 GNN backbone 的输出表示；启用域对抗/MMD/WB 时，可直接观察这些分支作用后的共享表征
- 可选导出 `z_clf`：即 task MLP 最后一层隐藏表示，维度由 `clf_hidden[-1]` 自动决定
- `z64` 作为兼容别名保留，仅适用于分类头 latent 维度确实为 64 的模型

依赖：
- `umap-learn`（UMAP）；t-SNE 来自 `scikit-learn`。`requirements.txt` 已包含 `umap-learn`。

### 9.1 导出 embedding（CSV）

输入可组合：
- 一组单切片 NPZ（推荐用于 val_npz/external_val）：
```bash
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir /path/to/artifacts \
  --npz_glob '/path/to/val_npz/*.npz' \
  --out_csv /path/to/artifacts/spot_embeddings_val_npz.csv \
  --embed_source h
```
- 若只想加几个指定切片，也可以重复传 `--npz`：
```bash
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir /path/to/artifacts \
  --npz /path/to/val_npz/BRCA1.npz \
  --npz /path/to/val_npz/BRCA4.npz \
  --out_csv /path/to/artifacts/spot_embeddings_specific_val.csv \
  --embed_source h
```
- 一个多切片训练 NPZ（可用 `--subset train|val|all` 按 meta.json 的 train/val ids 过滤）：
```bash
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir /path/to/artifacts \
  --train_npz /path/to/train_data.npz \
  --subset val \
  --out_csv /path/to/artifacts/spot_embeddings_train_npz_val.csv \
  --embed_source h
```
- 训练集和外部验证集也可以放在同一条命令里：
```bash
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir /path/to/artifacts \
  --train_npz /path/to/train_data.npz \
  --subset train \
  --npz_glob '/path/to/val_npz/*.npz' \
  --out_csv /path/to/artifacts/spot_embeddings_train_plus_val.csv \
  --embed_source h
```

说明：
- `--embed_source {h,z_clf,z64}`：
  - `h`：导出 `h_*` 列，维度随 backbone 结构变化，默认使用这个模式
  - `z_clf`：导出 `z_clf_*` 列，维度由训练产物自动识别
  - `z64`：旧兼容别名；仅当分类头 latent 维度为 64 时可用
- 至少提供一个输入源：`--train_npz` 和/或 `--npz` 和/或 `--npz_glob`
- `--subset` 只对 `--train_npz` 生效，对 `--npz/--npz_glob` 会忽略
- 导出 CSV 会额外写入 `embed_source`、`embed_dim` 与 `clf_latent_dim` 三列，便于后续区分来源和维度
- 若 `meta.json:cfg.use_image_features=1`，导出 embedding 时会加载图像预处理器（`img_scaler.joblib` + 可选 `img_pca.joblib`），并按 `x=[Xp_gene, Xp_img, img_mask, (LapPE)]` 构图。
- 输入 NPZ 若包含 `X_img/img_mask`（或训练 NPZ 的 `X_imgs/img_masks`），必须同时提供 `img_feature_names` 且与训练产物一致；否则直接报错。若缺失 image keys，则自动兜底为 `X_img=0, img_mask=0`。
- 若模型的 `clf_latent_dim != 64`，则：
  - `--embed_source z_clf` 正常可用
  - `--embed_source z64` 会直接报错，避免把非 64 维 latent 误当成历史 `z64`

### 9.2 UMAP + t-SNE 可视化（SVG）

```bash
python -m stonco.utils.visualize_umap_tsne \
  --embeddings_csv /path/to/artifacts/spot_embeddings_val_npz.csv \
  --out_dir /path/to/artifacts/embedding_plots \
  --max_points 50000 \
  --seed 42 \
  --embed_source h \
  --color_cols sample_id batch_id tumor_label \
  --highlight_col cancer_type \
  --highlight_values BRCA \
  --out_coords_csv /path/to/artifacts/embedding_plots/spot_embeddings_h_coords.csv
```

输出：
- `umap_tsne_h_by_sample_id.svg`
- `umap_tsne_h_by_batch_id.svg`
- `umap_tsne_h_by_tumor_label.svg`
- `spot_embeddings_h_coords.csv`（如果传了 `--out_coords_csv`）

说明：
- `--embed_source {h,z_clf,z64}` 用来指定读取哪类列：
  - `--embed_source h` -> 读取 `h_*`
  - `--embed_source z_clf` -> 优先读取 `z_clf_*`，若是旧 CSV 则兼容读取 `z64_*`
  - `--embed_source z64` -> 仅读取历史 `z64_*`
- 若切换到 `z_clf`，输出文件名会变为：
  - `umap_tsne_z_clf_by_sample_id.svg`
  - `umap_tsne_z_clf_by_batch_id.svg`
  - `umap_tsne_z_clf_by_tumor_label.svg`
- `h` 维度不是固定 64，会按实际导出的列数自适应读取；`visualize_umap_tsne.py` 不再对 embedding 维度写死为 64
- `--out_coords_csv` 建议在需要做 UMAP/t-SNE 空间 LISI 时一起保存，这样评估脚本使用的坐标会与图像完全一致
- 若不传 `--color_cols`，默认仍生成按 `tumor_label`、`batch_id`、`cancer_type` 上色的图
- `--highlight_col` 与 `--highlight_values` 必须成对使用；匹配到的 spot 会画成三角形，其他 spot 仍为圆点。该功能只影响 SVG，不改变 `spot_embeddings_*.csv`、`*_coords.csv` 或 LISI 结果。
- `--highlight_col` 常用值为 `cancer_type`、`sample_id`、`batch_id`；也兼容简写 `cancer`、`sampleid`、`batchid`。`--highlight_values` 支持空格分隔或逗号分隔，例如 `--highlight_values BRCA GBM` 或 `--highlight_values BRCA,GBM`。

### 9.3 LISI 混合评估（`evaluate_embedding_mixing.py`）

主评估推荐在原始 embedding 空间完成，UMAP / t-SNE 空间只作为辅助解释。

`group_role` 与 `metric_name` 的关系：

- `integration` -> `iLISI`
- `conservation` -> `cLISI`

常用示例：

```bash
python -m stonco.utils.evaluate_embedding_mixing \
  --embeddings_csv /path/to/artifacts/spot_embeddings_val_npz.csv \
  --coords_csv /path/to/artifacts/embedding_plots/spot_embeddings_h_coords.csv \
  --out_csv /path/to/artifacts/embedding_plots/mixing_metrics_h.csv \
  --out_spot_csv /path/to/artifacts/embedding_plots/mixing_metrics_h_spot.csv \
  --embed_source h \
  --spaces embedding umap tsne \
  --group_cols sample_id batch_id tumor_label \
  --group_roles sample_id:integration batch_id:integration tumor_label:conservation \
  --k_values 15 30 50 \
  --max_points 50000 \
  --seed 42
```

说明：
- `--embeddings_csv` 必需；用于读取 metadata 和 embedding 列
- `--coords_csv` 可选；若提供，则 `umap_1/umap_2` 与 `tsne_1/tsne_2` 从这里读取
- `--spaces` 可选 `embedding`、`umap`、`tsne`
- `--group_cols` 为要评估的 metadata 列
- `--group_roles` 允许显式指定列的角色，格式例如 `sample_id:integration tumor_label:conservation`
- 若不传 `--group_roles`，脚本会自动推断常见列：
  - integration：`sample_id`、`batch_id`、`slide_id`
  - conservation：`tumor_label`、`cell_type`、`region`
  - 其他列如 `cancer_type` 建议显式传入，避免误判实验目标
- `--k_values` 默认是 `15 30 50`
- 对缺失列、唯一值列或有效组数 `<=1` 的列，脚本会 warning 并跳过，不会让整个评估流程失败

汇总 CSV 的关键字段：
- `group_col`
- `group_role`
- `metric_name`
- `space`
- `embed_source`
- `embed_dim`
- `k`
- `n_spots_total`
- `n_valid`
- `n_groups`
- `lisi_mean`
- `lisi_median`
- `lisi_q25`
- `lisi_q75`

BRCA-only 去批次推荐默认配置：

```bash
--group_cols sample_id batch_id tumor_label \
--group_roles sample_id:integration batch_id:integration tumor_label:conservation \
--spaces embedding umap \
--k_values 15 30 50
```

建议解读顺序：
- 先看 `space=embedding` 下 `sample_id` / `batch_id` 的 `iLISI`
- 再看 `space=umap` 或 `space=tsne` 下对应的辅助 `iLISI`
- 同时检查 `tumor_label` 的 `cLISI` 是否被异常拉高

### 9.4 一键三步流程（`run_embedding_analysis_pipeline.py`）

如果日常实验希望“一条命令跑完导出、作图、量化”，可以直接使用 pipeline：

```bash
python -m stonco.utils.run_embedding_analysis_pipeline \
  --artifacts_dir /path/to/artifacts \
  --out_dir /path/to/artifacts/embedding_analysis \
  --npz_glob '/path/to/val_npz/*.npz' \
  --embed_source h \
  --group_cols sample_id batch_id tumor_label \
  --group_roles sample_id:integration batch_id:integration tumor_label:conservation \
  --metric_spaces embedding umap \
  --k_values 15 30 50 \
  --max_points 50000 \
  --seed 42 \
  --color_cols sample_id batch_id tumor_label \
  --highlight_col cancer_type \
  --highlight_values BRCA \
  --save_spot_metrics
```

默认产物会统一落在 `--out_dir` 下：
- `spot_embeddings_h.csv`
- `spot_embeddings_h_coords.csv`
- `mixing_metrics_h.csv`
- `mixing_metrics_h_spot.csv`（仅 `--save_spot_metrics` 时输出）
- `umap_tsne_h_by_sample_id.svg`
- `umap_tsne_h_by_batch_id.svg`
- `umap_tsne_h_by_tumor_label.svg`

说明：
- pipeline 只是编排层，本身不重复实现算法，而是顺序调用三个工具
- 默认 `--metric_spaces embedding umap`，因此输出天然区分“主指标”和“辅助指标”
- 若希望纳入 t-SNE 辅助指标，可显式传 `--metric_spaces embedding umap tsne`
- 输入侧与 `export_spot_embeddings.py` 一致，支持混合使用 `--train_npz`、重复 `--npz` 和重复 `--npz_glob`
- 可直接传 `--highlight_col` / `--highlight_values`，pipeline 会把高亮参数传给 `visualize_umap_tsne.py`；高亮不参与 LISI 计算

### 9.5 LOCO UMAP/t-SNE 与留出癌种高亮

当 `--leave_one_cancer_out` 已生成 `loco_eval/{CancerType}/` 后，可以用仓库脚本批量绘制每个 LOCO fold 的 embedding UMAP/t-SNE。默认用于检查“训练癌种和留出癌种是否混在一起”：

```bash
bash /apps/users/sky_luozhihui/STOnco/model/STOnco/scripts/plot_loco_umap_tsne.sh \
  --loco_dir /path/to/loco_eval \
  --train_npz /path/to/train_data.npz
```

默认行为：
- `--subset all`：每个 fold 绘制该 fold 的训练样本 + 留出样本，适合检查训练癌种与留出癌种是否混合
- `--embed_source h`：默认观察 GNN latent `h`
- 默认输出到 `loco_eval/loco_umap_tsne/{CancerType}/final_all_h/`
- 默认 `--highlight_col cancer_type --highlight_values {CancerType}`：当前 fold 的留出癌种画为三角形，训练癌种仍为圆点
- `plot_loco_umap_tsne.sh` 的 `--metric_spaces` 与 `--k_values` 需要传逗号分隔字符串，例如 `embedding,umap,tsne` 与 `15,30,50`
- 若已有 `mixing_metrics_h.csv`，脚本默认跳过该目录；需要重画 SVG 时加 `--force`

输出示例：
- `loco_umap_tsne/BRCA/final_all_h/spot_embeddings_h.csv`
- `loco_umap_tsne/BRCA/final_all_h/spot_embeddings_h_coords.csv`
- `loco_umap_tsne/BRCA/final_all_h/mixing_metrics_h.csv`
- `loco_umap_tsne/BRCA/final_all_h/umap_tsne_h_by_cancer_type.svg`
- `loco_umap_tsne/BRCA/final_all_h/umap_tsne_h_by_tumor_label.svg`

绘制全部或指定 checkpoints：

```bash
# 全部 checkpoint
bash /apps/users/sky_luozhihui/STOnco/model/STOnco/scripts/plot_loco_umap_tsne.sh \
  --loco_dir /path/to/loco_eval \
  --train_npz /path/to/train_data.npz \
  --include_checkpoints

# 只绘制部分 checkpoint
bash /apps/users/sky_luozhihui/STOnco/model/STOnco/scripts/plot_loco_umap_tsne.sh \
  --loco_dir /path/to/loco_eval \
  --train_npz /path/to/train_data.npz \
  --include_checkpoints \
  --checkpoint_epochs 20,25,30,50
```

手动指定高亮对象：

```bash
# 高亮指定癌种
--highlight_col cancer_type --highlight_values BRCA,GBM

# 高亮指定样本或 batch；sampleid/batchid 会自动映射为 sample_id/batch_id
--highlight_col sampleid --highlight_values sample_1,sample_2
--highlight_col batchid --highlight_values batch_1,batch_2

# 禁用 LOCO 默认高亮
--no_highlight_loco
```

解读建议：
- 看 `umap_tsne_h_by_cancer_type.svg`：三角形是当前 fold 的留出癌种，圆点是训练癌种；三角形若分散在圆点中，说明留出癌种与训练癌种在该 embedding 中混合较好
- 看 `umap_tsne_h_by_tumor_label.svg`：确认 tumor/normal 判别结构没有被 WB/Sinkhorn 过度抹平
- 正式量化仍优先看 `mixing_metrics_h.csv` 中 `space=embedding` 的 LISI；UMAP/t-SNE 图主要用于诊断和解释

LOCO 汇总指标与空间预测图可用：

```bash
bash /apps/users/sky_luozhihui/STOnco/model/STOnco/scripts/eval_visualize_loco.sh \
  --loco_dir /path/to/loco_eval \
  --train_npz /path/to/train_data.npz
```

### 9.6 embedding 域统计与局部二维诊断（`analyze_spot_embedding_domains.py`）

当你已经有 `spot_embeddings_h.csv` 或 `spot_embeddings_z_clf.csv`，并且想快速回答下面这些问题时，可以直接使用这个脚本：

- 不同域（如 `cancer_type`、`batch_id`）在 embedding 上的逐维均值和方差分别是多少
- 某几个维度（例如 `h_7`、`h_20`）在不同域上的均值分布是否仍有明显偏移
- 某几个维度在 spot 级别按 `cancer_type` 分组后的 KDE 是什么样
- 两个指定维度在二维平面上的 spot 分布是否存在癌种聚类，以及均衡抽样后视觉上是否仍然分离

常用示例：

```bash
python -m stonco.utils.analyze_spot_embedding_domains \
  --embeddings_csv /path/to/artifacts/spot_embeddings_h.csv \
  --out_dir /path/to/artifacts/embedding_analysis \
  --group_cols cancer_type batch_id \
  --selected_dims 7 20 \
  --spot_kde_group_col cancer_type \
  --joint_scatter_group_col cancer_type \
  --joint_scatter_sample_caps 3000 1000
```

默认会生成的产物包括：
- `domain_stats_by_cancer_type.csv`、`domain_stats_by_batch_id.csv`
- `domain_stats_by_cancer_type_overview.svg`、`domain_stats_by_batch_id_overview.svg`
- `selected_means_cancer_type_h7_h20.csv`、`selected_means_batch_id_h7_h20.csv`
- `mean_h7_h20_density_compare.svg`
- `h_7_spot_level_kde_by_cancer_type.svg`、`h_20_spot_level_kde_by_cancer_type.svg`
- `h_7_h_20_spot_scatter_by_cancer_type.svg`
- `h_7_h_20_spot_scatter_by_cancer_type_sampled_1000_per_group.svg`（以及对应抽样 CSV）

说明：
- `--embedding_prefix` 默认是 `h_`，因此 `--selected_dims 7 20` 对应读取 `h_7`、`h_20`；若分析 `z_clf_*`，可改为 `--embedding_prefix z_clf_`
- `--group_cols` 控制会输出哪些域统计表和 overview 图；默认是 `cancer_type batch_id`
- `--spot_kde_group_col` 控制 spot 级别 KDE 的分组列；默认是 `cancer_type`
- `--joint_scatter_group_col` 控制二维散点图上色分组列
- `--joint_scatter_sample_caps` 可以同时给多个值，例如 `3000 1000`，脚本会分别导出多版均衡抽样二维散点图
- 该脚本是“诊断型”工具，适合回答“域偏移主要落在哪些维度、在二维投影上是否仍可见”；正式量化比较仍建议结合 9.3 节的 LISI 指标一起看

---

## 10. 合成/模拟数据（可选）

可用 generate_synthetic_data.py 生成示例数据，快速端到端验证：
```bash
python examples/generate_synthetic_data.py --output_dir STOnco/synthetic_data --n_genes 2000 --seed 42
python -m stonco.core.train --train_npz STOnco/synthetic_data/train_data.npz --artifacts_dir STOnco/artifacts --epochs 2 --early_patience 2 --batch_size_graphs 1
python -m stonco.core.infer --npz STOnco/synthetic_data/infer_slide_1.npz --artifacts_dir STOnco/artifacts --out_csv STOnco/synthetic_data/preds_infer_slide_1.csv
python -m stonco.utils.visualize_prediction --train_npz STOnco/synthetic_data/train_data.npz --artifacts_dir STOnco/artifacts --slide_idx -1 --out_svg STOnco/synthetic_data/vis_val_slide.svg
```

---

若需了解模型内部与双域自适应实现，请参考：
- `stonco/core/models.py`
- `docs/Dual-Domain Adversarial Learning.md`
- 训练入口与参数：`stonco/core/train.py`
