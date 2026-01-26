# STOnco 命令行流程图与产物目录地图（train / kfold / LOCO / HPO）

本文从“命令行使用”的角度，梳理四类主要流程的**执行路径**、**会写出的产物目录/文件**，以及它们在下游（推理/批量推理/评估/复现）中的**依赖关系**。

---

## 0. 路径与符号约定

- `TRAIN_NPZ`：`python -m stonco.utils.prepare_data build-train-npz` 生成的训练 NPZ（包含 `Xs/xys/ys/slide_ids/gene_names`）。
- `VAL_NPZ_DIR`：`build-val-npz` 生成的目录（每个 `*.npz` 一张切片，含 `X/xy/gene_names/y`）。
- `ARTIFACTS_DIR`：训练输出目录（默认 `./artifacts`；你也可以指定任意路径）。
- `ARTIFACTS_PARENT`：`ARTIFACTS_DIR` 的上级目录（代码里很多“汇总目录”会写到这个层级）。

> 说明：`train.py` / `train_hpo.py` 会把部分产物写到 `ARTIFACTS_PARENT`（例如 `tuning/`, `kfold_val/`, `loco_eval/`），而不是完全写在 `ARTIFACTS_DIR` 里。

---

## 1. 数据准备（prepare_data）

### 1.1 训练 NPZ（多切片打包）

```bash
python -m stonco.utils.prepare_data build-train-npz \
  --train_dir /path/to/train_slides \
  --out_npz ./processed_data/train_data.npz \
  --xy_cols row col \
  --label_col true_label
```

产物：
- `./processed_data/train_data.npz`（训练主输入，后续 `train.py`/`train_hpo.py` 都依赖它）

### 1.2 验证/外部 NPZ（单切片多文件）

```bash
python -m stonco.utils.prepare_data build-val-npz \
  --val_dir /path/to/val_slides \
  --out_dir ./processed_data/val_npz \
  --xy_cols row col \
  --label_col true_label
```

产物：
- `./processed_data/val_npz/*.npz`（每个文件一个切片；`train.py --val_sample_dir` 与 `batch_infer.py --external_val_dir` 会用到）

### 1.3 单样本 NPZ（用于推理/可视化）

```bash
python -m stonco.utils.prepare_data build-single-npz \
  --exp_csv /path/to/sample_exp.csv \
  --coord_csv /path/to/sample_coordinates.csv \
  --out_npz ./sample.npz \
  --xy_cols row col
```

产物：
- `./sample.npz`（常用于 `infer.py`/`visualize_prediction.py`）

---

## 2. 单次训练（train：默认模式）

### 2.1 命令

```bash
python -m stonco.core.train \
  --train_npz ./processed_data/train_data.npz \
  --artifacts_dir ./artifacts
```

可选外部验证并入验证阶段（内部 val + 外部 val 一起算指标）：

```bash
python -m stonco.core.train \
  --train_npz ./processed_data/train_data.npz \
  --artifacts_dir ./artifacts \
  --val_sample_dir ./processed_data/val_npz
```

### 2.2 执行流程图（高层）

```text
TRAIN_NPZ
  -> (按癌种分层) 生成 train_ids / val_ids  [--val_ratio, --split_seed, --no_stratify_by_cancer]
  -> 拟合 Preprocessor（HVG/Scaler/(可选PCA)）
  -> 构图（KNN + (可选) LapPE + concat_lap_pe）
  -> train_and_validate（task + (可选) slide/cancer 双域对抗）
      -> 每 epoch 验证：内部 val_graphs + (可选) 外部 external_val_graphs
      -> early stopping 依据 val_accuracy（按 slide 平均）
  -> 写出 ARTIFACTS_DIR（模型、预处理器、meta、曲线、可选解释性输出）
```

### 2.3 产物目录与下游依赖

写出到 `ARTIFACTS_DIR`（训练主产物）：
- **必须（下游推理依赖）**
  - `ARTIFACTS_DIR/model.pt`：模型权重（`infer.py`/`batch_infer.py` 依赖；默认保存验证集最优模型；若训练启用 `--save_last` 且实际跑满 `--epochs`，则会用最后一轮覆盖）
  - `ARTIFACTS_DIR/meta.json`：训练配置与划分信息（推理会读 `cfg`；批量推理内部验证会读 `val_ids`；并记录 best/last 的 epoch 与指标，以及 `saved_checkpoint`）
  - `ARTIFACTS_DIR/genes_hvg.txt`：HVG 列表（由 `Preprocessor.save` 写）
  - `ARTIFACTS_DIR/scaler.joblib`：标准化器（由 `Preprocessor.save` 写）
  - `ARTIFACTS_DIR/pca.joblib`：PCA 信息（可能是 dict：包含 `use_pca` 与 `pca`）
- **可选（辅助分析/复现）**
  - `ARTIFACTS_DIR/model_best.pt`：最优模型权重（仅当训练使用 `--save_last` 时额外保存）
  - `ARTIFACTS_DIR/train_loss.svg`：训练曲线图（默认写出）
  - `ARTIFACTS_DIR/train_val_metrics.svg`：验证指标曲线图（默认写出）
  - `ARTIFACTS_DIR/loss_components.csv`：loss 组件记录（需 `--save_loss_components 1`）
  - 基因重要性（若 `--explain_saliency` 开启）：在 `ARTIFACTS_DIR` 下写出相应 CSV（具体文件名以代码为准）

下游依赖关系：
- `python -m stonco.core.infer`：依赖 `ARTIFACTS_DIR/{model.pt, meta.json, genes_hvg.txt, scaler.joblib, pca.joblib}`
- `python -m stonco.core.batch_infer`：同上；并可选读取 `meta.json` 的 `val_ids` 来做内部验证集推理
- `python -m stonco.utils.visualize_prediction`：通常依赖 `ARTIFACTS_DIR` + 单样本 `*.npz`

---

## 3. K 次分层随机组合训练（train：`--kfold_cancer K`）

### 3.1 命令

```bash
python -m stonco.core.train \
  --train_npz ./processed_data/train_data.npz \
  --artifacts_dir ./artifacts \
  --kfold_cancer 10
```

### 3.2 执行流程图（高层）

```text
TRAIN_NPZ
  -> 构建全部切片图（一次性）
  -> 生成 K 组 (train_ids_k, val_ids_k)  [随机组合，尽量不重复；不是严格互斥K折]
  -> 对每个 fold_k：
      -> 写出 ARTIFACTS_PARENT/kfold_val/fold_k/...
      -> (可选) 从 fold_k 目录加载预处理器，构建 external_val_graphs 参与验证
      -> train_and_validate
      -> 保存 fold_k 的 model.pt / meta.json / (可选) model_best.pt / (可选) 训练曲线与 loss CSV
  -> 汇总写出 ARTIFACTS_PARENT/kfold_val/kfold_summary.csv
```

### 3.3 产物目录与下游依赖

写出到 `ARTIFACTS_PARENT/kfold_val/`：
- `ARTIFACTS_PARENT/kfold_val/fold_1/` ... `fold_K/`
  - `fold_i/model.pt`
  - `fold_i/model_best.pt`（仅当训练使用 `--save_last` 时额外保存）
  - `fold_i/meta.json`
  - `fold_i/genes_hvg.txt`, `fold_i/scaler.joblib`, `fold_i/pca.joblib`（预处理器产物）
  - `fold_i/train_loss.svg`, `fold_i/train_val_metrics.svg`（默认）
  - `fold_i/loss_components.csv`（可选）
- `ARTIFACTS_PARENT/kfold_val/kfold_summary.csv`：每折 best 指标与样本数汇总

下游依赖关系：
- 对每个 fold 进行推理/批量推理：将 `--artifacts_dir` 指向对应 `fold_i/` 即可，例如：
  - `python -m stonco.core.batch_infer --artifacts_dir ARTIFACTS_PARENT/kfold_val/fold_3 ...`

重要注意（口径/严格性）：
- 当前 `train.py --kfold_cancer` 更像 Monte Carlo CV（随机 K 次组合），不是严格互斥覆盖的 K 折。
- 当前实现里预处理器可能对“全体切片”拟合后复用到各 fold（代码亦有提示：若要严格无泄漏，应改为每折重拟合）。

---

## 4. 留一癌种评估（train：`--leave_one_cancer_out`，LOCO）

### 4.1 命令

```bash
python -m stonco.core.train \
  --train_npz ./processed_data/train_data.npz \
  --artifacts_dir ./artifacts \
  --leave_one_cancer_out
```

### 4.2 执行流程图（高层）

```text
TRAIN_NPZ
  -> 枚举每个 cancer_type = ct：
      -> val_ids = ct 的全部切片；train_ids = 其余切片
      -> 在 train_ids 上拟合 Preprocessor（严格避免泄漏）
      -> 构图（train/val）
      -> (可选) external_val_graphs 并入验证
      -> train_and_validate
      -> 写出 ARTIFACTS_PARENT/loco_eval/ct/（模型、预处理器、meta、曲线等）
  -> 写出 ARTIFACTS_PARENT/loco_eval/ 的顶层汇总 CSV
```

### 4.3 产物目录与下游依赖

写出到 `ARTIFACTS_PARENT/loco_eval/`：
- `ARTIFACTS_PARENT/loco_eval/<cancer_type>/`
  - `model.pt`, `meta.json`
  - `model_best.pt`（仅当训练使用 `--save_last` 时额外保存）
  - `genes_hvg.txt`, `scaler.joblib`, `pca.joblib`
  - `train_loss.svg`, `train_val_metrics.svg`（默认）
  - `loss_components.csv`（可选）
- 顶层汇总（用于分析/画图/对比）：
  - `ARTIFACTS_PARENT/loco_eval/loco_summary.csv`
  - `ARTIFACTS_PARENT/loco_eval/loco_per_slide.csv`
  - `ARTIFACTS_PARENT/loco_eval/loco_per_slide_summary.csv`

下游依赖关系：
- 可对每个癌种子目录分别做推理/批量推理（`--artifacts_dir` 指向 `loco_eval/<ct>/`）。
- 顶层 CSV 适合直接交给 `stonco/utils` 里的绘图/评估脚本做汇总分析。

---

## 5. 超参优化（HPO：`train_hpo.py`）

### 5.1 命令（示例）

```bash
python -m stonco.core.train_hpo \
  --train_npz ./processed_data/train_data.npz \
  --artifacts_dir ./hpo_results \
  --tune all \
  --n_trials 100
```

### 5.2 执行流程图（高层）

```text
TRAIN_NPZ
  -> prepare_graphs（一次内部划分，得到 train_graphs/val_graphs）
  -> Optuna study（sqlite）
  -> objective(trial):
      -> 采样一组超参 trial_cfg
      -> (stage3 可选) 可能重建图（lap_pe/concat 变化）
      -> train_and_validate
      -> 返回 best['accuracy'] 作为 trial value
  -> 写出 ARTIFACTS_PARENT/tuning/<stage>/（study.db, trials.csv, best_config_*.json）
  -> 若 --tune all：最终写出 ARTIFACTS_PARENT/tuning/best_config.json
```

### 5.3 产物目录与下游依赖

写出到 `ARTIFACTS_PARENT/tuning/`（注意：HPO 默认不产出 `model.pt`）：
- `ARTIFACTS_PARENT/tuning/stage1/study.db`
- `ARTIFACTS_PARENT/tuning/stage1/trials.csv`
- `ARTIFACTS_PARENT/tuning/stage1/best_config_stage1.json`
- `ARTIFACTS_PARENT/tuning/stage2/...`
- `ARTIFACTS_PARENT/tuning/stage3/...`
- `ARTIFACTS_PARENT/tuning/best_config.json`（当 `--tune all` 时）
- 若启用复评：`topk_rescore.json`、`best_config_rescored.json`（按 stage 写入）

下游依赖关系（把 HPO 结果用于正式训练）：
- `train.py` 支持 `--config_json`，因此可以用 HPO 的最佳配置直接开训：
  - `python -m stonco.core.train --train_npz TRAIN_NPZ --artifacts_dir ARTIFACTS_DIR --config_json ARTIFACTS_PARENT/tuning/best_config.json`
- 训练完成后，推理/批量推理仍然只依赖最终训练写出的 `ARTIFACTS_DIR`。

> 对照文档 `docs/PLAN_hpo_stratified_kfold_cv_external_val.md`：该文档提出的“trial 内 K 次分层 split + 外部验证参与 objective”的逻辑目前主要是**设计目标**，与当前 `train_hpo.py` 现状存在差距（例如缺少 `--val_sample_dir` 的 objective 接入与 CV objective）。

---

## 6. 外部验证 / 批量推理的输入差异（避免混淆）

- `train.py --val_sample_dir VAL_NPZ_DIR`：
  - 用于**训练过程的验证阶段**（内部 val + 外部 val 一起参与验证指标与 early stopping）
  - 依赖 `VAL_NPZ_DIR/*.npz` 中含 `X, xy, gene_names, y`（可选 `sample_id`）
- `batch_infer.py --external_val_dir VAL_NPZ_DIR`：
  - 用于**训练完成后的批量推理评估**（输出 spot 级预测 CSV + slide 级汇总）
  - 同样读取 `VAL_NPZ_DIR/*.npz`

---

## 7. 一页速查：各模式“会写到哪里、下游用什么”

| 模式 | 主要命令 | 主要写出位置 | 下游最关键依赖 |
|---|---|---|---|
| 单次训练 | `python -m stonco.core.train ...` | `ARTIFACTS_DIR/` | `model.pt` + `meta.json` + 预处理器文件（如启用 `--save_last`，还会写出 `model_best.pt`，且 `model.pt` 可能是最后一轮） |
| KFold | `train --kfold_cancer K` | `ARTIFACTS_PARENT/kfold_val/fold_i/` + `kfold_summary.csv` | 每折 `fold_i/` 作为独立 `--artifacts_dir` |
| LOCO | `train --leave_one_cancer_out` | `ARTIFACTS_PARENT/loco_eval/<ct>/` + 顶层汇总 CSV | 每癌种目录 `<ct>/` + 顶层 CSV |
| HPO | `python -m stonco.core.train_hpo --tune ...` | `ARTIFACTS_PARENT/tuning/` | `best_config*.json`（供 `train.py --config_json` 使用） |
