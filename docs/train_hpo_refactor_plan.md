# 训练与超参数优化（HPO）功能彻底分离重构计划

本计划用于指导将当前 `train.py`（训练 + HPO 混合）重构为：
- `train.py`: 仅负责模型训练与评估，不依赖 Optuna；
- `train_hpo.py`: 仅负责超参数搜索与（可选）多阶段重评分相关逻辑，按需从 `train.py` 调用训练与评估函数。

任何时候卡住，可打开本文件继续按步骤推进。

---

## 1) 目标
- 将训练与 HPO 完全解耦，保证：
  - 训练路径不再依赖 `optuna`，环境更轻；
  - HPO 路径独立、清晰，专注搜索流程；
  - 两者共享训练核心函数，避免重复实现和行为漂移；
- 保持既有 CLI 行为的主要一致性（训练相关参数不变）。

---

## 2) 最终文件结构（核心相关）
- <project_root>/Spotonco/
  - train.py（训练与评估脚本，仅训练参数）
  - train_hpo.py（HPO 专用脚本，含搜索与重评分 CLI）
  - 其它模块（models.py, utils.py, prepare_data.py, preprocessing.py, …）保持不变

---

## 3) 职责边界

### 3.1 train.py（保留/聚焦）
- CLI: 仅包含训练/评估所需参数（数据、模型、优化器、K 折、分层划分、可视化/输出路径等）。
- 功能：
  - 数据/图构建与预处理（如 `prepare_graphs` 等）；
  - 训练与验证主循环（如 `train_and_validate`、`run_single_training`）；
  - K 折与癌种分层划分支持（如已实现的功能）；
  - 非 HPO 的 split test only 流程（如果存在）；
  - 主入口 `main()`：解析训练参数 -> 执行训练/评估流程。
- 绝不：
  - 引入/导入 `optuna`；
  - 包含 HPO 或重评分逻辑/参数；
  - 调度 HPO 分支。

### 3.2 train_hpo.py（新增/重构）
- CLI: 仅包含 HPO/重评分所需参数（搜索空间、试验数、时间限制、study/storage、top-k 重评分、多阶段控制、seed 列表等）+ 必要的训练核心参数（如模型名、数据路径、输出目录等）。
- 功能：
  - 定义/构建搜索空间与目标函数（objective）；
  - 运行单阶段或多阶段 HPO；
  - 保存最佳配置与指标；
  - 可选：对候选配置进行多 seed 再评估/重评分（如现有逻辑）；
  - 通过 `from train import prepare_graphs, train_and_validate, …` 复用训练核心。
- 绝不：
  - 定义或复制训练核心实现（只调用）。

---

## 4) CLI 参数调整

### 4.1 train.py 保留（示例，按现有为准）
- 数据与路径：`--train_dir`, `--val_dir`, `--out_dir`, `--pred_dir`, `--vis_dir` 等；
- 模型与优化：`--model`, `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--hidden_dim`, `--heads`, `--dropout`, `--use_pca`, `--pca_dim` 等；
- 设备与复现：`--device`, `--seed`, `--num_workers` 等；
- 划分策略：`--stratify_by_cancer`, `--folds` (K 折), `--split_test_only` 等；
- 可视化/日志：现有的相关参数。

### 4.2 从 train.py 移除（全部为 HPO/重评分相关）
- 例如：`--tune`, `--n_trials`, `--timeout`, `--study_name`, `--storage`, `--sampler`, `--pruner`, `--multi_stage`, `--stage_list`, `--rescore_topk`, `--rescore_stages`, `--seed_list`, `--candidate_dir` 等（以实际代码为准逐项移除）。

### 4.3 train_hpo.py 新增（示例，按现有为准）
- 基础：`--train_dir`, `--val_dir`, `--out_dir`, `--model`, `--device`；
- 搜索控制：`--n_trials`, `--timeout`, `--sampler`, `--pruner`, `--study_name`, `--storage`；
- 多阶段：`--multi_stage`, `--stage_list`；
- 重评分：`--rescore_topk`, `--rescore_stages`, `--seed_list`；
- 输出：`--hpo_dir`/沿用 `--out_dir` 子目录，`--save_best_config` 等。

---

## 5) 代码迁移映射（示例名，按实际函数/类名对照）
- 留在 train.py：
  - 数据/图准备：`prepare_graphs`, 及其依赖；
  - 训练与验证：`train_one_epoch`, `validate`, `train_and_validate`, `early_stopping` 等；
  - 单次训练调度：`run_single_training`；
  - 划分工具：癌种分层、K 折相关函数；
  - 入口：`main()`（仅训练/评估）。

- 移到 train_hpo.py：
  - 目标函数/搜索：`objective`, `run_hyperparameter_optimization`, `run_multi_stage_hpo`；
  - 再评估/重评分：`run_rescore_multiple_stages`；
  - 保存/加载最佳配置：相关辅助函数；
  - 入口：`main()`（仅 HPO/重评分）。

说明：若函数命名不同，请以“功能归属”为准，按职责划分归档。

---

## 6) 依赖与导入策略
- train.py 不得 `import optuna`；
- train_hpo.py 可 `import optuna`，并从 `train.py` 顶层导入训练核心函数：
  - `from train import prepare_graphs, train_and_validate, run_single_training, …`
- 避免循环依赖：
  - train.py 不导入 train_hpo.py；
  - train_hpo.py 只调用训练核心，不向 train.py 反向导入 HPO 内容。

---

## 7) 简单测试用例（验证功能分离）

假设数据/输出目录有效，以下命令只作为流程验证建议：

1) 快速训练（gpu，少量 epoch）
- 命令示例：
  ```bash
  python train.py \
    --train_dir data/ST_train_datasets \
    --val_dir data/ST_validation_datasets \
    --model gcn \
    --epochs 2 \
    --batch_size 8 \
    --out_dir test_train0820/gcn_quick \
    --device cuda
  ```
- 期望：完成训练并在 `out_dir` 生成 artifacts/metrics，无需安装 optuna。

2) 快速 HPO（小试验数）
- 命令示例：
  ```bash
  python train_hpo.py \
    --train_dir data/ST_train_datasets \
    --val_dir data/ST_validation_datasets \
    --model gcn \
    --n_trials 3 \
    --timeout 180 \
    --out_dir test_train0820/gcn_hpo
  ```
- 期望：运行 3 次试验，输出 study 结果并保存最佳配置，不修改训练核心代码。

3) （可选）重评分/多阶段
- 若实现：
  ```bash
  python train_hpo.py \
    --train_dir data/ST_train_datasets \
    --val_dir data/ST_validation_datasets \
    --model gcn \
    --rescore_topk 5 \
    --rescore_stages stage1,stage2 \
    --seed_list 0,1
  ```
- 期望：按 top-k 进行多 seed 复评估并产出对比结果。

---

## 8) 实施步骤（按顺序进行）
1) 清理 train.py 顶层导入：移除与 HPO/optuna 相关的所有导入；
2) 从 train.py 移除所有 HPO/重评分参数与分支逻辑；
3) 确认 train.py 仅保留训练/评估参数与流程，主入口 `main()` 不再调度 HPO；
4) 完善/创建 train_hpo.py：
   - 顶层导入所需包（含 `optuna`）与 `from train import …`；
   - 定义 CLI：HPO/重评分参数 + 必要训练核心参数；
   - 实现 objective、单/多阶段搜索、保存最佳配置、（可选）重评分；
   - 主入口 `main()`；
5) 跑“简单测试用例”1）与 2），确认两条路径均可独立运行；
6) 检查输出目录结构、日志、指标与既有行为一致性；
7) 如有需要，再添加 3）重评分用例验证。

---

## 9) 潜在陷阱与对策
- 循环依赖：务必确保 train.py 不导入 train_hpo.py；
- 参数漂移：将 HPO 参数彻底从 train.py 删除，避免“隐藏开关”；
- 路径一致性：HPO 输出目录和训练输出目录使用清晰分区（如 `…/tuning/` 子目录）；
- 可复现性：HPO 与训练的随机种子管理要清晰、互不干扰；
- 资源控制：HPO 时谨慎设置并发、num_workers、GPU 可见性，避免 OOM；
- 结果格式：最佳配置保存格式（json/yaml）与加载逻辑要稳定一致。

---

## 10) 成功标准（勾选）
- [ ] `train.py` 不再导入 optuna，且无 HPO 相关参数与分支；
- [ ] `train_hpo.py` 可以独立运行 HPO 并保存最佳配置；
- [ ] 训练与 HPO 共用同一套核心训练/评估函数；
- [ ] 简单测试用例 1）与 2）均能成功完成；
- [ ] 现有训练指标与输出结构未回退；
- [ ] 无循环依赖或重复函数定义。

