# HPO 改造方案：每个 Trial 使用「癌种分层随机 K 次 split 均值」作为目标 + 外部验证集参与 Optuna objective（可选）

> 本文档是**设计/实施方案**（不是代码改动）。你确认关键决策点后，我再开始修改 `stonco/core/train_hpo.py` 等代码。

## 0. 背景：当前实现的关键事实（便于对齐）

- `train_hpo.py` 当前 Optuna 目标：`best['accuracy']`（来自 `train_and_validate` 返回的 `val_accuracy`，按 slide 平均）。  
- 当前 HPO 的内部验证集来自 `prepare_graphs()` 对 `train_npz` 的**一次内部划分**（默认癌种分层单次 split）。  
- `train_hpo.py` 当前**不支持** `--val_sample_dir` 外部验证集（该能力在 `train.py` 里有）。  
- `train.py` 的 `--kfold_cancer` 目前实现是 “K 个癌种分层的随机组合 split（Monte Carlo）”，并非严格意义的 “K 折互斥覆盖”。代码里也写了“若需严格无泄漏，可改为每折重拟合”。  

本次改造目标：在 HPO 阶段把 “单次 split” 变成 “每个 trial 进行癌种分层随机 K 次 split 并取均值”，并支持外部验证集输入（当提供时直接参与 Optuna objective；不提供则仅内部验证）。

---

## 1. 目标与非目标

### 1.1 目标

1) **HPO 目标改为**：对每个 trial 的配置，在内部数据上做癌种分层随机 K 次 split，取每次 split 的 `best_accuracy` 均值作为 Optuna objective（maximize）。  
2) **外部验证集输入**：支持从目录读取外部 NPZ（单切片）作为额外评估集；当设置 `--val_sample_dir` 时，外部验证集 accuracy 将**直接参与** Optuna objective（未设置则仅内部验证）。  
3) **复现性**：fold 划分与 trial 内部种子可控、可复现；输出中能追踪每个 trial 的折内表现。  
4) **与现有多阶段 HPO 保持兼容**：`--tune stage1|2|3|all` 行为不被破坏；仅当用户显式启用 CV 模式时才改变 objective。

### 1.2 非目标（本次不做/可后续做）

- 不引入新的模型结构或新损失；不改变现有 domain adv 的默认行为。  
- 不改变训练指标定义（仍以 `val_accuracy` 为早停与 HPO 主指标），除非你要求。  

---

## 2. CLI 设计（建议）

为了最大兼容现有用法，我建议如下：

### 2.1 启用随机 K 次 split 的方式（已确认：复用现有参数）

- `--kfold_cancer K`：当 `--tune ...` 且 `K>0` 时，HPO objective 使用 “trial 内 K 次癌种分层随机 split 的均值”；不传（或不设置该参数）则保持当前单次分层 split objective。

### 2.2 外部验证集输入

在 `train_hpo.py` 新增（与 `train.py` 对齐）：
- `--val_sample_dir /path/to/external_npz_dir`：目录下每个 `*.npz` 表示 1 个外部样本/切片。

外部 NPZ 约定（沿用 `train.py` 的外部验证读取逻辑）：
- 必需键：`X`, `xy`, `gene_names`, `y`  
- 可选键：`sample_id`（否则使用文件名 stem）

### 2.3 外部验证集是否参与 Optuna 目标（已确认：参与）

- 若设置了 `--val_sample_dir`：外部验证集的 accuracy **直接参与** Optuna objective。  
- 若未设置 `--val_sample_dir`：objective 仅由内部验证（K 次随机 split 的均值）构成。

仍有 1 个细节需要你确认：当外部验证存在时，内部/外部 accuracy 的合成方式（见第 8 节）。

---

## 3. 划分算法：K 次癌种分层随机 split（已确认）

采用“Monte Carlo CV 风格”的随机 K 次 split：
- 固定 `val_ratio`（沿用现有 `--val_ratio`），每次 split 按癌种分层抽取验证集（每癌种保底 1 张，n=1 则全进训练）
- **允许重复**：当某些癌种切片较少时，允许不同 split 中出现相同 val 样本（符合“随机 kfold 即可”的需求）

实现建议（优先满足“允许重复”这一点）：
- 在 HPO 中直接重复调用 `stonco/core/train.py::_stratified_single_split(present_ids, rng, val_ratio)` 共 K 次，得到 K 组 `(train_ids, val_ids)`。

---

## 4. 每个 Trial 的训练/评估流程（随机 K 次 split objective）

假设启用了随机 K 次 split（`--kfold_cancer K` 且 `K>0`），则 objective(trial) 变为：

1) 根据 stage 构造 `trial_cfg`（沿用现有 `_get_stage_search_space` + 多阶段参数累积）。  
2) 准备 splits：`[(train_ids_1, val_ids_1), ..., (train_ids_K, val_ids_K)]`（固定不随 trial 变化）。  
3) 对每个 split：
   - **预处理器拟合仅用该 split 的 train slides**（避免泄漏）
   - 用该预处理器 transform train/val slides，并构图（LapPE 如需）
   - 若提供 `--val_sample_dir`：构建外部验证图 `external_val_graphs`（同样使用该 split 的预处理器 transform），并传给 `train_and_validate(...)`
   - 调用 `train_and_validate(...)`，拿到该 split 的 `best_accuracy`（accuracy 的定义见第 5 节）
   - 记录该 split 的其它指标（auroc/auprc/macro_f1，可选）
4) trial 的 objective value = `mean(best_accuracy over K splits)`（可选同时输出 std）。

### 4.1 剪枝（pruning）建议

现在代码是按 epoch `trial.report(accuracy, epoch)`。在 CV 下更推荐按 fold 进行 report：
- fold 1 完成：report 当前 mean（只含 fold1）
- fold 2 完成：report fold1-2 mean
- ...
这样 pruner 能在较早 fold 就剪掉明显差的 trial，节省时间。

---

## 5. 外部验证集：如何接入（已确认：参与 objective）

### 5.1 说明

你已确认：当设置 `--val_sample_dir` 时，外部验证集 accuracy 将直接参与 Optuna objective（未设置则不参与）。

### 5.2 推荐实现（与现有 `train.py` 评估方式对齐）

在每个 split 的训练中，把外部验证集构建为 `external_val_graphs` 传给 `train_and_validate(...)`，让其在每个 epoch 的验证阶段同时评估：
- 内部 `val_graphs`（来自本 split 的 val_ids）
- 外部 `external_val_graphs`（来自 `--val_sample_dir`）

这样 `train_and_validate` 返回的 `best['accuracy']` 会自然包含外部验证的贡献（见 `train.py` 当前实现：内部 val + 外部 val 会一起进入验证循环）。

### 5.3 外部验证与“预处理器一致性”

无论外部验证在哪里计算，都必须使用与训练一致的特征变换：
- 若模型是 fold 训练得到：外部样本必须用该 fold 的预处理器 transform 才能输入模型
- 因此，若要在 HPO 阶段每折都算 external，则外部特征会被 transform K 次（每折一次）

---

## 6. 输出与可追踪性（建议新增）

当前 `trials.csv` 只有 params/value/state/number。CV 后建议增强记录：

- `tuning/<stage>/trials.csv` 新增列（不破坏原列）：
  - `cv_mean_accuracy`, `cv_std_accuracy`
  - （可选）`cv_mean_auroc`, `cv_mean_macro_f1` 等
  - （可选）`external_accuracy` 等（若启用外部评估）

- 额外保存一个更详细文件（便于排查）：
  - `tuning/<stage>/trials_detail.jsonl`：每行一个 trial，包含 fold-level 指标与 fold sizes（train/val 样本数、各癌种计数摘要）。

---

## 7. 性能与缓存策略（避免 CV 过慢）

CV 计算会把每个 trial 的训练次数放大 K 倍。建议的优化顺序：

1) **fold 划分固定**（由 `split_seed` 决定），避免每 trial 重新划分带来的噪声与开销。  
2) stage1/2 图不依赖搜索参数（除非你把图相关参数也纳入搜索空间），可以：
   - 预先为每个 fold 构建一次 graphs（基于 base cfg），objective 里复用，只训练模型
3) stage3 需要变动 LapPE/concat 等，仍可能需要 per trial 重建图；但可考虑缓存 “预处理后 Xz” 或 “边/坐标” 以减少重复工作（可选，后续再做）。

---

## 8. 需要你再确认 1 个细节（确认后我再改代码）

你已确认：
- 使用 **K 次癌种分层随机 split**（允许重复）
- 复用 `--kfold_cancer K`
- 外部验证集 `--val_sample_dir` **参与** Optuna objective（未设置则仅内部）
- Optuna 目标仍为 `accuracy`


- 当设置了 `--val_sample_dir` 时，objective 中内部/外部 accuracy 的“合成方式”我希望：**合并评估（推荐、改动最小）**：像 `train.py` 那样把外部样本也加入验证循环，最终 `accuracy` 在“内部 val + 外部 val”所有验证 slide 上统一计算（等价于按 slide 数量隐式加权）。

---

## 9. 预计改动的文件（你确认后才会动）

- `stonco/core/train_hpo.py`：新增 CV objective、外部验证输入与结果记录逻辑
- （可能）`stonco/core/train.py`：抽取/复用 split 与外部验证构图工具，避免复制代码
- （可选新增）`stonco/utils/splits.py`：把“癌种分层 K 折”工具函数做成可复用模块
