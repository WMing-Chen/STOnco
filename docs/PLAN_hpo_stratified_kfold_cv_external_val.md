# HPO 改造方案：每个 Trial 使用「癌种分层 K 折（CV）平均」作为目标 + 支持外部验证集

> 本文档是**设计/实施方案**（不是代码改动）。你确认关键决策点后，我再开始修改 `stonco/core/train_hpo.py` 等代码。

## 0. 背景：当前实现的关键事实（便于对齐）

- `train_hpo.py` 当前 Optuna 目标：`best['accuracy']`（来自 `train_and_validate` 返回的 `val_accuracy`，按 slide 平均）。  
- 当前 HPO 的内部验证集来自 `prepare_graphs()` 对 `train_npz` 的**一次内部划分**（默认癌种分层单次 split）。  
- `train_hpo.py` 当前**不支持** `--val_sample_dir` 外部验证集（该能力在 `train.py` 里有）。  
- `train.py` 的 `--kfold_cancer` 目前实现是 “K 个癌种分层的随机组合 split（Monte Carlo）”，并非严格意义的 “K 折互斥覆盖”。代码里也写了“若需严格无泄漏，可改为每折重拟合”。  

本次改造目标：在 HPO 阶段把 “单次 split” 变成 “每个 trial 进行癌种分层 CV 并取均值”，并支持可选外部验证集输入（用于记录/筛选）。

---

## 1. 目标与非目标

### 1.1 目标

1) **HPO 目标改为**：对每个 trial 的配置，在内部数据上做癌种分层 K 折（CV），取各折 `best_val_accuracy` 的均值作为 Optuna objective（maximize）。  
2) **外部验证集输入**：支持从目录读取外部 NPZ（单切片）作为额外评估集，并在 HPO 期间记录（默认不用于优化目标，避免“用测试集调参”）。  
3) **复现性**：fold 划分与 trial 内部种子可控、可复现；输出中能追踪每个 trial 的折内表现。  
4) **与现有多阶段 HPO 保持兼容**：`--tune stage1|2|3|all` 行为不被破坏；仅当用户显式启用 CV 模式时才改变 objective。

### 1.2 非目标（本次不做/可后续做）

- 不引入新的模型结构或新损失；不改变现有 domain adv 的默认行为。  
- 不改变训练指标定义（仍以 `val_accuracy` 为早停与 HPO 主指标），除非你要求。  

---

## 2. CLI 设计（建议）

为了最大兼容现有用法，我建议如下：

### 2.1 启用 CV 的方式（两种选一）

方案 A（复用现有参数，最少改动）：
- `--kfold_cancer K`：当 `--tune ...` 时，若 `K>1` 则启用 “trial 内 K 折平均” 作为 objective；不传则保持当前单次 split objective。

方案 B（语义更清晰，避免歧义）：
- 新增 `--hpo_cv_k K`：仅对 HPO 生效；与 `train.py` 的 `--kfold_cancer` 训练评估逻辑互不影响。

> 需要你确认：你更倾向 A 还是 B？（见文末“需要确认”）

### 2.2 外部验证集输入

在 `train_hpo.py` 新增（与 `train.py` 对齐）：
- `--val_sample_dir /path/to/external_npz_dir`：目录下每个 `*.npz` 表示 1 个外部样本/切片。

外部 NPZ 约定（沿用 `train.py` 的外部验证读取逻辑）：
- 必需键：`X`, `xy`, `gene_names`, `y`  
- 可选键：`sample_id`（否则使用文件名 stem）

### 2.3 外部验证集是否参与 Optuna 目标（默认不参与）

默认建议：**外部验证只记录，不参与 objective**。  
可选增强（如果你明确希望）：新增 `--hpo_objective`：
- `internal_cv`（默认）
- `internal_cv_plus_external`（比如：`0.8*internal + 0.2*external` 或直接平均）

> 需要你确认：外部验证集到底是“调参依据”还是“最终挑选/报告”？（见文末）

---

## 3. 折划分算法：建议用“癌种分层、互斥覆盖”的真 K 折

你现在提的是“癌种分层 K 折平均”，我建议实现为**严格 K 折**（每个样本最多出现在 1 个 fold 的 val 中，尽可能覆盖全体），而不是现有的“随机组合 split”。

### 3.1 真 K 折（推荐）

对每个癌种 type：
- 收集该癌种的 slide_ids 列表并 shuffle（受 `--split_seed` 控制）
- 将该列表按 round-robin 或 numpy array_split 切成 K 份
- fold i 的 val_ids = 各癌种第 i 份的并集；train_ids = 其余

优点：
- 折间互斥、覆盖更完整，CV 均值更稳定、更像“泛化估计”

需要处理的边界情况：
- 如果某癌种样本数 `< K`，则该癌种在部分 fold 的 val 会为空，这是正常的；但会导致 fold 间癌种分布略不均。
- 也可以选择强约束：`K <= min_count_per_cancer`，否则直接报错/自动降 K。

### 3.2 继续沿用现有“随机组合 split”（备选）

如果你认为“每折 val_ratio 固定（比如 0.2）更重要”，那我们可以沿用 `train.py::_k_random_combinations` 的逻辑（K 个不同的分层随机验证组合）来做 trial 内均值。

> 需要你确认：你说的 “K 折” 是想要 **真 K 折互斥覆盖**，还是想要 **K 次分层随机 split**？（见文末）

---

## 4. 每个 Trial 的训练/评估流程（CV objective）

假设启用了 CV（K>1），则 objective(trial) 变为：

1) 根据 stage 构造 `trial_cfg`（沿用现有 `_get_stage_search_space` + 多阶段参数累积）。  
2) 准备 folds：`[(train_ids_1, val_ids_1), ..., (train_ids_K, val_ids_K)]`（固定不随 trial 变化）。  
3) 对每个 fold：
   - **预处理器拟合仅用该 fold 的 train slides**（避免泄漏）
   - 用该预处理器 transform train/val slides，并构图（LapPE 如需）
   - 调用 `train_and_validate(...)`，拿到该 fold 的 `best_val_accuracy`
   - 记录该 fold 的其它指标（auroc/auprc/macro_f1，可选）
4) trial 的 objective value = `mean(best_val_accuracy over folds)`（可选同时输出 std）。

### 4.1 剪枝（pruning）建议

现在代码是按 epoch `trial.report(accuracy, epoch)`。在 CV 下更推荐按 fold 进行 report：
- fold 1 完成：report 当前 mean（只含 fold1）
- fold 2 完成：report fold1-2 mean
- ...
这样 pruner 能在较早 fold 就剪掉明显差的 trial，节省时间。

---

## 5. 外部验证集：如何接入（建议默认“只记录不优化”）

### 5.1 关键原则

- 外部验证如果代表“真正的外部泛化测试”，原则上**不应参与 HPO 目标**，否则会变成“在测试集上调参”。  
- 但外部验证非常适合作为：
  - trial 记录项（user_attr）
  - Top-K 复评/最终挑选依据（例如 internal_cv 接近时，用 external 更好者）

### 5.2 推荐实现（成本较低且更合理）

实现两个层次：

1) **HPO 阶段（每个 trial）**：
   - 仅计算 internal CV mean accuracy 作为 objective
   - 可选：在 trial 完成后，对该 trial 的“每折 best model”（或最后一次训练出的 best_state）跑外部验证并记录 external 指标  
     - 这一步会明显增加成本；建议默认关闭或只对 Top-K 做

2) **复评阶段（`--rescore_topk`）**：
   - 对 Top-K trial：多种子 ×（可选）CV 或全量训练
   - 在此阶段**额外计算 external 指标**，并输出一个“internal + external”的综合表格，帮助最终选择

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

## 8. 需要你确认的决策点（确认后我再改代码）

请你逐条回复编号即可：

1) 你要的 “K 折” 是哪种？
   - B. **K 次癌种分层随机 split**（更接近当前 `--kfold_cancer` 的实现语义）

2) 启用 CV 的 CLI 你更希望：
   - A. 复用 `--kfold_cancer K`（HPO 里也用它）

3) 外部验证集 `--val_sample_dir` 的用途：
   - B. **直接参与 Optuna objective**（会有“用外部集调参”的风险）

4) 若选择 “真 K 折”，当某些癌种样本数 `< K` 时你希望：
   - A. 允许该癌种在部分 fold 的 val 为空（继续跑，并提示 warning）
   - B. 直接报错要求你调小 K
   - C. 自动把 K 下调到可行值（并打印最终 K）

5) 你希望 Optuna 目标仍然是 `accuracy` 吗？
   - A. 是（保持一致）
   - B. 改成 `macro_f1` 或 `auroc`（需要你指定）

---

## 9. 预计改动的文件（你确认后才会动）

- `stonco/core/train_hpo.py`：新增 CV objective、外部验证输入与结果记录逻辑
- （可能）`stonco/core/train.py`：抽取/复用 split 与外部验证构图工具，避免复制代码
- （可选新增）`stonco/utils/splits.py`：把“癌种分层 K 折”工具函数做成可复用模块

