# STOnco 训练流程调整方案（Batch Domain + 方差风险 + 新指标图 + 外部验证集）

本文基于当前 `stonco/core/train.py` 的实现（含 kfold/LOCO 复用 `train_and_validate`），提出可落地的调整方案与关键确认点。**请先确认“待确认问题”后再动代码**。

---

## 目标与变更概览

1) **Slide Domain → Batch Domain**
- 用 `data/cancer_sample_labels.csv` 中的 `Batch_id` 作为真实 Batch 域标签。
- 与 Cancer Domain 一样从 CSV 读取，并在训练图上记录 `batch_dom`（替代 `slide_dom`）。

2) **方差风险 Var_risk（总体方差）**
- 以 **batch 维度**计算每个 epoch 的风险方差，并写入 `loss_components.csv`。
- 同时简化 `train_and_validate` 中重复的 loss 计算逻辑。

3) **训练指标图拆分**
- `train_loss.svg`：训练相关 6 项（2 行 × 3 列）：
  - `avg_total_loss`（原 train_loss）
  - `avg_task_loss`
  - `var_risk`
  - `avg_cancer_domain_loss`
  - `avg_batch_domain_loss`（原 slide_domain_loss 的新名称）
  - `train_accuracy`
- `train_val_metrics.svg`：验证相关 4 项（2 行 × 2 列）：
  - `val_accuracy`
  - `val_macro_f1`
  - `val_auroc`
  - `val_auprc`
- **以上所有指标均写入 `loss_components.csv`**（按你给的命名）。


---

## 具体实现方案

### A. Batch Domain 标签读取与映射

**现状**
- `prepare_graphs` 与 kfold/LOCO 中使用 `slide_dom`，由 `slide_id` → index 映射；
- Cancer Domain 由 `cancer_sample_labels.csv` 读取 `cancer_type` 字段；
- `slide_dom` 用于 domain head 的切片域对抗。

**改动**
- 新增读取 `Batch_id` 的逻辑（来源同一 `cancer_sample_labels.csv`）。
- 为每个 `slide_id` 关联 `batch_id`，并建立 `batch_to_idx`。
- 训练图上新增 `batch_dom`（替代 `slide_dom`），训练与评估均使用 `batch_dom`。
- 变量命名统一为 Batch Domain（`batch_dom`、`batch_domain_loss` 等）。

**兼容建议**
- 代码内保留旧字段名 `slide_dom` 的兼容路径（仅在缺少 `Batch_id` 时回退），或直接强制要求 `Batch_id` 必填（需你确认）。

### B. 方差风险 Var_risk（总体方差）

**定义（batch 维度）**
- 对每个 epoch 内所有 batch 的 **平均总风险** 记为 `L_i`。
- 平均风险：`avg_total_loss = mean(L_i)`  
- 方差风险（总体方差）：  
  `var_risk = mean((L_i - avg_total_loss)^2)`

**落地点**
- 在训练循环中记录每个 batch 的 `total_loss`，epoch 结束后计算 `var_risk`。
- `loss_components.csv` 新增 `var_risk` 列。

### C. 训练指标图拆分

**输出文件**
- `train_loss.svg`（2×3）
  1. `avg_total_loss`
  2. `avg_task_loss`
  3. `var_risk`
  4. `avg_cancer_domain_loss`
  5. `avg_batch_domain_loss`
  6. `train_accuracy`
- `train_val_metrics.svg`（2×2）
  1. `val_accuracy`
  2. `val_macro_f1`
  3. `val_auroc`
  4. `val_auprc`

**指标来源**
- `avg_total_loss`：原 `hist['train_loss']`
- `avg_task_loss`：原 `hist['task_loss']`
- `avg_cancer_domain_loss`：原 `hist['cancer_domain_loss']`
- `avg_batch_domain_loss`：原 `hist['slide_domain_loss']`（重命名）
- `train_accuracy`：训练时对每个 batch 做预测，汇总 epoch accuracy
- 验证指标：`eval_logits` 中已有 accuracy/macro_f1/auroc/auprc

**loss_components.csv 列表（建议）**
- `epoch`
- `avg_total_loss`
- `avg_task_loss`
- `var_risk`
- `avg_cancer_domain_loss`
- `avg_batch_domain_loss`
- `train_accuracy`
- `val_accuracy`
- `val_macro_f1`
- `val_auroc`
- `val_auprc`
- （如需保留历史列名，可附加 `train_loss` 等兼容列）

### D. `train_and_validate` 简化与去重

**思路**
- 把 “计算 task loss + domain loss + 统计指标” 合并为一个小函数，避免在 train/val/kfold/LOCO 多处重复。
- 保留现有 `eval_logits` 用于 AUROC/AUPRC/accuracy/macro_f1 计算。
- 训练阶段增加 `train_accuracy` 统计：在 batch 级别产生 logits 并累积。

### E. 外部验证集 `--val_sample_dir`

**输入格式**
- 目录包含多个单切片 NPZ（`build-val-npz` 输出）：
  - `X, xy, gene_names, y`（y 必需用于 accuracy）

**评估逻辑**
- 每个 epoch 构建外部验证图列表（或训练前构建并缓存图结构）。
- **最终 val 指标为内部 val + 外部 val 的合并结果**：
  - `val_accuracy`：按切片 accuracy 取平均（内部+外部）
  - `val_macro_f1` / `val_auroc` / `val_auprc`：基于合并后的所有 spot logits/labels 计算（内部+外部）

**性能**
- 推荐训练开始前就对外部验证集做图构建（使用当前 preprocessor），每 epoch 只做模型前向。

---

## 待确认问题（已确认）

1) **Batch_id 缺失时怎么办？**
   - B. 回退为 `slide_id`（每切片一个 batch），附带输出提示即可。

2) **命名与参数兼容**
   - 新增 `--use_domain_adv_batch/--lambda_batch` 并弃用旧名

3) **var_risk 列名**
   - 为 `var_risk` 即可

4) **val_sample_dir 的 NPZ 结构**
   - 默认按 `build-val-npz` 输出格式处理

5) **验证指标合并**
   - 默认仅记录一个“总的” `val_*` 指标（内部+外部合并），不区分 internal/external。

---

## 影响范围

- `stonco/core/train.py`：Batch domain 逻辑、loss/metrics 统计、图保存逻辑、外部验证集支持。
- `stonco/core/train_hpo.py`：若复用相同训练逻辑，需同步 Batch Domain 标签与 `loss_components.csv` 列名。
- 文档：`docs/Tutorial.md`、`README.md`（指标图名、loss_components 字段）。

