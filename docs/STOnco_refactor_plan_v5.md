# STOnco 改造方案（v5 · 基于现有代码对齐版）

> 说明：本文件为 **v5**，已按你的要求直接覆盖原 v4 内容。
> v5 的目标是在不“凭空设计”的前提下，**严格对齐当前仓库实现**（`stonco/core/models.py`、`stonco/core/train.py`、`data/cancer_sample_labels.csv` 等），给出可落地的结构性改造规范。

---

## 0. v5 相对当前代码的关键对齐点（先读这个）

当前代码现实情况（用于避免“文档写了但代码并不是那样”）：

1. **主任务已是 spot-level 二分类**：`STOnco_Classifier` 输出 `logits` 为每个节点/spot 的 logit；训练用 `BCEWithLogitsLoss`（`pos_weight=1.0` 等价于不加权）。
2. **域对抗当前是 graph-level**：代码对 `h` 做 `global_mean_pool` 后再进域头；且 `lambda_slide/lambda_cancer` 同时用作 GRL 强度与 loss 权重（未解耦）。
3. **域标签来源已固定**：`data/cancer_sample_labels.csv` 中有 `Batch_id`（27）与 `cancer_type`（11）；训练时按当前 fold 出现的样本动态映射到连续索引（K 动态）。
4. **依赖缺失 UMAP**：当前 `requirements.txt` 没有 `umap-learn`，因此要实现“UMAP + t-SNE”，需补充依赖。

v5 下面的方案全部以这些事实为基础展开。

---

## 一、总体目标（v5 冻结）

对 STOnco 模型做四项核心改造（与 v4 同方向，但对齐代码与参数约束）：

1. **主任务分类头升级为固定 MLP**
   - Spot 级二分类（tumor vs normal）
   - MLP 结构固定为 **[256, 128, 64, 1]**（含 BN/ReLU/Dropout）
2. **导出可视化潜变量**
   - 使用 **主任务 MLP 的 64 维隐藏层输出**作为 spot embedding
3. **域分类器改为 spot-level 域判别（两个域头）**
   - Domain A：`Batch_id`（**K 动态**：按 train/fold 出现的 Batch 类别数）
   - Domain B：`cancer_type`（**K 动态**：按 train/fold 出现的癌种类别数）
4. **解耦 GRL 强度（beta）与域损失权重（alpha）**
   - 保留现有 CLI：`lambda_slide/lambda_cancer` 作为 **alpha（loss 权重）**，减少破坏性改动
   - 新增 `grl_beta_*_target + grl_beta_gamma` 控制 **GRL 对 encoder 的对抗强度**（schedule 固定为 DANN-style）

---

## 二、数据与任务定义（v5 冻结）

### 2.1 Spot / Graph 定义
- 每个 graph 对应一张切片/样本（`slide_id` / `sample_id`）
- 节点为 spot；主任务标签为 `y ∈ {0,1}`（normal/tumor）
- 当前训练代码仍保留 `mask = (y >= 0)` 的兼容逻辑；**v5 假设训练数据为全标注**（mask 全 True），但不强制删掉 mask

### 2.2 主任务（Task）
- 类型：spot-level 二分类
- Loss：`BCEWithLogitsLoss`（不使用 `pos_weight` 或等价 `pos_weight=1.0`）

### 2.3 域标签（Domain）
- 域标签来自 `data/cancer_sample_labels.csv`：
  - `Batch_id` → Domain A（Batch 域）
  - `cancer_type` → Domain B（Cancer 域）
- K 动态：每次训练/每折只对 **当前 train/fold 出现的类别**做映射（与当前代码一致）
- Spot-level 域判别时，**同一张图内所有 spot 共享同一个域标签**；实现上可在 loss 处把图级标签 expand 成每个 spot 的标签向量
- `Batch_id` 缺失时：回退为 `slide_id` 作为 batch 域（保持现有代码行为；可能导致 `K_batch` 变大）

---

## 三、模型结构设计（v5 冻结）

### 3.1 Encoder（保持不变）
- 保持现有 GNN backbone（GATv2/GCN/WeightedSAGE）不变
- 输出 spot embedding：
  - `h ∈ R^{N_spots × d}`

### 3.2 主任务分类头（Spot-level Task Head）

**输入**：spot embedding `h`

**固定网络结构**
```
Linear(d → 256)
BatchNorm1d(256)
ReLU
Dropout(0.1)

Linear(256 → 128)
BatchNorm1d(128)
ReLU
Dropout(0.1)

Linear(128 → 64)
BatchNorm1d(64)
ReLU
Dropout(0.1)

Linear(64 → 1) → logit
```

**潜变量（冻结）**
- `z_64`：上述第三段输出（64-d，BN+ReLU 之后、Dropout 之前或之后需实现时统一约定；建议 Dropout 之后，保证训练/导出一致）
- `z_64 ∈ R^{N_spots × 64}`

**主任务损失**
```
L_task = mean_spot( BCEWithLogits(logit, y) )
```

### 3.3 域分类器（Spot-level Domain Heads）

**域头设置**
- 两个完全独立的 MLP：
  - Batch 域头：输出 `K_batch`（动态）
  - Cancer 域头：输出 `K_cancer`（动态）

**输入**
- `h_adv = GRL(h, beta_*)`

**网络结构（可配置，默认 2 层 MLP）**
- 例（默认）：
  - `Linear(d → domain_hidden)`
  - `ReLU`
  - `Linear(domain_hidden → K)`
- 不强制 norm（后续可扩展）

**输出**
- `logits_batch ∈ R^{N_spots × K_batch}`
- `logits_cancer ∈ R^{N_spots × K_cancer}`

---

## 四、域损失与对抗机制（v5 冻结）

### 4.1 alpha/beta 解耦（v5 冻结）

每个域头分别定义：
- **alpha**：域损失权重（loss scale）
  - 直接复用现有 CLI：`lambda_slide` / `lambda_cancer`
  - v5 默认：`lambda_slide = 1.0`，`lambda_cancer = 1.0`
- **beta**：GRL 强度（encoder 对抗强度）
  - 新增 `grl_beta_*_target` + `grl_beta_gamma`（schedule 固定为 DANN-style）

### 4.2 beta schedule（DANN-style，v5 冻结）

```
beta(p) = beta_target * (2 / (1 + exp(-gamma * p)) - 1)
p = current_step / total_steps
gamma = 10 (可配置)
```

> 说明：beta 只影响 encoder 的对抗强度；alpha（lambda_*) 只影响域 loss 在总 loss 中的权重。

### 4.3 域损失（Spot-level，聚合方式冻结）

聚合方式（你已确认）：
- **所有 spot 全局平均**
```
L_domain = mean_spot( CE(logits, domain_label), class_weighted )
```

### 4.4 class weight（v5 冻结）

统计口径（冻结）：
- **graph-frequency（按图计数）**：每类出现了多少张 graph/slide
- 仅用训练集（train graphs）统计一次，训练中固定不变

权重形式（冻结）：
```
w_c = sqrt( N_graph / (K * count_graph[c]) )
```

稳定化策略（你已确认，冻结）：
- `w = clamp(w, 0.5, 5.0)`
- `w = w / mean(w)`（保持平均权重为 1，避免整体学习率隐式变化）

---

## 五、总损失函数（v5 冻结）

```
L_total =
    L_task
  + lambda_slide  * L_domain_batch
  + lambda_cancer * L_domain_cancer
```

其中：
- `L_domain_*` 使用 GRL(beta_*) 进入域头，但其数值项仍由 `lambda_*` 加权进入总 loss

---

## 六、训练与日志策略（v5 建议对齐）

### 6.1 指标记录（建议按 spot-level 统一口径）
- 主任务：spot-level loss / accuracy（现有已基本具备）
- 域任务：
  - spot-level CE（建议记录 **raw CE** 与 **weighted CE** 两套，避免“乘了 lambda 看不清”）
  - spot-level accuracy（当前代码是 graph-level 统计，v5 需要改为 spot-level）

### 6.2 诊断可视化
- 保留 2×2 诊断图：
  - CE vs `log(K)`
  - Accuracy vs `1/K`
- 注意：K 为动态（每次训练/每折可能不同），图标题/元数据需记录当次的 `K_batch`/`K_cancer`

---

## 七、潜变量导出与可视化（UMAP / t-SNE，v5 规范）

### 7.1 导出策略（v5 冻结）
- 训练后独立脚本运行：`model.eval()` + 逐图前向得到 `z_64`
- 支持自由指定数据来源（你已要求“自由度大”）：
  - `--train_npz`（多图 NPZ：Xs/xys/ys/slide_ids）
  - `--npz_glob`（单图 NPZ 批量：X/xy/(y)/sample_id/barcodes）
  - 可选：`--external_val_dir` 或 `--val_npz_glob`（等价于 npz_glob）

### 7.2 导出内容（建议单 CSV，便于下游）
建议输出 `spot_embeddings.csv`（每行一个 spot）：
- `spot_id`（优先：`barcode`；若无则 `sample_id:spot_idx`）
- `sample_id`
- `spot_idx`
- `x`/`y`（坐标）
- `tumor_label`（若有 y）
- `batch_id` / `cancer_type`（若可解析）
- `p_tumor`（推理概率，便于可视化）
- `z64_0 ... z64_63`（64 维潜变量）

### 7.3 可视化（v5 冻结）
- 同时运行 **UMAP + t-SNE**
- 输出格式：SVG
- 默认输出 3 张图：
  1. 按 tumor / normal 上色
  2. 按 batch_id 上色
  3. 按 cancer_type 上色

### 7.4 依赖更新（v5 冻结）
- 在 `requirements.txt` 增加：`umap-learn`

---

## 八、实现策略（v5：对现有仓库的最小侵入改动）

### 8.1 代码改动位置

1. `stonco/core/models.py`
   - `ClassifierHead`：替换为固定 MLP，并能取出 `z_64`
   - 域头：改为 spot-level（输入 `h`，输出 `N_spots × K`）
   - GRL：forward 需要接收 **beta**（而不是直接复用 lambda）

2. `stonco/core/train.py` 与 `stonco/core/train_hpo.py`
   - `lambda_slide/lambda_cancer` 语义明确为 **alpha（loss 权重）**
   - 新增 beta schedule 参数（见下方“CLI 参数草案”）
   - 域损失改为 spot-level（domain label expand 到每个 spot）
   - 注入 class weights（graph-frequency 的 sqrt 反频率）
   - 日志：补齐 spot-level domain accuracy 与 raw CE

3. `stonco/utils/`（新增脚本，符合仓库结构）
   - `stonco/utils/export_spot_embeddings.py`
   - `stonco/utils/visualize_umap_tsne.py`

### 8.2 CLI 参数（v5 冻结）

保留现有（语义调整为 alpha）：
- `--lambda_slide`：batch 域 loss 权重（默认 **1.0**）
- `--lambda_cancer`：cancer 域 loss 权重（默认 **1.0**）

新增（控制 GRL beta；你已确认参数名前缀为 `grl_`）：
- `--grl_beta_slide_target`：batch 域 GRL 目标强度（默认 **1.0**）
- `--grl_beta_cancer_target`：cancer 域 GRL 目标强度（默认 **0.5**）
- `--grl_beta_gamma`：DANN schedule 的 gamma（默认 **10**）

说明（冻结）：
- `grl_beta_schedule` **不作为 CLI 参数暴露**，固定使用 DANN-style schedule（见 4.2）

---

## 九、最终确认清单（v5）

已冻结（你已确认）：
- [x] v5 直接改写 v4 文件
- [x] Domain A = `Batch_id`（批次效应）
- [x] 域输出维度 K 动态（按训练集/当前 fold 出现的类别）
- [x] spot-level 域损失聚合：所有 spot 全局 mean
- [x] alpha/beta 解耦：保留 `lambda_slide/lambda_cancer` 作为 alpha，并新增 GRL beta schedule（`grl_*`）
- [x] alpha 默认强度：`lambda_slide=lambda_cancer=1.0`
- [x] GRL beta 参数命名：新增参数统一加前缀 `grl_`
- [x] GRL beta schedule：固定 DANN-style（不暴露 `grl_beta_schedule` CLI）
- [x] class weight 稳定化：`clamp(0.5, 5.0)` + `mean-normalize`
- [x] 64-d 潜变量：取 task MLP 的 64 维隐藏层输出
- [x] 增加 `umap-learn` 依赖，保证 UMAP+t-SNE
- [x] embedding 导出：默认 train+val，支持可选 external 输入
- [x] `Batch_id` 缺失策略：继续 fallback 为 `slide_id`

> **你确认本 v5 文档无遗漏后，才进入代码实现阶段。**

- 由于本方案设计调整不少，每次做完阶段性调整记得回头看看这个文档确保调整符合要求。
