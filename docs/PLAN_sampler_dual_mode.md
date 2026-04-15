# STOnco Sampler 双模式完整方案

## 0. 目标

本文档给出一份可直接落地到当前 STOnco 代码中的 sampler 方案，覆盖两种训练模式：

- 一图一切片模式：每个 `PyGData` 对应一张切片
- 子图训练模式：训练集按父切片生成子图并采样，验证与测试保持整图

目标不是单纯让一个 batch 里出现更多 graph，而是让训练时的每个 minibatch 尽可能同时满足：

1. 包含多个 `cancer_dom`
2. 每个 cancer 内包含多个不同父切片
3. 在可行时，这些父切片尽量来自不同 `bat_dom`
4. 子图模式下，同一 batch 不重复父切片

这样才能更快地为当前 STOnco 中的 `cancer GRL / batch GRL / MMD` 提供有效的 batch 内对齐信号。

---

## 1. 现状与代码约束

### 1.1 当前训练入口

当前训练入口在 [`stonco/core/train.py`](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py) 中：

- `prepare_graphs(...)` 负责构图与 train/val split
- `train_and_validate(...)` 里直接使用：
  - `PyGDataLoader(train_graphs, batch_size=cfg['batch_size_graphs'], shuffle=True, ...)`
  - `PyGDataLoader(val_graphs, batch_size=cfg['batch_size_graphs'], shuffle=False, ...)`

当前代码没有自定义 sampler / batch sampler 层，因此本方案默认通过新增 `BatchSampler` 来替换训练集的 `shuffle=True`。

### 1.2 当前图对象已有字段

在 [`prepare_graphs(...)`](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py#L600) 中，当前每张图已具备：

- `g.slide_id`
- `g.bat_dom`
- `g.cancer_dom`

这意味着一图一切片模式不需要改模型结构即可实现分层采样。

### 1.3 子图模式的唯一键约束

当前代码在 split 后使用：

```python
id2graph = {str(g.slide_id): g for g in pyg_graphs}
```

见 [`train.py`](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py#L688)。

因此如果未来一张切片拆成多个子图，多个子图不能继续共用同一个 `slide_id`，否则会覆盖。子图模式必须新增：

- `parent_slide_id`
- 唯一 `slide_id`，例如 `BRCA3__sg07`

### 1.4 当前 `bat_dom` 的现实含义

当前 `bat_dom` 来自 `cancer_sample_labels.csv` 的 `Batch_id`。在目前这套数据上，`Batch_id` 基本接近 sample-level 唯一值，因此：

- 在现阶段，`distinct bat_dom` 经常近似等于 `distinct slide_id`
- sampler 中的 `bat_dom` 去重仍保留，但在实现上优先保证 `slide_id` 去重

本方案默认只围绕现有 `bat_dom` 设计，不额外抽象 study/platform/source 接口。

---

## 2. 设计原则

### 2.1 sampler 优先级

训练 sampler 的约束优先级固定为：

1. 优先保证 `K` 个不同 `cancer_dom`
2. 在每个 cancer 内优先保证 `M` 个不同父切片
3. 在此基础上尽量保证不同 `bat_dom`
4. 当样本不足时按层级降级补齐

这里：

- `K` 表示每个 batch 中的 cancer 数
- `M` 表示每个 cancer 中的父切片数
- 总 batch 大小满足 `K * M = batch_size_graphs`

### 2.2 层级降级策略

当目标 `K/M` 无法严格满足时，训练 sampler 采用固定回退顺序：

1. 先保住不同父切片
2. 再放宽 `bat_dom` 去重要求
3. 再放宽 cancer 去重，允许由其他 cancer 补足剩余位置
4. 最后才允许有放回重复父切片

不采用“直接丢弃不完整 batch”，也不采用“一开始就有放回采样”的简单方案。

### 2.3 split 与构图顺序

所有模式都必须先按父切片完成 train/val split，再进入训练采样。

尤其在子图模式下，顺序必须固定为：

1. 按父切片构建 train/val id 划分
2. 训练集父切片生成子图
3. 验证集保持整图
4. 训练 sampler 仅作用于训练图或训练子图

禁止先切子图再 split，避免 train/val 泄漏。

---

## 3. 一图一切片模式

### 3.1 训练样本单位

样本单位就是当前 `prepare_graphs(...)` 返回的 `train_graphs` 中的每个 `PyGData`。

每个 graph 对应一张父切片，已具备：

- `slide_id`
- `bat_dom`
- `cancer_dom`

### 3.2 默认 batch 结构

建议默认采用以下规则：

- `batch_size_graphs >= 8` 时：默认 `K=4, M=2`
- `batch_size_graphs == 6` 时：默认 `K=3, M=2`
- `batch_size_graphs == 4` 时：默认 `K=2, M=2`
- 其他情况：
  - 优先取 `M=2`
  - `K = floor(batch_size_graphs / 2)`
  - 若 `batch_size_graphs < 4`，退化为普通随机采样并给出 warning

原因：

- `M=2` 是能同时兼顾 `cancer GRL / batch GRL / cancer MMD` 的最低可用配置
- 在当前 11 个 cancer、77 张训练切片的分布下，`K=4, M=2` 是最稳妥的主配置

### 3.3 sampler 索引结构

实现时先在 `train_graphs` 上构建如下索引：

```text
cancer_dom -> list[graph_idx]
cancer_dom -> bat_dom -> list[graph_idx]
slide_id -> graph_idx
```

如后续需要日志，还可维护：

```text
graph_idx -> {
  slide_id,
  bat_dom,
  cancer_dom
}
```

### 3.4 采样算法

每个训练 batch 的采样步骤固定为：

1. 随机选择 `K` 个 candidate cancers
2. 对每个 cancer：
   - 优先从不同 `bat_dom` 中取 `M` 个 graph
   - 若该 cancer 的 `bat_dom` 不足，则放宽到同 cancer 内不同 `slide_id`
3. 合并得到 `K * M` 个 graph
4. 若仍不足，则按层级降级策略补齐

补齐时必须先检查：

- 不重复 `slide_id`
- `bat_dom` 去重尽可能保留

### 3.5 epoch 长度定义

一图一切片模式下，sampler 的每个 epoch 长度默认与当前训练集图数保持同量级。

建议定义为：

```text
num_batches_per_epoch = ceil(len(train_graphs) / batch_size_graphs)
```

这可以尽量保持与当前训练流程一致，避免 sampler 导致一个 epoch 过短或过长。

### 3.6 验证集

验证与测试不使用训练 sampler，继续保持：

- 整图输入
- 顺序 loader
- 当前 `val_accuracy / val_macro_f1 / val_auroc / val_auprc` 口径不变

---

## 4. 子图训练模式

本文档同时规划两种子图模式：

- 静态预切子图库：首版推荐实现
- 动态在线裁图：备选长期方案

### 4.1 共同规则

无论采用哪种子图模式，都必须满足：

1. 仅训练集切子图，验证与测试保持整图
2. `M` 的定义始终是“不同父切片数”，不是“子图数”
3. 同一 batch 内每个 `parent_slide_id` 最多出现 1 个子图
4. 子图继承父切片的：
   - `bat_dom`
   - `cancer_dom`
5. 子图必须有唯一 `slide_id`

### 4.2 推荐默认方案：静态预切子图库

#### 4.2.1 适用原因

静态预切更贴合当前 STOnco 的数据流：

- 现有 `prepare_graphs(...)` 本身就是离线构图
- 易于复用当前 `PyGDataLoader`
- 容易验证 sampler 是否真正提升了早期混合速度

因此首版推荐默认采用静态预切。

#### 4.2.2 子图生成顺序

训练时的顺序固定为：

1. 完成父切片级 train/val split
2. 对 train 父切片生成子图库
3. 为每个子图创建新的 `PyGData`
4. 构建子图级 sampler
5. 验证与测试仍使用整图

#### 4.2.3 子图元数据规范

每个子图必须携带：

- `slide_id`: 唯一子图 id，例如 `BRCA3__sg07`
- `parent_slide_id`: 例如 `BRCA3`
- `subgraph_id`: 例如 `sg07`
- `bat_dom`: 继承父切片
- `cancer_dom`: 继承父切片

### 4.3 备选方案：动态在线裁图

动态在线裁图定义为：

- 训练时不预先生成完整子图库
- 每个 epoch 或每个 batch，从父切片临时生成一个局部子图

优点：

- 同一父切片可在不同 epoch 暴露不同局部区域
- 不需要提前持久化整套子图库

缺点：

- 改动现有 `prepare_graphs(...) -> train_graphs -> DataLoader` 流水线更多
- 更难保证复现性与调试稳定性

因此动态在线裁图在本方案中仅作为备选，不作为首版默认落地目标。

### 4.4 子图切分算法

#### 4.4.1 默认算法：空间固定窗口切块

首版推荐默认算法为：

- 基于 spot 的二维坐标 `xy`
- 在空间平面上做固定窗口切块
- 每个窗口内的 spots 组成候选子图
- 每个子图内部重新构建 kNN 图

不采用随机抽点，也不采用表达聚类切块。

原因：

- 当前 STOnco 的图结构本来就是空间 kNN 图
- 固定窗口切块最容易控制子图大小
- 与现有 `assemble_pyg(...)` 的逻辑最一致

#### 4.4.2 子图大小

首版默认参数建议：

- 目标子图大小：`800-1200` spots
- 允许范围：`500-1500`
- 最小有效值：`300`

当候选子图低于最小 spot 数时：

- 默认丢弃
- 或并入最近邻窗口

实现时默认优先丢弃过小块，避免增加复杂的块合并逻辑。

#### 4.4.3 边与特征构建

子图必须在子图内部重新建图：

- 使用子图自己的 `xy`
- 重新计算 `edge_index`
- 重新计算 `edge_weight`

不能直接从整图图结构中裁边后复用。

### 4.5 子图模式的 sampler

子图模式下，训练 batch 的采样语义固定为：

1. 先选 `K` 个 cancer
2. 每个 cancer 选 `M` 个不同父切片
3. 每个父切片随机选 1 个子图
4. 合并成当前 batch

注意：

- 这里的 `M` 是父切片数，不是子图数
- 同一父切片即使有多个可用子图，在一个 batch 内也只能贡献 1 个

### 4.6 子图模式的默认 batch 结构

若子图模式使显存允许更大的 batch，则默认推荐：

- `batch_size_graphs = 12` 时：`K=4, M=3`
- `batch_size_graphs = 8` 时：`K=4, M=2`

在当前 11 个 cancer 的训练集分布下：

- `K=4, M=2` 稳定可行
- `K=4, M=3` 可行，但会对 `PC / GBM / PRAD / RCC` 这类小癌种带来更频繁的重复抽样

因此建议：

- 先用一图一切片或子图版 `K=4, M=2` 验证 sampler 效果
- 再升级到子图版 `K=4, M=3`

### 4.7 验证与测试

子图训练模式下，验证与测试默认仍然保持整图，不拆分、不聚合。

原因：

- 更容易直接比较与当前 STOnco 结果的差异
- 可以更明确地判断“训练 sampler 是否改善泛化”

本方案不把“验证也拆子图并按 `parent_slide_id` 聚合”作为首版默认行为。

---

## 5. 配置与接口方案

### 5.1 新增配置项

建议在 `cfg` 和 CLI 中新增以下配置：

```text
sampler_mode: random | cancer_balanced | cancer_balanced_subgraph
sampler_k_cancers: int
sampler_m_per_cancer: int
sampler_enforce_distinct_batch: 0 | 1

subgraph_mode: off | static | online
subgraph_target_spots: int
subgraph_min_spots: int
```

默认值建议：

```text
sampler_mode = random
sampler_k_cancers = 4
sampler_m_per_cancer = 2
sampler_enforce_distinct_batch = 1

subgraph_mode = off
subgraph_target_spots = 1000
subgraph_min_spots = 300
```

### 5.2 训练入口改造点

建议改造点集中在 [`stonco/core/train.py`](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py)：

1. 在 `prepare_graphs(...)` 之后，根据 `sampler_mode / subgraph_mode` 构建训练样本列表
2. 新增训练用 `BatchSampler`
3. 仅替换 train loader，val loader 保持不变

推荐逻辑：

```text
if sampler_mode == 'random':
    train_loader = PyGDataLoader(..., shuffle=True)
else:
    train_loader = PyGDataLoader(..., batch_sampler=custom_batch_sampler)
```

### 5.3 日志与诊断

为便于确认 sampler 是否按预期工作，建议在训练日志中增加每 epoch 的 batch 覆盖统计：

- 平均每 batch 的唯一 cancer 数
- 平均每 batch 的唯一父切片数
- 平均每 batch 的唯一 `bat_dom` 数
- 子图模式下平均每 batch 的唯一 `parent_slide_id` 数

这些统计应写入 `loss_components.csv` 或单独的 sampler diagnostics 文件。

---

## 6. 测试与验收标准

### 6.1 一图一切片模式

需要验证：

1. 每个训练 batch 的癌种覆盖符合 `K`
2. 每个 cancer 内样本数尽量符合 `M`
3. 不足时降级顺序符合本文定义
4. 验证集输出与当前整图评估口径一致

### 6.2 子图模式

需要验证：

1. train/val split 在切图前完成
2. 子图 `slide_id` 唯一
3. 子图可正确回溯 `parent_slide_id`
4. 同一 batch 内不重复 `parent_slide_id`
5. 验证集仍按整图评估

### 6.3 训练动力学验收

与当前 `shuffle=True` 基线相比，sampler 首版的验收目标是：

1. 在前 `50-100` epoch 内，更快降低：
   - `train_cancer_domain_acc`
   - `avg_mmd_raw_cancer`
2. 潜空间混合开始出现的时间明显提前
3. 不要求首版立即提升最终 `val_accuracy`
4. 但不应显著恶化当前 best checkpoint 的验证性能

---

## 7. 待确认项

以下问题不阻碍首版实现设计，但在真正开始编码前建议你确认：

### 7.1 子图是否允许 overlap

当前本文默认：

- 固定窗口切块
- 首版按无 overlap 或极小 overlap 设计

待你确认：

- 显式支持 `10%-20%` 的小重叠窗口

### 7.2 静态预切是否持久化

当前本文默认：

- 静态预切子图库可以先在训练启动时内存构建

待你确认：

- 不需要把子图库持久化到磁盘，供复用和调试

### 7.3 动态在线裁图的接口预留程度

当前本文默认：

- 动态在线裁图作为备选方案写入文档
- 首版实际实现以静态预切为主

待你确认：

- 首版编码时，是只实现静态预切

---

## 8. 推荐实施顺序

建议按以下顺序推进：

1. 先实现一图一切片模式的 `cancer-balanced BatchSampler`
2. 在整图模式下验证 `K=4, M=2` 是否能显著提前混合
3. 再实现静态预切子图库
4. 在子图模式下把目标 batch 升级到 `K=4, M=3`
5. 最后再评估是否需要动态在线裁图

这样最容易判断每一步到底带来了什么收益，也最符合当前 STOnco 的代码形态。
