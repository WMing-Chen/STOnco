# 在 STOnco 中以 Sliced Wasserstein 替换 Sinkhorn WB 主损失的调整方案

本文档基于当前 STOnco 仓库中的实际实现，给出一个完整的 `sliced_wasserstein` 调整方案，用于替换当前 generated-support WB 中计算成本较高的 `sinkhorn_divergence` 主损失。

目标不是改变 STOnco 的推理路径，也不是把当前 WB 模块改造成一个严格意义上的全局 barycenter solver，而是在保留现有 generated-support 设计的前提下，用更低成本、可扩展到更多 spot 的 Wasserstein-style 分布对齐损失替代当前的 batch 内 Sinkhorn divergence。

---

## 1. 当前实际代码状态

当前 WB 相关主逻辑位于以下文件：

- `stonco/core/train.py`
- `stonco/core/wb_potentials.py`
- `docs/Tutorial.md`
- `docs/PLAN_sinkhorn_divergence_WB_STOnco.md`

### 1.1 当前 WB loss 类型

当前 `train.py` 中 `wb_loss_type` 只支持：

- `dual_potential`
- `euclidean_pairwise`
- `sinkhorn_divergence`

对应位置：

- CLI 参数定义：`stonco/core/train.py`
- 默认配置：`stonco/core/train.py`
- 配置合法性校验：`stonco/core/train.py`
- `GeneratedSupportWBLoss` 内部分支：`stonco/core/wb_potentials.py`

### 1.2 当前 generated-support 路径

当前 generated-support WB 结构是：

```text
h = GNN_theta(...)
b = T_phi(h) = support_map(h)
```

其中 `support_map` 由 `GeneratedSupportMap` 定义，当前是 identity-initialized residual map：

```text
b = h + MLP(LayerNorm(h))
```

重要特点：

- `support_map` 参数 `phi` 是跨 batch 共享和更新的
- `support_map` 不决定点数，只对每个进入 WB 的 spot 做一对一映射
- 有多少个 selected spots，就会生成多少个 `b` 点

### 1.3 当前 WB spot 采样逻辑

当前 `GeneratedSupportWBLoss._prepare(...)` 的预处理顺序是：

1. 先按 `wb_spots_per_graph` 限制每张 graph/slide 进入 WB 的 spot 数
2. 可选按 `wb_spots_per_cancer` 再做每癌种 cap
3. 可选 `wb_label_balanced_sampling`
4. 对 `h` 和 `b` 都做 `LayerNorm`
5. 只保留满足 `wb_min_spots` 的 active cancers
6. 若 active cancer 数小于 `wb_min_cancers`，该 batch 的 WB 直接跳过

这套预处理逻辑对 `sliced_wasserstein` 应继续复用，不应重写另一套采样流程。

### 1.4 当前 Sinkhorn 路径

当前 `sinkhorn_divergence` 分支位于 `stonco/core/wb_potentials.py`：

- `_sinkhorn_ot(...)`
- `_sinkhorn_divergence_loss(...)`
- `model_loss(...)` 中 `self.loss_type == 'sinkhorn_divergence'` 分支

核心特点：

- 每个 batch 内部通过 `torch.cdist(...)` 构造代价矩阵
- 使用 log-domain Sinkhorn 迭代，迭代次数由 `wb_sinkhorn_iters` 控制
- 对每个 active cancer 的 `x_k = h[cancer == k]` 与共享 `b_support` 计算 debiased Sinkhorn divergence
- `sinkhorn_divergence` 不训练跨 batch potential bank
- anchor loss 仍由训练循环单独加到总损失里

### 1.5 当前训练循环如何使用 WB

训练中 WB 相关调用链为：

1. 主模型前向得到 `h`
2. `b_wb = support_map(h)`
3. `wb_module.model_loss(h, b_wb, cancer_dom, graph_nodes, y)`
4. 训练总损失增加：

```text
loss_total += wb_lambda_t * (loss_wb + wb_anchor_weight * loss_wb_anchor)
```

这里：

- `loss_wb` 是 WB 主分布对齐项
- `loss_wb_anchor` 是 `h` 和 `b` 的 anchor
- `wb_lambda_t` 由 warmup + ramp 调度给出

### 1.6 当前 diagnostics / artifacts

当前 WB 会记录并保存：

- `avg_wb_loss`
- `avg_wb_potential_loss`
- `avg_wb_dual_obj`
- `avg_wb_euclid_pairwise`
- `avg_wb_sinkhorn`
- `avg_wb_anchor`
- `avg_wb_state_direction`
- `avg_wb_active_cancers`
- `avg_wb_active_spots`
- `wb_lambda`

验证阶段在 `wb_eval_loss=1` 时也会计算对应 validation WB diagnostics。

当前 artifact 仍保存：

- `wb_support_map_last.pt`
- `wb_potentials_last.pt`
- `wb_config.json`

即便 `sinkhorn_divergence` 不训练 potential bank，当前实现仍保留 `wb_potentials_last.pt` 文件名以保持布局兼容。

---

## 2. 本次调整目标

本次方案的核心目标是：

1. 在 `wb_loss_type` 中新增 `sliced_wasserstein`
2. 保留当前 generated-support 结构，不引入 `pooled` 参考分布
3. 固定对齐到 `support_map` 生成的共享 support
4. 继续保留 `anchor loss`
5. 继续复用当前 `_prepare(...)` 里的 WB spot 采样逻辑
6. 支持 cancer 侧样本数与 support 侧样本数不相等
7. 用更低成本的 Wasserstein-style 损失替换 `sinkhorn_divergence`
8. 让每张 slide 可以支持显著更多 spot 参与 WB

---

## 3. 为什么用 Sliced Wasserstein

### 3.1 当前 Sinkhorn 的问题

当前 `sinkhorn_divergence` 的主要问题不是定义不合理，而是工程成本过高：

- 每个 active cancer 都要和 `b_support` 构造代价矩阵
- 每次 WB forward 都要做多轮 Sinkhorn 迭代
- 在 batch size、active cancers、spot 数上升时，耗时和显存都明显增加

直接后果是：

- `wb_spots_per_graph` 很难开大
- 每张 slide 实际参与 WB 的 spot 太少
- generated-support 几何虽然存在，但统计稳定性不够

### 3.2 与 MMD / Gaussian-Bures 的区别

相对当前 MMD：

- `sliced_wasserstein` 仍保留 Wasserstein 风格几何
- 不只是 kernel discrepancy
- 在高维分布对齐上更贴近“投影后一维 OT”的思想

相对 Gaussian/Bures-Wasserstein：

- `sliced_wasserstein` 不只看均值和协方差
- 更适合非高斯、多峰 latent 分布
- 更不容易把 tumor/normal 等细结构过度压缩为二阶统计

### 3.3 与 full Sinkhorn 的取舍

`sliced_wasserstein` 不会像 full Sinkhorn 那样显式求高维 transport plan，但它有两个关键工程优势：

- 没有 `n x m x iter` 的迭代型 OT 成本
- 在可接受的计算预算下，通常能让更多 spot 进入损失

因此，它在当前 STOnco 里更适合被定位为：

```text
低成本、可扩展的 Wasserstein-style generated-support alignment
```

而不是：

```text
严格求解 batch 内 entropy-regularized OT
```

---

## 4. 调整后的目标定义

### 4.1 保留当前 shared support 语义

本方案固定使用：

```text
b = T_phi(h)
```

不引入 `pooled h` 作为 reference，也不新增 `wb_sw_ref` 参数。

也就是说，`sliced_wasserstein` 版本下每个 cancer 分布对齐的对象固定是 generated support：

```text
P_k^h  <->  Q_phi^b
```

这和当前 Sinkhorn 版的 shared support 语义保持一致。

### 4.2 继续使用 support subset

当前 `wb_support_size` 逻辑是：

```text
b_support = subset(b)
```

本方案建议继续保留这一路径：

- 若 `wb_support_size <= 0` 或 selected `b` 数不超过该阈值，则使用全量 `b`
- 否则，从 selected `b` 中随机采样 `b_support`

这样可以保持：

- 与当前实现的参数语义兼容
- 在 `sliced_wasserstein` 下进一步控制开销
- 允许在更大 `wb_spots_per_graph` 下继续把 reference support 点数控制在合理范围

### 4.3 支持不等长样本

`sliced_wasserstein` 不要求：

```text
|x_k| == |b_support|
```

在 STOnco 中，这一点非常重要，因为：

- 不同 cancer 在一个 batch 内的有效 spot 数不同
- `wb_support_size` 可能截断 support 侧点数
- `wb_spots_per_cancer` 可能导致各癌种样本数不同

因此，本方案明确采用：

```text
weighted 1D Wasserstein via shared quantile / CDF alignment
```

而不是要求等长后逐位比较。

---

## 5. 数学定义

设当前 batch 经 `_prepare(...)` 后得到：

```text
h_sel in R^{N x d}
b_sel in R^{N x d}
cancer labels
active cancers = K_B
```

并令：

```text
b_support = subset(b_sel)
```

对某个 active cancer `k`：

```text
x_k = h_sel[cancer == k] in R^{n_k x d}
q   = b_support          in R^{m x d}
```

其中 `n_k` 与 `m` 不要求相等。

### 5.1 随机投影

采样 `L = wb_sw_num_projections` 个随机方向：

```text
theta_l in R^d, ||theta_l||_2 = 1, l = 1..L
```

对每个方向做投影：

```text
u_l = x_k @ theta_l   in R^{n_k}
v_l = q   @ theta_l   in R^{m}
```

### 5.2 一维 Wasserstein

对每个方向 `l`，计算一维 `W_2^2`：

```text
W_2^2(u_l, v_l)
```

这里的实现不假设 `n_k == m`，而是采用：

- 默认均匀权重
- 各自排序
- 构造各自经验 CDF
- 在共同 quantile / CDF 断点上对齐后积分平方差

即采用成熟 OT 工具常见的 1D Wasserstein 做法，而不是简单要求长度相同。

对 STOnco 第一版 `sliced_wasserstein` 的实现，本文档在这里进一步明确固定为：

- 使用**均匀权重的精确离散 1D `W_2^2`**
- 基于排序后的经验 CDF / quantile 对齐
- 支持不等长样本
- 不做随机下采样到 `min(n_k, m)`
- 不做固定 quantile 网格插值近似
- 不在每个投影方向上再跑 1D Sinkhorn

这里“不在每个投影方向上再跑 1D Sinkhorn”的含义是：

- 不把投影后的 `u_l`、`v_l` 再构造成 1D cost matrix 后做 entropy-regularized Sinkhorn 迭代
- 而是直接利用 1D OT 的排序 + 经验 CDF / quantile 对齐公式，计算精确离散 `W_2^2`

这样做的原因是：

- 1D OT 本身已有更便宜的精确解
- 没必要再引入 `epsilon`、迭代次数和 entropy bias
- 也更符合本次方案“降低开销、扩大可参与 spot 数”的工程目标

本方案中不再把 `p` 暴露为可调参数，而是直接固定为 `2`。原因是：

- 当前 STOnco 的 Sinkhorn 代价本身是平方欧氏距离
- 当前 anchor loss 也是平方差
- 主流 SW 工具和实现默认也多采用 `p=2`
- 先固定 `p=2` 可以减少一个超参数，避免第一版实现过度扩张

### 5.3 Sliced Wasserstein

对某个 cancer `k` 的 sliced Wasserstein 损失定义为：

```text
SW_{k} = (1 / L) * sum_l W_2^2(u_l, v_l)
```

对 STOnco 第一版 `sliced_wasserstein`，这里进一步明确：

- 直接使用投影后的一维精确离散 `W_2^2`
- 不再像当前高维 `sinkhorn_divergence` 一样额外按 latent 维度 `d` 做缩放
- 也就是说，不额外除以 `d`

这样设定的原因是：

- `sliced_wasserstein` 在每个方向上处理的是标量投影后的 1D 分布，而不是高维平方距离矩阵
- `_prepare(...)` 后的 `h`、`b` 已经做过 `LayerNorm`
- 标准 SW 定义本身通常也不会在投影后的 1D `W_2^2` 外再额外乘一个 `1/d`

当前 STOnco 更适合把各 cancer 的损失再做均值：

```text
L_WB_SW = mean_k SW_k
```

其中 `k` 遍历 active cancers。

### 5.3.1 support 侧样本过少时的边界行为

对 `sliced_wasserstein`，本文档进一步明确：

- 不再沿用当前 `sinkhorn_divergence` 中“`b_support.size(0) < wb_min_spots` 就跳过”的条件
- `wb_min_spots` 继续只用于判定 active cancer 是否成立
- support 侧只要求 `b_support.size(0) >= 2`

也就是说：

- 若 `b_support.size(0) < 2`，当前 batch 的 sliced Wasserstein 主项返回 invalid / zero，与其他“WB 不可计算”情形一致
- 若 `b_support.size(0) >= 2`，即便它小于 `wb_min_spots`，仍然照常计算 sliced Wasserstein

这样做的原因是：

- `wb_min_spots` 的语义是“每个 active cancer 至少有多少 spot 才参与对齐”
- 它不是为 shared support 侧定义的阈值
- 对精确离散 1D `W_2^2` 而言，support 侧只要不是退化到单点，就仍然可以作为非退化经验分布参与计算

### 5.4 总损失

训练总损失仍保持当前 WB 外层结构：

```text
L_total
= L_task
+ L_domain_optional
+ L_mmd_optional
+ lambda_wb(t) * (
    L_WB_SW
    + wb_anchor_weight * L_anchor
  )
```

若开启 `state_direction`，则与当前实现保持一致，作为 raw WB 分支内部附加项：

```text
raw_wb_loss = L_WB_SW + wb_state_direction_weight * L_state_direction
```

anchor 仍在训练循环中单独乘 `wb_anchor_weight`。

---

## 6. 参数设计

### 6.1 保留的参数

以下现有参数应继续保留，并对 `sliced_wasserstein` 生效：

- `use_wb_align`
- `wb_loss_type`
- `lambda_wb`
- `wb_warmup_epochs`
- `wb_ramp_epochs`
- `wb_support_hidden`
- `wb_support_dropout`
- `wb_anchor_weight`
- `wb_spots_per_graph`
- `wb_spots_per_cancer`
- `wb_support_size`
- `wb_min_cancers`
- `wb_min_spots`
- `wb_label_balanced_sampling`
- `wb_state_direction`
- `wb_state_direction_weight`
- `wb_eval_loss`

### 6.2 新增参数

本方案建议新增：

- `wb_sw_num_projections`
  - 含义：随机投影方向数
  - 作用：控制高维 Wasserstein 近似精度和开销

### 6.3 不新增的参数

本方案明确不新增：

- `wb_sw_ref`
- `wb_sw_p`

原因：

- 当前需求已经明确固定对齐到 generated support
- 当前需求已经明确固定 `p=2`
- 不需要 `pooled` reference
- 引入额外参数会增加分支复杂度，但不会解决当前核心问题

### 6.4 对现有参数的适用性变化

以下参数在 `sliced_wasserstein` 下不再使用或应视为无效：

- `wb_epsilon`
- `wb_sinkhorn_iters`

以下参数只对其他 loss 类型有意义：

- `wb_regularizer`
- `wb_potential_hidden`
- `wb_potential_lr`
- `wb_potential_weight_decay`
- `wb_pot_every_n_steps`
- `wb_euclid_pairwise_weight`
- `wb_potential_weight`
- `wb_potential_constraint_weight`

建议在配置校验阶段打印一次提示，说明这些参数在 `wb_loss_type=sliced_wasserstein` 下被忽略或不参与主逻辑。

---

## 7. 代码改造方案

### 7.1 `stonco/core/train.py`

#### 7.1.1 CLI 参数

需要修改：

- 扩展 `--wb_loss_type` choices，加入 `sliced_wasserstein`
- 新增：
  - `--wb_sw_num_projections`

#### 7.1.2 默认配置

默认配置建议新增：

```text
wb_sw_num_projections
```

同时保留原有 Sinkhorn 默认参数，以兼容旧实验。

#### 7.1.3 配置覆盖与校验

需要新增：

- 解析 `wb_sw_num_projections`
- 校验 `wb_sw_num_projections >= 1`

并在 `wb_loss_type == sliced_wasserstein` 时打印一次提示：

- `wb_epsilon` 忽略
- `wb_sinkhorn_iters` 忽略
- `wb_regularizer` 忽略

### 7.2 `stonco/core/wb_potentials.py`

#### 7.2.1 `GeneratedSupportWBLoss.__init__`

需要新增构造参数：

- `sw_num_projections`

并保存为成员变量；`p` 在模块内部固定为 `2`，不作为构造参数暴露。

#### 7.2.2 `loss_type` 分支扩展

当前：

- `euclidean_pairwise` -> `self.potentials = SinglePotentialBank(...)`
- `dual_potential` -> `self.potentials = DualPotentialBank(...)`
- `sinkhorn_divergence` -> `self.potentials = None`

调整后：

- `sliced_wasserstein` 也应设置为 `self.potentials = None`

因为它不需要跨 batch potential bank。

#### 7.2.3 新增内部函数

建议新增以下函数：

- `_sample_projection_directions(...)`
- `_wasserstein_1d_weighted(...)`
- `_sliced_wasserstein_loss(...)`

其中：

`_sample_projection_directions(...)`

- 生成标准高斯随机方向
- 对每个方向做 `L2` 归一化
- 形状可设计为 `R^{L x d}`

`_wasserstein_1d_weighted(...)`

- 输入两个 1D 向量
- 默认使用均匀权重
- 支持不等长样本
- 通过排序 + CDF / quantile 对齐计算 `W_2^2`

`_sliced_wasserstein_loss(...)`

- 输入 `prepared`
- 从 `prepared['b']` 得到 `b_support`
- 对每个 active cancer 计算 `SW_k`
- 对 active cancers 取均值

#### 7.2.4 `model_loss(...)`

新增：

```text
if self.loss_type == 'sliced_wasserstein':
    sw_loss = self._sliced_wasserstein_loss(prepared)
    raw_loss = sw_loss + self.state_direction_weight * shape_loss
```

并写入 stats。

### 7.3 `potential_loss(...)`

`sliced_wasserstein` 和 `sinkhorn_divergence` 一样，不需要 Step A potential update。

因此 `potential_loss(...)` 保持当前逻辑即可：

- `self.potentials is None` 时直接返回空统计

### 7.4 训练循环

训练主循环无需改动结构，只需支持新统计字段：

- `loss_wb` 仍来自 `wb_module.model_loss(...)`
- `loss_total` 的组合方式保持不变

### 7.5 验证循环

若 `wb_eval_loss=1`：

- 验证阶段也应支持计算 `sliced_wasserstein`
- 逻辑与当前 `sinkhorn_divergence` / `euclidean_pairwise` 一致

但对 `sliced_wasserstein`，为了降低 diagnostics 抖动，本文档进一步明确建议：

- 训练阶段继续保持随机 spot 采样、随机 support 子采样、随机投影方向
- 验证阶段改为 deterministic

具体建议为：

- `_select_indices(...)` 支持可选随机数发生器
- `_support_subset(...)` 支持可选随机数发生器
- `_sample_projection_directions(...)` 支持可选随机数发生器
- train 阶段传入 `None`，沿用当前随机行为
- val 阶段传入固定 seed 对应的 generator

这样做的原因是：

- 当前 validation WB diagnostics 已经依赖 spot 采样与 support 子采样
- `sliced_wasserstein` 再额外引入随机投影后，验证波动会进一步增大
- 把 validation 改为固定 seed 的 deterministic eval，更利于比较不同 epoch 和不同实验之间的 WB 指标

---

## 8. 统计输出与可视化改造

### 8.1 新增统计字段

当前有：

- `avg_wb_sinkhorn`
- `val_avg_wb_sinkhorn`

本方案建议新增独立字段：

- `avg_wb_sliced_wasserstein`
- `val_avg_wb_sliced_wasserstein`

理由：

- 避免复用 `avg_wb_sinkhorn` 导致语义混乱
- CSV、图表、后续分析更清晰
- 不破坏对旧结果的解释

### 8.2 保留旧字段

建议保留：

- `avg_wb_sinkhorn`
- `val_avg_wb_sinkhorn`

因为：

- 旧 checkpoint / 旧实验已有该语义
- 新旧 loss 可以并存于代码中
- `wb_loss_type != sinkhorn_divergence` 时旧字段继续为 `NaN`

### 8.3 `loss_components.csv`

需要在保存训练历史时新增两列：

- `avg_wb_sliced_wasserstein`
- `val_avg_wb_sliced_wasserstein`

### 8.4 `wb_train_loss.svg`

当前图表逻辑已支持按可用曲线自动布局。

也就是说，当前 `wb_train_loss.svg` 不是按 `wb_loss_type` 写死某一套固定面板，而是先检查哪些指标当前实际有有限值，再根据可用曲线数量自动决定：

- 需要绘制哪些子图
- 子图的总数量
- 行列排版

因此，`sliced_wasserstein` 方案中不需要为新的 `wb_loss_type` 单独写一套固定布局逻辑，只需要把新的曲线字段加入候选指标集合，继续沿用当前“按可用曲线自动筛选 + 自动网格排版”的机制即可。

本方案建议在可用曲线面板中新增：

- `avg_wb_sliced_wasserstein`
- `val_avg_wb_sliced_wasserstein`

并在 `wb_loss_type=sliced_wasserstein` 时：

- 显示 `wb_loss`
- 显示 `wb_anchor`
- 显示 `wb_sliced_wasserstein`
- 可选显示 `wb_state_direction`
- 显示 `wb_lambda`
- 显示 active cancers / active spots

---

## 9. artifact 兼容策略

当前实现保存：

- `wb_support_map_last.pt`
- `wb_potentials_last.pt`
- `wb_config.json`

对于 `sliced_wasserstein`，建议延续当前 `sinkhorn_divergence` 的兼容策略：

- 仍保留 `wb_potentials_last.pt` 文件名
- 其中保存 `wb_module.state_dict()`
- 虽然没有实际 potential bank，但不改变 WB artifact 布局

这样可以减少：

- `save/load` 逻辑分支
- 历史工具链适配成本
- 文档和结果目录差异

---

## 10. 推荐默认超参

基于当前 STOnco 默认 backbone 和当前 WB 设计，建议文档中的初始推荐值为：

- `wb_sw_num_projections = 64`
- `wb_spots_per_graph = 512`
- `wb_support_size = 256`
- `wb_anchor_weight = 0.5`
- `lambda_wb` 起点仍建议在较小范围搜索

补充说明：

- 若 latent 维度接近 `64`，且更在意节省算力，可把 `wb_sw_num_projections` 从 `64` 下调到 `32`

推荐调参顺序：

1. 先试 `wb_sw_num_projections = 64`
2. 先把 `wb_spots_per_graph` 设到 `512`
3. 若 latent 维度接近 `64` 且更关注吞吐，可把 `wb_sw_num_projections` 从 `64` 降到 `32`
4. 若后续还想继续放大每张 slide 的 spot 数，再向更高 `wb_spots_per_graph` 做增量试验

---

## 11. 预期收益

与当前 `sinkhorn_divergence` 相比，本方案的主要收益是：

- 每个 batch 不再进行高维 OT 迭代
- 更容易扩大 `wb_spots_per_graph`
- 更容易扩大 `wb_support_size`
- 每张 slide 能有更多 spot 参与 WB
- 保留 generated-support 设计
- 保留 Wasserstein-style 几何信息
- 不退化为纯 MMD 或纯二阶统计

工程上更实际的预期是：

- 从当前典型 `64~128` spot / slide 的 WB 规模
- 提升到更现实的 `256~512` spot / slide 区间

这也是本次改造的主要价值所在。

---

## 12. 风险与注意事项

### 12.1 它不是 full OT

`sliced_wasserstein` 仍然不是完整高维 OT。

因此需要明确：

- 它是对高维 Wasserstein 的投影近似
- 不是当前 Sinkhorn 的同义替换
- 它的目标是更优工程折中，不是严格保真替代

### 12.2 投影数太少会导致方差偏大

若 `wb_sw_num_projections` 太小：

- 方向覆盖不足
- loss 波动可能较大
- 对高维几何的近似偏粗

### 12.3 样本数极少时收益有限

如果某个 active cancer 在 batch 中本身就只有极少数 spot：

- 即使换成 `sliced_wasserstein`
- 一维经验分布分辨率仍然有限

因此仍应保留：

- `wb_min_spots`
- 合理的 `wb_spots_per_graph`
- 合理的 sampler 策略

### 12.4 仍需注意过度对齐

和当前 WB 一样，`sliced_wasserstein` 仍可能：

- 让癌种信息被过度抹平
- 干扰 tumor/normal 判别结构

因此仍应联动观察：

- 主任务指标
- embedding mixing
- cancer probe
- tumor/normal 结构可视化

---

## 13. 验证与验收标准

### 13.1 功能验收

至少满足：

1. `wb_loss_type=sliced_wasserstein` 能正常训练
2. `loss_components.csv` 正确记录 `avg_wb_sliced_wasserstein`
3. `wb_train_loss.svg` 在该模式下展示对应曲线
4. `wb_eval_loss=1` 时能在验证阶段得到对应统计
5. 不等长样本时不报 shape mismatch

### 13.2 工程验收

至少观察：

1. 在相同资源预算下，`wb_spots_per_graph` 能显著高于当前 Sinkhorn 版本
2. 单 epoch 时间明显低于 `sinkhorn_divergence`
3. 显存使用可接受

### 13.3 训练效果验收

至少比较：

- baseline
- MMD
- `wb_loss_type=euclidean_pairwise`
- `wb_loss_type=sinkhorn_divergence`
- `wb_loss_type=sliced_wasserstein`

重点看：

- `val_macro_f1`
- `val_auprc`
- `avg_wb_active_spots`
- embedding mixing
- tumor/normal 可分性是否保持

---

## 14. 实施顺序建议

建议按以下顺序改：

1. 先扩展 `wb_loss_type` 与 CLI / 默认配置
2. 在 `GeneratedSupportWBLoss` 中新增 `sliced_wasserstein` 分支
3. 先实现不等长 1D Wasserstein
4. 再实现 `_sliced_wasserstein_loss(...)`
5. 再接入训练 / 验证统计字段
6. 最后更新 `Tutorial.md` 和绘图逻辑

这样可以先保证：

- loss 跑通
- 数值定义正确
- 再逐步补齐训练可视化和文档

---

## 15. 与当前需求的对应关系

本方案已经明确满足你当前提出的需求：

- 继续基于 STOnco 当前实际代码结构
- 用 `sliced_wasserstein` 替换当前 `sinkhorn_divergence`
- 固定对齐到 `support_map` 生成的共享 reference
- 不引入 `pooled` reference
- 不新增 `wb_sw_ref`
- 允许 support 侧和 cancer 侧样本数不一致
- 明确采用 quantile / CDF 对齐的一维 Wasserstein
- 目标是显著提高每张 slide 可参与 WB 的 spot 数量级

---

## 16. 待确认事项

以下关键点请你确认；确认后再按本文档执行代码改动：

1. 文档文件名就使用：(已确认)

```text
docs/PLAN_sliced_wasserstein_WB_STOnco.md
```

2. `wb_loss_type` 按下面扩展为四选一：(已确认)

```text
dual_potential | euclidean_pairwise | sinkhorn_divergence | sliced_wasserstein
```

3. 新增参数只保留：(已确认)

```text
wb_sw_num_projections
```

并且不新增：

```text
wb_sw_ref
wb_sw_p
```

4. `wb_support_size` 继续保留并用于 `sliced_wasserstein`，即允许：(已确认)

```text
b_support = subset(selected b)
```

而不是强制始终使用全量 `b`

5. diagnostics 字段按本文档新增独立命名：(已确认)

```text
avg_wb_sliced_wasserstein
val_avg_wb_sliced_wasserstein
```

同时保留旧字段：

```text
avg_wb_sinkhorn
val_avg_wb_sinkhorn
```

6. 文档里的默认推荐值按本文档写为：(已确认)

- `wb_sw_num_projections = 64`
- `wb_spots_per_graph = 512`
- 当 latent 接近 `64` 时，可把 `wb_sw_num_projections` 降到 `32` 以节省算力

7. 对不等长样本的一维 Wasserstein 实现，是否按本文档采用： (已确认)

- 默认均匀权重
- quantile / CDF 对齐
- 精确离散 1D `W_2^2`
- 不在投影后的 1D 问题上再跑 Sinkhorn
- 不额外按 latent 维度 `d` 缩放

8. `sliced_wasserstein` 的 validation 计算方式是否固定为： (已确认)

- train 阶段随机采样 / 随机投影
- val 阶段固定 seed，做 deterministic eval

9. support 侧样本过少时的边界行为是否固定为： (已确认)

- `wb_min_spots` 只约束 active cancer
- `b_support.size(0) < 2` 时跳过当前 SW 主项
- `b_support.size(0) >= 2` 时即使小于 `wb_min_spots` 也照算

而不是采用：

- 先裁成相同长度再逐位比较
- 或随机下采样到 `min(n, m)` 作为默认实现

10. artifact 兼容策略继续保持当前形式：(已确认)

- 仍保存 `wb_support_map_last.pt`
- 仍保存 `wb_potentials_last.pt`
- 即便 `sliced_wasserstein` 不训练 potential bank，也不改单文件名布局
