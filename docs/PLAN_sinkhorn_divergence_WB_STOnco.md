# 在 STOnco 中加入 generated-support Sinkhorn divergence 作为 WB 主损失的方案 C

本文档给出一个独立的方案 C：

```text
--wb_loss_type sinkhorn_divergence
```

目标是在不改变 STOnco 推理路径、不引入 `z_wb`、不引入 finite learnable barycenter atoms 的前提下，用更标准、可解释的 debiased Sinkhorn divergence 替代当前 `dual_potential` 的 neural dual surrogate，作为 Wasserstein-style barycenter alignment 的主损失。

核心原则：

- `h` 仍然是 STOnco GNN 输出的 latent，也是 tumor/normal 分类 head 使用的表征。
- 继续保留 generated support：

```text
b_i = T_phi(h_i)
```

- `T_phi` 是跨 batch 共享和更新的可学习 support map。
- Sinkhorn 本身是每个 mini-batch 内的数值 OT 求解器，不保存跨 batch neural potentials。
- 不再需要 `f_k/g_k` potential bank、`opt_pot` 和 Step A potential update。
- 训练目标是让每个 cancer source 的 `h` 分布对齐到共享 generated support 分布 `Q_phi^b`。

## 1. 问题定义

当前 STOnco 已经有：

```text
h = GNN_theta(x, edge_index)
logits = ClassifierHead(h)
```

当前 generated-support WB 版本额外加入：

```text
b = T_phi(h)
```

其中当前实现中的 `GeneratedSupportMap` 是 identity-initialized residual map：

```text
b = h + MLP(LayerNorm(h))
```

方案 C 保留这一路径，但把 WB 主损失改为：

```text
L_WB_sink = mean_{k in active cancers} SinkhornD(P_k^h, Q_phi^b)
```

其中：

```text
P_k^h    = 当前 batch 中 cancer k 的 h 分布
Q_phi^b = 当前 selected spots 的 b=T_phi(h) 诱导的共享 support 分布
```

训练总损失为：

```text
L_total
= L_task
 + L_domain_optional
 + L_mmd_optional
 + lambda_wb(t) * (
     L_WB_sink
     + wb_anchor_weight * L_anchor
     + optional_state_direction
   )
```

其中：

```text
L_anchor = mean ||LayerNorm(h) - LayerNorm(b)||_2^2
```

`L_anchor` 必须保留，因为下游分类 head 使用的是 `h`，不是 `b`。如果只让 `b` 与各癌种分布对齐，`h` 仍可能保留 cancer-source 信息。

## 2. 为什么选择 Sinkhorn divergence

标准离散 OT / Wasserstein 对两个经验分布：

```text
P = sum_i a_i delta_{x_i}
Q = sum_j b_j delta_{y_j}
```

求解：

```text
min_pi sum_ij pi_ij c(x_i, y_j)
```

约束为：

```text
sum_j pi_ij = a_i
sum_i pi_ij = b_j
pi_ij >= 0
```

它不是要求所有 spot 两两靠近，而是通过 transport plan `pi_ij` 学习软匹配权重。

Cuturi 2013 提出用 entropic regularization 和 Sinkhorn matrix scaling 高效近似 OT。实际训练中更推荐使用 debiased Sinkhorn divergence：

```text
S_eps(P,Q)
= OT_eps(P,Q)
 - 1/2 OT_eps(P,P)
 - 1/2 OT_eps(Q,Q)
```

原因：

- raw entropic OT 有 entropy smoothing bias。
- debiased Sinkhorn divergence 满足 `S_eps(P,P) ≈ 0`，更适合作为分布差异。
- Feydy et al. 2019 将 Sinkhorn divergences 解释为 OT 和 MMD 之间的几何 divergence。
- Janati et al. 2020 指出 entropy bias 会造成 blurred barycenters，debiased 形式更适合 barycenter 学习。

对 STOnco 而言，Sinkhorn divergence 的作用是提供一个更标准的 Wasserstein-style 距离下降信号，用于评估和训练：

```text
P_k^h -> Q_phi^b
```

## 3. 与当前 dual_potential 的关系

当前 `dual_potential` 是 CWB-style neural dual-potential surrogate：

```text
f_k(h): source-side potential
g_k(b): support-side potential
v_k(b)=g_k(b)-mean_j g_j(b)
J_k = E[f_k(x)] - E[R_eps(f_k(x)+v_k(y)-c(x,y))]
```

训练时需要：

```text
Step A: update f_k/g_k potentials
Step B: update STOnco + support_map
```

方案 C 则是：

```text
直接计算 SinkhornD(P_k^h, Q_phi^b)
```

不需要长期保存的 neural potential bank。Sinkhorn 内部仍会产生 batch-level dual variables 或 scaling vectors，但这些是当前 batch 的数值求解变量，不是跨 batch 参数。

对比：

| 项目 | dual_potential | sinkhorn_divergence |
|---|---|---|
| 主机制 | neural dual surrogate | per-batch Sinkhorn divergence |
| potential bank | 需要 `f_k/g_k` | 不需要 |
| 额外 optimizer | 需要 `opt_pot` | 不需要 |
| 训练方式 | 交替 max-min | 单次 forward/backward |
| 是否更接近标准 OT 数值 | 中等 | 更强 |
| 是否保留 neural dual-potential 语义 | 是 | 否 |
| 是否保留 continuous generated support | 是 | 是 |

因此方案 C 应被命名为：

```text
generated-support Sinkhorn divergence WB alignment
```

而不是：

```text
CWB-style neural dual-potential barycenter
```

## 4. b=T_phi(h) 在方案 C 中的作用

`b=T_phi(h)` 是方案 C 的核心。它提供一个跨 batch 共享、可学习的 generated barycenter support。

如果没有 `T_phi`，只能做：

```text
SinkhornD(P_k^h, pooled H_batch)
```

这会退化为当前 batch 内的经验中心对齐，中心分布没有可学习参数。

保留 `T_phi` 后：

```text
Q_phi^b = distribution of T_phi(h)
```

`T_phi` 的参数 `phi` 会和 STOnco 主模型一起被 optimizer 更新。虽然 Sinkhorn 每个 batch 临时求解，但 `GNN_theta` 和 `T_phi` 是跨 batch 共享的：

```text
batch 1: BRCA/LUAD/COAD -> update theta, phi
batch 2: KIRC/STAD/PRAD -> update theta, phi
batch 3: OV/HCC/BRCA    -> update theta, phi
```

经过多个 batch，`T_phi` 被所有癌种组合共同约束，学习出全局共享的 generated-support geometry。

## 5. 计算流程

每个 training batch：

1. STOnco 前向：

```python
out = model(..., return_h=True)
h = out["h"]
```

2. 生成 support：

```python
b = support_map(h)
```

3. WB 内部沿用当前抽样逻辑：

```text
wb_spots_per_graph
wb_spots_per_cancer
wb_min_cancers
wb_min_spots
wb_label_balanced_sampling
```

4. 标准化：

```python
h_sel = LayerNorm(h_selected)
b_sel = LayerNorm(b_selected)
```

5. 取 active cancers：

```text
K_B = {k | cancer k selected spots >= wb_min_spots}
```

若：

```text
|K_B| < wb_min_cancers
```

则当前 batch 跳过 WB loss。

6. 对每个 active cancer：

```python
x_k = h_sel[cancer == k]
y = support_subset(b_sel)
loss_k = SinkhornD(x_k, y)
```

7. 多癌种平均：

```python
loss_sinkhorn = mean(loss_k for k in active_cancers)
```

8. anchor：

```python
loss_anchor = mean((h_sel - b_sel) ** 2)
```

9. 总损失：

```python
loss_total += lambda_wb_t * (
    loss_sinkhorn + wb_anchor_weight * loss_anchor
)
```

10. 反传更新：

```text
GNN_theta
ClassifierHead
support_map T_phi
optional state_direction
```

不更新：

```text
f_k/g_k potentials
opt_pot
```

## 6. Sinkhorn divergence 的最简实现

为减少依赖，第一版建议用内部 PyTorch log-domain Sinkhorn。复用现有：

```text
wb_epsilon
wb_support_size
```

最多新增一个：

```text
wb_sinkhorn_iters
```

如果不想新增 CLI 参数，可以先固定：

```text
sinkhorn_iters = 50
```

### 6.1 log-domain Sinkhorn OT

建议内部函数：

```python
def _sinkhorn_ot(self, x, y):
    eps = max(float(self.epsilon), 1e-6)
    n_iter = int(getattr(self, "sinkhorn_iters", 50))

    nx = x.size(0)
    ny = y.size(0)
    log_a = x.new_full((nx,), -math.log(nx))
    log_b = y.new_full((ny,), -math.log(ny))

    dim = max(int(x.size(-1)), 1)
    cost = torch.cdist(x, y, p=2).pow(2) / float(dim)

    u = x.new_zeros(nx)
    v = y.new_zeros(ny)
    for _ in range(n_iter):
        u = eps * (
            log_a - torch.logsumexp((v.view(1, -1) - cost) / eps, dim=1)
        )
        v = eps * (
            log_b - torch.logsumexp((u.view(-1, 1) - cost) / eps, dim=0)
        )

    log_pi = (
        log_a.view(-1, 1)
        + log_b.view(1, -1)
        + (u.view(-1, 1) + v.view(1, -1) - cost) / eps
    )
    pi = torch.exp(log_pi)
    return (pi * cost).sum()
```

### 6.2 debiased Sinkhorn divergence

```python
def _sinkhorn_divergence(self, x, y):
    ot_xy = self._sinkhorn_ot(x, y)
    ot_xx = self._sinkhorn_ot(x, x)
    ot_yy = self._sinkhorn_ot(y, y)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
```

### 6.3 multi-cancer WB loss

```python
def _sinkhorn_divergence_loss(self, prepared):
    h = prepared["h"]
    b_support = self._support_subset(prepared["b"])
    cancer = prepared["cancer"]
    active = prepared["active_domains"]

    losses = []
    for k in active:
        xk = h[cancer == int(k)]
        if xk.size(0) < int(self.min_spots):
            continue
        losses.append(self._sinkhorn_divergence(xk, b_support))

    if not losses:
        return self._zero(h, requires_grad=True)
    return torch.stack(losses).mean()
```

注意：上面是方案级伪代码，正式实现前需要用 toy tensors 检查：

- `x == y` 时 Sinkhorn divergence 是否接近 0。
- loss 是否可反传到 `x` 和 `y`。
- 不同 batch size 下是否出现 NaN。

## 7. 代码改造建议

### 7.1 train.py CLI

把：

```python
choices=["dual_potential", "euclidean_pairwise"]
```

改为：

```python
choices=["dual_potential", "euclidean_pairwise", "sinkhorn_divergence"]
```

配置校验同步扩展。

如果增加一个参数：

```python
parser.add_argument("--wb_sinkhorn_iters", type=int, default=None)
```

默认配置：

```python
"wb_sinkhorn_iters": 50
```

### 7.2 wb_potentials.py 初始化

`GeneratedSupportWBLoss.__init__()`：

```python
if self.loss_type == "euclidean_pairwise":
    self.potentials = SinglePotentialBank(...)
elif self.loss_type == "dual_potential":
    self.potentials = DualPotentialBank(...)
elif self.loss_type == "sinkhorn_divergence":
    self.potentials = None
else:
    raise ValueError(...)
```

`potential_parameters()`：

```python
def potential_parameters(self):
    if self.potentials is None:
        return []
    return self.potentials.parameters()
```

### 7.3 train.py potential optimizer

只有需要 potential 的 loss 才创建 `opt_pot`：

```python
needs_potential = cfg["wb_loss_type"] in {
    "dual_potential",
    "euclidean_pairwise",
}

if use_wb_align and needs_potential:
    opt_pot = torch.optim.AdamW(...)
else:
    opt_pot = None
```

Step A potential update 也只在 `opt_pot is not None` 时执行。

### 7.4 wb_potentials.py model_loss

新增分支：

```python
if self.loss_type == "sinkhorn_divergence":
    sink_loss = self._sinkhorn_divergence_loss(prepared)
    raw_loss = sink_loss + self.state_direction_weight * shape_loss
    stats.update({
        "valid": True,
        "wb_loss": raw_loss.detach(),
        "wb_sinkhorn": sink_loss.detach(),
        "wb_dual_obj": h.new_tensor(float("nan")),
        "wb_euclid_pairwise": h.new_tensor(float("nan")),
        "wb_anchor": anchor.detach(),
        "wb_state_direction": shape_loss.detach(),
        "wb_state_direction_count": float(shape_count),
    })
    return raw_loss, anchor, stats
```

### 7.5 训练曲线

当前已有 WB 曲线。新增统计项：

```text
avg_wb_sinkhorn
val_wb_sinkhorn
```

如果 `wb_loss_type != sinkhorn_divergence`，这些字段可为 NaN。

`wb_train_loss.svg` 需要根据不同 `wb_loss_type` 自动选择子图布局，而不是固定使用同一套面板：

```text
euclidean_pairwise:
  total/task/WB/anchor/euclidean_pairwise/potential_score/state_direction/lambda

dual_potential:
  total/task/WB/anchor/dual_obj/potential_loss/state_direction/lambda

sinkhorn_divergence:
  total/task/WB/anchor/sinkhorn/state_direction/lambda
```

具体子图数量和排列应根据可用曲线自动压缩，例如 `state_direction=0` 时对应子图可省略或显示 NaN，但整体图像比例要保持可读。开启 `sinkhorn_divergence` 时，`wb_train_loss.svg` 应单独包含 `avg_wb_sinkhorn`，若 `wb_eval_loss=1`，同一子图可同时显示 `val_wb_sinkhorn`。

## 8. 推荐最少参数

第一版尽量只新增：

```bash
--wb_loss_type sinkhorn_divergence
```

复用：

```bash
--wb_epsilon
--wb_support_size
--wb_spots_per_graph
--wb_spots_per_cancer
--wb_min_cancers
--wb_min_spots
--lambda_wb
--wb_anchor_weight
--wb_warmup_epochs
--wb_ramp_epochs
```

可选新增：

```bash
--wb_sinkhorn_iters 50
```

如果不新增，则固定为 50。

推荐初始命令片段：

```bash
--use_mmd 0 \
--use_wb_align 1 \
--wb_loss_type sinkhorn_divergence \
--lambda_wb 0.03 \
--wb_warmup_epochs 10 \
--wb_ramp_epochs 20 \
--wb_anchor_weight 0.5 \
--wb_spots_per_graph 64 \
--wb_spots_per_cancer 0 \
--wb_support_size 128 \
--wb_epsilon 0.1 \
--wb_min_cancers 2 \
--wb_min_spots 2
```

如果分类性能下降：

```text
lambda_wb: 0.03 -> 0.01
wb_epsilon: 0.1 -> 0.2 或 0.5
wb_anchor_weight: 0.5 -> 0.3
```

如果 cancer mixing 不够：

```text
lambda_wb: 0.03 -> 0.05
wb_epsilon: 0.2 -> 0.1 或 0.05
可考虑 wb_label_balanced_sampling=1
```

## 9. 计算复杂度

设：

```text
K = active cancer 数
N_k = cancer k 的 selected h spot 数
M = support b 点数
D = latent 维度
T = Sinkhorn 迭代次数
```

单个 cancer 的 debiased Sinkhorn divergence：

```text
S_eps(P_k, Q)
= OT_eps(P_k,Q)
 - 1/2 OT_eps(P_k,P_k)
 - 1/2 OT_eps(Q,Q)
```

复杂度近似：

```text
O(T * (N_k*M + N_k^2 + M^2) * D)
```

多癌种：

```text
O(sum_k T * (N_k*M + N_k^2 + M^2) * D)
```

如果 `Q-Q` 自相似项能在 batch 内复用，可降为：

```text
O(T*M^2*D + sum_k T*(N_k*M + N_k^2)*D)
```

第一版为了实现简单，可以不复用 `Q-Q`，先控制点数：

```bash
--wb_spots_per_graph 64
--wb_support_size 128
--wb_spots_per_cancer 0 或 128
```

与当前 `dual_potential` 相比，Sinkhorn 多了迭代次数 `T`，因此更重，但 WB 点数可控时仍可训练。

## 10. 保护 tumor/normal 分类性能的策略

Sinkhorn divergence 更接近标准 OT，可能比当前 surrogate 更强。为了避免过度消除分类相关信号：

1. 保留 warmup/ramp：

```bash
--wb_warmup_epochs 10
--wb_ramp_epochs 20
```

2. `lambda_wb` 从小开始：

```bash
--lambda_wb 0.01 或 0.03
```

3. `wb_anchor_weight` 不宜过大：

```bash
--wb_anchor_weight 0.3 或 0.5
```

4. `wb_epsilon` 不宜太小：

```bash
--wb_epsilon 0.1 或 0.2
```

5. 第一轮不要同时打开太多约束：

```bash
--wb_label_balanced_sampling 0
--wb_state_direction 0
--use_domain_adv_cancer 0
```

6. 继续使用：

```bash
--best_metric val_macro_f1
```

## 11. 实验设计

建议最少比较：

| 实验 | 配置 | 目的 |
|---|---|---|
| Baseline | `use_mmd=0`, `use_wb_align=0` | 主任务基线 |
| MMD cancer | `use_mmd=1`, `mmd_on=cancer` | 当前边缘对齐基线 |
| WB dual | `wb_loss_type=dual_potential` | 当前 CWB-style surrogate |
| WB sinkhorn | `wb_loss_type=sinkhorn_divergence` | 方案 C |

主要验收指标：

```text
val_macro_f1
val_auc
val_accuracy
cancer_type iLISI
batch_id iLISI
tumor_label conservation
cancer probe accuracy
avg_wb_sinkhorn / val_wb_sinkhorn 曲线
UMAP/TSNE cancer mixing 与 tumor/normal 可分性
```

预期：

```text
1. sinkhorn_divergence 的 wb_sinkhorn 曲线下降。
2. cancer_type iLISI 上升。
3. cancer probe accuracy 下降。
4. val_macro_f1 不明显低于 baseline / dual_potential。
5. tumor/normal 在 embedding 中仍可分。
```

## 12. 风险与失败模式

### 12.1 Sinkhorn 过强导致分类下降

表现：

```text
val_macro_f1 下降
tumor_label conservation 下降
UMAP 中 tumor/normal 混在一起
```

处理：

```text
lambda_wb 降低
wb_epsilon 增大
wb_anchor_weight 降低
延长 warmup/ramp
```

### 12.2 epsilon 太小导致数值不稳

表现：

```text
wb_sinkhorn NaN
loss sudden spike
gradient 爆
```

处理：

```text
wb_epsilon 从 0.2 开始
log-domain Sinkhorn
对 cost 做 D 维归一化
```

### 12.3 batch 内 active cancer 不足

表现：

```text
WB 经常 skipped
avg_wb_active_cancers 低
```

处理：

```text
sampler_mode=cancer_balanced
sampler_k_cancers >= 4
wb_min_cancers=2
```

### 12.4 support b 追逐 h，未形成稳定中心

表现：

```text
wb_sinkhorn 下降但 cancer mixing 不明显
h-b anchor 过小或过大
```

处理：

```text
检查 wb_anchor 曲线
调 wb_anchor_weight
必要时后续再考虑 memory support，但第一版不加
```

## 13. 不建议第一版加入的内容

第一版不建议：

```text
1. finite learnable barycenter atoms
2. z_wb 作为分类输入
3. memory bank support
4. dual_potential + sinkhorn 同时作为强主损失
5. unbalanced Sinkhorn
6. per-cancer learnable support map
7. 大量新增 CLI 参数
```

原因是这些都会让方案偏离“`h` 本身成为连续泛癌表征”的目标，或让失败原因难以定位。

## 14. 最终推荐实现路径

第一步：只实现独立主损失：

```text
wb_loss_type=sinkhorn_divergence
```

第二步：复用现有 sampling、support map、anchor、loss curves。

第三步：不引入 potential bank，不创建 `opt_pot`。

第四步：使用内部 PyTorch log-domain Sinkhorn，默认 `sinkhorn_iters=50`。

第五步：完成 smoke test：

```text
1. wb_loss_type=sinkhorn_divergence 能跑通。
2. loss 可反传到 h 和 support_map。
3. wb_sinkhorn 不为 NaN。
4. loss_components.csv 记录 avg_wb_sinkhorn。
5. wb_train_loss.svg 根据 wb_loss_type 自动布局；sinkhorn_divergence 模式下包含独立 sinkhorn 子图。
```

第六步：跑四组对照：

```text
baseline_no_align
mmd_cancer_only
wb_only_dual
wb_only_sinkhorn
```

## 15. 需要你确认的问题

下面这些细节建议在开始代码实现前确认。

### 15.1 新增 `--wb_sinkhorn_iters` 

```text
方案 B：新增 --wb_sinkhorn_iters，默认 50。
```

我建议方案 B。虽然多一个参数，但调试数值稳定和计算时间会方便很多。

确认方案 B

### 15.2 第一版是否只做内部 PyTorch Sinkhorn

两个选择：

```text
方案 A：内部 PyTorch log-domain Sinkhorn，不新增依赖。
方案 B：优先 GeomLoss，内部 PyTorch 作为 fallback。
```

我建议第一版用方案 A。等效果确认后，再考虑 GeomLoss backend。
确认方案 A


### 15.3 `wb_epsilon` 是否直接复用为 Sinkhorn epsilon

两个选择：

```text
方案 A：复用 wb_epsilon，减少参数。
方案 B：新增 wb_sinkhorn_epsilon，避免和 dual_potential regularizer 混用。
```

我建议方案 A，因为你的要求是尽量少加参数。
确认方案 A


### 15.4 debiased Sinkhorn 是否强制开启

两个选择：

```text
方案 A：强制 debiased，不新增参数。
方案 B：新增 wb_sinkhorn_debias。
```

我建议方案 A。STOnco 的方案 C 就定义为 debiased Sinkhorn divergence，不再开放 raw OT 以免实验解释混乱。
确认方案 A


### 15.5 复用 Q-Q 自相似项

两个选择：

```text
方案 A：第一版简单实现，每个 cancer 单独算 SinkhornD(x_k, b)，Q-Q 可能重复计算。
方案 B：缓存当前 batch 的 OT(b,b)，复用 Q-Q 项。
```

确认方案 B 复用 Q-Q 项


### 15.6 把 `wb_regularizer` 对 sinkhorn_divergence 置为无效

当前 dual_potential 使用：

```text
wb_regularizer = l2 或 entropy
```

Sinkhorn divergence 固定是 entropy-regularized OT，所以 `wb_regularizer` 对方案 C 没有意义。

建议：

```text
wb_loss_type=sinkhorn_divergence 时忽略 wb_regularizer，并打印一次提示。
```
这个确认没问题

### 15.7 在 validation 中默认计算 Sinkhorn loss

当前 `wb_eval_loss` 控制 validation 是否计算 WB loss。

建议：

```text
保持现有逻辑：只有 wb_eval_loss=1 时才在 validation 计算。
```

原因是 Sinkhorn validation 成本较高。
这个确认没问题

### 15.8 不加入 Sinkhorn auxiliary 到 dual_potential

本方案 C 不实现 auxiliary。后续如果需要，可另写方案：

```text
dual_potential + sinkhorn_aux
```

确认：只做独立主损失。

## 16. 参考文献

[Cuturi2013] Marco Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." Advances in Neural Information Processing Systems 26, 2013. https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport

[Feydy2019] Jean Feydy, Thibault Sejourne, Francois-Xavier Vialard, Shun-ichi Amari, Alain Trouve, and Gabriel Peyre. "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences." AISTATS, PMLR 89:2681-2690, 2019. https://proceedings.mlr.press/v89/feydy19a.html

[Janati2020] Hicham Janati, Marco Cuturi, and Alexandre Gramfort. "Debiased Sinkhorn Barycenters." Proceedings of the 37th International Conference on Machine Learning, PMLR 119:4692-4701, 2020. https://proceedings.mlr.press/v119/janati20a.html

[Li2020] Lingxiao Li, Aude Genevay, Mikhail Yurochkin, and Justin Solomon. "Continuous Regularized Wasserstein Barycenters." Advances in Neural Information Processing Systems 33, 2020. https://papers.nips.cc/paper/2020/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html

[Flamary2021] Remi Flamary et al. "POT: Python Optimal Transport." Journal of Machine Learning Research, 22(78):1-8, 2021. https://www.jmlr.org/papers/v22/20-451.html

[GeomLoss] Jean Feydy et al. "GeomLoss: Geometric Loss Functions between Sampled Measures, Images and Volumes." Documentation and PyTorch implementation. https://www.kernel-operations.io/geomloss/
