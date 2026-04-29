# STOnco prior-generator barycenter support 更新方案

本文档讨论在 STOnco 中加入三种 training-only generative barycenter support：

```text
h = Encoder_theta(x)
z ~ p(z)
b = G_psi(z)
Q_psi = Law(G_psi(z))

L_bary = sum_k beta_k D(P_k^h, Q_psi)
```

本次只考虑：

1. `prior_generator + sinkhorn_divergence`
2. `prior_generator + sliced_wasserstein`
3. `prior_generator + fixed-kernel MMD`

不考虑 `deep-kernel MMD / SMMD`，不改 STOnco 推理路径。

## 0. 确认摘要

已确认的第一版实现边界：

```text
1. 新增 wb_support_mode=prior_generator。
2. prior_generator 支持 wb_loss_type=sinkhorn_divergence / sliced_wasserstein / mmd。
3. mmd 只开放给 prior_generator，不开放 generated_support + mmd。
4. sliced_wasserstein 同时支持 generated_support 和 prior_generator。
5. prior_generator 下完全关闭 pointwise anchor，记录 wb_anchor=0。
6. prior generator 默认 prior_type=normal，prior_dim=128。
7. 不新增 wb_prior_support_size，复用 wb_support_size。
8. 不新增 wb_mmd_*，复用 mmd_num_kernels / mmd_kernel_mul / mmd_sigma。
9. 额外记录 6 个 support diagnostics，并生成 wb_support_diagnostics.svg。
10. 测试直接使用 /apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz 跑真实数据 1 epoch。
```

## 1. 参考思路

参考论文：

```text
Samuel Cohen, Michael Arbel, Marc Peter Deisenroth.
Estimating Barycenters of Measures in High Dimensions.
arXiv:2007.07105
https://arxiv.org/pdf/2007.07105
```

论文核心思想是不用固定 Dirac atoms 表示 barycenter，而是用 generator 的 push-forward distribution 参数化 barycenter：

```text
P_theta = G_theta#rho
min_theta sum_p beta_p D(P_theta, mu_p)
```

映射到 STOnco：

```text
mu_p                      -> cancer p 的 spot latent 分布 P_p^h
G_theta#rho               -> prior generator 产生的 shared latent support Q_psi
D                         -> Sinkhorn divergence 或 fixed-kernel MMD
h                         -> STOnco GNN encoder 输出，不是 z_clf
```

因此新方案比当前 `b=T_phi(h)` 更接近论文中的 generative barycenter 表达。

## 2. 当前 STOnco 代码事实

### 2.1 latent h 的来源

`stonco/core/models.py` 中：

```python
h = self.encode(...)
logits, z_clf = self.clf(h, return_z=return_z)
...
if return_h:
    out['h'] = h
```

`h` 是 GNN backbone 输出。当前 WB 和 MMD 对齐都使用这个 `h`，而不是分类头 latent `z_clf`。

### 2.2 当前 WB generated support

`stonco/core/wb_potentials.py` 中已有：

```python
class GeneratedSupportMap(nn.Module):
    """Identity-initialized residual map: b = h + MLP(LayerNorm(h))."""
```

当前训练循环中：

```python
b_wb = support_map(h)
loss_wb, loss_wb_anchor, wb_stats = wb_module.model_loss(
    h=h,
    b=b_wb,
    cancer_dom=dom_nodes_wb,
    graph_nodes=batch.batch,
    y=batch.y,
)
```

也就是说当前 `generated_support` 是：

```text
b_i = T_phi(h_i)
```

它默认 `h` 与 `b` 有 spot-level 一一对应关系。

### 2.3 当前 WB loss 的关键假设

`GeneratedSupportWBLoss._prepare(...)` 当前逻辑是：

```python
idx = self._select_indices(cancer_dom, graph_nodes=graph_nodes, y=y, generator=generator)
h_sel = self._normalize_latent(h[idx])
b_sel = self._normalize_latent(b[idx])
cancer_sel = cancer_dom[idx].long()
```

这说明当前 WB loss 假设：

```text
len(b) == len(h)
b[i] 对应 h[i]
```

prior-generator 模式下这个假设不成立，因为：

```text
h_i 来自真实 spot
b_j = G_psi(z_j) 是独立 shared support sample
```

因此不能只把 `support_map(h)` 替换成 `prior_generator.sample(...)`，必须扩展 loss prepare 逻辑。

### 2.4 当前 Sinkhorn divergence 结构

当前 `sinkhorn_divergence` 已经接近需要的形式：

```python
ot_qq = self._sinkhorn_ot(b_support, b_support)
for k in active:
    xk = h[cancer == int(k)]
    ot_kq = self._sinkhorn_ot(xk, b_support)
    ot_kk = self._sinkhorn_ot(xk, xk)
    losses.append(ot_kq - 0.5 * ot_kk - 0.5 * ot_qq)
```

这与新方案目标一致：

```text
sum_k SinkhornD(P_k^h, Q_psi)
```

需要改的是 `b_support` 的来源和 `_prepare()` 对 `b` 的索引方式。

### 2.5 当前 MMD 结构

当前 `--use_mmd` 是训练函数内部的 multi-domain pairwise MMD：

```text
slide/cancer domain pairwise MMD(P_a^h, P_b^h)
```

它不是 barycenter-to-generator MMD。

当前 MMD helper 包括：

```python
_build_sigma_list(...)
_rbf_kernel(...)
_mmd2_unbiased(...)
_multi_domain_mmd(...)
```

新方案的 fixed-kernel MMD 应该是：

```text
sum_k MMD^2(P_k^h, Q_psi)
```

不要混入旧 `--use_mmd`，否则日志和语义会混乱。

## 3. 总体设计

新增一个 WB support mode：

```text
wb_support_mode = prior_generator
```

保留现有模式：

```text
wb_support_mode = generated_support
```

两种模式语义：

```text
generated_support:
    b = T_phi(h)
    b 与 h 一一对应
    适用于当前 euclidean_pairwise / dual_potential / sinkhorn_divergence / sliced_wasserstein

prior_generator:
    z ~ p(z)
    b = G_psi(z)
    b 是独立 shared support sample
    第一版支持 sinkhorn_divergence / sliced_wasserstein / mmd
```

推理路径保持不变：

```text
x -> Encoder_theta -> h -> ClassifierHead -> logits
```

`G_psi` 只参与训练阶段对齐，不参与常规预测。

## 4. 数学目标

### 4.1 shared support generator

定义：

```text
z_j ~ p(z)
b_j = G_psi(z_j), j = 1,...,M
Q_psi = empirical distribution over {b_j}
```

默认 prior：

```text
p(z) = N(0, I)
```

可选 prior：

```text
Uniform(-1, 1)
```

第一版实现 `normal` 和 `uniform`，默认 `normal`。

### 4.2 Sinkhorn divergence barycenter loss

对 batch 内 active cancer 集合 `K_batch`：

```text
L_prior_sinkhorn =
    sum_{k in K_batch} beta_k SinkhornD_epsilon(P_k^h, Q_psi)
```

其中：

```text
P_k^h = cancer k 的 selected spot latent empirical distribution
Q_psi = prior generator 生成的 support empirical distribution
```

沿用当前 debiased Sinkhorn divergence：

```text
SinkhornD(P, Q) = OT_epsilon(P, Q)
                - 0.5 OT_epsilon(P, P)
                - 0.5 OT_epsilon(Q, Q)
```

统一总 loss：

```text
L_total = L_task
        + L_domain_optional
        + L_mmd_optional_existing
        + lambda_wb(t) * L_prior_bary
```

其中 `L_prior_bary` 由 `wb_loss_type` 决定。

建议实验中关闭旧 MMD：

```text
--use_mmd 0
```

### 4.3 Sliced Wasserstein barycenter loss

定义：

```text
L_prior_sw =
    sum_{k in K_batch} beta_k SW_2^2(P_k^h, Q_psi)
```

复用当前 `sliced_wasserstein` 的随机投影逻辑：

```text
1. 对 h_k 和 b_support 使用同一组随机方向投影。
2. 每个方向上计算 1D Wasserstein。
3. 对方向和 active cancers 取均值。
```

相关参数：

```text
wb_sw_num_projections
wb_support_size
wb_spots_per_graph
wb_spots_per_cancer
```

### 4.4 Fixed-kernel MMD barycenter loss

定义：

```text
L_prior_mmd =
    sum_{k in K_batch} beta_k MMD^2(P_k^h, Q_psi)
```

使用当前 STOnco 已有的 multi-band RBF kernel 参数，第一版直接复用 legacy MMD 参数：

```text
mmd_num_kernels
mmd_kernel_mul
mmd_sigma
```

说明：

```text
--use_mmd=1 仍表示 legacy pairwise domain MMD；
--use_wb_align=1 --wb_loss_type=mmd 表示 prior-generator barycenter MMD；
mmd_* 只复用 RBF kernel 超参，不改变 use_mmd 的开关语义。
```

fixed-kernel MMD 的解释要保守：

```text
它学习的是 shared mixture-style latent target；
它不是 Wasserstein 几何意义下的 barycenter。
```

## 5. 建议 CLI / config 扩展

### 5.1 扩展现有参数

当前：

```python
parser.add_argument(
    '--wb_support_mode',
    choices=['generated_support'],
    ...
)

parser.add_argument(
    '--wb_loss_type',
    choices=['dual_potential', 'euclidean_pairwise', 'sinkhorn_divergence', 'sliced_wasserstein'],
    ...
)
```

建议改为：

```python
--wb_support_mode {generated_support,prior_generator}
--wb_loss_type {dual_potential,euclidean_pairwise,sinkhorn_divergence,sliced_wasserstein,mmd}
```

约束：

```text
if wb_support_mode == prior_generator:
    wb_loss_type must be sinkhorn_divergence, sliced_wasserstein, or mmd
```

### 5.2 最小新增 prior generator 参数

第一版建议只新增：

```text
--wb_prior_type {normal,uniform}
--wb_prior_dim int
```

默认值建议：

```text
wb_prior_type = normal
wb_prior_dim = 128
```

其余 prior generator 超参复用现有 WB support 参数：

```text
prior generator hidden dim     = wb_support_hidden
prior generator dropout        = wb_support_dropout
prior generator lr             = wb_support_lr
prior generator weight_decay   = wb_support_weight_decay
prior generator support count  = wb_support_size
```

`wb_support_size` 在 `generated_support` 下表示 support-side b 子采样上限；在 `prior_generator` 下表示每个 batch 生成多少个 `b_j=G_psi(z_j)`。这样可以少引入一个新参数，同时保留“WB support size”的统一语义。

### 5.3 复用 fixed-kernel MMD 参数

第一版不新增 `wb_mmd_*`。prior-generator barycenter MMD 复用：

```text
mmd_num_kernels
mmd_kernel_mul
mmd_sigma
```

默认值继续沿用当前 STOnco 默认：

```text
mmd_num_kernels = 5
mmd_kernel_mul = 2.0
mmd_sigma = None
```

这样会让 CLI 更短。文档和 help 文案中需要明确：`mmd_*` 是 RBF kernel 参数，可以同时被 legacy MMD 和 WB barycenter MMD 读取；是否启用 legacy MMD 仍由 `use_mmd` 决定。

### 5.4 anchor 和 potential 参数解释

prior-generator 模式下：

```text
wb_anchor_weight ignored
wb_potential_* ignored
wb_pot_every_n_steps ignored
wb_regularizer ignored, unless future dual_potential support is added
```

原因：

```text
L_anchor = ||h_i - b_i||^2
```

在 `b=G_psi(z)` 时没有 spot-level 配对，不成立。

potential bank 也不是第一版目标，因为本次只做 Sinkhorn 和 fixed-kernel MMD。

## 6. 模块设计

### 6.1 新增 PriorSupportGenerator

建议放在：

```text
stonco/core/wb_potentials.py
```

原因：

```text
GeneratedSupportMap 和 GeneratedSupportWBLoss 已在此文件；
prior generator 属于同一 WB support 体系；
不新增文件可以降低改动面。
```

示意：

```python
class PriorSupportGenerator(nn.Module):
    def __init__(self, prior_dim, out_dim, hidden=128, num_layers=2, dropout=0.3, prior_type='normal'):
        ...

    def sample_z(self, n, device, dtype, generator=None):
        ...

    def forward(self, z):
        return self.net(z)

    def sample(self, n, device, dtype, generator=None):
        z = self.sample_z(n, device=device, dtype=dtype, generator=generator)
        return self.forward(z)
```

MLP 建议：

```text
Linear(prior_dim, hidden)
LayerNorm(hidden)
LeakyReLU
Dropout
[repeat hidden blocks]
Linear(hidden, out_dim)
```

输出不建议加 `tanh`。原因是 STOnco 的 `h` 没有固定范围，WB loss 内部已经会做 `layer_norm`。

初始化建议：

```text
普通 Kaiming/Xavier 初始化即可；
不要像 GeneratedSupportMap 一样 zero-init 最后一层。
```

zero-init 会让所有初始 support 接近同一点，不利于 Sinkhorn/MMD 的早期信号。

### 6.2 扩展 GeneratedSupportWBLoss

当前类名可以保留，也可以改名。为了减少改动，建议保留：

```text
GeneratedSupportWBLoss
```

但内部增加：

```text
support_mode
mmd_num_kernels
mmd_kernel_mul
mmd_sigma
```

更清晰的内部结构：

```python
def _prepare_paired_support(...):
    # 当前 generated_support 逻辑

def _prepare_independent_support(...):
    # prior_generator 逻辑
```

prior-generator prepare 逻辑：

```text
1. 对 h 按 graph/cancer/label 规则选 spot，得到 h_sel 和 cancer_sel。
2. 不对 b 用 idx。
3. 对 h_sel 和 b 分别 layer_norm。
4. 根据 cancer_sel 识别 active cancers。
5. 返回：
   prepared = {
       'h': h_sel,
       'b': b_norm,
       'cancer': cancer_sel,
       'active_domains': active,
       'y': y_sel,
       'support_is_independent': True,
   }
```

对于 prior-generator：

```text
b.size(0) 可以不同于 h.size(0)
cancer 不应用于 b
```

### 6.3 扩展 model_loss

当前 `model_loss(...)` 已有：

```text
dual_potential
sinkhorn_divergence
sliced_wasserstein
euclidean_pairwise
```

新增：

```text
mmd
```

prior-generator 模式下允许：

```text
sinkhorn_divergence
sliced_wasserstein
mmd
```

建议：

```python
if self.support_mode == 'prior_generator':
    if self.loss_type == 'sinkhorn_divergence':
        anchor = zero
        raw_loss = self._sinkhorn_divergence_loss(prepared)
    elif self.loss_type == 'sliced_wasserstein':
        anchor = zero
        raw_loss = self._sliced_wasserstein_loss(prepared, generator=generator)
    elif self.loss_type == 'mmd':
        anchor = zero
        raw_loss = self._mmd_barycenter_loss(prepared)
```

统计字段：

```text
wb_loss
wb_sinkhorn
wb_sliced_wasserstein
wb_mmd
wb_anchor = 0 or NaN
wb_active_cancers
wb_active_spots
wb_support_points
```

建议 `wb_anchor` 记录为 `0.0` 而不是 NaN，表示该项明确关闭。

### 6.4 fixed-kernel MMD helper

建议把当前 train.py 里的 RBF helper 迁移或复制到 `wb_potentials.py`，作为 WB module 内部方法：

```text
_pairwise_sq_dists
_build_sigma_list
_rbf_kernel
_mmd2_unbiased
_mmd_barycenter_loss
```

为了外科化实现，第一版可以复制一份小 helper 到 `wb_potentials.py`，不重构 train.py 旧 MMD。

`_mmd_barycenter_loss(prepared)`：

```python
b_support = self._support_subset(prepared['b'], generator=generator)
losses = []
for k in active:
    xk = h[cancer == k]
    sigma_list = self._build_sigma_list(xk, b_support, ...)
    losses.append(self._mmd2_unbiased(xk, b_support, sigma_list))
return mean(losses)
```

如果 `xk` 或 `b_support` 点数少于 2，则跳过该 cancer。

### 6.5 sliced Wasserstein prior-generator loss

当前 `GeneratedSupportWBLoss` 已有 `_sliced_wasserstein_loss(prepared, generator=None)`，其结构天然适合 independent support：

```text
1. 从 prepared['b'] 得到 shared support b_support。
2. 随机采样投影方向。
3. 对每个 cancer 的 xk = h[cancer == k] 和 b_support 做同一组方向投影。
4. 在每个一维投影上计算 weighted 1D Wasserstein。
5. 对方向和 active cancers 取均值。
```

因此 prior-generator 模式下可以复用该函数，只需确保 independent prepare 不再对 `b` 使用 `idx`。

对应目标：

```text
L_prior_sw =
    sum_{k in K_batch} beta_k SW_2^2(P_k^h, Q_psi)
```

相关参数复用现有 WB 参数：

```text
wb_sw_num_projections
wb_support_size
wb_spots_per_graph
wb_spots_per_cancer
wb_min_cancers
wb_min_spots
```

## 7. 训练循环接入

### 7.1 module construction

当前：

```python
support_map = GeneratedSupportMap(...).to(device)
wb_module = GeneratedSupportWBLoss(...).to(device)
```

建议改为：

```python
if wb_support_mode == 'generated_support':
    support_map = GeneratedSupportMap(...).to(device)
    support_generator = None
elif wb_support_mode == 'prior_generator':
    support_map = None
    support_generator = PriorSupportGenerator(...).to(device)
```

`wb_module` 始终创建，但传入：

```text
support_mode=cfg['wb_support_mode']
```

### 7.2 optimizer groups

当前 main optimizer 包括：

```text
gnn
clf
dom
wb_support
wb_main
```

prior-generator 模式新增参数组：

```text
wb_prior_generator
```

学习率使用：

```text
wb_support_lr
```

如果不传，则沿用现有规则：

```text
wb_support_lr = lr
```

weight decay 使用 `wb_support_weight_decay`，未显式传入时沿用 `weight_decay`。

potential optimizer：

```text
prior_generator + sinkhorn_divergence: 不需要
prior_generator + sliced_wasserstein: 不需要
prior_generator + mmd: 不需要
```

因此：

```text
opt_pot = None
```

并跳过 Step A potential update。

### 7.3 training forward

当前 Step B：

```python
h = out['h']
b_wb = support_map(h)
loss_wb, loss_wb_anchor, wb_stats = wb_module.model_loss(...)
```

建议改为：

```python
h = out['h']
if wb_support_mode == 'generated_support':
    b_wb = support_map(h)
elif wb_support_mode == 'prior_generator':
    b_wb = support_generator.sample(
        int(cfg['wb_support_size']),
        device=h.device,
        dtype=h.dtype,
    )

loss_wb, loss_wb_anchor, wb_stats = wb_module.model_loss(
    h=h,
    b=b_wb,
    cancer_dom=batch.cancer_dom[batch.batch],
    graph_nodes=batch.batch,
    y=batch.y,
)
```

loss addition 保持：

```python
loss_total = loss_total + lambda_wb * (
    loss_wb + wb_anchor_weight * loss_wb_anchor
)
```

但 prior-generator 模式下 `loss_wb_anchor == 0`。

### 7.4 validation

当前 `wb_eval_loss=False` 默认关闭。

建议保持默认关闭。

如果开启：

```text
prior-generator validation 每个 val batch 重新采样 b。
```

为了可重复性，可以使用已有或新增的 `wb_generator_seed`，但第一版不强制。验证 WB loss 只是 diagnostic，不参与 best model selection。

## 8. 日志、CSV、曲线和 artifacts

### 8.1 hist 新增字段

建议新增：

```text
avg_wb_mmd
val_avg_wb_mmd
avg_wb_support_points
val_avg_wb_support_points
avg_wb_support_norm
avg_wb_support_std
avg_wb_h_norm
avg_wb_h_std
avg_wb_mean_gap
avg_wb_std_gap
```

现有：

```text
avg_wb_sinkhorn
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
```

可以继续复用。

### 8.2 loss_components.csv

新增列：

```text
avg_wb_mmd
val_avg_wb_mmd
avg_wb_support_points
val_avg_wb_support_points
avg_wb_support_norm
avg_wb_support_std
avg_wb_h_norm
avg_wb_h_std
avg_wb_mean_gap
avg_wb_std_gap
```

不适用时写 NaN。

### 8.3 wb_train_loss.svg

现有 WB 曲线图应增加 `wb_mmd` 面板。

推荐图中包含：

```text
avg_wb_loss
avg_wb_sinkhorn
avg_wb_mmd
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
avg_wb_support_points
avg_wb_lambda
```

具体布局可沿用当前 `_save_wb_train_loss_plot(...)` 自动按可用曲线布局的风格。

### 8.4 wb_support_diagnostics.svg

启用：

```text
use_wb_align=1
wb_support_mode=prior_generator
```

时，额外生成：

```text
wb_support_diagnostics.svg
```

该图单独画 prior generator support 与 selected `h` 的尺度和分布差异，避免把诊断曲线塞进 `wb_train_loss.svg` 导致过于拥挤。

推荐布局：

```text
Panel 1: norm scale
  avg_wb_support_norm
  avg_wb_h_norm

Panel 2: std scale
  avg_wb_support_std
  avg_wb_h_std

Panel 3: distribution gap
  avg_wb_mean_gap
  avg_wb_std_gap
```

这些曲线用于快速判断：

```text
support_std << h_std: Q_psi 可能 collapse
support_std >> h_std: Q_psi 可能过度发散
support_norm 持续远离 h_norm: generator 可能 scale drift
mean_gap / std_gap 不下降: Q_psi 与当前 h 分布仍未对齐
```

### 8.5 artifacts

当前保存：

```text
wb_support_map_last.pt
wb_potentials_last.pt
wb_config.json
```

prior-generator 模式建议保存：

```text
wb_prior_generator_last.pt
wb_potentials_last.pt
wb_support_diagnostics.svg
wb_config.json
```

其中 `wb_potentials_last.pt` 在 prior-generator + sinkhorn/sliced_wasserstein/mmd 下仍可保存 `wb_module.state_dict()`，但里面没有 potential bank 参数。也可以保留文件名以减少 artifact 分支逻辑。

如果希望语义更清楚：

```text
wb_module_last.pt
```

但这会改变既有 artifact 命名，第一版不建议。

## 9. 配置兼容性和报错规则

建议新增 validation：

```text
if wb_support_mode == 'generated_support':
    allow existing loss types:
        dual_potential
        euclidean_pairwise
        sinkhorn_divergence
        sliced_wasserstein
    disallow:
        mmd

if wb_support_mode == 'prior_generator':
    allow:
        sinkhorn_divergence
        sliced_wasserstein
        mmd
    disallow:
        dual_potential
        euclidean_pairwise
```

原因：

```text
dual_potential 当前实现依赖 generated support 和 potential bank 设计；
euclidean_pairwise 当前按 cancer 对 b 分组，prior support 没有 cancer label；
sliced_wasserstein 只需要比较每个 cancer 的 h_k 与 shared support b_support 的投影分布，因此适合 prior_generator。
```

建议 warning：

```text
if use_wb_align and use_mmd:
    warning 继续保留，但说明 prior-generator MMD 属于 wb_loss_type=mmd，
    旧 use_mmd 是 pairwise domain MMD，两个目标会叠加。

if wb_support_mode == prior_generator and wb_anchor_weight was explicitly set:
    print warning: ignored because no pointwise h-b pairing.

if wb_support_mode == prior_generator and wb_potential_* explicitly set:
    print warning: ignored.
```

## 10. 推荐实验命令

### 10.1 prior-generator + Sinkhorn divergence

```bash
python -m stonco.core.train \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type sinkhorn_divergence \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 10 \
  --wb_ramp_epochs 20 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --wb_epsilon 0.1 \
  --wb_sinkhorn_iters 50 \
  --wb_spots_per_graph 64 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --use_mmd 0
```

### 10.2 prior-generator + sliced Wasserstein

```bash
python -m stonco.core.train \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type sliced_wasserstein \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 10 \
  --wb_ramp_epochs 20 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --wb_sw_num_projections 64 \
  --wb_spots_per_graph 64 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --use_mmd 0
```

### 10.3 prior-generator + fixed-kernel MMD

```bash
python -m stonco.core.train \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type mmd \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 10 \
  --wb_ramp_epochs 20 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --mmd_num_kernels 5 \
  --mmd_kernel_mul 2.0 \
  --wb_spots_per_graph 64 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --use_mmd 0
```

## 11. 预期收益

### 11.1 相比当前 `b=T_phi(h)`

当前：

```text
Q_phi = Law(T_phi(h))
```

新方案：

```text
Q_psi = Law(G_psi(z))
```

差异：

```text
当前 Q_phi 依赖当前 batch 的 h；
新 Q_psi 是独立可采样 shared support distribution。
```

新方案更像论文中的 generative barycenter，也更容易解释为“学习一个全局 shared latent support”。

### 11.2 Sinkhorn 版本

优点：

```text
更接近 Wasserstein barycenter；
保留 debiased Sinkhorn divergence；
与当前 STOnco sinkhorn_divergence 实现高度兼容。
```

风险：

```text
计算量较高；
support_size、spots_per_graph、sinkhorn_iters 需要控制；
generator 可能 collapse，需要监控 support variance / wb_support_points / loss 曲线。
```

### 11.3 MMD 版本

优点：

```text
实现简单；
计算比 Sinkhorn 轻；
可作为 prior-generator support 的 baseline。
```

风险：

```text
fixed-kernel MMD barycenter 更接近 mixture target；
如果 kernel 带宽不合适，训练信号可能弱；
不能解释为 Wasserstein 几何 barycenter。
```

## 12. Anti-collapse 设计

第一版建议不加额外 anti-collapse loss，只记录诊断。

建议记录：

```text
wb_support_std = mean(std(b, dim=0))
wb_support_norm = mean(||b||_2)
wb_h_norm = mean(||h_sel||_2)
```

如果出现明显 collapse，再考虑加入弱 regularizer：

```text
L_var = mean(relu(target_std - std(b_dim)))
```

或者：

```text
L_moment = ||mean(h_sel) - mean(b)||^2
```

但第一版不建议加入这些项，避免目标函数过早复杂化。

## 13. 实现步骤

### Step 1. 配置和 CLI

修改：

```text
stonco/core/train.py
```

内容：

```text
1. 扩展 wb_support_mode choices。
2. 扩展 wb_loss_type choices 加入 mmd。
3. 新增最小 prior 参数：wb_prior_type、wb_prior_dim。
4. 复用 wb_support_hidden/dropout/lr/weight_decay/size、wb_sw_num_projections 和 mmd_*。
5. 增加 config validation 和 ignored-parameter warning。
```

验证：

```text
python -m stonco.core.train --help
```

### Step 2. prior generator module

修改：

```text
stonco/core/wb_potentials.py
```

内容：

```text
1. 新增 PriorSupportGenerator。
2. 支持 normal/uniform prior。
3. sample() 返回 [wb_support_size, encoder_out_dim]。
```

验证：

```text
构造 generator，sample 128 个 support，检查 shape 和 requires_grad。
```

### Step 3. loss prepare 分支

修改：

```text
stonco/core/wb_potentials.py
```

内容：

```text
1. GeneratedSupportWBLoss 增加 support_mode。
2. generated_support 继续走当前 paired prepare。
3. prior_generator 走 independent prepare。
4. prior_generator 下不计算 pointwise anchor。
```

验证：

```text
h shape = [N, D], b shape = [M, D] 且 M != N 时 model_loss 可正常运行。
```

### Step 4. Sinkhorn prior-generator loss

修改：

```text
stonco/core/wb_potentials.py
```

内容：

```text
复用当前 _sinkhorn_divergence_loss(prepared)。
```

验证：

```text
loss finite；
loss.backward() 后 h 和 generator 参数都有梯度。
```

### Step 5. sliced Wasserstein prior-generator loss

修改：

```text
stonco/core/wb_potentials.py
```

内容：

```text
复用当前 _sliced_wasserstein_loss(prepared, generator=generator)。
```

验证：

```text
loss finite；
loss.backward() 后 h 和 generator 参数都有梯度。
```

### Step 6. MMD prior-generator loss

修改：

```text
stonco/core/wb_potentials.py
```

内容：

```text
新增 _mmd_barycenter_loss(prepared)。
复用 multi-band RBF MMD helper。
```

验证：

```text
loss finite；
loss.backward() 后 h 和 generator 参数都有梯度。
```

### Step 7. train loop 接入

修改：

```text
stonco/core/train.py
```

内容：

```text
1. 根据 wb_support_mode 创建 support_map 或 prior_generator。
2. optimizer 加入 wb_prior_generator 参数组，复用 wb_support_lr / wb_support_weight_decay。
3. prior-generator 模式跳过 potential update。
4. Step B 中采样 b_wb = prior_generator.sample(...)。
5. validation WB diagnostics 同步支持 prior-generator。
```

验证：

```text
短 epoch smoke test:
  prior_generator + sinkhorn_divergence
  prior_generator + sliced_wasserstein
  prior_generator + mmd
```

### Step 8. logging 和 artifact

修改：

```text
stonco/core/train.py
```

内容：

```text
1. hist 增加 wb_mmd 和 6 个 support diagnostics，复用现有 wb_sliced_wasserstein 记录。
2. loss_components.csv 增加对应列。
3. wb_train_loss.svg 增加 WB loss 曲线。
4. prior_generator 模式额外生成 wb_support_diagnostics.svg。
5. 保存 wb_prior_generator_last.pt。
6. wb_config.json 包含 wb_prior_type、wb_prior_dim，以及复用的 wb_support_* / mmd_* 参数。
```

验证：

```text
训练结束 artifacts 目录包含:
  loss_components.csv
  wb_train_loss.svg
  wb_support_diagnostics.svg
  wb_prior_generator_last.pt
  wb_config.json
```

## 14. 测试计划

本方案实现后，测试计划直接使用真实数据跑一轮端到端训练，不以 synthetic tensor / 单元级 smoke 为主。这样可以同时验证数据读取、图构建、batch 采样、prior generator、WB loss、日志和 artifact 保存。

真实数据目录：

```text
/apps/users/sky_luozhihui/STOnco/data/data_best_11
```

训练 NPZ：

```text
/apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz
```

### 14.1 prior-generator + Sinkhorn divergence 真实数据一轮测试

从仓库模型目录运行：

```bash
cd /apps/users/sky_luozhihui/STOnco/model/STOnco

python -m stonco.core.train \
  --train_npz /apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz \
  --artifacts_dir artifacts/prior_generator_realdata_smoke/sinkhorn \
  --epochs 1 \
  --early_patience 0 \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --num_workers 0 \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type sinkhorn_divergence \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 0 \
  --wb_ramp_epochs 1 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --wb_epsilon 0.1 \
  --wb_sinkhorn_iters 30 \
  --wb_spots_per_graph 64 \
  --use_mmd 0 \
  --save_loss_components 1
```

检查：

```text
1. 训练能完整跑完 1 epoch。
2. 不出现 h/b shape mismatch。
3. loss_components.csv 中 avg_wb_loss 非 NaN。
4. loss_components.csv 中 avg_wb_sinkhorn 非 NaN。
5. avg_wb_active_cancers >= 2，说明真实 batch 内有可计算 WB 的多癌种样本。
6. artifacts/prior_generator_realdata_smoke/sinkhorn/ 中写出 model.pt、meta.json、loss_components.csv、wb_train_loss.svg、wb_support_diagnostics.svg、wb_prior_generator_last.pt、wb_config.json。
```

### 14.2 prior-generator + sliced Wasserstein 真实数据一轮测试

从仓库模型目录运行：

```bash
cd /apps/users/sky_luozhihui/STOnco/model/STOnco

python -m stonco.core.train \
  --train_npz /apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz \
  --artifacts_dir artifacts/prior_generator_realdata_smoke/sliced_wasserstein \
  --epochs 1 \
  --early_patience 0 \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --num_workers 0 \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type sliced_wasserstein \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 0 \
  --wb_ramp_epochs 1 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --wb_sw_num_projections 64 \
  --wb_spots_per_graph 64 \
  --use_mmd 0 \
  --save_loss_components 1
```

检查：

```text
1. 训练能完整跑完 1 epoch。
2. 不出现 h/b shape mismatch。
3. loss_components.csv 中 avg_wb_loss 非 NaN。
4. loss_components.csv 中 avg_wb_sliced_wasserstein 非 NaN。
5. avg_wb_active_cancers >= 2。
6. artifacts/prior_generator_realdata_smoke/sliced_wasserstein/ 中写出 model.pt、meta.json、loss_components.csv、wb_train_loss.svg、wb_support_diagnostics.svg、wb_prior_generator_last.pt、wb_config.json。
```

### 14.3 prior-generator + fixed-kernel MMD 真实数据一轮测试

从仓库模型目录运行：

```bash
cd /apps/users/sky_luozhihui/STOnco/model/STOnco

python -m stonco.core.train \
  --train_npz /apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz \
  --artifacts_dir artifacts/prior_generator_realdata_smoke/mmd \
  --epochs 1 \
  --early_patience 0 \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --num_workers 0 \
  --use_wb_align 1 \
  --wb_support_mode prior_generator \
  --wb_loss_type mmd \
  --lambda_wb 0.01 \
  --wb_warmup_epochs 0 \
  --wb_ramp_epochs 1 \
  --wb_prior_type normal \
  --wb_prior_dim 128 \
  --wb_support_size 128 \
  --mmd_num_kernels 5 \
  --mmd_kernel_mul 2.0 \
  --wb_spots_per_graph 64 \
  --use_mmd 0 \
  --save_loss_components 1
```

检查：

```text
1. 训练能完整跑完 1 epoch。
2. 不出现 h/b shape mismatch。
3. loss_components.csv 中 avg_wb_loss 非 NaN。
4. loss_components.csv 中 avg_wb_mmd 非 NaN。
5. avg_wb_active_cancers >= 2。
6. artifacts/prior_generator_realdata_smoke/mmd/ 中写出 model.pt、meta.json、loss_components.csv、wb_train_loss.svg、wb_support_diagnostics.svg、wb_prior_generator_last.pt、wb_config.json。
```

### 14.4 真实数据回归检查

新增 prior-generator 路径通过后，仍需用同一个真实数据文件确认旧路径没有被破坏：

```bash
cd /apps/users/sky_luozhihui/STOnco/model/STOnco

python -m stonco.core.train \
  --train_npz /apps/users/sky_luozhihui/STOnco/data/data_best_11/train.npz \
  --artifacts_dir artifacts/prior_generator_realdata_smoke/baseline_no_wb \
  --epochs 1 \
  --early_patience 0 \
  --batch_size_graphs 8 \
  --sampler_mode cancer_balanced \
  --sampler_k_cancers 4 \
  --sampler_m_per_cancer 2 \
  --num_workers 0 \
  --use_wb_align 0 \
  --use_mmd 0 \
  --save_loss_components 1
```

回归检查标准：

```text
1. baseline_no_wb 能完整跑完 1 epoch。
2. model.pt、meta.json、loss_components.csv 正常写出。
3. 未启用 WB 时，WB 相关列为 NaN 或 0，不影响 task loss 和验证指标。
4. 若时间允许，再补跑 generated_support + sinkhorn_divergence 1 epoch，确认原有 WB 模式仍可用。
```

## 15. 推荐第一版默认行为

如果实现后不给额外参数，只改：

```text
--use_wb_align 1
--wb_support_mode prior_generator
--wb_loss_type sinkhorn_divergence
```

则默认：

```text
prior_type = normal
prior_dim = 128
prior_support_size = wb_support_size（默认 128）
prior_hidden = wb_support_hidden（默认 128）
prior_num_layers = 2
prior_dropout = wb_support_dropout（默认跟随 dropout）
lambda_wb = 0.01
wb_epsilon = 0.1
wb_sinkhorn_iters = 50
```

对 MMD：

```text
--wb_loss_type mmd
```

默认：

```text
mmd_num_kernels = 5
mmd_kernel_mul = 2.0
mmd_sigma = None
```

对 sliced Wasserstein：

```text
--wb_loss_type sliced_wasserstein
```

默认：

```text
wb_sw_num_projections = 64
wb_support_size = 128
```

## 16. 已确认决策

1. `wb_loss_type=mmd` 只允许在 `wb_support_mode=prior_generator` 下使用。

   `generated_support + mmd` 不是第一版目标，先不开放。

2. `wb_loss_type=sliced_wasserstein` 同时允许 `generated_support` 和 `prior_generator`。

   当前代码已有 generated-support sliced Wasserstein；新增 prior-generator 后，两种 support mode 共用同一个 loss type。

3. prior generator 默认维度为 `128`。

   CLI 仍保留 `--wb_prior_dim` 供实验调整。

4. prior-generator 模式下完全关闭 pointwise anchor。

   记录 `wb_anchor=0`。如果后续发现 collapse 或 scale drift，再加 distribution-level regularizer。

5. artifact 文件名使用 `wb_prior_generator_last.pt`。

   不把 generator state 塞进 `wb_support_map_last.pt`。

6. validation 不强制固定 prior 采样 seed。

   `wb_eval_loss` 默认关闭；validation WB loss 仅作为 diagnostic。

7. 保留 `use_wb_align=1 and use_mmd=1` 的叠加能力。

   但训练时打印 warning；实验 baseline 推荐 `--use_mmd 0`。

8. 保留 `wb_prior_type`。

   第一版只支持 `normal` 和 `uniform`，默认 `normal`。
