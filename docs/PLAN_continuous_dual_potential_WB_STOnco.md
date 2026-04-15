# 在 STOnco 中以 generated-support Wasserstein barycenter 替代/升级当前 MMD 对齐

本文档基于当前 `model/STOnco` 子项目的真实代码结构、数据组织和训练流程，给出更新后的主方案 B：参考 BaryIR 的形式，不再把当前 mini-batch 的 pooled `h` 直接当作 barycenter support，而是由 STOnco 当前潜变量 `h` 输入一个可学习的 generated-support map，生成连续 barycenter support：

```text
h_i = G_theta(x_i)
b_i = T_phi(h_i)
```

其中 `h` 仍是 STOnco 下游 tumor/normal 分类使用的 latent 表征，`b=T_phi(h)` 是训练阶段生成的 continuous learnable Wasserstein barycenter support。训练时通过 BaryIR-style 交替迭代更新 generated support、potentials 和 STOnco 主模型，使 `h` 被拉向泛癌共享的 generated barycenter geometry，从而替代或升级当前 MMD 多癌种 latent 分布对齐。

核心实现原则：

- 不改 STOnco 推理结构：`GNNBackbone -> h -> ClassifierHead -> tumor/normal`。
- 训练阶段只复用当前已有的 `return_h=True` 取出 `h`。
- 训练时新增一个 generated-support map `T_phi`，由 `h` 生成 `b=T_phi(h)`。
- 训练时新增 potential 网络，主要读取 generated support `b` 和 `cancer_dom`；`dual_potential` 中还使用 source-side `h`。
- 每个 mini-batch 采用 BaryIR 风格交替更新：先 update potentials，再 update STOnco 主模型。
- loss 提供两个选项：
  - `dual_potential + generated_support`：较正规的 dual-potential regularized Wasserstein barycenter 损失，source-side 使用 `h`，support-side 使用 `b=T_phi(h)`，计算形式参考 CWB。
  - `euclidean_moment + generated_support`：轻量版欧氏 anchor / moment 约束与 potential critic，计算形式参考 BaryIR 的 latent 欧氏距离，计算量明显更低，可作为替代 Wasserstein 距离的工程选项。

## 1. 问题定义

当前 STOnco 已有三类相关机制：

1. `GNNBackbone` 输出 spot-level latent `h`。
2. `ClassifierHead(h)` 输出 tumor/normal logits。
3. 可选 `DomainHead(GRL(h))` 做 batch/cancer 域对抗。

当前 MMD 对齐也直接作用在 `h` 上：

```text
MMD(h, cancer_dom)
```

因此升级路径是把它替换为：

```text
GeneratedSupportWB(h, b=T_phi(h), cancer_dom)
```

目标仍然是让 `h` 本身成为一个连续、可学习的泛癌 Wasserstein barycenter 表征空间：

$$
h_i = G_\theta(x_i), \quad h_i \in \mathbb{R}^{D}.
$$

但 barycenter support 不再由 pooled `h` 直接给出，而是由一个共享的可学习映射生成：

$$
b_i = T_\phi(h_i), \quad b_i \in \mathbb{R}^{D}.
$$

对每个癌种 source domain $k$，当前 batch 或训练集中的 `h` 诱导经验分布：

$$
P_k^h = \mathcal{L}(h_i \mid cancer_i=k).
$$

generated support 诱导一个共享 barycenter support 分布：

$$
Q_\phi^b = \mathcal{L}(T_\phi(h_i) \mid i \text{ from mixed cancers}).
$$

希望所有 $P_k^h$ 共同靠近这个由 `T_phi` 生成的连续中心分布：

$$
Q_\phi^{b,*}
\approx
\arg\min_Q \sum_{k=1}^{K}\omega_k W_2^2(P_k^h, Q).
$$

这里的 $Q_\phi^b$ 不显式写成均值向量，也不是有限 learnable atoms。它由连续 map `T_phi` 对所有癌种 `h` 的输出隐式表示，并由 potentials 提供分布级约束。为了避免只让辅助 support `b` 对齐而 `h` 仍保留 cancer-source 信息，训练中必须加入 `h` 与 `b` 的 anchor 约束，使最终分类 head 使用的 `h` 也被拉向 generated barycenter support。

## 2. 为什么选择 generated-support Wasserstein barycenter

### 2.1 保留当前 STOnco 推理路径，同时让 barycenter 可学习

当前 STOnco 的 MMD 代码已经在 `train.py::train_and_validate(...)` 内部 `_compute_losses(...)` 中读取：

```python
h = out.get("h", None)
dom_nodes_cancer = batch.cancer_dom[batch.batch]
```

因此 generated-support WB 仍可以沿用同一 `h` 入口：

```python
return_h = bool(cfg.get("use_mmd", False) or cfg.get("use_wb_align", False))
```

但与 pooled-support 方案不同，generated-support 方案新增训练模块：

```python
b = support_map(h)
```

`support_map` 只在训练阶段参与 WB loss。默认推理仍保持：

```text
GNNBackbone -> h -> ClassifierHead
```

这样既不把 barycenter 降级成简单 batch mean / pooled center，也不把分类 head 改成依赖新的推理 embedding。

### 2.2 比 pooled support 更接近“连续、可学习”的 barycenter space

pooled-support 方案把当前 batch 的混合 `h` 当作 support：

```text
Q_B = pooled h.detach()
```

它实现简单，但每个 batch 的 support 只是随机经验集合，barycenter 本身没有可学习参数。generated-support 方案改为：

```text
Q_phi = distribution of b = T_phi(h)
```

其中 `T_phi` 是共享的可学习映射，随训练持续更新。因此它更符合“连续、可学习的 Wasserstein barycenter space”这一目标。

### 2.3 让分类 head 仍然直接使用被对齐的泛癌表征

generated support 会引入 `b=T_phi(h)`。如果分类 head 改为使用 `b`，就会变成新的推理 embedding 路径。为保持目标清晰，第一版推荐：

```text
ClassifierHead(h)
WB potentials / generated-support losses act on b and h
anchor loss forces h close to b
```

这样验收仍然看 `h`：如果 post-hoc cancer probe 无法从 `h` 判断癌种，同时 tumor/normal AUPRC 仍高，就说明最终下游使用的 `h` 成为了泛癌判别表征。

### 2.4 更贴合 BaryIR/CWB 可迁移部分

BaryIR 的工程重点是由 source latent 生成 bary latent，并用 `Pots` 作为训练时 critic，通过 freeze/unfreeze 交替更新主网络与 potentials。CWB 的理论重点是用 dual potentials 隐式刻画 continuous regularized Wasserstein barycenter。STOnco 最适合组合二者：

```text
BaryIR-style alternating update
+
BaryIR-style generated support b=T_phi(h)
+
CWB-style dual-potential barycenter objective
+
anchor loss forcing h toward generated support
```

## 3. 在 STOnco 中的具体插入位置

### 3.1 模型侧

第一版不需要改 `stonco/core/models.py`。

当前 `STOnco_Classifier.forward(..., return_h=True)` 已经返回：

```python
out["h"] = h
```

WB 对齐只需要训练代码拿到 `out["h"]`，模型侧第一版保持不动。

generated-support map 建议作为训练模块放在 `stonco/core/wb_potentials.py`，不塞进 `STOnco_Classifier`：

```python
b = support_map(h)
```

这样 checkpoint 推理路径不变；需要复现实验时，可以额外保存 `support_map` 和 potentials 的训练 artifacts。

### 3.2 训练侧

主要改 `stonco/core/train.py::train_and_validate(...)`：

- 新增 potential 网络。
- 新增 generated-support map `T_phi`。
- 新增 `opt_pot`。
- generated-support map 参数默认并入主优化器 `opt`，与 STOnco 一起在 Step B 更新。
- 在每个 training batch 中：
  1. freeze STOnco 与 `T_phi`，update potentials；
  2. freeze potentials，update STOnco 与 `T_phi`。
- 在 `_compute_losses(...)` 或训练循环中加入 `loss_wb`。

推荐将 WB 逻辑放在训练循环中，而不是完全塞进 `_compute_losses(...)`，因为 potential update 需要单独 optimizer 和 freeze/unfreeze。

### 3.3 数据和采样侧

generated support 是全局可学习映射，所以单个 batch 不会“生成一个永久独立 barycenter”。每个 batch 只是对全局目标做一次随机估计。为了让每个 step 具有直接跨癌种 coupling，仍建议同一 batch 至少覆盖 2 个癌种，最好 3-4 个癌种。当前已有 `CancerBalancedBatchSampler`，推荐实验默认：

```bash
--batch_size_graphs 8 \
--sampler_mode cancer_balanced \
--sampler_k_cancers 4 \
--sampler_m_per_cancer 2 \
--sampler_enforce_distinct_batch 1
```

如果仍用 `batch_size_graphs=2` 和 random sampler，WB 仍可运行，但许多 step 只看到单癌种，跨癌种 barycenter 信号会变弱、方差变大。

## 4. 推荐主方案 B：BaryIR 风格 generated support + 交替训练

### 4.1 总体结构

```text
x
  -> GNNBackbone
  -> h
  -> ClassifierHead
  -> tumor/normal logits

h
  -> GeneratedSupportMap T_phi
  -> b = T_phi(h)

(h, b, cancer_dom)
  -> Potential module
  -> dual_potential 或 euclidean_moment generated-support loss
```

`GeneratedSupportMap` 和 potential module 第一版只在训练时使用，推理不需要。为了防止 `b` 成为一个“干净但与分类无关”的辅助空间，必须保留：

```text
L_anchor(h,b) = ||norm(h) - norm(b)||^2
```

或者 BaryIR 风格的 RMSE anchor：

```text
L_anchor(h,b) = sqrt(mean((norm(h)-norm(b))^2))
```

第一版建议使用 identity-initialized residual MLP：

```text
b = h + alpha * MLP(LayerNorm(h))
```

最后一层零初始化，训练初期 `b≈h`，避免 generated support 在 warmup 后突然偏离当前分类表征。

### 4.2 每个 batch 的更新顺序

采用 BaryIR 风格交替：

```text
Step A: update potentials
  freeze STOnco
  freeze GeneratedSupportMap
  unfreeze potentials
  maximize potential objective 或 minimize potential critic loss

Step B: update STOnco
  unfreeze STOnco
  unfreeze GeneratedSupportMap
  freeze potentials
  minimize task + domain + WB alignment + anchor
```

对应伪代码：

```python
for batch in train_loader:
    batch = batch.to(device)

    # Step A: update potentials
    if use_wb_align and lambda_wb_t > 0:
        freeze(model)
        freeze(support_map)
        unfreeze(potentials)
        opt_pot.zero_grad()

        with torch.no_grad():
            # 直接调用 gnn，避免 classifier head 的 BatchNorm running stats
            # 在 potential update 阶段被无意义更新。
            h_detached = model.gnn(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
            b_detached = support_map(h_detached).detach()

        pot_loss, pot_stats = wb_module.potential_loss(
            h=h_detached,
            b=b_detached,
            cancer_dom=batch.cancer_dom[batch.batch],
            graph_nodes=batch.batch,
            y=batch.y,
        )
        pot_loss.backward()
        opt_pot.step()

    # Step B: update STOnco
    unfreeze(model)
    unfreeze(support_map)
    freeze(potentials)
    opt_main.zero_grad()

    out = model(..., return_h=True)
    h = out["h"]
    b = support_map(h)

    loss_task = bce(out["logits"][mask], batch.y[mask].float())
    loss_domain = compute_domain_losses(out, batch)
    loss_wb, wb_stats = wb_module.model_loss(
        h=h,
        b=b,
        cancer_dom=batch.cancer_dom[batch.batch],
        graph_nodes=batch.batch,
        y=batch.y,
    )
    loss_anchor = anchor_loss(h, b)

    total = loss_task + loss_domain + lambda_wb_t * (loss_wb + wb_anchor_weight * loss_anchor)
    total.backward()
    opt_main.step()
```

## 5. Loss 选项一：`dual_potential + generated_support`

### 5.1 什么时候用

当目标是更严格地贴近 continuous regularized Wasserstein barycenter，并且可以接受较高计算量时，使用：

```text
--wb_loss_type dual_potential
```

该模式参考 CWB：每个 cancer source domain 使用一对 dual potentials；但 support-side 不再用 pooled `h`，而是使用 generated support：

```text
b = T_phi(h)
```

### 5.2 两个势函数如何使用

对每个癌种 $k$，建立：

```text
f_k: source-side potential，作用在 cancer k 的 h 上
g_k: support-side potential，作用在 generated support b 上
```

设：

```text
x ~ P_k^h            # cancer k 的 h
y ~ Q_phi^b          # generated support b = T_phi(h)
```

ground cost：

$$
c(x,y)=\frac{1}{D}\|x-y\|_2^2.
$$

CWB 中 `calc_distances(ps, qs)` 使用的也是平方欧氏距离。STOnco 中建议除以维度 $D$，让 cost scale 更稳定。

CWB 还会对 $g_k$ 做中心化，使多个 source 共享同一个 barycenter support：

$$
\bar g(y)=\sum_{j=1}^{K}\omega_j g_j(y),
\quad
v_k(y)=g_k(y)-\bar g(y).
$$

然后对第 $k$ 个癌种的 regularized dual objective 可写为：

$$
J_k(f_k,g_k)
=
\mathbb{E}_{x\sim P_k^h}[f_k(x)]
-
\mathbb{E}_{x\sim P_k^h,y\sim Q_\phi^b}
\left[
R_\epsilon(f_k(x)+v_k(y)-c(x,y))
\right].
$$

其中 $R_\epsilon$ 是 regularizer。两种可选：

entropy regularizer：

$$
R_\epsilon(t)=\epsilon\exp(t/\epsilon)
$$

l2 regularizer：

$$
R_\epsilon(t)=\frac{\max(0,t)^2}{2\epsilon}.
$$

CWB 代码中就支持 entropy 与 l2 两类 regularizer。

### 5.3 Potential update

Potential update 要最大化：

$$
\sum_{k\in\mathcal K_B}\omega_k J_k.
$$

PyTorch 中写成最小化负号：

```python
dual_obj = sum(weight[k] * J_k for k in active_cancers)
pot_loss = -dual_obj
pot_loss.backward()
opt_pot.step()
```

此时 `h` 必须 detach，防止更新 STOnco：

```python
h_detached = h.detach()
b_detached = support_map(h_detached).detach()
```

在 Step A 中同时 freeze `support_map`。也就是说，potential update 只让 `f_k/g_k` 变成更强 critic，不改变 STOnco encoder 或 generated support map。

### 5.4 STOnco update

冻结 potentials 后，把 dual objective 作为分布距离近似，更新 STOnco 和 `support_map`：

```python
loss_wb = dual_obj
loss_anchor = anchor_loss(h, b)
total = loss_task + loss_domain + lambda_wb_t * (loss_wb + wb_anchor_weight * loss_anchor)
```

直观理解：

- potentials 尝试最大化各 cancer distribution `P_k^h` 与 generated support `Q_phi^b` 的 regularized OT dual gap；
- STOnco 和 `T_phi` 尝试改变 `h` 与 `b`，让这个 gap 变小；
- 所有 `g_k` 通过 $\bar g$ 耦合，避免每个癌种学到自己的独立中心。
- anchor loss 防止只让 `b` 对齐，而 `h` 仍然保留 cancer-source 信息。

### 5.5 generated support 如何初始化和采样

本方案不初始化有限 atoms，不采样固定高斯或均匀 prior。support 来自：

```text
b_i = T_phi(h_i)
```

`T_phi` 建议 identity-initialized：

```text
b = h + alpha * MLP(LayerNorm(h))
last linear layer zero-init
alpha 初始为 1 或小常数
```

因此训练初期：

```text
b ≈ h
```

然后通过 alternating update 逐渐把 `b` 学成更泛癌共享的 barycenter support。

`dual_potential` 仍需要抽样 source-side `x` 和 support-side `y`，但这里的 `y` 是从 generated support set `{b_i}` 中抽样：

```text
x = h[cancer == k]
y = b[sampled mixed indices]
```

这不是 pooled mean，也不是有限 atoms。

### 5.6 计算量

`dual_potential` 需要 source-support cost matrix：

$$
O(K_B \cdot n \cdot m \cdot D),
$$

其中：

- $K_B$：batch 内 active cancer 数；
- $n$：每个 cancer 抽样 spot 数；
- $m$：generated support 抽样 spot 数；
- $D$：`h` 维度。

推荐默认：

```text
wb_spots_per_cancer = 64
wb_support_size = 128 或 256
wb_epsilon = 0.05-0.2
wb_regularizer = entropy 或 l2
wb_anchor_weight = 0.1-1.0
```

## 6. Loss 选项二：`euclidean_moment + generated_support`

### 6.1 什么时候用

当主要目标是先以低计算量替换 MMD，并快速验证“barycenter-style 统一中心形状”是否有用时，使用：

```text
--wb_loss_type euclidean_moment
```

该模式参考 BaryIR 的工程思路：由 source latent `h` 生成 bary latent `b=T_phi(h)`，用欧氏 anchor / moment 约束提供 shape/center 约束，用 potential critic 提供训练时对抗信号。它不是严格 Wasserstein distance，但计算量小很多。

### 6.2 基本形式

对每个 active cancer $k$，记：

```text
H_k = {h_i | cancer_i = k}
B_k = {b_i=T_phi(h_i) | cancer_i = k}
B_Q = {b_i=T_phi(h_i) | i from mixed active cancers}
```

第一项是 BaryIR-style anchor，避免 generated support 与分类使用的 `h` 脱耦：

$$
\mathcal{L}_{anchor}
=
\frac{1}{N}
\sum_i
\frac{1}{D}\| \tilde h_i-\tilde b_i \|_2^2,
$$

其中 $\tilde h,\tilde b$ 建议为 WB loss 内部的 layer-normalized 表征。

第二项是 generated-support moment shape 约束。它不再对 pooled `h` 做中心，而是对 generated support `B_k` 的形状做跨癌种统一：

$$
\mathcal{L}_{mean}
=
\frac{1}{|\mathcal K_B|}
\sum_{k\in\mathcal K_B}
\|\mu(B_k)-\mu(B_Q)\|_2^2.
$$

可选 diagonal covariance 形状项：

$$
\mathcal{L}_{var}
=
\frac{1}{|\mathcal K_B|}
\sum_{k\in\mathcal K_B}
\|\operatorname{diagCov}(B_k)-\operatorname{diagCov}(B_Q)\|_2^2.
$$

然后加入 BaryIR-style potential critic：

```text
Pot_k(b): scalar potential for cancer k
```

主模型更新时：

$$
\mathcal{L}_{pot-main}
=
-
\mathbb{E}_i[Pot_{c_i}(b_i)].
$$

总的轻量 WB surrogate：

$$
\mathcal{L}_{WB}^{euc}
=
\lambda_{anchor}\mathcal{L}_{anchor}
+
\mathcal{L}_{mean}
+
\beta_{var}\mathcal{L}_{var}
+
\beta_{pot}\mathcal{L}_{pot-main}.
$$

### 6.3 Potential update

Potential update 参考 BaryIR，冻结 STOnco，只更新 `Pot_k`。为了避免 potential score 无界漂移，建议加入零均值约束：

$$
\mathcal{L}_{pot}
=
\mathbb{E}_i[Pot_{c_i}(b_i)]
+
\gamma
\left(
\sum_k \omega_k \mathbb{E}_{b\sim B_Q}[Pot_k(b)]
\right)^2.
$$

对应直觉：

- potential 网络试图降低当前 generated support `b` 的 score；
- 主模型和 `T_phi` 在下一步通过 `-Pot_k(b)` 试图提高 score；
- 这形成 BaryIR 风格对抗训练；
- 约束项防止所有 potential 只靠整体平移作弊。
- anchor 项让这个对抗信号回到 `h`，避免只优化辅助 support。

### 6.4 计算量

`euclidean_moment` 不需要 source-support cost matrix，主要计算：

```text
mean: O(ND)
diag var: O(ND)
potential MLP: O(ND hidden)
support map MLP: O(ND hidden)
```

比 `dual_potential` 和 Sinkhorn 都便宜，适合作为第一轮工程验证。

### 6.5 局限

该模式不是严格 Wasserstein barycenter。它更像：

```text
center/shape Euclidean alignment + potential adversarial regularization
```

因此文档和代码命名中应避免把它报告为精确 Wasserstein distance。建议在实验表中标为：

```text
WB surrogate (Euclidean moment + potential)
```

## 7. 强形状统一约束

用户目标是让 tumor/normal spot 的 latent 处于同一连续分布的两种状态。generated-support 版本中建议把形状统一主要加在 `b=T_phi(h)` 上，同时用 anchor 约束把效果传回 `h`。

### 7.1 label-balanced sampling

每个 cancer 内参与 WB 的 spot 尽量 tumor/normal 各半。这样避免 cancer 间 label ratio 差异被误当作 domain shift 消除。

### 7.2 state direction 约束

学习一个共享方向 $v$，鼓励不同癌种中 tumor-normal 均值差方向一致：

$$
\Delta_k=\mu_{k,tumor}-\mu_{k,normal},
\quad
\mathcal{L}_{state-dir}
=
\frac{1}{|\mathcal K_B|}
\sum_k
\left(1-\cos(\Delta_k,v)\right).
$$

### 7.3 state-removed residual alignment

可选地从 `h` 或 `b` 中移除状态方向：

$$
r_i = h_i - \alpha_{y_i}v.
$$

然后对 `r_i` 或对应的 generated residual 做同样的 `dual_potential` 或 `euclidean_moment` 对齐。这样保留 tumor/normal 沿共享方向的判别，同时要求其余 base shape 跨癌种统一。

第一版建议只启用 label-balanced sampling；state direction 和 residual alignment 作为第二阶段增强。

## 8. 损失函数汇总

总损失：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{task}
+
\lambda_{slide}\mathcal{L}_{adv,slide}
+
\lambda_{cancer}\mathcal{L}_{adv,cancer}
+
\lambda_{WB}(t)\mathcal{L}_{WB}(h,b,cancer)
+
\lambda_{shape}(t)\mathcal{L}_{shape}.
$$

其中：

- `L_task`：当前 BCEWithLogitsLoss，不变。
- `L_adv_slide`, `L_adv_cancer`：当前 GRL domain loss，不变。
- `L_WB`：由 `--wb_loss_type` 决定：
  - `dual_potential + generated_support`
  - `euclidean_moment + generated_support`
- `b=T_phi(h)`：generated barycenter support。
- `L_anchor(h,b)`：第一版并入 `L_WB`，用于防止 generated support 与最终分类表征 `h` 脱耦。
- `L_shape`：可选强形状统一约束。

调度：

```text
wb_warmup_epochs = 10-20
wb_ramp_epochs = 20-40
lambda_wb = 0.005, 0.01, 0.02
```

线性 ramp：

$$
\lambda_{WB}(t)
=
\lambda_{WB}^{max}
\cdot
\operatorname{clip}
\left(
\frac{t-T_{warmup}}{T_{ramp}},0,1
\right).
$$

## 9. 代码改造建议

### 9.1 新增文件

新增：

```text
stonco/core/wb_potentials.py
```

建议包含：

```python
class GeneratedSupportMap(nn.Module):
    """Identity-initialized residual map b = T_phi(h)."""

class SinglePotentialBank(nn.Module):
    """For euclidean_moment."""

class DualPotentialBank(nn.Module):
    """For dual_potential. Contains f_k and g_k."""

class GeneratedSupportWBLoss(nn.Module):
    def potential_loss(...):
        ...

    def model_loss(...):
        ...
```

### 9.2 `GeneratedSupportMap`

第一版建议用同维度 residual MLP，而不是任意 MLP。这样初始化时 `b≈h`，训练更稳：

```python
class GeneratedSupportMap(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(self.norm(h))
```

`GeneratedSupportMap` 默认并入主优化器 `opt`，与 STOnco 一起在 Step B 更新；Step A 更新 potentials 时冻结。

### 9.3 `SinglePotentialBank`

```python
class SinglePotentialBank(nn.Module):
    def __init__(self, n_domains, in_dim, hidden=128):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_domains)
        ])

    def forward(self, b, domain_idx):
        return self.heads[int(domain_idx)](b).squeeze(-1)
```

### 9.4 `DualPotentialBank`

```python
class DualPotentialBank(nn.Module):
    def __init__(self, n_domains, in_dim, hidden=128):
        super().__init__()
        self.f = nn.ModuleList([make_mlp(in_dim, hidden) for _ in range(n_domains)])
        self.g = nn.ModuleList([make_mlp(in_dim, hidden) for _ in range(n_domains)])

    def g_centered(self, b_support, weights):
        all_g = torch.stack([gk(b_support).squeeze(-1) for gk in self.g], dim=0)
        g_mean = (weights.view(-1, 1) * all_g).sum(dim=0)
        return all_g - g_mean.unsqueeze(0)
```

### 9.5 修改 `train.py`

新增 CLI：

```text
--use_wb_align {0,1}
--wb_loss_type {dual_potential,euclidean_moment}
--wb_support_mode {generated_support}
--lambda_wb
--wb_warmup_epochs
--wb_ramp_epochs
--wb_support_hidden
--wb_support_dropout
--wb_anchor_weight
--wb_potential_hidden
--wb_potential_lr
--wb_pot_every_n_steps
--wb_spots_per_cancer
--wb_support_size
--wb_min_cancers
--wb_min_spots
--wb_regularizer {entropy,l2}
--wb_epsilon
--wb_label_balanced_sampling {0,1}
--wb_euclid_cov_weight
--wb_potential_constraint_weight
```

默认：

```python
"use_wb_align": False,
"wb_loss_type": "euclidean_moment",
"wb_support_mode": "generated_support",
"lambda_wb": 0.01,
"wb_warmup_epochs": 15,
"wb_ramp_epochs": 30,
"wb_support_hidden": 128,
"wb_support_dropout": 0.0,
"wb_anchor_weight": 0.5,
"wb_potential_hidden": 128,
"wb_potential_lr": 1e-3,
"wb_pot_every_n_steps": 1,
"wb_spots_per_cancer": 64,
"wb_support_size": 128,
"wb_min_cancers": 2,
"wb_min_spots": 16,
"wb_regularizer": "l2",
"wb_epsilon": 0.1,
"wb_label_balanced_sampling": True,
"wb_euclid_cov_weight": 0.1,
"wb_potential_constraint_weight": 10.0,
```

训练 forward：

```python
return_h = bool(cfg.get("use_mmd", False) or cfg.get("use_wb_align", False))
```

### 9.6 记录指标

`hist` 和 `loss_components.csv` 增加：

```text
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_mean
avg_wb_euclid_var
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
wb_lambda
```

验证阶段第一版可不计算 WB loss，避免额外开销。

## 10. 计算量控制策略

### 10.1 推荐先跑轻量版

第一轮建议：

```text
wb_loss_type = euclidean_moment
wb_support_mode = generated_support
wb_spots_per_cancer = 64
wb_euclid_cov_weight = 0.1
wb_anchor_weight = 0.5
lambda_wb = 0.005, 0.01
```

确认不伤任务后，再切：

```text
wb_loss_type = dual_potential
```

### 10.2 降低 `dual_potential` 开销

若使用 `dual_potential`：

- `wb_spots_per_cancer=32/64`
- `wb_support_size=128`，表示从 generated support `b` 中抽样的 support-side 点数；
- `wb_pot_every_n_steps=2`
- `wb_regularizer=l2` 可比 entropy 更稳一些，但要网格验证。
- 对 `h` 和 `b` 做 `F.layer_norm(..., (D,))` 后再算 cost。

## 11. 保护下游肿瘤分类性能的策略

1. WB warm-up：前 10-20 epoch 不启用。
2. WB ramp-up：20-40 epoch 线性增加。
3. 弱 cancer GRL：若 generated-support WB 已强，`lambda_cancer` 建议 0.05-0.1。
4. label-balanced WB sampling：避免 label ratio 被对齐损失抹掉。
5. checkpoint 仍按 `val_auprc`、`val_macro_f1` 或当前兼容的 `val_accuracy` 选择，不用 WB loss 早停。
6. 对 `h` 做 post-hoc cancer probe 和 tumor probe，确认 cancer 信息下降但 tumor 信息保留。

## 12. 实验设计与验收指标

建议实验组：

| 组别 | 配置 | 目的 |
| --- | --- | --- |
| A. 当前无 MMD | `use_mmd=0`, `use_wb_align=0` | 主任务 baseline |
| B. 当前 MMD | `use_mmd=1`, `mmd_on=cancer` | 现有对齐 baseline |
| C. Euclidean moment + generated support | `use_wb_align=1`, `wb_loss_type=euclidean_moment`, `wb_support_mode=generated_support` | 低计算量替换 MMD |
| D. Dual-potential + generated support | `use_wb_align=1`, `wb_loss_type=dual_potential`, `wb_support_mode=generated_support` | 正规 dual WB 对齐 |
| E. Euclidean + weak GRL | C + weak cancer GRL | 检查去 cancer 信息 |
| F. Dual-potential + state-shape | D + shape constraints | 检查强形状统一 |

主指标：

- `val_auprc`
- `val_macro_f1`
- `val_accuracy`
- `val_auroc`
- LOCO average AUPRC / macro-F1
- post-hoc cancer probe accuracy / macro-F1 on `h`
- `iLISI(cancer_type)` 和 `cLISI(tumor_label)`

验收建议：

- `euclidean_moment` 至少不弱于当前 MMD 的任务性能，并能降低 cancer probe。
- `dual_potential` 若计算可接受，应比 `euclidean_moment` 有更强 cancer mixing。
- 若 cancer mixing 提升但 AUPRC 下降超过 3 个百分点，判定为过对齐。

## 13. 风险与失败模式

### 13.1 Euclidean surrogate 被误读成严格 Wasserstein

`euclidean_moment` 是 WB-inspired surrogate，不是严格 Wasserstein distance。报告时要明确命名。

### 13.2 Potentials score 漂移

potential 输出可以整体平移，导致 loss 不稳定。需要：

- potential constraint；
- gradient clipping；
- potential lr 不高于主 lr；
- 记录 `avg_wb_potential_loss` 和 score mean/std。

### 13.3 generated support 脱耦或过对齐

如果 `T_phi` 太强而 anchor 太弱，可能出现 `b` 已经泛癌对齐，但最终分类使用的 `h` 仍保留 cancer-source 信息。反过来，如果 anchor / WB 太强，也可能洗掉 tumor signal。需要 identity initialization、warm-up、ramp-up、anchor weight 网格、label-balanced sampling 和任务指标早停。

### 13.4 batch 内癌种不足

使用 random sampler 时 active cancers 可能不足。需要 cancer-balanced sampler，并记录 `avg_wb_active_cancers`。

## 14. 最终推荐实现路径

第一阶段：低风险接入。

1. 新增 `stonco/core/wb_potentials.py`。
2. 实现 `GeneratedSupportMap`：
   - `b = h + MLP(LayerNorm(h))`；
   - 最后一层零初始化，使训练初期 `b≈h`。
3. 实现 `euclidean_moment + generated_support`：
   - 单 potential head；
   - anchor + generated-support mean/diagonal variance 欧氏形状约束；
   - BaryIR-style potential alternating update。
4. 在 `train.py` 接入：
   - `return_h = use_mmd or use_wb_align`；
   - 每 batch 先 update potential，再 update STOnco + support map。
5. 跑 A/B/C 对照。

第二阶段：正规 dual WB。

1. 实现 `dual_potential`：
   - 每 cancer 一对 `f_k/g_k`；
   - `h` 作为 source-side samples；
   - `b=T_phi(h)` 作为 generated support-side samples；
   - entropy/l2 regularized dual objective；
   - `g_k - weighted_mean(g)` 中心化。
2. 跑 D/E 对照。

第三阶段：强形状统一。

1. 加 label-balanced sampling。
2. 加 state direction 和 state-removed residual alignment。
3. 用 LOCO、cancer probe、LISI 和 tumor AUPRC 共同验收。

## 15. 参考文献

[Agueh2011] Martial Agueh and Guillaume Carlier. "Barycenters in the Wasserstein Space." SIAM Journal on Mathematical Analysis, 43(2):904-924, 2011. https://doi.org/10.1137/100805741

[Gretton2012] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schoelkopf, and Alexander Smola. "A Kernel Two-Sample Test." Journal of Machine Learning Research, 13:723-773, 2012. https://jmlr.org/papers/v13/gretton12a.html

[Cuturi2013] Marco Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." Advances in Neural Information Processing Systems 26, 2013. https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html

[Cuturi2014] Marco Cuturi and Arnaud Doucet. "Fast Computation of Wasserstein Barycenters." Proceedings of the 31st International Conference on Machine Learning, PMLR 32(2):685-693, 2014. https://proceedings.mlr.press/v32/cuturi14.html

[Ganin2016] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Francois Laviolette, Mario Marchand, and Victor Lempitsky. "Domain-Adversarial Training of Neural Networks." Journal of Machine Learning Research, 17(59):1-35, 2016. https://jmlr.org/papers/v17/15-239.html

[Feydy2019] Jean Feydy, Thibault Sejourne, Francois-Xavier Vialard, Shun-ichi Amari, Alain Trouve, and Gabriel Peyre. "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences." AISTATS, PMLR 89:2681-2690, 2019. https://proceedings.mlr.press/v89/feydy19a.html

[Li2020] Lingxiao Li, Aude Genevay, Mikhail Yurochkin, and Justin Solomon. "Continuous Regularized Wasserstein Barycenters." Advances in Neural Information Processing Systems 33, 2020. https://proceedings.neurips.cc/paper/2020/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html

[Janati2020] Hicham Janati, Marco Cuturi, and Alexandre Gramfort. "Debiased Sinkhorn Barycenters." Proceedings of the 37th International Conference on Machine Learning, PMLR 119:4692-4701, 2020. https://proceedings.mlr.press/v119/janati20a.html

[Fan2021] Jiaojiao Fan, Amirhossein Taghvaei, and Yongxin Chen. "Scalable Computations of Wasserstein Barycenter via Input Convex Neural Networks." ICML, PMLR 139:1571-1581, 2021. https://proceedings.mlr.press/v139/fan21d.html

[Korotin2021] Alexander Korotin, Lingxiao Li, Justin Solomon, and Evgeny Burnaev. "Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization." ICLR, 2021. https://openreview.net/forum?id=3tFAs5E-Pe

[Flamary2021] Remi Flamary et al. "POT: Python Optimal Transport." Journal of Machine Learning Research, 22(78):1-8, 2021. https://jmlr.org/papers/v22/20-451.html

[Tang2026] Xiaole Tang, Xiaoyi He, Jiayi Xu, Xiang Gu, and Jian Sun. "Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration." arXiv:2602.23169, submitted 2026-02-26. https://arxiv.org/abs/2602.23169

## 16. 代码实现前需要用户确认的细节

本节是代码实现前的确认清单。下面每一项都来自当前 STOnco 实际代码与本文档方案之间的对照，目的是避免后续实现偏离最初目标：让 `h` 本身成为连续、可学习、泛癌共享的 Wasserstein barycenter 表征空间，同时尽量消除 cancer-source 信息并保护 tumor/normal 分类性能。

### 16.1 是否确认 generated support 是训练阶段 support，而不是推理 embedding

当前文档主方案是：

```text
GNNBackbone -> h -> ClassifierHead
              |
              +-> GeneratedSupportMap T_phi -> b
                                      |
                                      +-> WB potentials during training
```

需要确认：

- `h` 本身就是 continuous learnable Wasserstein barycenter space。
- 不新增独立 `barycenter atoms`。
- 新增 `GeneratedSupportMap`，但它默认只在训练阶段用于生成 support `b=T_phi(h)`。
- 推理阶段仍默认使用 `GNNBackbone -> h -> ClassifierHead`，不把分类 head 改成依赖 `b`。

这与 finite-atom barycenter 不同。这里的 barycenter support 由连续可学习映射 `T_phi` 从 `h` 生成，不是一组离散 atoms，也不是固定高斯/均匀 prior。

需要确认：

```text
确认 GeneratedSupportMap 只作为训练阶段 support generator；
推理与下游分类仍以 h 为准。
```

### 16.2 `h` 的实际维度是不是用户希望对齐的空间

当前 `STOnco_Classifier.forward(..., return_h=True)` 返回的是 GNN backbone 输出：

```python
h = self.gnn(x, edge_index, edge_weight=edge_weight)
```

默认 `gatv2` 下，`GNN_hidden=[256,128,64]` 且 `heads=4` 时，最后一层 `h` 维度通常是：

```text
D = 64 * 4 = 256
```

它不是 classifier head 内部的 `z_clf`，也不是 `clf_hidden` 最后一层的 64 维表征。

需要确认：

- WB 对齐是否就作用在当前 GNN 输出 `h`。
- 不改成 classifier latent `z_clf`。
- 不新增一个 64 维压缩空间来降低计算量。

确认：

```text
对齐 GNN 输出 h；不使用 z_clf。
```


### 16.3 generated support 是否替代 pooled support

当前文档推荐：

```text
b = T_phi(h)
```

旧版候选方案包括：

```text
current-batch pooled h.detach()
FIFO memory bank pooled h.detach()
```

但当前更新后的主方案 B 不再把 pooled `h` 作为主 support，而是：

```text
Q_phi = distribution of T_phi(h)
```

需要确认：

- 第一版是否只实现 `generated_support`。
- pooled support / memory bank 是否只保留为后续 ablation，不进入第一版代码。

建议默认确认：

```text
第一版只实现 generated_support。
不实现 batch_mixed_h / memory_bank support。
```

### 16.4 barycenter 是否需要显式采样

本方案不显式采样独立 barycenter 点，不调用类似 `sample_barycenter()` 的生成过程。

但 `dual_potential` 需要从 support measure 中取 `y`：

```text
x ~ P_k^h
y ~ Q_phi^b
```

实现上这个 `y` 是从当前 batch 生成的 support set `{b_i=T_phi(h_i)}` 中 subsample，不是从 pooled mean、固定分布或 finite atoms 中采样。

需要确认：

```text
确认不显式采样外部 barycenter；
dual_potential 的 y 只从 generated support b 中抽样。
```

### 16.5 potential update 时如何处理 STOnco 的 train/eval mode

当前 `ClassifierHead` 使用 `BatchNorm1d`。如果 potential update 阶段只是 `torch.no_grad()`，但模型仍处于 `train()`，classifier head 的 BatchNorm running stats 仍可能被更新。

这会让 Step A “只更新 potentials、不更新 STOnco”的语义不严格。

可选实现：

```text
方案 A：potential update 阶段调用 model.eval()
优点：不会更新 dropout/BN 状态。
缺点：GNN dropout 关闭，h support 与主训练 forward 略有差异。

方案 B：potential update 阶段只调用 model.gnn(...)
优点：直接取得 h，绕开 classifier head 的 BatchNorm。
缺点：绕过 STOnco_Classifier.forward，代码侵入性略高。

方案 C：保持 model.train()，但临时冻结 BatchNorm running stats
优点：dropout 行为与主训练一致。
缺点：实现更复杂，容易漏掉模块状态。
```

建议默认确认：

```text
采用方案 B：potential update 只调用 model.gnn(...) 取得 h_detached，
再由 frozen support_map 生成 b_detached，避免 classifier BN 被无意义更新。
```

### 16.6 WB 与当前默认 cancer GRL 是否同时强启用

当前 STOnco 默认配置是：

```python
"use_domain_adv_cancer": True
"lambda_cancer": 1.0
"grl_beta_cancer_target": 0.5
```

文档目标是用 WB 更强地消除 cancer-source domain 信息。如果同时保留强 cancer GRL，可能出现：

```text
WB alignment + cancer GRL 双重去癌种
```

这可能有利于消除 cancer 信息，但也更容易过对齐并损伤 tumor/normal 判别。

需要确认：

- 启用 WB 时，是否默认关闭 cancer GRL。
- 或者保留 cancer GRL，但把 `lambda_cancer` 降到 0.05-0.1。
- slide/batch GRL 是否仍保持默认开启。

建议默认确认：

```text
WB 实验第一版：use_domain_adv_cancer=1，但 lambda_cancer=0.05 或 0.1。
如果任务性能下降，再尝试关闭 cancer GRL。
```

### 16.7 WB 与 MMD 是否允许同时启用

当前 MMD 已经在 `_compute_losses(...)` 中直接加到 `total`。如果后续实现只是新增 WB，而不做互斥控制，则用户可能同时打开：

```text
use_mmd=1
use_wb_align=1
```

这会变成：

```text
task + domain + MMD + WB
```

而用户最初目标是“替换或升级当前 MMD”。因此需要确认第一版实现是否默认互斥。

建议默认确认：

```text
WB 主实验默认 use_mmd=0。
允许高级用户手动同时开启，但 CLI 打印 warning。
```

### 16.8 `mmd_on` 当前默认是 slide，不是 cancer

当前代码默认：

```python
"mmd_on": "slide"
```

但本方案关注多癌种 latent 边缘分布对齐，实验 baseline 应该使用：

```text
use_mmd=1, mmd_on=cancer
```

需要确认：

```text
文档中的 MMD baseline 是否固定使用 mmd_on=cancer，而不是沿用代码默认 mmd_on=slide。
```

建议默认确认：

```text
WB 对照实验中，MMD baseline 使用 mmd_on=cancer；若要同时评估 batch/slide 去除，再加 mmd_on=both。
```

### 16.9 sampler 默认值与 WB 需求不一致

当前代码默认：

```python
"batch_size_graphs": 2
"sampler_mode": "random"
```

generated-support WB 在参数层面是全局共享的，但有效跨癌种对齐仍依赖 batch 内出现多个 cancer source。文档推荐：

```bash
--batch_size_graphs 8
--sampler_mode cancer_balanced
--sampler_k_cancers 4
--sampler_m_per_cancer 2
```

需要确认：

- 启用 `use_wb_align=1` 时，代码是否自动检查 sampler 设置。
- 如果用户仍用 random sampler，是否只是跳过 active cancer 不足的 batch。
- 是否自动 warning，而不是强制报错。

建议默认确认：

```text
启用 WB 时不强制改 sampler，但若 avg active cancers < wb_min_cancers，打印 warning，并在对应 batch 跳过 WB loss。
```

### 16.10 graph-level balance 不等于 spot-level balance

当前 `CancerBalancedBatchSampler` 平衡的是 graph/slide 数，而 WB loss 作用在 spot-level `h` 和 generated support `b` 上。不同切片 spot 数可能差异很大，导致 generated support set 被大图或大癌种主导。

需要确认：

- WB 内部是否必须按 cancer 限制 spot 数。
- 是否还需要按 graph/slide 限制 spot 数。

建议默认确认：

```text
实现 wb_spots_per_cancer，并在每个 cancer 内随机抽 spot。
暂不新增 wb_spots_per_graph；如果大图支配明显，再加。
```

### 16.11 WB 是否使用 unlabeled spot

当前 task loss 会跳过：

```python
mask = batch.y >= 0
```

但 WB 对齐如果直接使用所有 `h`，会包含 `y < 0` 的 spot。这样更接近 source distribution alignment，但无法进行 label-balanced shape 约束。

需要确认：

```text
WB base alignment 是否使用全部 spot，还是只使用 y>=0 的 labeled spot。
```

建议默认确认：

```text
base WB alignment 使用全部 spot。
label-balanced 和 state-shape 约束只在 y>=0 的 spot 上启用。
```

### 16.12 `euclidean_moment` 是否符合用户对“欧氏距离替代 Wasserstein”的理解

当前文档中的 `euclidean_moment` 是：

```text
anchor h -> b
mean alignment
optional diagonal variance alignment
potential critic
```

它不是 pairwise Euclidean OT，也不是完整 energy distance。它是低成本 moment-level shape surrogate。

需要确认：

- 这个轻量版本是否就命名为 `euclidean_moment`。
- 是否希望增加另一个更直接的 `euclidean_pairwise`，用跨癌种 pairwise Euclidean/energy distance 做分布距离。

建议默认确认：

```text
第一版只实现 euclidean_moment + generated_support。
若效果不足，再增加 euclidean_pairwise。
```

### 16.13 `dual_potential` 的符号和 min-max 更新需要固定

文档当前定义：

```text
potential update: maximize dual objective
STOnco update: minimize dual objective
```

PyTorch 实现应为：

```python
pot_loss = -dual_obj
model_loss = +dual_obj
```

需要确认：

```text
采用这套符号约定，并在实现后用一个 toy batch sanity check 检查 loss 方向。
```

建议默认确认：

```text
代码中记录 avg_wb_dual_obj、avg_wb_potential_loss、avg_wb_loss，避免符号错误隐藏。
```

### 16.14 `dual_potential` 的 regularizer 默认选 entropy 还是 l2

文档当前默认配置写的是：

```python
"wb_regularizer": "l2"
"wb_epsilon": 0.1
```

但计算量控制章节提到 `l2` 可能更稳。两者行为不同：

```text
entropy: 更接近 entropic regularized OT，但 exp 可能数值敏感。
l2: hinge-squared 形式更稳定，梯度更稀疏。
```

需要确认第一版默认值。

建议默认确认：

```text
第一版 dual_potential 默认 wb_regularizer=l2, wb_epsilon=0.1。
entropy 作为网格实验选项。
```

如果用户希望更贴近 CWB 默认设置，则把默认改回 entropy。

### 16.15 `h` 和 `b` 是否在 WB loss 内部归一化

当前 GNN backbone 已经使用 `LayerNorm`，但 `h` 和 generated support `b` 的尺度仍会随模型训练改变。`dual_potential` 的 cost：

```text
c(x,y)=||x-y||^2/D
```

对尺度敏感。

可选实现：

```text
raw: 直接用 h 和 b。
layer_norm: 只在 WB loss 内对 h 和 b 做 F.layer_norm。
stop_scale: 用 detach 的 batch std 做尺度归一化。
```

需要确认：

```text
WB loss 内是否使用 layer_norm(h) 与 layer_norm(b)。
```

建议默认确认：

```text
WB loss 内使用 F.layer_norm(h, (D,)) 与 F.layer_norm(b, (D,)) 计算 cost 和 potential 输入；
ClassifierHead 仍使用 raw h。
```

这仍会让 WB 梯度回传到 `h`，但对齐的是归一化后的几何。

### 16.16 state-shape 约束第一版做到哪一步

文档提出：

```text
label-balanced sampling
state direction
state-removed residual alignment
```

这些约束用于实现“tumor 和 normal spot 是同一连续分布中的两种状态”。但当前 STOnco 代码里还没有这些组件。

需要确认第一版范围：

```text
方案 A：第一版只做 label-balanced WB sampling。
方案 B：第一版同时实现 state direction。
方案 C：第一版直接实现 state-removed residual alignment。
```

建议默认确认：

```text
第一版只实现 label-balanced sampling。
state direction 和 residual alignment 作为第二阶段。
```

原因是 generated-support WB 与 cancer GRL 已经会强约束 `h`，一次性加入 residual 约束会让失败原因难以定位。

### 16.17 validation 阶段是否计算 WB loss

当前文档建议第一版验证阶段不计算 WB loss，避免额外开销。当前 STOnco 验证阶段会在 `use_mmd=1` 时计算 MMD，因此如果 WB 也在验证阶段计算，会增加明显开销。

需要确认：

```text
validation 是否只评估 task/domain/MMD，不计算 WB。
```

建议默认确认：

```text
第一版 validation 不计算 WB loss。
只在 train hist 记录 WB 训练指标。
```

### 16.18 best checkpoint 当前按 val_accuracy，不是 AUPRC

当前 `train_and_validate(...)` 里最佳模型选择是：

```python
improved = val_accuracy > best["accuracy"]
```

文档保护下游任务时提到可按 `val_auprc` 或 `val_macro_f1`。这与当前代码不一致。

需要确认：

- 是否在实现 WB 时顺手增加 `--best_metric`。
- 还是保持当前 `val_accuracy`，避免扩大改动范围。

建议默认确认：

```text
增加 best_metric 参数，默认保持 val_accuracy。
WB 实验推荐 best_metric=val_auprc 或 val_macro_f1。
```

否则“保护下游肿瘤分类性能”的验收与 checkpoint 选择可能不一致。

### 16.19 loss_components 和曲线图是否同步扩展

当前 `loss_components.csv` 和训练曲线固定记录 MMD、domain、task loss。WB 实现后需要新增：

```text
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_mean
avg_wb_euclid_var
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
wb_lambda
```

需要确认：

```text
第一版是否只写入 CSV，不改 SVG 图布局。
```

建议默认确认：

```text
先扩展 CSV 和 hist；SVG 只做最小兼容，避免重排所有旧图。
```

### 16.20 support map 和 potential 参数是否需要保存

GeneratedSupportMap 和 potential module 第一版只在训练时使用，推理不需要。但如果要完整复现实验或中断恢复，support map、potential state 与 optimizer state 也有价值。

当前 STOnco 保存 best state 时主要保存：

```text
model.state_dict()
```

需要确认：

- 最终 `model.pt` 是否只保存 STOnco 主模型。
- 是否额外保存 `wb_support_map.pt` 和 `wb_potentials.pt` 用于复现或恢复训练。

建议默认确认：

```text
推理模型只保存 STOnco 主模型。
训练 artifacts 额外保存 wb_support_map_last.pt、wb_potentials_last.pt 和 wb_config.json。
```

### 16.21 WB 的 cancer 权重 `ω_k` 如何定义

当前 domain CE 使用 graph-frequency 权重。WB barycenter 目标中的 `ω_k` 可以有两种：

```text
uniform over active cancers: 每个癌种同权。
frequency weighted: 按 batch 或训练集频率加权。
```

用户目标是消除 cancer-source 信息，因此频率加权可能让大癌种主导 barycenter。

需要确认：

```text
WB 中 ω_k 是否统一使用 active-cancer uniform weights。
```

建议默认确认：

```text
WB 使用 uniform over active cancers。
```

### 16.22 LOCO / k-fold 下 potential bank 的 cancer 数量

当前 `train_and_validate(...)` 接收 `n_domains_cancer`，通常按当前训练/验证构建好的 cancer label space 传入。LOCO 模式下某个癌种作为验证癌种，训练 batch 中不会出现该癌种。

需要确认：

- potential bank 是否按全局 `n_domains_cancer` 建立，未出现癌种的 potentials 不更新。
- 还是只按训练集 active cancers 建立映射。

建议默认确认：

```text
按 n_domains_cancer 建立 potential bank，batch 内只更新 active cancers。
```

这样与当前 `batch.cancer_dom[batch.batch]` 编码最兼容。

### 16.23 文档中的 post-hoc probe / LISI 是否纳入第一版代码

文档验收指标包含：

```text
post-hoc cancer probe
iLISI / cLISI
LOCO average AUPRC / macro-F1
```

当前训练代码已有 domain head 训练准确率，但这不是严格 post-hoc probe。LISI 也不是当前训练流程的一部分。

需要确认：

```text
第一版实现是否只提供 WB 训练机制，probe/LISI 作为后续分析脚本。
```

建议默认确认：

```text
第一版不把 probe/LISI 接进 train.py。
先保存 h embedding 或复用现有 embedding 分析脚本做 post-hoc 验收。
```

### 16.24 实现文件边界是否接受新增 `wb_potentials.py`

用户希望 loss 直接写进训练函数中，以保持简洁。文档当前建议新增：

```text
stonco/core/wb_potentials.py
```

折中实现可以是：

```text
wb_potentials.py: 放 GeneratedSupportMap、potential modules 和纯 loss helper。
train.py: 保留 alternating update 控制流、lambda schedule、metrics 汇总。
```

需要确认：

```text
是否接受新增 wb_potentials.py，而不是把 GeneratedSupportMap、potentials 和 loss helper 全部塞进 train.py。
```

建议默认确认：

```text
接受新增 wb_potentials.py；训练流程仍显式写在 train.py 中。
```

### 16.25 建议的第一版实现边界

如果以上默认建议被确认，第一版代码实现边界是：

```text
实现 use_wb_align。
实现 wb_loss_type={euclidean_moment,dual_potential}。
实现 wb_support_mode=generated_support。
新增 GeneratedSupportMap，但不改 models.py。
训练阶段 return_h = use_mmd or use_wb_align。
potential update 使用 model.gnn(...) 取 h_detached，再生成 b_detached。
main update 使用 model(..., return_h=True)，再生成 b。
base WB 使用全部 spot。
label-balanced sampling 仅用于 euclidean_moment / optional state-aware sampling。
WB 默认 use_mmd=0。
WB 默认建议弱 cancer GRL，而不是默认强 lambda_cancer=1.0。
第一版不实现 pooled support、memory bank、state residual alignment、post-hoc probe/LISI 集成。
```

需要用户最终确认的最小问题集：

1. 是否确认 `h` 仍是最终分类表征，`b=T_phi(h)` 只作为训练阶段 generated support？
2. 第一版是否只实现 `generated_support`，不实现 pooled support / memory bank？
3. `GeneratedSupportMap` 是否采用 identity-initialized residual MLP：`b=h+MLP(LayerNorm(h))`？
4. potential update 是否可以直接调用 `model.gnn(...)` 取 `h`，再用 frozen `support_map` 生成 `b`，以避免 classifier BatchNorm 状态被更新？
5. 启用 WB 时，cancer GRL 默认是弱保留、关闭，还是保持当前强默认？
6. WB 和 MMD 是否默认互斥？
7. 第一版 `euclidean_moment` 是否只做 anchor + generated-support mean/diag-var + potential critic，不实现 pairwise Euclidean distance？
8. 第一版是否只做 label-balanced sampling，不做 state direction / residual alignment？
9. 是否新增 `best_metric`，让 WB 实验能按 `val_auprc` 或 `val_macro_f1` 选 checkpoint？
10. 是否接受新增 `stonco/core/wb_potentials.py`，但把 alternating update 保留在 `train.py`？
