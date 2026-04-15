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
  - `euclidean_pairwise + generated_support`：轻量版欧氏 anchor / pairwise distribution 约束与 potential critic，计算形式参考 BaryIR 的 latent 欧氏距离，但用样本两两欧氏距离刻画癌种分布与共享 support 的差异。

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
  -> dual_potential 或 euclidean_pairwise generated-support loss
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
b = h + MLP(LayerNorm(h))
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
            h_detached = model.gnn(
                batch.x,
                batch.edge_index,
                edge_weight=getattr(batch, "edge_weight", None),
            )
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
    # model_loss 返回 raw WB alignment 项，不包含 anchor，避免重复计入。
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
# loss_wb 是 raw dual alignment，不包含 anchor。
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
b = h + MLP(LayerNorm(h))
last linear layer zero-init
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
wb_spots_per_graph = 64
wb_spots_per_cancer = 0
wb_support_size = 128 或 256
wb_epsilon = 0.05-0.2
wb_regularizer = entropy 或 l2
wb_anchor_weight = 0.1-1.0
```

## 6. Loss 选项二：`euclidean_pairwise + generated_support`

### 6.1 什么时候用

当主要目标是先以低计算量替换 MMD，并快速验证“barycenter-style 统一中心形状”是否有用时，使用：

```text
--wb_loss_type euclidean_pairwise
```

该模式参考 BaryIR 的工程思路：由 source latent `h` 生成 bary latent `b=T_phi(h)`，用欧氏 anchor 约束 `h` 与 `b`，再用 generated support 上的两两欧氏距离刻画每个 cancer 分布与共享 support 分布的差异，并用 potential critic 提供训练时对抗信号。它不是严格 Wasserstein distance，但比均值/方差矩匹配更直接地比较分布形状。

### 6.2 基本形式

对每个 active cancer $k$，记：

$$
H_k = \{h_i \mid \text{cancer}_i = k\}
\\[10pt]
B_k = \{b_i = T_\phi(h_i) \mid \text{cancer}_i = k\}
\\[10pt]
B_Q = \{b_i = T_\phi(h_i) \mid i \text{ from mixed active cancers}\}
$$


训练循环额外加 BaryIR-style anchor，避免 generated support 与分类使用的 `h` 脱耦。该项不包含在 `model_loss()` 返回的 raw WB loss 内：

$$
\mathcal{L}_{anchor}
=
\frac{1}{N}
\sum_i
\frac{1}{D}\| \tilde h_i-\tilde b_i \|_2^2,
$$

其中 $\tilde h,\tilde b$ 建议为 WB loss 内部的 layer-normalized 表征。

第二项是 generated-support pairwise Euclidean distribution loss。第一版建议用 energy-distance 形式，因为它同时包含跨分布距离和分布内部距离，比单向 nearest-neighbor 更不容易塌缩：

$$
\mathcal{L}_{pair}
=
\frac{1}{|\mathcal K_B|}
\sum_{k\in\mathcal K_B}
\left[
2\mathbb{E}_{u\sim B_k,v\sim B_Q}d(u,v)
-
\mathbb{E}_{u,u'\sim B_k}d(u,u')
-
\mathbb{E}_{v,v'\sim B_Q}d(v,v')
\right],
$$

其中：

$$
d(u,v)=\frac{1}{\sqrt D}\|u-v\|_2.
$$

实现时用 `torch.cdist(B_k, B_Q, p=2)`、`torch.cdist(B_k, B_k, p=2)` 和 `torch.cdist(B_Q, B_Q, p=2)` 计算两两欧氏距离矩阵。若某个集合只有 1 个点，内部距离项可置 0 或跳过该 cancer。

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

总的轻量 raw WB surrogate 不包含 anchor。anchor 统一由训练循环在 `model_loss()` 外加一次，避免与 `wb_anchor_weight` 重复计入：

$$
\mathcal{L}_{WB,raw}^{euc}
=
\beta_{pair}\mathcal{L}_{pair}
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
- 训练循环外单独加入的 anchor 项让这个对抗信号回到 `h`，避免只优化辅助 support。

### 6.4 计算量

`euclidean_pairwise` 不需要 dual potentials 的 source-support regularized OT cost，但需要 generated support 的两两欧氏距离矩阵，主要计算：

```text
pairwise distance: O(K_B * n * m * D) 或 O(N^2D)
potential MLP: O(ND hidden)
support map MLP: O(ND hidden)
```

它比 `dual_potential` 少了 `f_k/g_k` dual regularizer 的 min-max OT 计算，训练更直接；但它不是 $O(ND)$ 的矩匹配 loss，计算量高于均值/方差对齐。第一版应配合 `wb_spots_per_graph` 和可选 `wb_spots_per_cancer` 控制 selected spot 数。

### 6.5 局限

该模式不是严格 Wasserstein barycenter。它更像：

```text
pairwise Euclidean distribution alignment + potential adversarial regularization
```

因此文档和代码命名中应避免把它报告为精确 Wasserstein distance。建议在实验表中标为：

```text
WB surrogate (Euclidean pairwise + potential)
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

然后对 `r_i` 或对应的 generated residual 做同样的 `dual_potential` 或 `euclidean_pairwise` 对齐。这样保留 tumor/normal 沿共享方向的判别，同时要求其余 base shape 跨癌种统一。

按第 16 节确认结果，第一版实现 `label-balanced sampling` 与 `state direction`，但二者都作为可选项，默认关闭；`state-removed residual alignment` 放到第二阶段增强。

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
- `L_WB`：由 `--wb_loss_type` 决定，表示 raw WB alignment 项，不包含 anchor：
  - `dual_potential + generated_support`
  - `euclidean_pairwise + generated_support`
- `b=T_phi(h)`：generated barycenter support。
- `L_anchor(h,b)`：在训练循环中单独加一次，用于防止 generated support 与最终分类表征 `h` 脱耦；`model_loss()` 内不再重复包含 anchor。
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
    """For euclidean_pairwise."""

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

    def forward_one_domain(self, b, domain_idx):
        return self.heads[int(domain_idx)](b).squeeze(-1)

    def forward_by_domain(self, b, domain_idx):
        """domain_idx 是 per-sample cancer_dom，shape=[N]。"""
        out = b.new_empty(b.size(0))
        for k in torch.unique(domain_idx).tolist():
            mask = domain_idx == int(k)
            out[mask] = self.forward_one_domain(b[mask], int(k))
        return out
```

`SinglePotentialBank.forward_one_domain()` 只接收 scalar domain id。训练时若输入 `batch.cancer_dom[batch.batch]` 这种 per-sample 向量，必须使用 `forward_by_domain()` 或在 loss helper 中先按 cancer 分组，再把每个 head 的输出填回原 spot 顺序，避免整个 batch 错误地走同一个 potential head。

### 9.4 `DualPotentialBank`

```python
class DualPotentialBank(nn.Module):
    def __init__(self, n_domains, in_dim, hidden=128):
        super().__init__()
        self.f = nn.ModuleList([make_mlp(in_dim, hidden) for _ in range(n_domains)])
        self.g = nn.ModuleList([make_mlp(in_dim, hidden) for _ in range(n_domains)])

    def g_centered(self, b_support, active_domains, weights=None):
        active_domains = [int(k) for k in active_domains]
        if weights is None:
            weights = b_support.new_full((len(active_domains),), 1.0 / len(active_domains))
        else:
            weights = weights.to(device=b_support.device, dtype=b_support.dtype)
        all_g = torch.stack([self.g[k](b_support).squeeze(-1) for k in active_domains], dim=0)
        g_mean = (weights.view(-1, 1) * all_g).sum(dim=0)
        return all_g - g_mean.unsqueeze(0)
```

`g_centered()` 只对当前 batch 的 `active_domains` 做 stack 和中心化。`weights` 采用 active-cancer uniform weights，长度必须等于 `len(active_domains)`；未出现在当前 batch 的 cancer potential 不参与本 batch 的中心化，也不更新。

### 9.5 修改 `train.py`

新增 CLI：

```text
--use_wb_align {0,1}
--wb_loss_type {dual_potential,euclidean_pairwise}
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
--wb_spots_per_graph
--wb_spots_per_cancer
--wb_support_size
--wb_min_cancers
--wb_min_spots
--wb_regularizer {entropy,l2}
--wb_epsilon
--wb_label_balanced_sampling {0,1}
--wb_state_direction {0,1}
--wb_eval_loss {0,1}
--wb_euclid_pairwise_weight
--wb_potential_constraint_weight
--best_metric {val_macro_f1,val_auprc,val_accuracy,val_auroc}
```

默认：

```python
"use_wb_align": False,
"wb_loss_type": "euclidean_pairwise",
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
"wb_spots_per_graph": 64,
"wb_spots_per_cancer": 0,
"wb_support_size": 128,
"wb_min_cancers": 2,
"wb_min_spots": 16,
"wb_regularizer": "l2",
"wb_epsilon": 0.1,
"wb_label_balanced_sampling": False,
"wb_state_direction": False,
"wb_eval_loss": False,
"wb_euclid_pairwise_weight": 1.0,
"wb_potential_constraint_weight": 10.0,
"best_metric": "val_macro_f1",
```

训练 forward：

```python
return_h = bool(cfg.get("use_mmd", False) or cfg.get("use_wb_align", False))
```

WB 的 cancer potential bank 依赖 `n_domains_cancer`，不能只在 `use_domain_adv_cancer=True` 时才返回 cancer domain 数。`prepare_graphs()` 中应改为：

```python
n_domains_cancer = len(cancer_to_idx) if (
    cfg.get("use_domain_adv_cancer", False) or cfg.get("use_wb_align", False)
) else None
```

这样即使后续为了保护分类性能关闭 cancer GRL，只要 `use_wb_align=1`，WB 仍能建立完整 cancer potential bank。

### 9.6 记录指标

`hist` 和 `loss_components.csv` 增加：

```text
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_pairwise
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
wb_lambda
```

验证阶段新增可选 `wb_eval_loss`：默认不计算 WB loss，避免额外开销；开启后计算并保存 validation WB diagnostics。

开启 `use_wb_align=1` 时，额外保存一张 WB 专用曲线图：

```text
wb_train_loss.svg
```

建议包含：

```text
avg_total_loss
avg_task_loss
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_pairwise
avg_wb_anchor
wb_lambda
avg_wb_active_cancers
avg_wb_active_spots
```

其中不适用于当前 `wb_loss_type` 的曲线用 NaN 或跳过绘制。

## 10. 计算量控制策略

### 10.1 推荐先跑轻量版

第一轮建议：

```text
wb_loss_type = euclidean_pairwise
wb_support_mode = generated_support
wb_spots_per_graph = 64
wb_spots_per_cancer = 0
wb_euclid_pairwise_weight = 1.0
wb_anchor_weight = 0.5
lambda_wb = 0.005, 0.01
```

确认不伤任务后，再切：

```text
wb_loss_type = dual_potential
```

### 10.2 降低 `dual_potential` 开销

若使用 `dual_potential`：

- `wb_spots_per_graph=64`：每张 graph/slide 最多贡献 64 个 spot，优先避免大切片主导；
- `wb_spots_per_cancer=0`：默认关闭 cancer-level cap；若某些 batch 内癌种 spot 数仍明显不平衡，可设为 128/256；
- `wb_support_size=128`，表示从 generated support `b` 中抽样的 support-side 点数；
- `wb_pot_every_n_steps=2`
- `wb_regularizer=l2` 可比 entropy 更稳一些，但要网格验证。
- 对 `h` 和 `b` 做 `F.layer_norm(..., (D,))` 后再算 cost。

抽样顺序建议固定为：

```text
1. graph-balanced sampling:
   每张 graph/slide 最多抽 wb_spots_per_graph 个 spot；
2. optional cancer cap:
   若 wb_spots_per_cancer > 0 且某 cancer 候选 spot 数超过 cap，
   在该 cancer 内按 graph-balanced 方式再抽到 cap；
3. euclidean_pairwise:
   直接使用上述 selected h/b 全部点计算 anchor、pairwise Euclidean loss 和 potential；
4. dual_potential:
   source-side 使用 selected h；
   support-side 再从 selected b 中抽最多 wb_support_size 个点作为 y，
   以控制 source-support pairwise cost。
```

例如 `batch_size_graphs=8`、`wb_spots_per_graph=64` 时，WB 最多先得到 `8*64=512` 个 selected spot。`euclidean_pairwise` 可以直接在这 512 个 `b` 上计算 cancer-specific support 与 mixed support 的两两欧氏距离；`dual_potential` 需要计算 source-side `h` 与 support-side `b` 的 regularized dual cost，因此再从这 512 个 `b` 中抽 `wb_support_size=128` 个作为 support-side `y`。

## 11. 保护下游肿瘤分类性能的策略

1. WB warm-up：前 10-20 epoch 不启用。
2. WB ramp-up：20-40 epoch 线性增加。
3. 弱 cancer GRL：若 generated-support WB 已强，`lambda_cancer` 建议 0.05-0.1。
4. label-balanced WB sampling 与 state direction 作为可选增强，默认关闭；开启时只在 `y>=0` 的 spot 上使用，避免 label ratio 被对齐损失抹掉。
5. checkpoint 新增 `best_metric`，默认 `val_macro_f1`，WB 实验推荐 `val_auprc` 或 `val_macro_f1`，不用 WB loss 早停。
6. 对 `h` 做 post-hoc cancer probe 和 tumor probe，确认 cancer 信息下降但 tumor 信息保留。

## 12. 实验设计与验收指标

建议实验组：

| 组别 | 配置 | 目的 |
| --- | --- | --- |
| A. 当前无 MMD | `use_mmd=0`, `use_wb_align=0` | 主任务 baseline |
| B. 当前 MMD | `use_mmd=1`, `mmd_on=cancer` | 现有对齐 baseline |
| C. Euclidean pairwise + generated support | `use_wb_align=1`, `wb_loss_type=euclidean_pairwise`, `wb_support_mode=generated_support` | 欧氏两两距离替换 MMD |
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

- `euclidean_pairwise` 至少不弱于当前 MMD 的任务性能，并能降低 cancer probe。
- `dual_potential` 若计算可接受，应比 `euclidean_pairwise` 有更强 cancer mixing。
- 若 cancer mixing 提升但 AUPRC 下降超过 3 个百分点，判定为过对齐。

## 13. 风险与失败模式

### 13.1 Euclidean surrogate 被误读成严格 Wasserstein

`euclidean_pairwise` 是 WB-inspired surrogate，不是严格 Wasserstein distance。报告时要明确命名。

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
3. 实现 `euclidean_pairwise + generated_support`：
   - 单 potential head；
   - anchor + generated-support pairwise Euclidean / energy-distance 形状约束；
   - BaryIR-style potential alternating update。
4. 在 `train.py` 接入：
   - `return_h = use_mmd or use_wb_align`；
   - 每 batch 先 update potential，再 update STOnco + support map。
5. 同步接入可选 `label-balanced sampling` 和 `state direction`，默认关闭；不实现 residual alignment。
6. 跑 A/B/C 对照。

第二阶段：正规 dual WB。

1. 实现 `dual_potential`：
   - 每 cancer 一对 `f_k/g_k`；
   - `h` 作为 source-side samples；
   - `b=T_phi(h)` 作为 generated support-side samples；
   - entropy/l2 regularized dual objective；
   - `g_k - weighted_mean(g)` 中心化。
2. 跑 D/E 对照。

第三阶段：强形状统一增强。

1. 调参启用 label-balanced sampling。
2. 调参启用 state direction。
3. 再实现 state-removed residual alignment。
4. 用 LOCO、cancer probe、LISI 和 tumor AUPRC 共同验收。

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

## 16. 代码实现前已确认的细节

本节是代码实现前的确认清单。下面每一项都来自当前 STOnco 实际代码与本文档方案之间的对照，目的是避免后续实现偏离最初目标：让 `h` 本身成为连续、可学习、泛癌共享的 Wasserstein barycenter 表征空间，同时尽量消除 cancer-source 信息并保护 tumor/normal 分类性能。

状态：16.1-16.24 已由用户确认。后续代码实现按本节确认结果执行；16.25 汇总为第一版实现边界。

### 16.1 确认 generated support 是训练阶段 support，而不是推理 embedding （已确认）

当前文档主方案是：

```text
GNNBackbone -> h -> ClassifierHead
              |
              +-> GeneratedSupportMap T_phi -> b
                                      |
                                      +-> WB potentials during training
```

已确认内容：

- `h` 本身就是 continuous learnable Wasserstein barycenter space。
- 不新增独立 `barycenter atoms`。
- 新增 `GeneratedSupportMap`，但它默认只在训练阶段用于生成 support `b=T_phi(h)`。
- 推理阶段仍默认使用 `GNNBackbone -> h -> ClassifierHead`，不把分类 head 改成依赖 `b`。

这与 finite-atom barycenter 不同。这里的 barycenter support 由连续可学习映射 `T_phi` 从 `h` 生成，不是一组离散 atoms，也不是固定高斯/均匀 prior。

确认结果：

```text
确认 GeneratedSupportMap 只作为训练阶段 support generator；
推理与下游分类仍以 h 为准。
```

### 16.2 `h` 的实际维度是用户希望对齐的空间 （已确认）

当前 `STOnco_Classifier.forward(..., return_h=True)` 返回的是 GNN backbone 输出：

```python
h = self.gnn(x, edge_index, edge_weight=edge_weight)
```

默认 `gatv2` 下，`GNN_hidden=[256,128,64]` 且 `heads=4` 时，最后一层 `h` 维度通常是：

```text
D = 64 * 4 = 256
```

它不是 classifier head 内部的 `z_clf`，也不是 `clf_hidden` 最后一层的 64 维表征。

确认结果：

- WB 对齐作用在当前 GNN 输出 `h`。
- 不改成 classifier latent `z_clf`。
- 不新增一个 64 维压缩空间来降低计算量。

确认结果：

```text
对齐 GNN 输出 h；不使用 z_clf。
```


### 16.3 generated support 是替代 pooled support （已确认）

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

确认结果：

- 第一版只实现 `generated_support`。
- pooled support / memory bank 不保留。

确认结果：

```text
第一版只实现 generated_support。
不实现 batch_mixed_h / memory_bank support。
```

### 16.4 不使用外部 barycenter sampler （已确认）

本方案不显式采样独立 barycenter 点，不调用类似 `sample_barycenter()` 的生成过程。

但 `dual_potential` 需要从 support measure 中取 `y`：

```text
x ~ P_k^h
y ~ Q_phi^b
```

实现上这个 `y` 是从当前 batch 生成的 support set `{b_i=T_phi(h_i)}` 中 subsample，不是从 pooled mean、固定分布或 finite atoms 中采样。

确认如下：

```text
确认不使用 finite atoms、固定 prior 或 z->generator 的外部 barycenter sampler；  support-side y 只从当前 batch 生成的 b=T_phi(h) 中 subsample。
```

### 16.5 potential update 时如何处理 STOnco 的 train/eval mode （已确认）

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

确认如下：

```text
采用方案 B：potential update 只调用 model.gnn(...) 取得 h_detached，
再由 frozen support_map 生成 b_detached，避免 classifier BN 被无意义更新。
```

### 16.6 WB 与当前默认 cancer GRL 的并用策略（已确认）

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

已确认内容：

- 启用 WB 时，第一版默认弱保留 cancer GRL，而不是默认关闭。
- 或者保留 cancer GRL，但把 `lambda_cancer` 降到 0.05-0.1。
- slide/batch GRL 保持默认开启。

确认如下：

```text
WB 实验第一版：use_domain_adv_cancer=1，但 lambda_cancer=0.05 或 0.1。
如果任务性能下降，再尝试关闭 cancer GRL。
```

实现约束：

```text
即使 use_domain_adv_cancer=0，只要 use_wb_align=1，prepare_graphs() 仍必须返回 n_domains_cancer=len(cancer_to_idx)，以便建立 WB potential bank。
```

### 16.7 WB 与 MMD 的互斥策略（已确认）

当前 MMD 已经在 `_compute_losses(...)` 中直接加到 `total`。如果后续实现只是新增 WB，而不做互斥控制，则用户可能同时打开：

```text
use_mmd=1
use_wb_align=1
```

这会变成：

```text
task + domain + MMD + WB
```

而用户最初目标是“替换或升级当前 MMD”。因此第一版实现默认采用互斥策略。

确认结果：

```text
WB 主实验默认 use_mmd=0。
允许高级用户手动同时开启，但 CLI 打印 warning。
```

### 16.8 `mmd_on` 当前默认是 slide，不是 cancer （已确认）

当前代码默认：

```python
"mmd_on": "slide"
```

但本方案关注多癌种 latent 边缘分布对齐，实验 baseline 应该使用：

```text
use_mmd=1, mmd_on=cancer
```

已确认内容：

```text
文档中的 MMD baseline 固定使用 mmd_on=cancer，而不是沿用代码默认 mmd_on=slide。
```

确认结果：

```text
MMD baseline 实验显式设置 mmd_on=cancer。
不改全局 mmd_on 默认值，避免改变普通 MMD / 非 WB 实验的历史行为。
```

### 16.9 sampler 默认值与 WB 需求不一致 （已确认）

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

已确认内容：

- 启用 `use_wb_align=1` 时，代码自动检查 sampler 设置。
- 如果用户仍用 random sampler，则跳过 active cancer 不足的 batch。
- 自动 warning，而不是强制报错。

确认如下：

```text
启用 WB 时不强制改 sampler，但若 avg active cancers < wb_min_cancers，打印 warning，并在对应 batch 跳过 WB loss。
WB 实验推荐显式使用下面四个参数；不改非 WB baseline 的全局默认值：
--batch_size_graphs 8
--sampler_mode cancer_balanced
--sampler_k_cancers 4
--sampler_m_per_cancer 2

```

### 16.10 graph-level balance 不等于 spot-level balance （已确认）

当前 `CancerBalancedBatchSampler` 平衡的是 graph/slide 数，而 WB loss 作用在 spot-level `h` 和 generated support `b` 上。不同切片 spot 数可能差异很大，导致 generated support set 被大图或大癌种主导。

确认的抽样原则：

- WB 内部优先按 graph/slide 限制 spot 数，抽样时尽量照顾到当前 batch 内全部切片。
- `wb_spots_per_cancer` 作为可选二级 cap；默认关闭。
- `h` 与 `b=T_phi(h)` 必须使用同一组 selected indices，不能分别抽样。

建议默认设置：

```text
wb_spots_per_graph = 64
wb_spots_per_cancer = 0
wb_support_size = 128
```

抽样流程：

```text
Step 1. graph-balanced sampling
    对每张 graph/slide：
        从该 graph 内最多抽 wb_spots_per_graph 个 spot。

Step 2. optional cancer cap
    如果 wb_spots_per_cancer > 0：
        对每个 cancer 检查候选 spot 数；
        若超过 wb_spots_per_cancer，
        在该 cancer 内按 graph-balanced 方式再抽到 cap。

Step 3. build selected tensors
    h_sel = h[selected_idx]
    b_sel = b[selected_idx]
    cancer_sel = cancer_dom[selected_idx]

Step 4. loss-specific use
    euclidean_pairwise:
        直接使用全部 h_sel / b_sel 计算 anchor、pairwise Euclidean loss 和 potential。

    dual_potential:
        source-side x 使用 h_sel[cancer_sel == k]；
        support-side y 从 b_sel 中再抽最多 wb_support_size 个点。
```

如果 `batch_size_graphs=8` 且 `wb_spots_per_graph=64`，graph-level sampling 后最多得到 512 个 selected spot。`euclidean_pairwise` 可以直接在这 512 个 `b` 上计算 cancer-specific support 与 mixed support 的两两欧氏距离；`dual_potential` 需要计算 source-support regularized dual cost，所以再从这 512 个 `b` 里抽 `wb_support_size=128` 个作为 support-side `y`，避免显存和时间随 support 数过快增长。

### 16.11 WB 使用 unlabeled spot 的策略 （已确认）

当前 task loss 会跳过：

```python
mask = batch.y >= 0
```

但 WB 对齐如果直接使用所有 `h`，会包含 `y < 0` 的 spot。这样更接近 source distribution alignment，但无法进行 label-balanced shape 约束。

已确认内容：

```text
WB base alignment 使用全部 spot，不只使用 y>=0 的 labeled spot。
```

确认结果：

```text
base WB alignment 使用全部 spot。
label-balanced 和 state-shape 约束只在 y>=0 的 spot 上启用。
```

### 16.12 `euclidean_pairwise` 是唯一保留的欧氏替代项 （已确认）

当前文档不再保留 mean/variance 矩匹配版本。欧氏替代 Wasserstein 的轻量项只保留：

```text
euclidean_pairwise + generated_support
```

其组成是：

```text
anchor(h, b)
+ pairwise Euclidean / energy-distance distribution loss on b
+ potential critic Pot_k(b)
```

它不是严格 Wasserstein distance，也不是 dual-potential barycenter；它是用 generated support 上的两两欧氏距离矩阵直接比较 `B_k` 与 mixed `B_Q` 的分布差异。

确认如下：

```text
只实现 euclidean_pairwise + generated_support。
不实现 mean/variance 矩匹配版本。
```

### 16.13 `dual_potential` 的符号和 min-max 更新需要固定 （已确认）

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

已确认内容：

```text
采用这套符号约定，并在实现后用一个 toy batch sanity check 检查 loss 方向。
```

确认如下：

```text
代码中记录 avg_wb_dual_obj、avg_wb_potential_loss、avg_wb_loss，避免符号错误隐藏。
```

### 16.14 `dual_potential` 的 regularizer 默认选 entropy 还是 l2 （已确认）

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

已确认第一版默认值。

确认如下：

```text
第一版 dual_potential 默认 wb_regularizer=l2, wb_epsilon=0.1。
entropy 作为网格实验选项。
```

如果用户希望更贴近 CWB 默认设置，则把默认改回 entropy。

### 16.15 `h` 和 `b` 在 WB loss 内部归一化（已确认）

当前 GNN backbone 已经使用 `LayerNorm`，但 `h` 和 generated support `b` 的尺度仍会随模型训练改变。`dual_potential` 的 cost：

```text
c(x,y)=||x-y||^2/D
```

对尺度敏感。

已确认内容：

```text
WB loss 内使用 layer_norm(h) 与 layer_norm(b)。
```

确认如下：

```text
WB loss 内使用 F.layer_norm(h, (D,)) 与 F.layer_norm(b, (D,)) 计算 cost 和 potential 输入；
ClassifierHead 仍使用 raw h。
```

这仍会让 WB 梯度回传到 `h`，但对齐的是归一化后的几何。

### 16.16 state-shape 约束第一版做到哪一步 （已确认）

文档提出：

```text
label-balanced sampling
state direction
state-removed residual alignment
```

这些约束用于实现“tumor 和 normal spot 是同一连续分布中的两种状态”。但当前 STOnco 代码里还没有这些组件。

已确认第一版范围：

```text
方案 B：第一版同时实现 state direction。
```

确认：

```text
第一版实现 label-balanced sampling 和 state direction 两个可选项，默认均关闭。
residual alignment 则作为第二阶段。
```

原因是 generated-support WB 与 cancer GRL 已经会强约束 `h`，一次性加入 residual 约束会让失败原因难以定位。

### 16.17 validation 阶段计算 WB loss 的可选策略 （已确认）

当前文档建议第一版验证阶段不计算 WB loss，避免额外开销。当前 STOnco 验证阶段会在 `use_mmd=1` 时计算 MMD，因此如果 WB 也在验证阶段计算，会增加明显开销。

已确认内容：

```text
validation 添加一个可选项计算保存 WB loss 。
```

确认：

```text
validation 添加一个可选项计算保存 WB loss 。

```

### 16.18 best checkpoint 问题 （已确认）

当前 `train_and_validate(...)` 里最佳模型选择是：

```python
improved = val_accuracy > best["accuracy"]
```

文档保护下游任务时提到可按 `val_auprc` 或 `val_macro_f1`。这与当前代码不一致。

已确认内容：

- 在实现 WB 时顺手增加 `--best_metric`。

确认如下：

```text
增加 best_metric 参数，默认改为 val_macro_f1。
WB 实验推荐 best_metric=val_auprc 或 val_macro_f1。
```

否则“保护下游肿瘤分类性能”的验收与 checkpoint 选择可能不一致。

### 16.19 loss_components 和曲线图同步扩展（已确认）

当前 `loss_components.csv` 和训练曲线固定记录 MMD、domain、task loss。WB 实现后，`hist` 与 `loss_components.csv` 需要新增：

```text
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_pairwise
avg_wb_anchor
avg_wb_active_cancers
avg_wb_active_spots
wb_lambda
```

确认：

```text
开启 use_wb_align=1 时，除继续写入 loss_components.csv 外，
额外保存一张 WB 专用 SVG，不重排现有 train_loss.svg / train_val_loss.svg。
```

新增 SVG 文件建议命名：

```text
wb_train_loss.svg
```

内容建议包含：

```text
avg_total_loss
avg_task_loss
avg_wb_loss
avg_wb_potential_loss
avg_wb_dual_obj
avg_wb_euclid_pairwise
avg_wb_anchor
wb_lambda
avg_wb_active_cancers
avg_wb_active_spots
```

绘图策略：

```text
1. 保持原有 train_loss.svg、train_val_loss.svg、train_val_metrics.svg 布局不变。
2. 仅在 use_wb_align=1 时额外生成 wb_train_loss.svg。
3. 绘图方式与现有其他损失图一致，复用当前 `_plot_train_metrics(...)` 的线型、颜色、平滑窗口、标题和坐标轴风格。
4. 子图布局和图像比例需要单独设置，避免 10 个指标挤在现有 3x3 布局中。建议使用 4x3 或 5x2 子图布局，figsize 按子图数量合理放大，保证标题、图例和坐标刻度不重叠。
5. 不适用于当前 wb_loss_type 的曲线用 NaN 或跳过绘制：
   - dual_potential 主要看 avg_wb_dual_obj；
   - euclidean_pairwise 主要看 avg_wb_euclid_pairwise；
   - 两者都看 avg_wb_loss、avg_wb_anchor、avg_wb_potential_loss、wb_lambda。
6. validation 第一版默认不计算 WB loss，因此该 SVG 只画训练端 WB diagnostics。
```

### 16.20 support map 和 potential 参数保存策略（已确认）

GeneratedSupportMap 和 potential module 第一版只在训练时使用，推理不需要。但如果要完整复现实验或中断恢复，support map、potential state 与 optimizer state 也有价值。

当前 STOnco 保存 best state 时主要保存：

```text
model.state_dict()
```

已确认内容：

- 最终 `model.pt` 只保存 STOnco 主模型。
- 额外保存 `wb_support_map.pt` 和 `wb_potentials.pt` 用于复现或恢复训练。

确认如下：

```text
推理模型保存 STOnco 主模型。
训练 artifacts 额外保存 wb_support_map_last.pt、wb_potentials_last.pt 和 wb_config.json。
```

### 16.21 WB 的 cancer 权重 `ω_k` 如何定义（已确认）

当前 domain CE 使用 graph-frequency 权重。WB barycenter 目标中的 `ω_k` 可以有两种：

```text
uniform over active cancers: 每个癌种同权。
frequency weighted: 按 batch 或训练集频率加权。
```

用户目标是消除 cancer-source 信息，因此频率加权可能让大癌种主导 barycenter。

已确认内容：

```text
WB 中 ω_k 统一使用 active-cancer uniform weights。
```

确认结果：

```text
WB 使用 uniform over active cancers。
```

### 16.22 LOCO / k-fold 下 potential bank 的 cancer 数量（已确认）

当前 `train_and_validate(...)` 接收 `n_domains_cancer`，通常按当前训练/验证构建好的 cancer label space 传入。LOCO 模式下某个癌种作为验证癌种，训练 batch 中不会出现该癌种。

已确认内容：

- potential bank 按全局 `n_domains_cancer` 建立，未出现癌种的 potentials 不更新。
- 不单独按训练集 active cancers 建立新的映射。
- `n_domains_cancer` 的返回条件改为 `use_domain_adv_cancer or use_wb_align`，不再只依赖 cancer GRL 是否开启。

确认结果：

```text
按 n_domains_cancer 建立 potential bank，batch 内只更新 active cancers。
```

这样与当前 `batch.cancer_dom[batch.batch]` 编码最兼容。

### 16.23 post-hoc probe / LISI 不纳入第一版训练代码 （已确认）

文档验收指标包含：

```text
post-hoc cancer probe
iLISI / cLISI
LOCO average AUPRC / macro-F1
```

当前训练代码已有 domain head 训练准确率，但这不是严格 post-hoc probe。LISI 也不是当前训练流程的一部分。

已确认内容：

```text
第一版实现只提供 WB 训练机制，probe/LISI 作为后续分析脚本。
```

确认结果：

```text
第一版不把 probe/LISI 接进 train.py。
先保存 h embedding 或复用现有 embedding 分析脚本做 post-hoc 验收。
```

### 16.24 实现文件边界接受新增 `wb_potentials.py` （已确认）

用户希望 loss 直接写进训练函数中，以保持简洁。文档当前建议新增：

```text
stonco/core/wb_potentials.py
```

折中实现可以是：

```text
wb_potentials.py: 放 GeneratedSupportMap、potential modules 和纯 loss helper。
train.py: 保留 alternating update 控制流、lambda schedule、metrics 汇总。
```

已确认内容：

```text
接受新增 wb_potentials.py，而不是把 GeneratedSupportMap、potentials 和 loss helper 全部塞进 train.py。
```

确认结果：

```text
接受新增 wb_potentials.py；训练流程仍显式写在 train.py 中。
```

### 16.25 建议的第一版实现边界

根据 16.1-16.24 已确认内容，第一版代码实现边界是：

```text
实现 use_wb_align。
实现 wb_loss_type={euclidean_pairwise,dual_potential}。
实现 wb_support_mode=generated_support。
新增 GeneratedSupportMap，但不改 models.py。
训练阶段 return_h = use_mmd or use_wb_align。
prepare_graphs 在 use_domain_adv_cancer or use_wb_align 时返回 n_domains_cancer。
potential update 使用 model.gnn(...) 取 h_detached，edge_weight 用 getattr(batch, "edge_weight", None)，再生成 b_detached。
main update 使用 model(..., return_h=True)，再生成 b。
model_loss 返回 raw WB alignment，不包含 anchor。
anchor_loss 在训练循环中通过 wb_anchor_weight 单独加一次。
base WB 使用全部 spot。
WB 内部 graph-level spot sampling 默认 wb_spots_per_graph=64。
wb_spots_per_cancer 默认 0，作为可选二级 cap。
euclidean_pairwise 使用 selected h/b 全部点。
dual_potential 从 selected b 中再抽 wb_support_size 个 support 点。
label-balanced sampling 与 state direction 作为可选项，默认关闭。
residual alignment 放到第二阶段。
WB 默认 use_mmd=0；如果同时开启 MMD，打印 warning。
MMD baseline 显式使用 mmd_on=cancer，但不改全局 mmd_on 默认值。
use_domain_adv_cancer=1，但 lambda_cancer 默认弱化为 0.05 或 0.1。
新增 best_metric，默认 val_macro_f1。
validation 增加可选项计算保存 WB loss。
use_wb_align=1 时额外保存 wb_train_loss.svg。
额外保存 wb_support_map_last.pt、wb_potentials_last.pt、wb_config.json。
第一版不实现 pooled support、memory bank、residual alignment、post-hoc probe/LISI 集成。
```

用户已确认的最小实现约束：

1. `h` 仍是最终分类表征，`b=T_phi(h)` 只作为训练阶段 generated support。
2. 第一版只实现 `generated_support`，不实现 pooled support / memory bank。
3. `GeneratedSupportMap` 采用 identity-initialized residual MLP：`b=h+MLP(LayerNorm(h))`。
4. potential update 直接调用 `model.gnn(...)` 取 `h`，`edge_weight` 使用 `getattr(batch, "edge_weight", None)`，再用 frozen `support_map` 生成 `b`，以避免 classifier BatchNorm 状态被更新。
5. 启用 WB 时弱保留 cancer GRL，`lambda_cancer=0.05` 或 `0.1`；如果 task performance 明显下降，再尝试关闭 cancer GRL。
6. WB 和 MMD 默认互斥；允许高级用户同时开启，但必须打印 warning。
7. `prepare_graphs()` 在 `use_domain_adv_cancer or use_wb_align` 时返回 `n_domains_cancer=len(cancer_to_idx)`，确保关闭 cancer GRL 但开启 WB 时仍能建立 potential bank。
8. 第一版实现 `euclidean_pairwise`：generated-support pairwise Euclidean / energy-distance + potential critic；anchor 在训练循环外单独加一次，不包含在 `model_loss()` 内。
9. 第一版实现 label-balanced sampling 和 state direction 两个可选项，默认关闭；residual alignment 放到第二阶段。
10. 新增 `best_metric`，默认 `val_macro_f1`，WB 实验推荐 `val_auprc` 或 `val_macro_f1`。
11. 接受新增 `stonco/core/wb_potentials.py`，但 alternating update 保留在 `train.py`。
