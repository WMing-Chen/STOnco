# 在 STOnco 中加入 MMD（Maximum Mean Discrepancy）方案

本文档给出一个可落地的 MMD 接入方案，目标是在你当前的 STOnco 训练流程中，显式对齐不同域（batch/slide、cancer type）的特征分布，提升跨域泛化能力。

---

## 0. 现状与改造目标

### 现状（基于现有代码）
- 训练入口：`stonco/core/train.py`，主训练函数 `train_and_validate(...)`。
- 当前总损失：
  - 任务损失：`BCEWithLogitsLoss`（spot 二分类）
  - 域对抗损失：`loss_batch` + `loss_cancer`（通过 GRL + domain head）
- 当前模型输出：`stonco/core/models.py` 中 `STOnco_Classifier.forward(...)` 返回：
  - `out['logits']`
  - `out['dom_logits_slide']`
  - `out['dom_logits_cancer']`
  - 若 `return_z=True`，还会返回 `out['z64']`（分类头中的中间表示），但目前训练中默认未开启。

### 改造目标
在保持现有任务损失 + 域对抗损失不变的前提下，新增 MMD 对齐项：

$$
\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_{slide}\mathcal{L}_{adv,slide}+\lambda_{cancer}\mathcal{L}_{adv,cancer}+\lambda_{mmd}\mathcal{L}_{mmd}
$$

其中各项含义在当前 STOnco 里可写成：

$$
\mathcal{L}_{task}
=
\frac{1}{|\Omega|}
\sum_{i \in \Omega}
\operatorname{BCEWithLogits}\bigl(\hat y_i, y_i\bigr)
$$

这里：
- $i$ 表示一个 spot/node
- $\Omega=\{i \mid y_i \ge 0\}$ 表示有监督标签的 spot 集合
- $\hat y_i$ 是模型输出 `out['logits'][i]`

作用：
- $\mathcal{L}_{task}$ 是主任务损失，直接约束每个 spot 的肿瘤/非肿瘤预测结果。
- 它保证模型首先学会分类，而不是只做域对齐。
- 在总损失里，这一项决定了任务性能的基本上限，是最核心的监督信号。

域对抗部分在现有实现里分别是：

$$
\mathcal{L}_{adv,slide}
=
\frac{1}{N}
\sum_{i=1}^{N}
\operatorname{CE}\bigl(\hat d^{(slide)}_i, d^{(slide)}_i\bigr)
$$

$$
\mathcal{L}_{adv,cancer}
=
\frac{1}{N}
\sum_{i=1}^{N}
\operatorname{CE}\bigl(\hat d^{(cancer)}_i, d^{(cancer)}_i\bigr)
$$

其中：
- $N$ 是当前 mini-batch 中的 spot 数
- $\hat d^{(slide)}_i$ 对应 `out['dom_logits_slide'][i]`
- $\hat d^{(cancer)}_i$ 对应 `out['dom_logits_cancer'][i]`
- $d^{(slide)}_i$ 由 `batch.bat_dom[batch.batch]` 展开得到
- $d^{(cancer)}_i$ 由 `batch.cancer_dom[batch.batch]` 展开得到

作用：
- $\mathcal{L}_{adv,slide}$ 用来区分每个 spot 来自哪个 slide/batch 域。
- $\mathcal{L}_{adv,cancer}$ 用来区分每个 spot 来自哪个 cancer 域。
- 这两项在数值上是普通的交叉熵，但由于前面接了 GRL，优化时会推动 encoder 学到“更难被域头区分”的表示。
- 因此它们的作用不是提升域分类精度，而是通过对抗训练让特征更域不变。

总损失中各权重项的作用如下：
- $\lambda_{slide}$：控制 slide/batch 域对抗项的强度。越大，越强调去除 batch effect。
- $\lambda_{cancer}$：控制 cancer 域对抗项的强度。越大，越强调跨癌种对齐。
- $\lambda_{mmd}$：控制 MMD 分布对齐项的强度。越大，越强迫不同域在特征空间内靠近。
- 这三个系数本质上是在“任务判别能力”和“域不变性”之间做权衡。

---

### MMD 模块的数学定义

记 GNN 编码器输出的节点表示为：

$$
H=\{h_i\}_{i=1}^{N}, \qquad h_i \in \mathbb{R}^{D}
$$

其中 $h_i$ 对应代码里的 `out['h'][i]`。

符号说明：
- $H$：当前一个 mini-batch 内所有 spot 的共享表示集合。
- $h_i$：第 $i$ 个 spot 经过 GNN backbone 后得到的特征向量。
- $D$：特征维度，即每个 $h_i$ 的长度。

作用：
- $h_i$ 是 MMD 真正作用的对象。
- 对 $h_i$ 做分布对齐，等价于直接约束不同域在共享表征空间中的几何分布。
- 这样梯度会回传到 GNN encoder，使 backbone 学到更稳定的域无关特征。

对任意两个域 $a,b$，设：

$$
X_a = \{h_i \mid d_i = a\}, \qquad X_b = \{h_j \mid d_j = b\}
$$

且 $|X_a|=n_a$，$|X_b|=n_b$。

符号说明：
- $d_i$：第 $i$ 个 spot 的域标签，可以是 slide 域，也可以是 cancer 域。
- $X_a$：域 $a$ 中所有 spot 的特征集合。
- $X_b$：域 $b$ 中所有 spot 的特征集合。
- $n_a$、$n_b$：对应两个域中的样本数。

作用：
- MMD 的基本单位不是单个样本，而是两个域对应的“样本分布”。
- 把特征按域分成 $X_a$ 和 $X_b$ 后，才能计算两个域之间的分布差异。

#### 1. RBF 核

单个带宽 $\sigma$ 下的 RBF 核定义为：

$$
k_{\sigma}(x,y)=\exp\left(-\frac{\|x-y\|_2^2}{2\sigma^2}\right)
$$

多带宽核则是多个 RBF 核的和：

$$
k(x,y)=\sum_{m=1}^{M} k_{\sigma_m}(x,y)
$$

这里：
- $M$ 对应 `mmd_num_kernels`
- $\sigma_m$ 由 `mmd_kernel_mul` 和中心带宽共同生成
- 若 `mmd_sigma is None`，则中心带宽由当前两域样本间距离自适应估计

作用：
- $k_\sigma(x,y)$ 用来衡量两个特征向量 $x,y$ 的相似度。
- $\sigma$ 越小，核越关注局部邻近关系；$\sigma$ 越大，核越关注更粗粒度的整体分布。
- 多带宽求和的作用，是同时在“局部尺度”和“全局尺度”上比较两个域的分布差异。
- 这样比只用单一带宽更稳，不容易因为某个 $\sigma$ 选得不合适而导致 MMD 失真。

#### 2. 两域之间的 unbiased MMD\(^2\)

对于域 $a,b$，采用无偏估计的 MMD\(^2\)：

$$
\operatorname{MMD}^2(X_a, X_b)
=
\frac{1}{n_a(n_a-1)}
\sum_{\substack{u,v=1 \\ u \ne v}}^{n_a}
k(x_u, x_v)
+
\frac{1}{n_b(n_b-1)}
\sum_{\substack{u,v=1 \\ u \ne v}}^{n_b}
k(y_u, y_v)
- 2
\frac{1}{n_a n_b}
\sum_{u=1}^{n_a}
\sum_{v=1}^{n_b}
k(x_u, y_v)
$$

其中：
- $x_u \in X_a$
- $y_v \in X_b$
- 第一、二项去掉了对角线项，所以是 unbiased estimator

逐项解释：
- 第一项

$$
\frac{1}{n_a(n_a-1)}
\sum_{\substack{u,v=1 \\ u \ne v}}^{n_a}
k(x_u, x_v)
$$

表示域 $a$ 内部样本两两之间的平均相似度，反映域 $a$ 自身分布的“凝聚程度”。

- 第二项

$$
\frac{1}{n_b(n_b-1)}
\sum_{\substack{u,v=1 \\ u \ne v}}^{n_b}
k(y_u, y_v)
$$

表示域 $b$ 内部样本两两之间的平均相似度，反映域 $b$ 自身分布的“凝聚程度”。

- 第三项

$$
\frac{1}{n_a n_b}
\sum_{u=1}^{n_a}
\sum_{v=1}^{n_b}
k(x_u, y_v)
$$

表示域 $a$ 和域 $b$ 之间跨域样本的平均相似度。

整体作用：
- 如果两个域分布相近，那么“域内相似度”和“跨域相似度”会接近，MMD\(^2\) 会变小。
- 如果两个域分布差异大，那么跨域相似度会偏低，MMD\(^2\) 会变大。
- 训练时最小化 MMD\(^2\)，本质上就是在推动不同域的特征分布彼此靠近。

文档后面代码里的 `_mmd2_unbiased(x, y, sigma_list)` 实现的就是这一式。

#### 3. 多域 pairwise 聚合

若当前 batch 中有效域集合为：

$$
\mathcal{D}=\{d \mid |\{i : d_i=d\}| \ge 2\}
$$

则域对集合为：

$$
\mathcal{P}=\{(a,b)\mid a,b \in \mathcal{D},\ a<b\}
$$

若设置了 `mmd_max_pairs=K`，则只取前 $K$ 个域对；否则使用全部域对。

符号说明：
- $\mathcal{D}$：当前 batch 里参与 MMD 计算的有效域集合。
- “有效域”的定义是该域至少有 2 个 spot，因为 unbiased MMD 需要去掉对角项。
- $\mathcal{P}$：所有可计算的域对集合。
- $|\mathcal{P}|$：域对数量。

最终 batch 内的多域 MMD 定义为：

$$
\mathcal{L}_{mmd}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\operatorname{MMD}^2(X_a, X_b)
$$

若没有有效域对，即 $|\mathcal{P}|=0$，则定义：

$$
\mathcal{L}_{mmd}=0
$$

#### 4. `slide / cancer / both` 三种模式

当 `mmd_on='slide'` 时：

$$
\mathcal{L}_{mmd}=\mathcal{L}_{mmd}^{(slide)}
$$

其中域标签 $d_i$ 取自 `batch.bat_dom[batch.batch]`。

作用：
- 对齐不同切片/批次的特征分布。
- 主要解决 batch effect、样本制备差异、测序批次差异带来的偏移。

当 `mmd_on='cancer'` 时：

$$
\mathcal{L}_{mmd}=\mathcal{L}_{mmd}^{(cancer)}
$$

其中域标签 $d_i$ 取自 `batch.cancer_dom[batch.batch]`。

作用：
- 对齐不同癌种的特征分布。
- 主要提升跨癌种泛化能力，减少模型只记住某一癌种特异模式的风险。

当 `mmd_on='both'` 时，按当前建议实现为两路直接相加：

$$
\mathcal{L}_{mmd}
=
\mathcal{L}_{mmd}^{(slide)}
+
\mathcal{L}_{mmd}^{(cancer)}
$$

因此实际进入总损失的 MMD 项是：

$$
\lambda_{mmd}\mathcal{L}_{mmd}
$$

作用：
- `slide` 模式强调去 batch effect。
- `cancer` 模式强调跨癌种对齐。
- `both` 模式同时约束两类域偏移，但约束更强，也更容易出现过对齐。
- $\lambda_{mmd}\mathcal{L}_{mmd}$ 是最终真正参与反向传播的 MMD 项；决定了 MMD 对 encoder 更新的影响强弱。

如果 `both` 模式下你后续觉得约束过强，也可以改成平均：

$$
\mathcal{L}_{mmd}
=
\frac{1}{2}
\left(
\mathcal{L}_{mmd}^{(slide)}
+
\mathcal{L}_{mmd}^{(cancer)}
\right)
$$

但首版实现建议保持“直接相加”，这样与后文代码和日志字段更一致。

---

## 1. 方案选择（先给结论）

### 推荐默认方案
- 对齐层位：**GNN backbone 输出 `h`（节点级）**
- 对齐域标签：优先用 `bat_dom`（batch/slide），可选 `cancer_dom`
- MMD 核：**RBF 多带宽核（multi-kernel RBF）**
- 估计方式：**mini-batch 内按域两两计算并平均**
- 和域对抗并用：**是**（先小权重，如 `lambda_mmd=0.05`）

### 为什么选 `h` 而不是 `z64`
- `h` 是 GNN 编码后的共享特征，位置更“底层”，更适合做域对齐。
- `z64` 更靠近分类头，语义偏任务判别，过强对齐更容易伤分类边界。

---

## 2. 代码改动总览（每一步做什么）

## Step 1: 在配置/CLI 增加 MMD 开关与超参数
目的：让 MMD 可控、可做 ablation。

在 `stonco/core/train.py` 的 `parse_args()` 中新增参数：
- `--use_mmd`：`0/1`，默认 `0`
- `--mmd_on`：`slide|cancer|both`，默认 `slide`
- `--lambda_mmd`：MMD 权重，默认 `0.05`
- `--mmd_num_kernels`：RBF 核个数，默认 `5`
- `--mmd_kernel_mul`：带宽倍率，默认 `2.0`
- `--mmd_sigma`：可选固定带宽（默认 `None` 表示自适应）
- `--mmd_max_pairs`：每个 batch 最多取多少域对做 MMD，默认 `8`（防止域很多时开销爆炸）
- `--mmd_spots_per_slide`：每张切片最多抽取多少个 spot 参与 MMD，默认 `0`（表示不抽样，使用该切片全部 spot）

并在 `build_cfg(args)` 合并到 `cfg`。

---

## Step 2: 在训练文件中新增 MMD 计算函数
目的：提供可复用、数值稳定的 MMD 实现。

建议在 `train.py` 中新增：
1. `_rbf_kernel(x, y, sigma_list)`
2. `_mmd2_unbiased(x, y, sigma_list)`
3. `_multi_domain_mmd(h, dom_nodes, ...)`

核心逻辑：
- 输入 `h`: `[N, D]`（节点特征），`dom_nodes`: `[N]`（每个节点的域 id）
- 若 `mmd_spots_per_slide > 0`，先在 mini-batch 内对每张切片独立抽样，最多保留固定数量的 spot；该抽样仅用于 MMD，不影响 BCE 和 domain head
- 按域分组后，对域两两组合 `(d_i, d_j)` 计算 `MMD^2`，取平均
- 只对样本数 >= 2 的域参与计算
- 无有效域对时返回 `0`

伪代码（简化版）：
```python

def _multi_domain_mmd(
    h,
    dom_nodes,
    graph_nodes=None,
    spots_per_slide=0,
    num_kernels=5,
    kernel_mul=2.0,
    sigma=None,
    max_pairs=8,
):
    if spots_per_slide > 0:
        keep = []
        for g in unique(graph_nodes):
            idx = where(graph_nodes == g)
            idx = random_choice(idx, size=min(spots_per_slide, len(idx)), replace=False)
            keep.append(idx)
        keep = concat(keep)
        h = h[keep]
        dom_nodes = dom_nodes[keep]

    uniq = unique(dom_nodes)
    groups = {d: h[dom_nodes == d] for d in uniq if (dom_nodes == d).sum() >= 2}
    pairs = all_pairs(groups.keys())
    pairs = pairs[:max_pairs] if max_pairs > 0 else pairs

    losses = []
    for a, b in pairs:
        xa, xb = groups[a], groups[b]
        sigma_list = build_sigma_list(xa, xb, num_kernels, kernel_mul, sigma)
        losses.append(_mmd2_unbiased(xa, xb, sigma_list))

    return mean(losses) if losses else torch.tensor(0.0, device=h.device)
```

说明：
- `graph_nodes` 建议直接使用 PyG batch 对象中的 `batch.batch`，它表示每个 node 属于当前 mini-batch 中哪一张图/切片。
- 这里的抽样是“每张切片先限流，再做域级聚合”，因此即使 `mmd_on=cancer`，单个癌种域内的 spot 总数也会被上界控制为 `该域在本 batch 中切片数 x mmd_spots_per_slide`。
- 对 32G 显卡，建议先尝试 `--mmd_spots_per_slide 256` 或 `512`。

---

## Step 3: `models.py` 当前已满足 `return_h`
目的：让损失函数拿到 `h`。

这里要先纠正文档和当前代码的差异：**你现在这份代码库里 `stonco/core/models.py` 已经实现了 `return_h`**，不需要重复修改。

当前 `STOnco_Classifier.forward(...)` 已经是：

```python
def forward(
    self,
    x,
    edge_index,
    batch=None,
    edge_weight=None,
    grl_beta_slide=1.0,
    grl_beta_cancer=1.0,
    return_z=False,
    return_h=False,
):
    h = self.gnn(x, edge_index, edge_weight=edge_weight)
    logits, z64 = self.clf(h, return_z=return_z)
    ...
    if return_h:
        out['h'] = h
    return out
```

因此本方案里真正要改的是：**`train.py` 在训练时把 `return_h=True` 打开，并基于 `out['h']` 计算 MMD。**

---

## Step 4: 在 `train.py` 中接入 MMD
目的：把 MMD 加进总损失并记录日志。

下面不再只写方案，而是直接给出**按当前代码基线可落地的代码段**。

---

## Step 5: 训练命令模板

```bash
python -m stonco.core.train \
  --train_npz /path/to/train.npz \
  --artifacts_dir /path/to/artifacts \
  --model gatv2 \
  --use_image_features 1 \
  --img_use_pca 1 \
  --img_pca_dim 256 \
  --device cuda \
  --use_domain_adv_slide 1 \
  --lambda_slide 0.1 \
  --use_domain_adv_cancer 1 \
  --lambda_cancer 0.1 \
  --use_mmd 1 \
  --mmd_on slide \
  --lambda_mmd 0.05 \
  --mmd_num_kernels 5 \
  --mmd_kernel_mul 2.0
```

---

## 3. 代码核心改动详细解释

### 改动核心 A：`h` 作为 MMD 对齐对象
MMD 对齐的是编码器表征空间。如果对齐的是 `h`，梯度会直接回传到 GNN 编码器，推动不同域在共享表示层收敛；这与 GRL 的目标一致，但优化行为更平滑。

### 改动核心 B：无偏 MMD 估计避免自项偏置
`MMD^2` 若直接用全核均值会有偏差（包含对角项）。使用 unbiased 形式：
- 去除同域核矩阵对角线影响
- 小 batch 时更稳

### 改动核心 C：多域 pairwise 而非只做二域
你的训练经常是多 slide/多癌种并存。二域 MMD 不够覆盖。pairwise 聚合可以同时拉近多域分布。

### 改动核心 D：MMD 与域对抗共存
- 域对抗：通过分类器博弈逼近域不可分
- MMD：直接约束分布距离
- 共用时常见收益：训练更稳，验证波动更小
- 风险：`lambda_mmd` 过大时可能损伤任务判别，需从小到大调参

---

## 4. 建议的实验顺序

1. Baseline（你当前配置，不开 MMD）
2. 仅开 MMD（关 domain adv）
3. Domain adv + MMD（推荐）
4. 比较 `mmd_on=slide` vs `cancer` vs `both`

建议重点看：`external val` 的 AUROC / AUPRC / Macro-F1 是否提升。

---

## 5. 风险与防护

- 风险 1：过对齐导致负迁移
  - 处理：从 `lambda_mmd=0.01~0.05` 起，逐步增大
- 风险 2：域极不平衡时 MMD 被大域主导
  - 处理：限制 pair 数、做域均匀采样
- 风险 3：训练开销增加
  - 处理：`mmd_max_pairs` 限流，必要时降低 `mmd_num_kernels`

---

## 6. 已确认选择（2026-03-06）

以下配置已由用户确认，后续实现按此执行：

1. MMD 对齐域：`slide`
- 作用：优先对齐切片/批次分布，降低 batch effect，提升跨批次稳定性。

2. 与现有 GRL 域对抗：`同时启用`
- 作用：MMD 做显式分布对齐，GRL 做对抗式域不可分约束，二者互补。

3. MMD 作用特征层：`h`（GNN 输出）
- 作用：在共享表示层做域不变学习，通常比在更靠后层更稳。

4. 接入范围：`仅 single train`
- 作用：先用最小改动验证收益，降低联调复杂度；有效后再扩展到 kfold/loco。

注意：从代码结构上说，`train_and_validate()` 是 `single/kfold/loco/train_hpo` 共用的。如果你按下文直接改这个公共函数，那么其他流程也会“具备 MMD 能力”；但因为 `use_mmd` 默认是 `False`，**不显式开启就不会改变现有行为**。所以这仍然符合“先只在 single train 里启用”的目标。

### 对应的首版训练建议

- `--use_mmd 1`
- `--mmd_on slide`
- `--lambda_mmd 0.05`（起始值，可在 `0.01~0.1` 微调）
- 保留现有：
  - `--use_domain_adv_slide 1`
  - `--use_domain_adv_cancer 1`

---

## 7. 按当前代码库的精确改动

下面的代码段按当前文件组织给出，默认目标文件是 `stonco/core/train.py`。如果你后续要真正落地实现，基本可以按块粘贴。

### 7.1 文件头部 import

当前第 1 行是：

```python
import argparse, os, numpy as np, torch, math
```

改为：

```python
import argparse, os, numpy as np, torch, math, itertools
```

---

### 7.2 `main()` 里新增 CLI 参数

插入位置：`--grl_beta_gamma` 后面、癌种分层参数前面。

```python
    # 新增：MMD 对齐
    parser.add_argument('--use_mmd', type=int, choices=[0,1], default=None, help='启用/关闭 MMD 对齐（1/0）')
    parser.add_argument('--mmd_on', choices=['slide', 'cancer', 'both'], default=None, help='MMD 对齐域：slide/cancer/both')
    parser.add_argument('--lambda_mmd', type=float, default=None, help='MMD 损失权重')
    parser.add_argument('--mmd_num_kernels', type=int, default=None, help='MMD 多带宽 RBF 的核个数')
    parser.add_argument('--mmd_kernel_mul', type=float, default=None, help='MMD 多带宽倍率')
    parser.add_argument('--mmd_sigma', type=float, default=None, help='固定 RBF 带宽；不传则自适应估计')
    parser.add_argument('--mmd_max_pairs', type=int, default=None, help='每个 batch 最多计算多少个域对')
    parser.add_argument('--mmd_spots_per_slide', type=int, default=None, help='每张切片最多抽取多少个 spot 参与 MMD；<=0 表示使用全部 spot')
```

---

### 7.3 `cfg` 默认值增加 MMD 配置

插入位置：默认 `cfg = {...}` 里，紧跟在现有 domain/GRL 配置后面即可。

```python
           # 新增：MMD 默认配置
           'use_mmd': False,
           'mmd_on': 'slide',
           'lambda_mmd': 0.05,
           'mmd_num_kernels': 5,
           'mmd_kernel_mul': 2.0,
           'mmd_sigma': None,
           'mmd_max_pairs': 8,
           'mmd_spots_per_slide': 0,
```

---

### 7.4 `cfg` 覆盖与校验

插入位置：现有 domain 参数覆盖块之后，也就是 `grl_beta_gamma` 相关赋值后面。

```python
    if getattr(args, 'use_mmd', None) is not None:
        cfg['use_mmd'] = bool(args.use_mmd)
    if getattr(args, 'mmd_on', None) is not None:
        cfg['mmd_on'] = str(args.mmd_on)
    if getattr(args, 'lambda_mmd', None) is not None:
        cfg['lambda_mmd'] = float(args.lambda_mmd)
    if getattr(args, 'mmd_num_kernels', None) is not None:
        cfg['mmd_num_kernels'] = int(args.mmd_num_kernels)
    if getattr(args, 'mmd_kernel_mul', None) is not None:
        cfg['mmd_kernel_mul'] = float(args.mmd_kernel_mul)
    if getattr(args, 'mmd_sigma', None) is not None:
        cfg['mmd_sigma'] = float(args.mmd_sigma)
    if getattr(args, 'mmd_max_pairs', None) is not None:
        cfg['mmd_max_pairs'] = int(args.mmd_max_pairs)
    if getattr(args, 'mmd_spots_per_slide', None) is not None:
        cfg['mmd_spots_per_slide'] = int(args.mmd_spots_per_slide)

    cfg['use_mmd'] = bool(cfg.get('use_mmd', False))
    cfg['mmd_on'] = str(cfg.get('mmd_on', 'slide')).lower()
    if cfg['mmd_on'] not in {'slide', 'cancer', 'both'}:
        raise ValueError(f"cfg['mmd_on'] must be one of slide/cancer/both, got: {cfg['mmd_on']}")
    cfg['lambda_mmd'] = float(cfg.get('lambda_mmd', 0.05))
    cfg['mmd_num_kernels'] = int(cfg.get('mmd_num_kernels', 5))
    cfg['mmd_kernel_mul'] = float(cfg.get('mmd_kernel_mul', 2.0))
    cfg['mmd_sigma'] = None if cfg.get('mmd_sigma', None) is None else float(cfg['mmd_sigma'])
    cfg['mmd_max_pairs'] = int(cfg.get('mmd_max_pairs', 8))
    cfg['mmd_spots_per_slide'] = int(cfg.get('mmd_spots_per_slide', 0))
```

---

### 7.5 在 `train.py` 增加 MMD 工具函数

插入位置：建议放在 `train_and_validate()` 里的 `_make_graph_frequency_weights(...)` 后面、`model = STOnco_Classifier(...)` 前面。

```python
    def _sample_nodes_per_graph(h, dom_nodes, graph_nodes, spots_per_slide):
        if spots_per_slide is None or int(spots_per_slide) <= 0:
            return h, dom_nodes
        if graph_nodes is None:
            raise ValueError('graph_nodes is required when mmd_spots_per_slide > 0')

        keep_idx = []
        for gid in sorted(int(v) for v in torch.unique(graph_nodes.detach()).tolist()):
            idx = torch.where(graph_nodes == gid)[0]
            if idx.numel() <= int(spots_per_slide):
                keep_idx.append(idx)
            else:
                perm = torch.randperm(idx.numel(), device=idx.device)[:int(spots_per_slide)]
                keep_idx.append(idx[perm])

        if not keep_idx:
            return h, dom_nodes

        keep_idx = torch.cat(keep_idx, dim=0)
        return h[keep_idx], dom_nodes[keep_idx]

    def _pairwise_sq_dists(x, y):
        return torch.cdist(x, y, p=2).pow(2)

    def _build_sigma_list(x, y, num_kernels=5, kernel_mul=2.0, fixed_sigma=None):
        if fixed_sigma is not None:
            base_sigma = torch.tensor(float(fixed_sigma), dtype=x.dtype, device=x.device)
        else:
            z = torch.cat([x, y], dim=0)
            if z.size(0) <= 1:
                base_sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            else:
                sq = _pairwise_sq_dists(z, z)
                mask = ~torch.eye(sq.size(0), dtype=torch.bool, device=sq.device)
                vals = sq[mask]
                if vals.numel() == 0:
                    base_sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
                else:
                    base_sigma = torch.sqrt(vals.mean().clamp_min(1e-12))

        center = int(num_kernels) // 2
        sigma_list = []
        for i in range(int(num_kernels)):
            scale = float(kernel_mul) ** (i - center)
            sigma_list.append((base_sigma * scale).clamp_min(1e-6))
        return sigma_list

    def _rbf_kernel(x, y, sigma_list):
        sq = _pairwise_sq_dists(x, y)
        k = torch.zeros_like(sq)
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma.pow(2).clamp_min(1e-12))
            k = k + torch.exp(-sq * gamma)
        return k

    def _mmd2_unbiased(x, y, sigma_list):
        n = int(x.size(0))
        m = int(y.size(0))
        if n < 2 or m < 2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        k_xx = _rbf_kernel(x, x, sigma_list)
        k_yy = _rbf_kernel(y, y, sigma_list)
        k_xy = _rbf_kernel(x, y, sigma_list)

        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / float(n * (n - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / float(m * (m - 1))
        sum_xy = k_xy.mean()
        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        return torch.clamp(mmd2, min=0.0)

    def _multi_domain_mmd(h, dom_nodes, graph_nodes=None, spots_per_slide=0, num_kernels=5, kernel_mul=2.0, sigma=None, max_pairs=8):
        if h is None or dom_nodes is None:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        if h.size(0) != dom_nodes.size(0):
            raise ValueError(f"MMD shape mismatch: h={tuple(h.shape)}, dom_nodes={tuple(dom_nodes.shape)}")

        h, dom_nodes = _sample_nodes_per_graph(h, dom_nodes, graph_nodes, spots_per_slide)

        groups = {}
        for dom in sorted(int(v) for v in torch.unique(dom_nodes.detach()).tolist()):
            mask = dom_nodes == dom
            if int(mask.sum().item()) >= 2:
                groups[dom] = h[mask]

        pairs = list(itertools.combinations(sorted(groups.keys()), 2))
        if max_pairs is not None and int(max_pairs) > 0:
            pairs = pairs[:int(max_pairs)]

        if not pairs:
            return torch.tensor(0.0, dtype=h.dtype, device=h.device)

        losses = []
        for da, db in pairs:
            xa = groups[da]
            xb = groups[db]
            sigma_list = _build_sigma_list(
                xa,
                xb,
                num_kernels=num_kernels,
                kernel_mul=kernel_mul,
                fixed_sigma=sigma,
            )
            losses.append(_mmd2_unbiased(xa, xb, sigma_list))

        return torch.stack(losses).mean()
```

说明：
- 这里对 `slide/cancer` 都按 **spot-level 的 `h`** 计算；
- 域标签仍然沿用当前训练逻辑：`batch.bat_dom[batch.batch]` / `batch.cancer_dom[batch.batch]`；
- 若 `mmd_spots_per_slide > 0`，则先按 `batch.batch` 对每张切片随机抽样，再做域聚合；
- `max_pairs` 目前是确定性截断，不引入额外随机性。

---

### 7.6 替换 `_compute_losses(out, batch)`

把当前的 `_compute_losses` 整段替换成下面这个版本：

```python
    def _compute_losses(out, batch):
        logits = out['logits']
        mask = batch.y >= 0
        if mask.sum() > 0:
            loss_task = bce(logits[mask], batch.y[mask].float())
        else:
            loss_task = torch.tensor(0.0, device=device)

        total = loss_task
        loss_batch = torch.tensor(0.0, device=device)
        loss_cancer = torch.tensor(0.0, device=device)
        loss_mmd = torch.tensor(0.0, device=device)
        ce_batch = torch.tensor(float('nan'), device=device)
        ce_cancer = torch.tensor(float('nan'), device=device)
        raw_mmd_slide = torch.tensor(float('nan'), device=device)
        raw_mmd_cancer = torch.tensor(float('nan'), device=device)

        if out.get('dom_logits_slide', None) is not None and hasattr(batch, 'bat_dom') and hasattr(batch, 'batch'):
            if n_domains_batch is None:
                raise ValueError('n_domains_batch is None while batch domain logits are enabled.')
            dom_nodes = batch.bat_dom[batch.batch]
            dom_min = int(dom_nodes.min().item())
            dom_max = int(dom_nodes.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_batch):
                raise ValueError(
                    f"[DomainCheck] batch_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_batch={n_domains_batch}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            ce_batch = F.cross_entropy(out['dom_logits_slide'], dom_nodes)
            loss_dom_batch = cel_batch(out['dom_logits_slide'], dom_nodes)
            loss_batch = float(cfg.get('lambda_slide', 1.0)) * loss_dom_batch
            total = total + loss_batch

        if out.get('dom_logits_cancer', None) is not None and hasattr(batch, 'cancer_dom') and hasattr(batch, 'batch'):
            if n_domains_cancer is None:
                raise ValueError('n_domains_cancer is None while cancer domain logits are enabled.')
            dom_nodes = batch.cancer_dom[batch.batch]
            dom_min = int(dom_nodes.min().item())
            dom_max = int(dom_nodes.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_cancer):
                raise ValueError(
                    f"[DomainCheck] cancer_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_cancer={n_domains_cancer}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            ce_cancer = F.cross_entropy(out['dom_logits_cancer'], dom_nodes)
            loss_dom_cancer = cel_cancer(out['dom_logits_cancer'], dom_nodes)
            loss_cancer = float(cfg.get('lambda_cancer', 1.0)) * loss_dom_cancer
            total = total + loss_cancer

        if bool(cfg.get('use_mmd', False)):
            h = out.get('h', None)
            if h is None:
                raise ValueError("MMD is enabled but model output does not contain 'h'.")

            mmd_on = str(cfg.get('mmd_on', 'slide')).lower()
            lambda_mmd = float(cfg.get('lambda_mmd', 0.05))
            num_kernels = int(cfg.get('mmd_num_kernels', 5))
            kernel_mul = float(cfg.get('mmd_kernel_mul', 2.0))
            sigma = cfg.get('mmd_sigma', None)
            max_pairs = int(cfg.get('mmd_max_pairs', 8))
            spots_per_slide = int(cfg.get('mmd_spots_per_slide', 0))

            if mmd_on in {'slide', 'both'} and hasattr(batch, 'bat_dom') and hasattr(batch, 'batch'):
                dom_nodes_slide = batch.bat_dom[batch.batch]
                raw_mmd_slide = _multi_domain_mmd(
                    h,
                    dom_nodes_slide,
                    graph_nodes=batch.batch,
                    spots_per_slide=spots_per_slide,
                    num_kernels=num_kernels,
                    kernel_mul=kernel_mul,
                    sigma=sigma,
                    max_pairs=max_pairs,
                )
                loss_mmd = loss_mmd + lambda_mmd * raw_mmd_slide

            if mmd_on in {'cancer', 'both'} and hasattr(batch, 'cancer_dom') and hasattr(batch, 'batch'):
                dom_nodes_cancer = batch.cancer_dom[batch.batch]
                raw_mmd_cancer = _multi_domain_mmd(
                    h,
                    dom_nodes_cancer,
                    graph_nodes=batch.batch,
                    spots_per_slide=spots_per_slide,
                    num_kernels=num_kernels,
                    kernel_mul=kernel_mul,
                    sigma=sigma,
                    max_pairs=max_pairs,
                )
                loss_mmd = loss_mmd + lambda_mmd * raw_mmd_cancer

            total = total + loss_mmd

        return (
            total,
            loss_task,
            loss_batch,
            loss_cancer,
            loss_mmd,
            ce_batch,
            ce_cancer,
            raw_mmd_slide,
            raw_mmd_cancer,
        )
```

这里的实现选择是：
- `mmd_on='slide'` 时只计算 `bat_dom`
- `mmd_on='cancer'` 时只计算 `cancer_dom`
- `mmd_on='both'` 时两者直接相加
- `lambda_mmd` 同时作用于 `slide/cancer` 两路 raw MMD

---

### 7.7 训练循环里请求 `h`

把当前训练 forward：

```python
            out = model(
                batch.x,
                batch.edge_index,
                batch=getattr(batch, 'batch', None),
                edge_weight=getattr(batch, 'edge_weight', None),
                grl_beta_slide=grl_beta_slide,
                grl_beta_cancer=grl_beta_cancer,
            )
```

改成：

```python
            out = model(
                batch.x,
                batch.edge_index,
                batch=getattr(batch, 'batch', None),
                edge_weight=getattr(batch, 'edge_weight', None),
                grl_beta_slide=grl_beta_slide,
                grl_beta_cancer=grl_beta_cancer,
                return_h=bool(cfg.get('use_mmd', False)),
            )
```

验证阶段不用请求 `h`，因为验证指标里不需要 MMD。

---

### 7.8 `hist` 和 epoch 聚合增加 MMD 指标

把当前 `hist = {...}` 扩成：

```python
    hist = {
        'avg_total_loss': [],
        'avg_task_loss': [],
        'avg_batch_domain_loss': [],
        'avg_cancer_domain_loss': [],
        'avg_batch_domain_ce': [],
        'avg_cancer_domain_ce': [],
        'avg_mmd_loss': [],
        'avg_mmd_raw_slide': [],
        'avg_mmd_raw_cancer': [],
        'train_batch_domain_acc': [],
        'train_cancer_domain_acc': [],
        'var_risk': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_auroc': [],
        'val_auprc': [],
    }
```

在 epoch 累积变量区新增：

```python
        tot_mmd = 0.0
        tot_mmd_raw_slide = 0.0
        tot_mmd_raw_cancer = 0.0
        cnt_mmd_slide = 0
        cnt_mmd_cancer = 0
```

把 loss 解包改为：

```python
            (
                loss_total,
                loss_task,
                loss_batch,
                loss_cancer,
                loss_mmd,
                ce_batch,
                ce_cancer,
                raw_mmd_slide,
                raw_mmd_cancer,
            ) = _compute_losses(out, batch)
```

在 batch 累积区新增：

```python
            tot_mmd += float(loss_mmd.item())
            if torch.isfinite(raw_mmd_slide):
                tot_mmd_raw_slide += float(raw_mmd_slide.item())
                cnt_mmd_slide += 1
            if torch.isfinite(raw_mmd_cancer):
                tot_mmd_raw_cancer += float(raw_mmd_cancer.item())
                cnt_mmd_cancer += 1
```

在 epoch 平均区新增：

```python
        avg_mmd = tot_mmd / max(1, num_batches)
        avg_mmd_raw_slide = (tot_mmd_raw_slide / cnt_mmd_slide) if cnt_mmd_slide > 0 else float('nan')
        avg_mmd_raw_cancer = (tot_mmd_raw_cancer / cnt_mmd_cancer) if cnt_mmd_cancer > 0 else float('nan')
```

在 `hist[...]append(...)` 区新增：

```python
        hist['avg_mmd_loss'].append(avg_mmd)
        hist['avg_mmd_raw_slide'].append(avg_mmd_raw_slide)
        hist['avg_mmd_raw_cancer'].append(avg_mmd_raw_cancer)
```

如果你希望 `tqdm` 里直接看到 MMD，可把：

```python
                epoch_iter.set_postfix(
                    train_loss=f'{avg_total:.3f}',
                    val_acc=f'{val_accuracy:.3f}'
                )
```

改成：

```python
                epoch_iter.set_postfix(
                    train_loss=f'{avg_total:.3f}',
                    mmd=f'{avg_mmd:.3f}',
                    val_acc=f'{val_accuracy:.3f}',
                )
```

---

### 7.9 `_save_loss_components_csv(...)` 新增 MMD 列

把 DataFrame 构造改成：

```python
    df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'avg_total_loss': hist.get('avg_total_loss', [float('nan')] * n_epochs),
        'avg_task_loss': hist.get('avg_task_loss', [float('nan')] * n_epochs),
        'Var_risk': hist.get('var_risk', [float('nan')] * n_epochs),
        'avg_cancer_domain_ce': hist.get('avg_cancer_domain_ce', [float('nan')] * n_epochs),
        'avg_batch_domain_ce': hist.get('avg_batch_domain_ce', [float('nan')] * n_epochs),
        'avg_cancer_domain_loss': hist.get('avg_cancer_domain_loss', [float('nan')] * n_epochs),
        'avg_batch_domain_loss': hist.get('avg_batch_domain_loss', [float('nan')] * n_epochs),
        'avg_mmd_loss': hist.get('avg_mmd_loss', [float('nan')] * n_epochs),
        'avg_mmd_raw_slide': hist.get('avg_mmd_raw_slide', [float('nan')] * n_epochs),
        'avg_mmd_raw_cancer': hist.get('avg_mmd_raw_cancer', [float('nan')] * n_epochs),
        'train_batch_domain_acc': hist.get('train_batch_domain_acc', [float('nan')] * n_epochs),
        'train_cancer_domain_acc': hist.get('train_cancer_domain_acc', [float('nan')] * n_epochs),
        'train_accuracy': hist.get('train_accuracy', [float('nan')] * n_epochs),
        'val_accuracy': hist.get('val_accuracy', [float('nan')] * n_epochs),
        'val_macro_f1': hist.get('val_macro_f1', [float('nan')] * n_epochs),
        'val_auroc': hist.get('val_auroc', [float('nan')] * n_epochs),
        'val_auprc': hist.get('val_auprc', [float('nan')] * n_epochs),
    })
```

---

### 7.10 `_plot_train_metrics(...)` 增加 MMD 子图

当前是 2x3。要把 MMD 放进去，最简单是改成 3x3：

```python
    fig1, axes1 = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    metrics_train = [
        ('avg_total_loss', 'avg_total_loss'),
        ('avg_task_loss', 'avg_task_loss'),
        ('avg_mmd_loss', 'avg_mmd_loss'),
        ('var_risk', 'Var_risk'),
        ('avg_cancer_domain_loss', 'avg_cancer_domain_loss'),
        ('avg_batch_domain_loss', 'avg_batch_domain_loss'),
        ('avg_mmd_raw_slide', 'avg_mmd_raw_slide'),
        ('avg_mmd_raw_cancer', 'avg_mmd_raw_cancer'),
        ('train_accuracy', 'train_accuracy'),
    ]
```

这样图上能同时看到：
- MMD 加权后的总贡献：`avg_mmd_loss`
- slide/cancer 两路原始分布距离：`avg_mmd_raw_slide` / `avg_mmd_raw_cancer`

如果首版只开 `mmd_on=slide`，那 `avg_mmd_raw_cancer` 会是 `NaN`，这是正常的。

---

### 7.11 `models.py` 是否要改

当前分支下：
- `return_h=False` 已存在
- `out['h']` 已存在

所以 **`models.py` 不需要改**。

只有在你切到更老的分支、`forward()` 还没有 `return_h` 时，才需要补下面这段：

```python
        return_z=False,
        return_h=False,
    ):
        h = self.gnn(x, edge_index, edge_weight=edge_weight)
        logits, z64 = self.clf(h, return_z=return_z)
        ...
        if return_z:
            out['z64'] = z64
        if return_h:
            out['h'] = h
        return out
```

---

## 8. 最小落地顺序

如果你接下来要真正改代码，建议按这个顺序：

1. 先加 CLI/config 字段，保证 `cfg['use_mmd']` 能正确落到训练函数。
2. 再加 MMD helper 函数和 `_compute_losses` 替换。
3. 然后补训练循环里的 `return_h`、`hist`、CSV。
4. 最后再改曲线图；这一步不是训练正确性的前置条件。

---

## 9. 首版参数建议

- `use_mmd = 1`
- `mmd_on = slide`
- `lambda_mmd = 0.05`
- `mmd_num_kernels = 5`
- `mmd_kernel_mul = 2.0`
- `mmd_sigma = None`
- `mmd_max_pairs = 8`

如果首版出现任务指标下降，优先把 `lambda_mmd` 降到 `0.01`，不要先改核参数。
