# MDAN 多域对抗设计总结与 STOnco 实际实现对比

## 1. 文档目的与来源

本文档分两部分：

1. 总结 NeurIPS 2018 论文 **Adversarial Multiple Source Domain Adaptation** 及其官方实现 `hanzhaoml/MDAN` 的多域对抗设计。
2. 基于 **STOnco 当前代码实现**，而不是基于旧说明文档，分析 STOnco 的域对抗网络，并与 MDAN 做结构化对比。

外部来源：

- 论文摘要页：<https://proceedings.neurips.cc/paper_files/paper/2018/hash/717d8b3d60d9eea997b35b02b6a4e867-Abstract.html>
- 论文 PDF：<https://proceedings.neurips.cc/paper_files/paper/2018/file/717d8b3d60d9eea997b35b02b6a4e867-Paper.pdf>
- 官方代码仓库：<https://github.com/hanzhaoml/MDAN>

STOnco 本地代码依据：

- `stonco/core/models.py`
- `stonco/core/train.py`

下文中，“代码事实”表示可以直接从代码读出的结论；“基于代码的推断”表示对设计意图或训练行为的合理解释。

## 2. MDAN 的多域对抗设计

### 2.1 问题设定

MDAN 解决的是 **multiple-source unsupervised domain adaptation**：

- 有 `k` 个带标签源域 `S1 ... Sk`
- 有 `1` 个不带标签目标域 `T`
- 目标是在保留源域任务判别能力的同时，让表示对每个 `Si` 和 `T` 的域差异都尽量不敏感

论文第 3 节给出了多源情形下的泛化界。核心点不是“把所有源域简单拼起来”，而是显式考虑多个源域与目标域之间的差异项。

### 2.2 网络结构

MDAN 的结构由三部分组成：

- 一个共享特征提取器 `G_f`
- 一个任务分类器 `G_y`
- `k` 个域分类器 `G_d^i`

其中第 `i` 个域分类器只负责一个二分类问题：区分样本来自 `Si` 还是来自 `T`。

对应到官方实现 `model.py`：

- `self.hiddens` 是共享特征提取器
- `self.softmax` 是共享任务分类器
- `self.domains = nn.ModuleList([... for _ in range(self.num_domains)])` 是 `k` 个二分类域头

这意味着 MDAN 的“多域”不是一个多类域分类器，而是 **多组 source-vs-target 二分类器并行存在**。

### 2.3 目标函数设计

论文第 4 节提出了两个版本。

#### Hard-Max MDAN

硬版本直接近似优化 worst-case bound，可写成：

`min_h max_i [ source_task_loss_i - domain_confusion_i ]`

直观上：

- 任务部分希望每个源域都可判别
- 对抗部分希望 `Si` 与 `T` 在表示空间中不可区分
- 最终重点优化“最难的那个源域”

论文指出这个版本的问题是：

- 过于关注最差源域
- 每步只主要依赖一个域的梯度，数据利用率不高

#### Soft-Max MDAN

软版本用 `log-sum-exp` 平滑 Hard-Max：

`(1/gamma) * log sum_i exp(gamma * e_i)`

其中 `e_i` 表示第 `i` 个源域对应的“任务误差 + 域差异项”组合。

论文给出的解释是：

- 这相当于对多个源域梯度做自适应加权
- 哪个源域当前更难，哪个源域的权重就更大
- 因而它既保留多源信息，又避免 Hard-Max 只看单个最差源域的问题

### 2.4 官方代码中的实现方式

官方 `main_amazon.py` 与论文设计是一致的。

每个 iteration 中：

1. 从多个源域各采样一个 batch
2. 从目标域采样一个无标签 batch
3. 前向得到：
   - 每个源域的任务分类损失
   - 每个 `Si` 对应的 source-vs-target 域损失
4. 组合成最终目标

代码中的两种组合规则为：

- `mode == "maxmin"`:
  `loss = max(task_losses) + mu * min(domain_losses)`
- `mode == "dynamic"`:
  `loss = log(sum(exp(gamma * (task_losses + mu * domain_losses)))) / gamma`

这里可以看出两个关键实现事实：

- MDAN 的域对抗聚合是 **按源域维度聚合** 的，不是把所有域样本直接塞进一个共享多类域头。
- Soft-Max 版本的核心不是简单加和，而是 **带自适应权重的 log-sum-exp 聚合**。

### 2.5 GRL 与优化方式

MDAN 官方实现使用 GRL 来做同步更新：

- 前向时 GRL 是恒等映射
- 反向时对特征提取器收到的域分类梯度乘以 `-1`

实现上，`GradientReversalLayer.backward` 直接返回负梯度，因此训练是单阶段的 simultaneous update，而不是交替训练 feature extractor 与 domain classifier。

### 2.6 设计要点总结

MDAN 的域对抗设计可以概括为 5 点：

1. 多源单目标设定，目标域在训练时无标签。
2. 每个源域各有一个 **source-vs-target 的二分类域头**。
3. 任务损失也是按源域分别计算，而不是先混合再算一个总任务损失。
4. 多源信息的关键在于 **Hard-Max / Soft-Max 聚合规则**。
5. Soft-Max MDAN 的本质是对多源梯度做自适应加权，而不是均匀平均。

## 3. STOnco 当前代码中的域对抗实现

本节只依据 STOnco 当前代码。

### 3.1 域标签是如何定义的

代码事实：

- 在构图阶段，每张图会被注入两个**图级**域标签：
  - `bat_dom`: slide/batch 域
  - `cancer_dom`: cancer 域
- 见 `stonco/core/train.py:670-675`

这说明 STOnco 的“域”定义不是 `source_i vs target`，而是：

- 一个 slide/batch 多类域分类问题
- 一个 cancer type 多类域分类问题

### 3.2 模型结构

代码事实：

- 共享编码器是 `GNNBackbone`
- 主任务头是 `ClassifierHead`
- 域头是两个可选的 `DomainHead`
- 两个域头都直接接在同一个节点表示 `h` 上
- 见 `stonco/core/models.py:139-195`

也就是说，STOnco 当前结构是：

`h -> task head`

同时

`h -> GRL(beta_slide) -> slide domain head`

`h -> GRL(beta_cancer) -> cancer domain head`

这里没有 MDAN 那种 “`k` 个 source-vs-target 二分类头”的结构，而是两个并行的**多类**域头。

### 3.3 域损失是如何计算的

代码事实：

- `loss_task` 是主任务 BCE
- `loss_batch = lambda_slide * CE(dom_logits_slide, dom_nodes)`
- `loss_cancer = lambda_cancer * CE(dom_logits_cancer, dom_nodes)`
- 总损失直接相加：
  `total = loss_task + loss_batch + loss_cancer + loss_mmd(optional)`
- 见 `stonco/core/train.py:1073-1161`

因此 STOnco 当前的域对抗目标不是 Hard-Max，也不是 Soft-Max，而是一个**线性加和目标**。

### 3.4 域标签如何从图级变成训练监督

代码事实：

- `bat_dom` 和 `cancer_dom` 是图级标签
- 真正计算 CE 时，使用 `batch.bat_dom[batch.batch]` 与 `batch.cancer_dom[batch.batch]`
- 见 `stonco/core/train.py:1089-1118`

这表示图级域标签被**复制到图中每一个节点**，最终域分类监督是节点级的。

直接含义：

- 一个节点更多的图，在域损失里贡献更大
- 域损失的统计单位实际是 node/spot，而不是 graph/slide

### 3.5 域损失的类别权重

代码事实：

- STOnco 会基于训练图中的图频率构造域类别权重
- 权重形式是 `sqrt(n_graph / (k * count_c))`
- 再做 `clamp(0.5, 5.0)` 和均值归一化
- 见 `stonco/core/train.py:835-849`

这说明 STOnco 尝试缓解图级类别不平衡，但由于域标签在损失里是节点级展开，图大小差异仍会影响每步实际梯度贡献。

### 3.6 GRL 调度

代码事实：

- STOnco 的 GRL 强度 `beta` 不是固定写死的
- 支持 `constant`、`dann`、`linear`
- 默认是 `dann`
- slide 与 cancer 两个域头有不同 target/delay/warmup
- 默认：
  - `slide target = 1.0`
  - `cancer target = 0.5`
  - `slide delay = 1 epoch`
  - `cancer delay = 3 epochs`
- 见 `stonco/core/train.py:314-327` 与 `stonco/core/train.py:1252-1319`

这点比 MDAN 官方代码更工程化，因为 STOnco 对两个域头分别控制对抗强度和启动时机。

### 3.7 MMD 与域对抗并存

代码事实：

- STOnco 还支持在同一个表示 `h` 上叠加 MMD 正则
- MMD 可作用在 slide 域、cancer 域或两者
- 总损失会继续加上 `loss_mmd`
- 见 `stonco/core/train.py:1120-1161`

因此 STOnco 当前并不是“纯 MDAN 式域对抗网络”，而是：

**主任务 + 双域对抗 + 可选 MMD** 的组合式正则训练框架。

## 4. 与 STOnco 的对比

这一节是基于上面的代码事实做出的对比。

### 4.1 核心结论

STOnco 当前实现 **不是 MDAN 的直接复现，也不是轻微改写版 MDAN**。  
更准确地说，它是一个：

- 单数据集内的多因素去偏训练框架
- 使用两个并行多类域头
- 在共享表示上同时去除 slide/batch 与 cancer 信息

而 MDAN 是一个：

- 多源单目标的无监督域适配模型
- 使用多个 source-vs-target 二分类域头
- 通过 Hard-Max / Soft-Max 在源域维度上做自适应聚合

### 4.2 结构化对比

| 维度 | MDAN | STOnco 当前实现 |
| --- | --- | --- |
| 训练设定 | `k` 个带标签源域 + 1 个无标签目标域 | 单训练集内，同时对抗 `slide/batch` 与 `cancer` 两类混杂因素 |
| 域头形式 | `k` 个二分类头，每个头判别 `Si` vs `T` | 2 个多分类头：一个 slide/batch，多一个 cancer |
| 任务损失 | 每个源域各算一份任务损失 | 所有样本共享一个主任务 BCE |
| 域损失聚合 | Hard-Max 或 Soft-Max(log-sum-exp) | 线性加和 `loss_task + lambda_slide * CE + lambda_cancer * CE (+ MMD)` |
| 对抗对象 | 每个源域和目标域之间的分布差异 | 表示中与 batch / cancer 相关的可预测信息 |
| 域监督粒度 | 样本级 | 节点级，图级域标签被展开到所有节点 |
| 权重机制 | Soft-Max 自适应强调更难源域 | 域损失权重为手动 `lambda`，类别权重按图频率构造 |
| GRL | 官方代码中固定 GRL 反转 | 支持 `constant/dann/linear`，且 slide 与 cancer 分别调度 |
| 额外正则 | 论文主线是多域对抗 | 还可叠加 MMD |

### 4.3 最重要的设计差异

#### 差异 1：MDAN 的关键是“按源域聚合”，STOnco 没有这层结构

MDAN 的创新点不只是“多个域头”，而是：

- 每个源域单独形成一个任务项和一个域差异项
- 再在源域维度做 Hard-Max 或 Soft-Max 聚合

STOnco 当前没有这层 source-wise aggregation。  
它只有两个固定语义的域头，并把它们的损失直接相加。

#### 差异 2：STOnco 的域定义更像“混杂因素去除”，不是“多源到目标适配”

MDAN 的域来自多个数据分布与一个目标分布。  
STOnco 当前的两个域分别来自：

- `slide/batch`
- `cancer type`

这更接近 nuisance removal / confounder suppression，而不是经典的 multisource UDA。

#### 差异 3：STOnco 当前域监督单位是节点，不是图

从 `batch.bat_dom[batch.batch]` 和 `batch.cancer_dom[batch.batch]` 可以直接看出，STOnco 把图级域标签复制给图内所有节点。  
因此它的域对抗强度会随图内 spot 数变化，这与 MDAN 的样本级 source-target 判别机制不同。

#### 差异 4：STOnco 有更强的工程控制，但理论对应关系更弱

STOnco 当前实现提供了：

- 双域头
- 不同 `lambda`
- 不同 GRL beta schedule
- 可选 MMD

这些都很实用，但它们与 MDAN 论文中的多源泛化界并不是一一对应的。  
换言之，STOnco 当前实现更像工程化的 domain-invariance regularization，而不是由 MDAN 理论直接推导出的目标函数。

### 4.4 如果要说“STOnco 借鉴了 MDAN 的什么”

可以成立的说法：

- 借鉴了 **GRL 驱动的域不变表示学习** 思路
- 借鉴了“共享表示 + 主任务头 + 域头”的对抗训练框架
- 借鉴了“通过域分类器反向约束编码器”的基本机制

不准确的说法：

- “STOnco 当前实现就是 MDAN”
- “STOnco 现在已经实现了 MDAN 的多源对抗聚合”
- “STOnco 的两个域头等价于 MDAN 的多个 source-vs-target 头”

### 4.5 基于代码的推断：为什么两者训练行为会不同

基于代码，STOnco 与 MDAN 在训练动态上大概率会有以下不同：

1. STOnco 的两个域头共享同一个 `h`，其反向梯度会直接叠加到同一编码器上。
2. slide 域与 cancer 域都使用多类 CE，而不是 source-vs-target 二分类，因此优化景观与 MDAN 不同。
3. 节点级域监督会让大图在域对抗中占更大比重。
4. 由于没有 MDAN 的 Soft-Max 源域自适应聚合，STOnco 当前无法自动把更多训练注意力分配给“更难对齐”的域组合，只能依赖 `lambda` 和 `beta` 调参。

这些结论是从代码结构推出来的，不是论文作者显式声明的理论性质。

## 5. 一句话总结

MDAN 的核心是 **“多源单目标 + 多个 source-vs-target 域头 + Hard/Soft 聚合”**；  
STOnco 当前代码的核心是 **“共享 GNN 表示上的双多类域对抗 + 可调 GRL + 可选 MMD”**。  
两者共享的是 GRL 域不变学习思想，但目标函数形式、域定义方式和训练粒度都明显不同。
