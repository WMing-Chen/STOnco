# Wasserstein Barycenter Methods: Dual Potentials, ICNN, and Generative Models

本文解释 Wasserstein barycenter 相关文献中常见的三类神经网络化方法：dual potentials、ICNN、generative model，并说明它们和 BaryIR 代码中 `barylatent + Pots` 设计的关系。

## 1. 问题背景：barycenter 到底是什么

给定多个源分布 $P_1, \ldots, P_K$ 和权重 $\lambda_k$，Wasserstein barycenter 要找一个分布 $Q$，使它到所有源分布的加权 Wasserstein 距离最小：

$$
Q^* = \arg\min_Q \sum_k \lambda_k W(P_k, Q)
$$

直观上，$Q^*$ 不是欧氏空间里的普通均值点，而是考虑最优传输几何后的“分布平均”。在 BaryIR 语境里，不同 $P_k$ 可以理解为不同退化域的 latent 特征分布；理想的 $Q^*$ 是退化无关的公共特征分布。

代码中的重要区别是：BaryIR 没有显式维护一个 $Q$ 或 `center` 参数。$\mathrm{bary\_latent} = \mathrm{barylatent}(\mathrm{source\_latent})$ 是网络前向生成的特征；训练过程中所有样本生成的 `bary_latent` 的整体分布，才是隐式的 barycenter 表示。

## 2. Dual Potentials：用势函数刻画分布对齐

最优传输有 primal 和 dual 两种视角。primal 视角关心“如何把一个分布的质量搬到另一个分布”；dual 视角则引入势函数 potential，用函数值来刻画运输代价的下界或对偶目标。

对于多个源域到一个 barycenter 的问题，每个源域通常需要自己的势函数：

$$
P_1 \to Q: f_1,\quad P_2 \to Q: f_2,\quad \ldots,\quad P_K \to Q: f_K
$$

这就是为什么 BaryIR 里的 `Pots` 是多 head 结构：每个 head 对应一个源域的 scalar potential。它不是分类器，不输出域概率；训练代码已经根据 `de_id` 指定使用哪个 head。`Pots(bary_latent_i, k)` 的输出只是第 $k$ 个势函数对当前 `bary_latent_i` 的标量评分。

在 BaryIR 中，更新 `BaryIR` 时使用：

$$
\mathrm{mse\_loss} + \alpha(\mathrm{ort\_loss} + \mathrm{contra\_loss}) - \mathrm{potential\_loss}
$$

这里的负号意味着：固定 `Pots` 时，`BaryIR` 会调整 `bary_latent`，让当前势函数给出更高分。随后固定 `BaryIR` 更新 `Pots`，又会让势函数适应当前的 `bary_latent` 分布。这是一个对抗式或 minimax 风格的分布约束。

相关文献中，Li et al. 的 Continuous Regularized Wasserstein Barycenters 使用正则化 barycenter 的 dual formulation，通过 dual potentials 隐式参数化 barycenter，并用随机梯度下降处理连续分布样本。这和 BaryIR 的共同点是：都不是直接枚举一个离散中心点集合，而是通过势函数给出分布级训练信号。不同点是：BaryIR 的 `Pots` 是任务网络里的经验性 neural critic，并没有完整复现某个严格的 Sinkhorn 或 regularized barycenter solver。

## 3. ICNN：用输入凸神经网络表示凸势函数

ICNN 是 input convex neural network。它的目标是让网络输出关于输入 $x$ 是凸函数。实现上通常会对部分权重施加非负约束，并使用保持凸性的激活函数与结构，使网络整体满足输入凸性。

ICNN 在 W2 barycenter 中很重要，是因为二次代价下的最优传输映射和凸势函数关系密切。根据 Brenier 理论，在合适条件下，从一个绝对连续分布到另一个分布的最优传输映射可以写成某个凸函数的梯度：

$$
T(x) = \nabla \phi(x)
$$

因此，如果用 ICNN 表示 $\phi$，就可以通过 $\nabla \phi$ 表示运输映射。Fan, Taghvaei, and Chen 在 ICML 2021 的方法基于 Kantorovich dual formulation 和 ICNN 来近似高维 Wasserstein barycenter，并用 generative model 表示 barycenter。Korotin et al. 的 ICLR 2021 工作也使用 ICNN 与 cycle-consistency regularization，目标是在连续样本访问设置下估计 W2 barycenter，同时避免一些正则化方法带来的偏差。

ICNN 的优点：

- 有明确的 OT 几何动机，尤其适合 W2 和凸势函数相关的设置。
- 相比完全自由的 MLP，凸性约束让势函数更贴近理论形式。
- 可以通过梯度映射构造或约束 transport map。

ICNN 的代价：

- 表达能力受凸性结构限制，复杂高维图像任务中可能不够灵活。
- 训练和实现比普通 MLP 更繁琐，需要处理权重非负、梯度映射、cycle consistency 等约束。
- 如果只是想给任务模型一个 latent-level 分布正则，ICNN 可能过重。

BaryIR 的 `Pots` 没有使用 ICNN。它只是普通 `Conv2d + Linear` 网络，输出 scalar potential。这说明作者更关注图像复原任务中的可训练分布约束，而不是严格实现 ICNN-based W2 barycenter solver。

## 4. Generative Model：直接学习能采样 barycenter 的生成器

另一类方法直接用生成模型表示 barycenter 分布：

$$
z \sim \text{noise}, \quad x = G_\theta(z), \quad x \sim Q_\theta
$$

这里 $G_\theta$ 是生成器，$Q_\theta$ 是它诱导出的分布。优化目标是让 $Q_\theta$ 成为多个源分布的 Wasserstein barycenter。这样做的好处是，一旦训练完成，可以无限采样 barycenter，而不用每次查询所有源分布样本。

Fan et al. 的 ICML 2021 方法提到用 generative model 表示 barycenter，并结合 ICNN 势函数做优化。Korotin et al. 的 NeurIPS 2022 Wasserstein Iterative Networks 则强调用 generative model 近似连续测度的 W2 barycenter，并指出相比 ICNN 限制，任意神经网络生成器能提供更强表达能力。

这种 generative view 和 BaryIR 的关系需要小心区分：

- 标准生成式 barycenter：$z \sim \text{noise} \to G_\theta(z) \to$ barycenter sample
- BaryIR：degraded image $\to$ encoder $\to$ `source_latent` $\to$ `barylatent` $\to$ `bary_latent`

BaryIR 的 `barylatent` 更像一个条件特征映射，而不是无条件生成器。它不从随机噪声采样 barycenter，而是把每个输入图像的 `source_latent` 映射到公共 WB latent。因此它是“隐式分布生成”：所有训练样本经过 `barylatent` 后形成的 `bary_latent` 分布，才对应任务中的隐式 barycenter space。

## 5. 三类方法的对比

| 方法 | 核心对象 | 输出 | 优点 | 局限 | 与 BaryIR 的关系 |
| --- | --- | --- | --- | --- | --- |
| Dual potentials | 每个源域的势函数 $f_k$ | scalar potential | 直接来自 OT dual，可做分布级约束 | 需要和 primal/生成器交替优化，可能不稳定 | `Pots` 最接近这一类，是 multi-head scalar critic |
| ICNN | 输入凸势函数 $\phi$ | convex scalar，常用其梯度表示 map | 更贴近 W2 凸势理论 | 结构受限，复杂高维任务表达力可能不足 | BaryIR 没用 ICNN，只用普通 Conv/MLP potential head |
| Generative model | barycenter 生成器 $G_\theta$ | barycenter samples | 能显式采样 $Q_\theta$，表达力强 | 需要训练生成器和 OT 目标，优化复杂 | `barylatent` 类似条件生成 barycenter feature，但不是无条件生成器 |

## 6. 回到 BaryIR：应该如何理解 `barylatent + Pots`

BaryIR 的设计可以概括为：

$$
\mathrm{source\_latent} = \mathrm{bary\_latent} + \mathrm{res\_bary}
$$

其中：

- `bary_latent`：希望表示退化无关的公共内容，近似 WB space 中的样本特征。
- `res_bary`：希望表示退化特异残差，通过 contrastive loss 让同域 residual 更接近、异域 residual 更分开。
- `Pots`：多域 potential head，对 `bary_latent` 提供分布级评分，推动不同退化域的公共特征分布向共享 barycenter 表示靠拢。

训练时的交替逻辑是：

1. `freeze(Pots)`, `unfreeze(BaryIR)`：BaryIR 通过 $-\mathrm{Pots}(\mathrm{bary\_latent})$ 接收势函数梯度，调整 `barylatent` 生成的公共表示。
2. `unfreeze(Pots)`, `freeze(BaryIR)`：`Pots` 根据当前 `bary_latent` 更新各域势函数，让评分器继续保持有效约束。

因此，BaryIR 借鉴了 dual-potential / neural barycenter 的思想，但做了任务化改造：它不求一个可独立采样的全局 barycenter 生成器，也不使用 ICNN 严格参数化凸势函数，而是在图像复原网络的 latent 空间中学习一个隐式的、条件化的公共 WB 表示。

## 7. 实用理解

如果把三类方法映射到一句话：

- Dual potentials：用多个“评分函数”衡量各源域和中心分布之间的关系。
- ICNN：把评分函数限制成凸函数，以更贴近 W2 最优传输理论。
- Generative model：直接训练一个生成器，让它产生 barycenter 分布的样本。
- BaryIR：用 `barylatent` 从输入图像特征中生成公共 latent，用 `Pots` 对这个 latent 加分布级约束，用 `res_bary` 承担退化特异信息。

所以，BaryIR 的“隐式 barycenter”不是完全凭空原创；它继承了连续 Wasserstein barycenter 中 dual potential、神经网络势函数、生成式表示等已有思路。它的任务创新点在于把这些思想嵌入 all-in-one image restoration 的 latent 分解框架中。

## References

- Agueh and Carlier, "Barycenters in the Wasserstein Space", SIAM Journal on Mathematical Analysis, 2011.
- Li, Genevay, Yurochkin, and Solomon, "Continuous Regularized Wasserstein Barycenters", NeurIPS 2020: https://research.ibm.com/publications/continuous-regularized-wasserstein-barycenters
- Fan, Taghvaei, and Chen, "Scalable Computations of Wasserstein Barycenter via Input Convex Neural Networks", ICML 2021: https://proceedings.mlr.press/v139/fan21d.html
- Korotin, Li, Solomon, and Burnaev, "Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization", ICLR 2021: https://openreview.net/forum?id=3tFAs5E-Pe
- Korotin, Egiazarian, Li, and Burnaev, "Wasserstein Iterative Networks for Barycenter Estimation", NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/6489f2c6ac6420124fcef2a489615a97-Abstract-Conference.html
- Tang et al., "Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration", arXiv 2026: https://arxiv.org/abs/2602.23169

## 8. 逐篇文章详细总结

### 8.1 Agueh and Carlier, 2011: Wasserstein barycenter 的理论起点

这篇文章是 Wasserstein barycenter 理论的经典来源之一。它把欧氏空间中点的加权均值推广到概率测度空间：欧氏 barycenter 是最小化 $\sum_i \lambda_i \|x - x_i\|^2$ 的点；Wasserstein barycenter 则是最小化 $\sum_i \lambda_i W_2^2(\nu_i, \nu)$ 的概率分布。这个定义把 McCann interpolation 从两个测度的情形推广到多个测度。

文章主要解决理论问题，而不是神经网络计算问题。核心贡献包括：证明 barycenter 的存在性；在至少一个输入测度满足一定绝对连续/小集条件时给出唯一性；从 dual problem 推导最优性条件；把 barycenter 问题联系到 multi-marginal optimal transport；讨论正则性；并严格求解 Gaussian 输入分布的情形。Gaussian 情形很重要，因为它说明 Wasserstein barycenter 不只是形式定义，在某些分布族中可以得到结构化解：Gaussian 的 barycenter 仍是 Gaussian，均值和协方差有明确关系。

对 BaryIR 的意义：BaryIR 使用的“多个源域分布到一个公共 WB 分布”的概念，理论根基来自这类定义。BaryIR 并不直接求解 Agueh-Carlier 的原始变分问题，而是把它移植到深度特征空间：不同退化域的 latent 分布对应 $\nu_i$，网络生成的 `bary_latent` 分布对应隐式的 $\nu$。

### 8.2 Li, Genevay, Yurochkin, and Solomon, 2020: 用 dual potentials 隐式参数化连续 barycenter

这篇 NeurIPS 2020 论文关注一个计算瓶颈：传统 barycenter 算法常把 barycenter 的 support 限制为有限点集，这在连续分布和高维任务中不理想。作者提出 regularized Wasserstein barycenter 的新 dual formulation，并用 dual potentials 隐式参数化 barycenter。重点不是先假设 barycenter 有一组固定 support 点，而是通过求解 dual potentials，再从 primal-dual 关系中恢复 barycenter 或对应 transport plan。

方法上，文章引入 regularizing measure 作为 barycenter support 的 proxy；对 entropic 或 quadratic regularization 的 OT 问题写出 unconstrained dual objective；再把积分写成 expectation，用随机采样和 SGD 训练一组 dual potential 函数。函数参数化可以用神经网络，也可以用 random Fourier features。训练完成后，可通过恢复 transport plan、Monge map 或采样过程得到连续 barycenter 样本。

它和 BaryIR 的关系非常直接：BaryIR 的 `Pots` 也是“用势函数给 barycenter 表示提供分布级信号”的思路。但二者差别也很大。Li 等人的方法是一个较完整的 regularized barycenter solver，强调 strong duality、regularization、primal-dual recovery；BaryIR 的 `Pots` 则是任务网络中的经验性 multi-head critic，只在 latent space 中通过 $-\mathrm{potential\_loss}$ 和交替更新为 `barylatent` 提供约束，并没有显式恢复 transport plan 或求解标准 Sinkhorn/regularized barycenter。

### 8.3 Fan, Taghvaei, and Chen, 2021: ICNN + generative model 的可扩展 barycenter 估计

这篇 ICML 2021 论文从 W2 的 Kantorovich dual / semi-dual 出发，利用 W2 最优传输中的凸势函数结构，用 input convex neural network 表示 convex potentials。同时，它用生成器表示 barycenter 分布：从简单 latent distribution 采样，经 generator $h$ 得到 barycenter 样本。因此，该方法既有 dual potential 的理论结构，又有 generative model 的可采样能力。

关键思想是：对固定候选 barycenter 分布，可以用 W2 的 semi-dual 计算它到每个源分布的距离；对未知 barycenter，则用 generator 诱导的分布作为可优化对象。ICNN 负责参数化凸函数 $f_i, g_i$，generator 负责参数化 barycenter $\nu$。训练形式接近 min-sup-inf：generator 寻找 barycenter，ICNN potentials 估计对应的 W2 代价。论文强调三点：只需要源分布样本；生成器可以无限采样 barycenter；在单源边界情况下和 GAN 有相似形式。

对 BaryIR 的启发：BaryIR 的 `barylatent` 也像一个“生成 barycenter feature 的模块”，但它不是从噪声生成无条件样本，而是从每张图像的 `source_latent` 条件生成 `bary_latent`。BaryIR 的 `Pots` 也不是 ICNN，没有凸性约束；它牺牲部分 W2 理论结构，换取更简单、更贴近图像复原任务的特征空间约束。

### 8.4 Korotin, Li, Solomon, and Burnaev, 2021: 无 minimax 的连续 W2 barycenter 估计

这篇 ICLR 2021 论文针对连续 W2 barycenter 的另一个痛点：许多方法依赖 entropic/quadratic regularization，会引入 bias；而一些直接对抗式方法又会进入复杂的 minimax 优化。作者提出使用 ICNN 和 cycle-consistency regularization，构造一种不需要 minimax 优化、同时避免 regularization bias 的连续 W2 barycenter 算法。

它的核心对象仍是 W2 的 convex potentials。ICNN 用来表达凸势函数；cycle-consistency regularization 用来让势函数对之间更接近 Legendre conjugate / transport map 的一致关系；congruence 约束则来自 barycenter 最优性条件，即各源域到 barycenter 的最优映射在加权意义下满足一致性。这样，算法不直接训练一个生成器去对抗判别器，而是通过势函数和正则约束恢复 barycenter。

对 BaryIR 的意义：这篇文章说明“用 neural potentials 学连续 barycenter”不必总是 GAN 式生成器-判别器对抗，也可以用 ICNN + consistency/congruence 的非 minimax 方式。BaryIR 没采用这条路线：它没有 ICNN 和 cycle-consistency，而是用普通 `Pots` 与 `BaryIR` 交替优化。换句话说，BaryIR 更像任务驱动的 adversarial potential regularization，而不是严格的 non-minimax W2 barycenter solver。

### 8.5 Korotin, Egiazarian, Li, and Burnaev, 2022: Wasserstein Iterative Networks

这篇 NeurIPS 2022 论文认为已有连续 barycenter 方法仍有两类限制：regularization 会带来 bias，ICNN 的凸性结构在大规模图像任务上可能表达力不足。作者提出 Wasserstein Iterative Networks，用普通神经网络和生成模型来近似连续 W2 barycenter，并通过固定点迭代思想更新 barycenter generator。

方法上，论文从 barycenter 的 fixed-point condition 出发。若当前候选 barycenter 分布为 $P$，对每个源分布 $P_n$ 计算从 $P$ 到 $P_n$ 的 OT map $T_{P\to P_n}$，然后把当前分布通过加权平均映射 $\sum_n \alpha_n T_{P\to P_n}$ 推到新的分布。这一过程的固定点就是 barycenter。实际实现中，作者用 generator $G_\xi$ 表示当前 barycenter 分布；先训练多组 OT map 网络把 $G_\xi$ 的样本映射到各源分布；再用回归把 generator 输出更新为这些 OT maps 的加权平均结果。

这篇文章和 BaryIR 的共同点是：都避免把 barycenter 写成离散 support；都接受“barycenter 是由网络诱导的隐式分布”。不同点是：Wasserstein Iterative Networks 的目标是训练一个可采样的 barycenter generator，并且显式近似 fixed-point iteration；BaryIR 的 `barylatent` 是条件特征映射，服务图像复原，不输出独立可采样的 barycenter 图像分布。

### 8.6 Tang et al., 2026: BaryIR 在图像复原中的任务化改造

BaryIR 的出发点是 all-in-one image restoration 的泛化问题：统一模型虽然能处理多种训练退化，但面对未知退化类型、未知退化强度或真实混合退化时容易过拟合训练域。作者的假设是：多源退化特征分布可以看作由一个退化无关的底层内容分布经过不同退化特异偏移形成；因此，恢复这个共享分布有助于提升泛化。

方法上，BaryIR 把编码器得到的 `source_latent` 分解成两个空间：`bary_latent` 对应 WB space，负责退化无关共享内容；$\mathrm{res\_bary} = \mathrm{source\_latent} - \mathrm{bary\_latent}$ 对应 residual subspace，负责退化特异信息。训练中，`Pots` 为不同退化源域设置 potential head，对 `bary_latent` 施加分布级约束；`ort_loss` 让 `bary_latent` 和 `res_bary` 正交；`contra_loss` 让同域 residual 更接近、异域 residual 更分开；重建 L1 loss 保证输出图像质量。

和前面几篇 barycenter 计算论文相比，BaryIR 的重点不是提出一个通用 barycenter solver，而是把 Wasserstein barycenter 表示学习嵌入 restoration 网络。它的 `Pots` 不是严格 ICNN，也不恢复 transport plan；它的 `barylatent` 不是无条件 generator，而是条件化 latent projector。其创新点更像是“WB latent decomposition for restoration generalization”：把连续 barycenter / dual potential 的思想变成一个可端到端训练的图像复原表示学习模块。

### 8.7 这些论文之间的演进关系

可以按如下脉络理解：

- Agueh & Carlier 2011：定义并证明 Wasserstein barycenter 的基本理论。
- Li et al. 2020：用 regularized dual potentials 隐式表示连续 barycenter，避免固定离散 support。
- Fan et al. 2021：用 ICNN 表示 W2 凸势函数，并用 generator 表示可采样 barycenter。
- Korotin et al. 2021：用 ICNN + cycle/congruence regularization，避免 minimax 和 regularization bias。
- Korotin et al. 2022：放宽 ICNN 限制，用普通神经网络和 fixed-point iteration 训练 barycenter generator。
- BaryIR 2026：不求通用 barycenter generator，而是在图像复原 latent space 中学习隐式 WB 表示和 residual 表示。

因此，BaryIR 的 `barylatent + Pots` 可以被看作对前述连续 barycenter 思想的任务化、工程化使用：它吸收了“隐式 barycenter 分布”和“dual potential 约束”的思想，但没有严格采用 ICNN、显式 transport map 或生成式 fixed-point solver。
