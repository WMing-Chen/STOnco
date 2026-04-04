# PyGDA 仓库导读

## 文档目的

这份文档不是对 `pygda-team/pygda` 全仓库的逐字转录，而是一个后续精读前的导读提纲。
目标是先快速明确：

- 这个项目解决什么问题
- GitHub 首页已经暴露了哪些关键信息
- 后续应该按什么顺序继续读
- 哪些内容对 STOnco 的域自适应设计最值得重点关注

---

## 1. 项目一句话总结

PyGDA 是一个基于 **PyTorch** 和 **PyTorch Geometric (PyG)** 的 **图域自适应（Graph Domain Adaptation, GDA）** Python 库，主打：

- 以接近 `sklearn` 的方式调用图域自适应模型
- 统一封装 20+ 个 GDA 模型
- 支持节点级、图级、source-free、multi-source-free 等多种设置
- 提供数据、模型、评估和示例的相对完整工具链

GitHub 首页把它定位成一个“可直接训练和评估图域自适应模型的通用库”，而不是单篇论文代码。

---

## 2. 首页已经能确认的核心信息

### 2.1 核心定位

首页 README 明确写到：

- 项目名：`pygda`
- 组织：`pygda-team`
- 代码仓库：<https://github.com/pygda-team/pygda>
- 官方文档：<https://pygda.readthedocs.io/en/stable/>

其核心卖点包括：

- API 风格统一
- 文档相对完善
- 覆盖 20+ 图域自适应模型
- 支持大图上的 mini-batch / sampling
- 与 PyG 数据结构兼容

### 2.2 使用方式非常直接

README 首页强调的调用范式非常简单：

1. 从 `pygda.models` 导入模型
2. 初始化模型
3. 调 `fit(source_data, target_data)`
4. 调 `predict(target_data)` 获取结果

这说明它非常适合作为：

- 基线复现工具
- 不同 GDA 方法的统一实验入口
- 后续迁移到自定义图任务时的参考接口

### 2.3 已支持的任务设置

从首页更新日志可见，PyGDA 不只覆盖最传统的无监督节点级 GDA，还逐步扩展到：

- 图级 domain adaptation
- source-free GDA
- multi-source-free GDA
- 更新到 2025 年前后的较新方法

这点很重要，因为它说明仓库不只是“老方法集合”，而是在持续补齐更接近现实部署的域迁移设置。

---

## 3. README 中最值得记住的内容

### 3.1 安装依赖有一个关键前提

PyGDA **不会自动帮你安装底层图学习依赖**。README 明确要求用户自行准备：

- `torch`
- `torch_geometric`
- `torch_sparse`
- `torch_scatter`
- 以及常见科学计算依赖如 `numpy`、`scipy`、`sklearn`、`cvxpy`、`tqdm`

也就是说，后续如果要在本地真正跑通 PyGDA，环境兼容性会是第一道门槛。

### 3.2 Quick Start 反映了它的标准工作流

首页 Quick Start 很清楚地给出 4 步：

1. 加载 source / target 数据集
2. 构建模型
3. 训练 `fit`
4. 用 `predict` 和指标函数评估

这说明 PyGDA 的抽象大概分成三层：

- `datasets`
- `models`
- `metrics`

后续读源码时，可以优先围绕这三层理解仓库设计。

### 3.3 支持自定义模型扩展

README 明确说，用户如果想实现自己的 GDA 方法，需要：

- 继承 `BaseGDA`
- 实现 `fit()`
- 实现 `forward_model()`
- 实现 `predict()`

这对后续很关键，因为它直接告诉我们：

- PyGDA 有统一的模型抽象基类
- 二次开发入口很可能就在 `BaseGDA`
- 如果 STOnco 后面想借鉴其接口，最该先读的不是全部模型，而是基础抽象层

---

## 4. 仓库结构的初步理解

GitHub 首页文件树已经暴露出几个关键目录：

- `.github/workflows`：CI 或发布流程
- `benchmark`：基准实验相关内容
- `data`：数据资源或示例数据
- `docs`：文档站点源文件
- `examples`：用法示例
- `pygda`：核心源码包
- `pyproject.toml`：打包与依赖定义
- `mkdocs.yml`：说明文档站点基于 MkDocs

对后续阅读来说，可以把它理解为：

- `README` 负责“项目入口”
- `docs` 负责“系统说明”
- `examples` 负责“最短落地路径”
- `pygda` 负责“真正的实现逻辑”
- `benchmark` 负责“实验体系和方法比较”

---

## 5. 首页列出的模型范围

README 的参考表基本等于一个“已纳入方法清单”。从首页可直接看到，仓库至少覆盖以下代表性方法：

- DANE
- ACDNE
- UDAGCN
- ASN
- AdaGCN
- GRADE
- SpecReg
- StruRW
- JHGDA
- KBL
- DMGNN
- CWGCN
- SAGDA
- GTrans
- DGDA
- A2GNN
- PairAlign
- SEPA
- SOGA
- GraphCTA
- TDSS
- GraphATA
- DGSDA

这说明 PyGDA 的价值不只是“一个模型”，而是一个相对系统的 GDA 方法集合。

如果后续目标是做方法比较、迁移实验或找适合 STOnco 的域适配思路，这个仓库很适合当作方法地图。

---

## 6. 对 STOnco 最相关的阅读关注点

结合 STOnco 当前的双域对抗框架，后续读 PyGDA 时建议优先盯住下面几个问题：

### 6.1 域定义方式

要重点看 PyGDA 里“source / target domain”在图任务中是怎么定义的：

- 不同图是不同域？
- 同一大图中的子图是不同域？
- 节点分类场景下标签与域标签如何组织？

这和 STOnco 中“癌种域 + batch 域”的双域设计能否映射，直接相关。

### 6.2 训练目标组成

重点看每种方法如何组合：

- 任务损失
- 域对齐损失
- 对抗损失
- 原型/谱/结构约束
- source-free 情况下的伪标签或自训练机制

这可以帮助判断 STOnco 后续是否值得从“单纯 DANN 风格 GRL”扩展到更结构化的图域对齐策略。

### 6.3 图级 vs 节点级任务

STOnco 当前本质上更接近“spot/node-level classification on graphs”。
所以后续读 PyGDA 时，优先级建议是：

- 先看节点级 GDA
- 再看 source-free 节点级 GDA
- 最后再看图级 GDA

除非后面要把“整张切片”视为图实例做图级迁移，否则图级部分可以晚一点读。

### 6.4 扩展接口

如果未来希望：

- 复用 PyGDA 的训练框架
- 借鉴其 model base class
- 对接自己的图数据

那么 `BaseGDA` 及其训练/预测接口会是最优先的源码入口。

---

## 7. 建议的后续精读顺序

建议下一轮按这个顺序读，而不是直接扎进所有模型文件：

1. `README.md`
   - 先把支持的任务类型、安装前提、Quick Start 和模型清单吃透。

2. `docs`
   - 看官方文档怎样组织 API、教程和任务设置。

3. `examples`
   - 找最接近“节点分类 + 域迁移”的最小可运行脚本。

4. `pygda` 包里的基础抽象
   - 优先找 `BaseGDA`、数据接口、评估接口。

5. 代表性模型
   - 先选 2 到 3 个模型读实现，不要一次全读。
   - 推荐优先：`A2GNN`、`UDAGCN`、`GraphCTA` 或 `GraphATA`。

6. `benchmark`
   - 最后再看实验配置、数据拆分和性能比较。

---

## 8. 下一轮阅读时建议重点回答的问题

后续真正精读 PyGDA 时，可以围绕下面这些问题做笔记：

- 它的数据输入最终要求是什么 PyG 对象或张量格式？
- source / target 图数据是如何封装与对齐的？
- `fit()` 内部是否统一处理训练循环、验证和 early stopping？
- `predict()` 返回的是什么，是否统一为 logits + labels？
- 不同模型之间复用的公共模块有哪些？
- source-free / multi-source-free 设置相比传统 UDA 多了哪些接口参数？
- benchmark 中使用了哪些图数据集与评价指标？
- 哪些方法最适合迁移到 STOnco 这种空间转录组图场景？

---

## 9. 结论

如果只看 GitHub 首页，可以先得出一个足够稳定的判断：

PyGDA 是一个面向 **图域自适应方法统一实现与实验** 的成熟工具库，核心价值在于：

- 方法覆盖面广
- 任务设置逐步扩展
- 接口统一
- 适合作为基线平台和二次开发起点

对 STOnco 来说，它最值得借鉴的不是某一个模型名字，而是：

- 它如何抽象 GDA 任务
- 它如何封装统一训练接口
- 它如何处理 source-free / multi-source-free 等更贴近真实迁移场景的问题

---

## 10. 来源与快照信息

- 访问页面：<https://github.com/pygda-team/pygda>
- 访问日期：2026-04-03
- 当时 GitHub 首页可见信息包括：
  - 仓库简介：PyGDA is a Python library for Graph Domain Adaptation
  - 代码主分支：`main`
  - 官方文档站点：`pygda.readthedocs.io/en/stable`
  - 最新 release（页面显示）：`v1.2.1`，日期为 2025-08-05

后续如果要继续深入，建议下一步直接读：

- README 全文
- `docs` 目录
- `examples` 目录
- `pygda` 包内的基础类与 2 到 3 个代表模型
