# STOnco `GNN_hidden` 列表化支持方案

本文基于当前实现，给出一份仅涉及设计与改造范围说明的方案文档，**暂不修改代码**。目标是在保持旧配置可用的前提下，为 GNN 主干新增 `GNN_hidden` 列表输入能力，使不同层可使用不同 hidden 宽度，例如 3 层时可写为 `[128, 45, 64]`。

---

## 目标与已确认决策

### 目标

1. 支持为每一层单独指定 hidden 宽度，而不是当前全层共享一个标量 `hidden`。
2. 允许 hidden 宽度为任意正整数，不要求是 32 的倍数。
3. 当 `model='gatv2'` 时，每层实际输出维度按 `GNN_hidden[i] * heads` 计算。
4. 当 `model in {'gcn', 'sage'}` 时，每层实际输出维度直接使用 `GNN_hidden[i]`。

### 已确认决策

1. 外部新增参数名：`GNN_hidden`
2. 若同时传入旧参数 `hidden` 和新参数 `GNN_hidden`：直接报错
3. `HPO` 这一轮保持现状，仅继续使用标量 `hidden`
4. 新配置以 `GNN_hidden` 为准
5. 当前代码中的层数字段统一按现有命名 `num_layers` 处理

---

## 当前实现现状

### 现状 1：主干网络只支持标量 `hidden`

当前 `GNNBackbone` 在循环构建每一层时，始终复用同一个 `hidden` 值：

- `stonco/core/models.py`
- `GNNBackbone.__init__`

现状行为：

1. `gatv2`：每层都执行 `GATv2Conv(dim_prev, hidden, heads=heads, concat=True)`，输出维度恒为 `hidden * heads`
2. `gcn`：每层输出维度恒为 `hidden`
3. `sage`：每层输出维度恒为 `hidden`

因此当前结构无法表达类似 `[128, 45, 64]` 这样的逐层配置。

### 现状 2：训练/推理/导出链路都默认 `hidden` 是标量

当前以下入口均把 `hidden` 当作单个整数使用：

1. `stonco/core/train.py`
2. `stonco/core/infer.py`
3. `stonco/utils/export_spot_embeddings.py`
4. `stonco/utils/visualize_prediction.py`

这意味着本次改造不能只改模型定义，必须同步覆盖配置解析和下游加载逻辑。

### 现状 3：HPO 的搜索空间也是标量 `hidden`

`stonco/core/train_hpo.py` 当前的 stage2 搜索空间中，`hidden` 仍是单值类别搜索：

1. `hidden ∈ {64, 128, 192, 256}`
2. `num_layers ∈ [2, 5]`

本方案确认：这一轮不扩展 HPO 到列表 hidden。

---

## 参数语义设计

### 新参数：`GNN_hidden`

新增 `GNN_hidden` 作为 GNN 主干宽度配置的主入口，支持以下输入语义：

1. 标量
   - 示例：`128`
   - 含义：所有层共享同一 hidden 宽度

2. 列表
   - 示例：`[128, 45, 64]`
   - 含义：每一层使用对应位置的 hidden 宽度

3. CLI 字符串
   - 示例：`--GNN_hidden 128,45,64`
   - 含义：按逗号分隔解析为整数列表

### 与旧参数 `hidden` 的关系

1. 旧参数 `hidden` 仅作为向后兼容读取入口保留
2. 新训练产物和新配置逻辑以 `GNN_hidden` 为主
3. 若合并后的配置中同时出现 `hidden` 和 `GNN_hidden`：直接报错，不做优先级覆盖

### `num_layers` 与 `GNN_hidden` 的关系

统一按以下规则规范化：

1. `GNN_hidden` 是标量时
   - 依赖 `num_layers`
   - 扩展为长度等于 `num_layers` 的列表
   - 示例：`GNN_hidden=128, num_layers=3 -> [128, 128, 128]`

2. `GNN_hidden` 是列表时
   - 默认以列表长度作为有效 `num_layers`
   - 示例：`GNN_hidden=[128,45,64] -> num_layers=3`

3. 若用户显式传入了 `num_layers`，且其值与 `len(GNN_hidden)` 不一致
   - 直接报错
   - 不自动裁剪
   - 不自动补齐
   - 不静默容错

### 合法性校验

`GNN_hidden` 规范化时建议增加以下校验：

1. 必须为整数或整数列表
2. 列表不能为空
3. 每个元素必须为正整数
4. `num_layers` 必须为正整数

---

## 模型构建方案

### 总体思路

将当前“单个 `hidden` 贯穿所有层”的构建方式，改为“先把 `GNN_hidden` 规范化为 `hidden_list`，再按层构建”。

### `GNNBackbone` 的目标行为

设：

1. `GNN_hidden = [128, 45, 64]`
2. `num_layers = 3`
3. `heads = 4`
4. `model = 'gatv2'`

则每层应按以下方式构建：

1. 第 1 层
   - `GATv2Conv(dim_prev, 128, heads=4, concat=True)`
   - 输出维度：`128 * 4 = 512`

2. 第 2 层
   - `GATv2Conv(512, 45, heads=4, concat=True)`
   - 输出维度：`45 * 4 = 180`

3. 第 3 层
   - `GATv2Conv(180, 64, heads=4, concat=True)`
   - 输出维度：`64 * 4 = 256`

最终：

1. `self.out_dim = 256`
2. 分类头和域头继续以 `self.gnn.out_dim` 作为输入维度

### 对非 GATv2 模型的行为

若：

1. `model='gcn'`
2. 或 `model='sage'`

则每层输出维度直接等于该层 `hidden_list[i]`，不乘 `heads`。

---

## 配置流改造方案

### A. 训练入口

涉及文件：

1. `stonco/core/train.py`

计划改造：

1. 新增 CLI 参数 `--GNN_hidden`
2. 保留旧参数 `--hidden` 作为兼容输入
3. 在配置合并完成后，统一执行一次规范化
4. 规范化产出：
   - `cfg['GNN_hidden']`: 最终层宽列表
   - `cfg['num_layers']`: 最终有效层数
5. 若 `hidden` 与 `GNN_hidden` 同时出现：直接抛错

### B. 推理入口

涉及文件：

1. `stonco/core/infer.py`

计划改造：

1. 读取 `meta.json` 时优先识别 `GNN_hidden`
2. 兼容旧模型仅存在 `hidden` 的情况
3. 同样通过统一的规范化逻辑得到最终 `GNN_hidden` 列表与有效 `num_layers`

### C. embedding 导出与可视化

涉及文件：

1. `stonco/utils/export_spot_embeddings.py`
2. `stonco/utils/visualize_prediction.py`

计划改造：

1. 和推理入口保持同一套兼容逻辑
2. 禁止这些工具各自写一套局部解析规则
3. 统一依赖一个公共的 hidden 规范化函数，避免行为漂移

### D. 配置持久化

训练输出的 `meta.json` 建议采用以下原则：

1. 新配置以 `GNN_hidden` 为主字段保存
2. 保存规范化后的 `num_layers`
3. 不再把 `hidden` 作为主配置字段写回
4. 加载旧 `meta.json` 时仍兼容读取 `hidden`

这样可以保证：

1. 新模型的结构信息表达完整
2. 推理阶段不会因 `num_layers` 和实际层宽列表脱节而歧义

---

## 建议的规范化入口

建议新增一个统一的辅助函数，负责处理以下问题：

1. 识别标量 / 列表 / 逗号分隔字符串
2. 检查 `hidden` 与 `GNN_hidden` 是否同时出现
3. 处理 `num_layers` 与列表长度的一致性
4. 返回最终 `hidden_list` 与有效 `num_layers`

建议输出形式：

```python
hidden_list, effective_num_layers = normalize_gnn_hidden(...)
```

建议该函数在以下场景复用：

1. `train.py`
2. `infer.py`
3. `export_spot_embeddings.py`
4. `visualize_prediction.py`
5. 模型初始化前的任何配置加载路径

这样可以避免：

1. 某些入口接受字符串列表，另一些入口不接受
2. 某些入口自动容错，另一些入口直接报错
3. 新旧模型在训练与推理时结构不一致

---

## HPO 范围控制

本轮明确不扩展 `HPO` 到列表 hidden。

### 保持现状

`stonco/core/train_hpo.py` 中：

1. `hidden` 继续是标量搜索空间
2. `num_layers` 继续独立搜索
3. `heads` 对 `gatv2` 继续独立搜索

### 原因

1. 列表化 hidden 会显著扩大搜索空间
2. 还会引入“层宽列表长度”和 `num_layers` 的联合约束
3. 当前这轮需求重点是手动配置与推理兼容，不是 HPO 结构升级

### 兼容建议

HPO 生成的配置仍然可以沿用旧风格：

1. `hidden=128`
2. `num_layers=3`

进入训练主流程后，可由规范化逻辑自动扩成：

1. `GNN_hidden=[128,128,128]`

---

## 错误处理策略

以下情况直接报错，不做自动容错：

1. 同时传入 `hidden` 和 `GNN_hidden`
2. `GNN_hidden` 列表为空
3. `GNN_hidden` 中含非整数
4. `GNN_hidden` 中含非正数
5. `num_layers <= 0`
6. `GNN_hidden` 为列表，且用户显式传入的 `num_layers != len(GNN_hidden)`
7. `GNN_hidden` 为标量，但缺失可用的 `num_layers`

这样做的目的是让网络结构定义保持显式、可复现、无歧义。

---

## 验证方案

### 基础用例

1. 标量输入
   - 输入：`GNN_hidden=128, num_layers=3`
   - 期望：规范化为 `[128,128,128]`

2. 列表输入
   - 输入：`GNN_hidden=[128,45,64]`
   - 期望：有效 `num_layers=3`

3. 列表 + 一致层数
   - 输入：`GNN_hidden=[128,45,64], num_layers=3`
   - 期望：通过

4. 列表 + 不一致层数
   - 输入：`GNN_hidden=[128,45,64], num_layers=2`
   - 期望：直接报错

5. 新旧参数同时存在
   - 输入：`hidden=128, GNN_hidden=[128,45,64]`
   - 期望：直接报错

### 结构维度用例

1. `model='gatv2', heads=4, GNN_hidden=[128,45,64]`
   - 期望层输出维度：`512 -> 180 -> 256`

2. `model='gcn', GNN_hidden=[128,45,64]`
   - 期望层输出维度：`128 -> 45 -> 64`

3. `model='sage', GNN_hidden=[128,45,64]`
   - 期望层输出维度：`128 -> 45 -> 64`

### 兼容性用例

1. 旧 `meta.json` 仅有 `hidden=128`
   - 期望：推理仍正常

2. 新 `meta.json` 仅有 `GNN_hidden=[128,45,64]`
   - 期望：推理仍正常

3. `export_spot_embeddings.py` 与 `visualize_prediction.py`
   - 期望：可正确加载新旧两类配置

---

## 影响范围

本方案预计影响以下模块：

1. `stonco/core/models.py`
2. `stonco/core/train.py`
3. `stonco/core/infer.py`
4. `stonco/utils/export_spot_embeddings.py`
5. `stonco/utils/visualize_prediction.py`

可能需要补充更新但不一定作为首轮必改项：

1. `README.md`
2. `docs/Tutorial.md`
3. 任何直接展示 `hidden`/`num_layers` 关系的说明文档

---

## 待确认问题

当前没有阻塞性未决项；若开始实现，默认按本文执行。

仍建议在正式改代码前最后确认以下两个实现细节：

1. `--GNN_hidden` 的 CLI 形式是否只支持逗号分隔字符串，如 `128,45,64`
   - 本方案默认：是

2. 新训练产物 `meta.json` 是否完全不再写回旧字段 `hidden`
   - 本方案默认：是，只保留 `GNN_hidden` 与规范化后的 `num_layers`

---

## 实施顺序建议

1. 先补统一的 `GNN_hidden` 规范化函数
2. 再改 `GNNBackbone`，让其按层构建
3. 接着改训练入口配置合并逻辑
4. 然后改推理、embedding 导出、可视化的配置读取逻辑
5. 最后补最小验证用例，确保新旧配置都能通过

这样可以把“结构定义”和“配置兼容”拆开处理，降低回归风险。
