# STOnco GRL 线性 Warm-up 四参数改造方案

本文档基于当前 STOnco 项目中的真实实现，提出一个只涉及 GRL beta 调度方式的更新方案：

- 保留现有 `constant` / `dann` 两种模式
- 新增一个线性 warm-up 模式
- 采用 4 个新参数，分别控制 slide 和 cancer 两路对抗的 `delay` 与 `warmup`
- 其中 `delay` 同时作用于 `linear` 和 `dann`
- 对 `dann` 的具体定义改为：前面完全不开对抗，`delay` 之后在剩余训练过程上重新归一化，走完整的 DANN 曲线

本文档先用于确认方案，不直接改代码。请先看完文末的“待确认问题”。

---

## 1. 当前代码情况回顾

### 1.1 当前支持的 GRL 模式

当前 `train.py` 和 `train_hpo.py` 里，`grl_beta_mode` 只支持两种：

- `dann`
- `constant`

对应位置：

- `stonco/core/train.py`
- `stonco/core/train_hpo.py`

在 `train.py` 中：

- CLI 参数定义：`train.py:244-247`
- cfg 默认值：`train.py:307-310`
- cfg 合法性检查：`train.py:450-459`
- 训练时实际使用：`train.py:707-709`, `train.py:1028-1035`

在 `train_hpo.py` 中：

- CLI 参数定义：`train_hpo.py:589-592`
- cfg 默认值：`train_hpo.py:609-612`
- cfg 合法性检查：`train_hpo.py:682-691`

说明：

- 当前代码状态里，`train.py` 和 `train_hpo.py` 都有 GRL 相关参数入口
- 但本次方案的实现范围只覆盖 `train.py`
- `train_hpo.py` 本轮不改

### 1.2 当前 DANN 公式

当前 `train.py` 中的 helper 为：

```python
def _dann_beta(p, beta_target, gamma):
    return beta_target * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
```

位置：

- `stonco/core/train.py:701-705`

### 1.3 当前 `p` 的定义方式

当前 `dann` 模式下：

```python
total_steps = int(cfg['epochs']) * max(1, len(train_loader))
p = float(global_step) / float(total_steps)
```

位置：

- `stonco/core/train.py:1002-1003`
- `stonco/core/train.py:1032`

这意味着：

- beta 的增长速度直接依赖 `epochs * len(train_loader)`
- 同一个 `gamma=10`，换一个数据划分、batch size、LOCO cancer type，beta 曲线都会变
- 用户无法直接指定“前多少个 step 拉满”

### 1.4 当前 `global_step` 的时序

当前 `global_step` 是在前向和 loss 计算之前读出，在本 batch 结束后才 `+1`：

```python
if grl_mode == 'constant':
    ...
else:
    p = float(global_step) / float(total_steps)
    ...
out = model(...)
global_step += 1
```

位置：

- `stonco/core/train.py:1025-1035`
- `stonco/core/train.py:1045`

所以：

- 第 1 个 batch 使用的 `global_step=0`
- 第 10 个 batch 使用的实际上是 `global_step=9`

这个细节会影响“delay / warmup 到底在哪个 batch 生效”的定义。

### 1.5 当前 slide / cancer 两路已经分开 target，但没有分开调度窗口

当前已有：

- `grl_beta_slide_target`
- `grl_beta_cancer_target`

但没有：

- slide 单独的 delay
- slide 单独的 warmup
- cancer 单独的 delay
- cancer 单独的 warmup

因此两路虽然 target 不同，但时间调度完全共用同一个全局训练进度。

---

## 2. 这次改造的目标

本次改造只聚焦一个问题：

把当前“依赖总训练步数的 DANN beta 增长”补充为“显式、可控、与 `total_steps` 解耦、用户按 epoch 指定的线性 warm-up”。

目标是：

1. 让用户直接控制每一路域对抗何时开始
2. 让用户直接控制每一路域对抗延迟多少个 epoch 开始、再用多少个 epoch 拉到 target
3. 保持 `constant` 行为不变
4. 让 `dann` 支持显式的 `delay`
5. 先补充新的线性模式，不动 loss 结构、不动 model 结构

---

## 3. 建议方案：新增 `linear` 模式

### 3.1 模式设计

建议把 `grl_beta_mode` 从：

- `dann | constant`

扩展为：

- `dann | constant | linear`

含义：

- `constant`：全程固定为 `beta_target`
- `dann`：新增 `delay` 前缀；`delay` 之后在剩余训练过程上重新归一化，走完整 DANN 曲线
- `linear`：新增线性 warm-up 调度；`delay` 之后按线性方式升到 target

这样兼容性最好：

- `constant` 完全不受影响
- `linear` 与 `dann` 共用同一套 `delay` 参数语义
- 如果用户希望复现旧的 DANN 无延迟行为，只需把两路 `delay_epochs` 设为 `0`

---

## 4. 四个新参数

建议新增以下 4 个参数：

- `grl_beta_slide_delay_epochs`
- `grl_beta_slide_warmup_epochs`
- `grl_beta_cancer_delay_epochs`
- `grl_beta_cancer_warmup_epochs`

### 4.1 参数语义

`grl_beta_slide_delay_epochs`

- slide 域对抗开始前，保持 `beta_slide=0` 的 epoch 数
- 对 `linear` 和 `dann` 都生效

`grl_beta_slide_warmup_epochs`

- slide 域对抗从 0 线性涨到 `grl_beta_slide_target` 所用的 epoch 数
- 只对 `linear` 生效，`dann` 下忽略

`grl_beta_cancer_delay_epochs`

- cancer 域对抗开始前，保持 `beta_cancer=0` 的 epoch 数
- 对 `linear` 和 `dann` 都生效

`grl_beta_cancer_warmup_epochs`

- cancer 域对抗从 0 线性涨到 `grl_beta_cancer_target` 所用的 epoch 数
- 只对 `linear` 生效，`dann` 下忽略

补充说明：

- 用户侧按 epoch 配置
- 训练开始后，基于当前 `len(train_loader)` 自动换算成 step
- 真正进入 beta 调度计算的仍然是换算后的 `delay_steps` / `warmup_steps`

### 4.2 为什么是 4 个，不是 1 个或 3 个

本项目当前已经把两路 target 分开：

- `grl_beta_slide_target`
- `grl_beta_cancer_target`

所以继续把两路的时间调度也分开，是更一致的设计。

这样能表达下面这种更符合 STOnco 现状的需求：

- slide 对抗可以更早开始，但缓慢升温
- cancer 对抗可以更晚开始，避免太早抹掉癌种相关判别信息

---

## 5. `linear` 与 `dann` 的调度定义

### 5.1 公共换算

先在训练开始后定义：

```text
steps_per_epoch = max(1, len(train_loader))
delay_steps = delay_epochs * steps_per_epoch
warmup_steps = warmup_epochs * steps_per_epoch
```

### 5.2 `linear` 的定义

再对某一路域对抗定义：

```text
step_eff = max(global_step - delay_steps, 0)
progress = min(step_eff / warmup_steps, 1.0)
beta = beta_target * progress
```

为了处理边界，建议补充两条规则：

```text
if global_step < delay_steps:
    beta = 0

elif warmup_steps <= 0:
    beta = beta_target

else:
    step_eff = global_step - delay_steps
    progress = min(step_eff / warmup_steps, 1.0)
    beta = beta_target * progress
```

### 5.3 `linear` 的直观含义

如果：

- `steps_per_epoch = 20`
- `delay_epochs = 5`
- `warmup_epochs = 10`

那么自动换算后：

- `delay_steps = 100`
- `warmup_steps = 200`

那么：

- `global_step < 100` 时，beta 为 0
- `global_step = 100` 时，beta 刚开始，仍然是 0
- `global_step = 200` 时，beta 到 target 的 50%
- `global_step = 300` 时，beta 到 target
- `global_step > 300` 后，beta 保持在 target

### 5.4 `dann` 的 delay 版本定义

对某一路 `dann` 分支，建议使用下面的形式：

```text
if global_step < delay_steps:
    beta = 0

else:
    rem_steps = max(total_steps - delay_steps, 1)
    p = min((global_step - delay_steps) / rem_steps, 1.0)
    beta = beta_target * (2 / (1 + exp(-gamma * p)) - 1)
```

直观含义：

- `delay` 之前完全不开对抗
- `delay` 开始时，`p=0`，DANN 曲线从起点重新开始
- 之后在“剩余训练过程”上走完完整的 sigmoid 曲线

这样可以避免一种不自然的实现：

- 前面先把若干 step 强行置零
- 后面却继续沿用旧的 `p = global_step / total_steps`

后一种做法会导致：

- 名义上有 delay
- 实际上 DANN 曲线已经偷偷走过一段
- `delay` 之后不是从真正的起点进入对抗

### 5.5 与当前 DANN 的本质区别

当前 DANN：

- beta 的变化速度依赖 `total_steps`
- 很难从命令行直接控制“前多少个训练阶段拉满”

新的 `linear` / `delay-aware dann`：

- 用户直接指定 `delay_epochs` 和 `warmup_epochs`
- 内部自动换算为当前运行的 step 数
- 对 LOCO 更直观，因为用户仍然按 epoch 思考
- 对实现层仍然保持 step 级控制
- 其中 `dann` 的 `delay` 会在剩余训练过程上重新归一化，而不是简单置零前缀

---

## 6. 推荐 helper 形式

建议在 `train.py` 的 `train_and_validate(...)` 内，新增一个 helper，例如：

```python
def _epochs_to_steps(value_epochs, steps_per_epoch):
    ...
```

以及：

```python
def _linear_warmup_beta(global_step, beta_target, delay_steps, warmup_steps):
    ...
```

以及：

```python
def _dann_beta_with_delay(global_step, total_steps, beta_target, gamma, delay_steps):
    ...
```

建议输入输出关系：

- 输入：
  - `global_step`
  - `beta_target`
  - `delay_steps`
  - `warmup_steps`
- 输出：
  - 当前 step 应使用的 beta

`_dann_beta_with_delay(...)` 建议输入：

- `global_step`
- `total_steps`
- `beta_target`
- `gamma`
- `delay_steps`

建议做法：

- 在 `train_loader` 构建完成后先得到 `steps_per_epoch`
- 把 4 个 `*_epochs` 配置先换算成对应的 `*_steps`
- 训练循环里只使用换算后的 step 变量

建议保持与 `_dann_beta(...)` 同级，方便训练循环根据 `grl_beta_mode` 分支调用。

---

## 7. 对训练循环的建议更新方式

当前训练循环在每个 batch 中这样选择 beta：

- `constant`：直接取 target
- `dann`：按 `global_step / total_steps` 算

建议更新为三路分支：

```text
if grl_mode == 'constant':
    slide_beta = slide_target
    cancer_beta = cancer_target

elif grl_mode == 'dann':
    先用当前 len(train_loader) 把两路 *_delay_epochs 换算成 *_delay_steps
    slide_beta = dann_with_delay(...)
    cancer_beta = dann_with_delay(...)

elif grl_mode == 'linear':
    先用当前 len(train_loader) 把 *_epochs 换算成 *_steps
    slide_beta = linear_warmup(...)
    cancer_beta = linear_warmup(...)
```

这样可以做到：

- `constant` 完全不变
- `dann` 和 `linear` 对 `delay` 的语义一致
- 新实验只需切换 mode

---

## 8. 对 CLI / cfg 的建议更新

### 8.1 `train.py`

建议在现有参数旁边新增：

- `--grl_beta_slide_delay_epochs`
- `--grl_beta_slide_warmup_epochs`
- `--grl_beta_cancer_delay_epochs`
- `--grl_beta_cancer_warmup_epochs`

并把 `--grl_beta_mode` 的候选从：

- `['dann', 'constant']`

扩成：

- `['dann', 'constant', 'linear']`

### 8.2 本轮范围说明

本轮只更新：

- `train.py`

本轮不更新：

- `train_hpo.py`

也就是说，这次方案落地后会暂时出现：

- `train.py` 支持 `linear`
- `train_hpo.py` 仍只支持现有 `dann/constant`

这是本轮明确接受的范围边界。

### 8.3 cfg 默认值建议

建议把下面这组值作为新调度方案下的推荐默认值：

```text
grl_beta_slide_delay_epochs = 1
grl_beta_slide_warmup_epochs = 8
grl_beta_cancer_delay_epochs = 3
grl_beta_cancer_warmup_epochs = 12
```

含义：

- `constant` 下，这 4 个值都不会生效
- `dann` 下，只有两路 `delay_epochs` 生效，`warmup_epochs` 忽略
- `linear` 下，4 个值全部生效
- slide 对抗延后 1 个 epoch 启动，并在 `linear` 下用 8 个 epoch 线性升温
- cancer 对抗延后 3 个 epoch 启动，并在 `linear` 下用 12 个 epoch 更缓慢地升温

推荐这组默认值的原因：

- 当前训练集共有 11 个 cancer 域、25 个 batch 域，而 batch 域里有不少单切片或极小样本 batch
- slide/batch 更像技术域，应早于 cancer 介入，但不能升温过快
- cancer 域与主任务信号更容易纠缠，适合更晚、更慢地升温
- 这组值仍属于前中期介入，不会把对抗启动推得过晚

补充说明：

- 这组值首先是面向 `linear` 模式设计的
- 但如果你选择 `dann`，其中两路 `delay_epochs` 也会生效
- 如果你想让 `linear` 更接近当前 `constant` 行为，可以把 4 个参数都设成 `0`
- 如果你想严格复现旧的 DANN 无延迟行为，需要把：
  - `grl_beta_slide_delay_epochs = 0`
  - `grl_beta_cancer_delay_epochs = 0`

---

## 9. 与现有 `grl_beta_gamma` 的关系

在线性 warm-up 模式下，`grl_beta_gamma` 不再参与计算。

建议文档和代码都明确：

- `constant` 使用：`target`
- `dann` 使用：`target + gamma + delay_epochs`
- `linear` 使用：`target + delay_epochs + warmup_epochs`

也就是说：

- `grl_beta_gamma` 在 `linear` 模式下应被忽略

这个行为需要在 help 文案中写清楚，否则很容易让用户误以为 `gamma` 仍然有效。

---

## 10. backward compatibility 建议

建议遵循以下兼容原则：

1. `constant` 行为完全不变
2. `dann` 新增 `delay` 语义，但仅当对应 `delay_epochs > 0` 时改变行为
3. `dann` 读取两路 `delay_epochs`，忽略两路 `warmup_epochs`
4. `linear` 读取 4 个新参数
5. 老的配置文件如果没有这 4 个字段，不报错，自动走默认值
6. `meta.json` 中照常保存新字段，保证实验可追溯

---

## 11. 建议的日志与文档更新范围

如果后续确认并开始改代码，建议至少同步更新：

- `stonco/core/train.py`
- 相关 CLI help 文案
- 如有需要，补充到 `docs/Tutorial.md` 或新的训练说明文档

本次改动不要求修改：

- model 结构
- loss 结构
- 域头结构
- MMD 结构

---

## 12. 预期收益

如果按该方案落地，主要收益是：

1. slide / cancer 两路 GRL 调度从“绑在总训练进度上”改成“用户显式可控”
2. 便于把 cancer 对抗延后，减少过早损伤主任务表示的风险
3. `dann` 的 delay 语义也与 `linear` 对齐，减少两种模式之间的理解断层
4. 便于后续做更系统的 ablation
5. 训练行为在不同 LOCO cancer type / batch size 下更一致

---

## 13. 已确认事项

以下事项已确认，本次方案按此执行：

1. `grl_beta_mode` 保留 `constant` / `dann`，并新增 `linear`
2. 用户侧参数按 `epoch` 配置，训练开始后基于当前 `len(train_loader)` 自动换算成 `step`
3. `*_epochs` 仅接受整数，不接受小数 epoch
4. 线性 warm-up 的边界定义采用：
   - `global_step < delay_steps -> beta = 0`
   - `global_step = delay_steps -> beta = 0`
   - `global_step = delay_steps + warmup_steps -> beta = target`
5. `dann` 模式下两路 `delay_epochs` 也生效，且采用“delay 之后在剩余训练过程上重新归一化并走完整 DANN 曲线”的定义
6. `warmup_steps <= 0` 时，采用“delay 结束后立即跳到 target”
7. `linear` 模式下直接忽略 `grl_beta_gamma`，不报错，但文档和 help 要写明其不生效
8. 本轮不修改 `train_hpo.py`

---

## 14. 推荐默认值

如果后续把这 4 个参数正式接入代码，建议默认值采用：

```text
grl_beta_slide_delay_epochs = 1
grl_beta_slide_warmup_epochs = 8
grl_beta_cancer_delay_epochs = 3
grl_beta_cancer_warmup_epochs = 12
```

对应解释：

- slide：
  - 第 1 个 epoch 后开始介入
  - 用 8 个 epoch 线性涨到 `grl_beta_slide_target`
- cancer：
  - 前 3 个 epoch 不对 encoder 施加 cancer 对抗
  - 从第 4 个 epoch 开始，用 12 个 epoch 线性涨到 `grl_beta_cancer_target`

这组默认值的设计意图是：

- slide/batch 技术域先介入，尽早抑制明显的技术偏移
- cancer 域晚一点启动，降低过早损伤主任务判别信号的风险
- cancer 的 warm-up 更长，让共享表示有更平滑的适应过程
- 如果用户选择 `dann`，这组默认值也意味着：
  - slide DANN 从第 2 个 epoch 左右开始进入 sigmoid 曲线
  - cancer DANN 从第 4 个 epoch 左右开始进入 sigmoid 曲线

如果后续实验发现：

- 主任务前期仍被明显扰动
  - 优先进一步增大 `grl_beta_cancer_delay_epochs`
- batch effect 仍然很重
  - 优先增大 `grl_beta_slide_target` 或缩短 `grl_beta_slide_warmup_epochs`

---

## 15. 推荐的初始使用方式

在你确认方案并开始改代码后，我建议第一批实验先从下面这类设置开始：

- slide：
  - `target` 保持较低
  - `delay=0`
  - `warmup` 中等长度
- cancer：
  - `target` 更保守
  - `delay` 晚于 slide
  - `warmup` 长于 slide

原因：

- slide 更像技术域，通常可以更早介入
- cancer 更容易和主任务判别信号纠缠，应该更晚、更慢

这一部分暂时只作为训练策略建议，不在本次文档里固化成默认参数。

---

## 16. 本文档的结论

建议采用以下方案进入代码实现阶段：

1. 保留 `constant` / `dann`
2. 新增 `linear` 模式
3. 新增 4 个参数：
   - `grl_beta_slide_delay_epochs`
   - `grl_beta_slide_warmup_epochs`
   - `grl_beta_cancer_delay_epochs`
   - `grl_beta_cancer_warmup_epochs`
4. 用户侧按 epoch 配置，内部基于当前 `len(train_loader)` 换算成 steps
5. 推荐默认值为：
   - `slide_delay_epochs = 1`
   - `slide_warmup_epochs = 8`
   - `cancer_delay_epochs = 3`
   - `cancer_warmup_epochs = 12`
6. `delay_epochs` 同时作用于 `linear` 和 `dann`，其中 `dann` 在 delay 后对剩余训练过程重新归一化并走完整曲线
7. `gamma` 在 `linear` 下忽略
8. 本轮只更新 `train.py`

文档层面的方案已确认完成；后续若开始实现，按本方案更新 `train.py` 即可。
