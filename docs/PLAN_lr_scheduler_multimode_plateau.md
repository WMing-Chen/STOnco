# STOnco 学习率调度方案确认稿（多模式 + ReduceLROnPlateau）

## 1. 背景与当前代码事实

基于当前 `stonco/core/train.py` 的真实实现，训练流程有以下几个关键事实：

1. 当前优化器只有一个：

```python
opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
```

2. 当前没有学习率 scheduler，`lr` 是全程固定值。
3. 当前已经存在按 `step` 计算的训练内调度基础：
   - `global_step`
   - `steps_per_epoch`
   - `total_steps`
   - GRL 的 `dann / constant / linear` 调度
4. `train_and_validate(...)` 是 single train / KFold / LOCO / train_hpo 共用的公共训练函数。
5. 当前 early stopping 依据是 `val_accuracy`：
   - 每轮验证后若 `val_accuracy` 没有提升，则 `patience += 1`
   - 当 `patience >= early_patience` 时停止训练
6. 当前最佳模型也是按 `val_accuracy` 选取。

因此，学习率调度最合适的接入点仍然是：

- `stonco/core/train.py` 中的 `train_and_validate(...)`

这样可以最小改动复用到现有训练流程。

## 2. 这次方案的目标

这次计划在不破坏历史默认行为的前提下，为 STOnco 增加可选学习率调度。

### 目标

- 默认仍保持固定学习率，不改变现有命令的训练行为
- 仍保持单一 optimizer
- 支持多种常用 scheduler 模式
- 记录学习率变化，便于确认 scheduler 是否按预期生效
- 对 `ReduceLROnPlateau` 和 early stopping 的耦合关系给出明确提示

### 本轮建议纳入的模式

- `none`
- `linear`
- `cosine`
- `warmup_cosine`
- `plateau`（即 `ReduceLROnPlateau`）

### 本轮不建议纳入的范围

- 不同模块分别设置完全独立的学习率
- 多个 scheduler 叠加
- `warmup + plateau` 组合模式
- `OneCycleLR`
- `CyclicLR`
- `CosineAnnealingWarmRestarts`
- 按参数组做复杂的 weight decay / lr 双重分组

## 3. 为什么第一版不建议做不同模块不同 lr

当前模型结构很清楚，主要由三块组成：

- `self.gnn`
- `self.clf`
- `self.dom_slide / self.dom_cancer`

但从训练机制看，暂时没有足够理由把它们拆成多组学习率。

### 3.1 现在已经有两套控制“域分支强度”的参数

当前域对抗相关强度已经由以下参数共同决定：

- `lambda_slide / lambda_cancer`
- `grl_beta_mode`
- `grl_beta_slide_target / grl_beta_cancer_target`

如果再引入：

- `lr_backbone`
- `lr_clf`
- `lr_domain`

就会出现三套同时影响主干与域分支相对强弱的旋钮，搜索空间会明显膨胀，而且实验结论会变得难解释。

### 3.2 当前并不存在预训练 backbone / 冻结解冻需求

通常需要分模块 lr 的典型场景包括：

- 预训练 backbone + 新分类头
- backbone 需要小 lr 微调
- 头部随机初始化，需要更大 lr
- 某些模块先冻结后解冻

STOnco 当前并不是这种结构；所有模块基本都在同一训练阶段从头学习。

### 3.3 当前更值得优先解决的是“整体训练节奏”

项目目前更明显的问题是：

- 固定 lr
- 多损失叠加（task + dual-domain + MMD）
- 训练前期可能不够稳定
- 长训练中后期缺少更平滑的收敛控制

这些问题更适合优先通过“全局 scheduler”解决，而不是先把优化器拆复杂。

### 3.4 结论

本轮建议：

- 继续保持单一全局 `lr`
- 不做参数组分 lr

若未来实验发现以下现象，再单独规划第二阶段：

- 域头收敛明显过快，持续压制主任务
- 分类头收敛明显慢于 backbone
- backbone 表示已稳定，但头部仍明显欠拟合

到那时也建议优先采用“倍率型参数组”而不是 3 个完全独立 lr，例如：

- `lr_backbone_ratio`
- `lr_clf_ratio`
- `lr_domain_ratio`

而不是直接暴露：

- `lr_backbone`
- `lr_clf`
- `lr_domain`

## 4. 推荐的 scheduler 接口

### 4.1 CLI 新增参数

建议在 `train.py` 新增以下参数：

- `--lr_scheduler`
  - 类型：`choices=['none', 'linear', 'cosine', 'warmup_cosine', 'plateau']`
  - 默认：`none`

- `--lr_warmup_epochs`
  - 类型：`int`
  - 默认：`10`
  - 仅 `warmup_cosine` 使用

- `--min_lr_ratio`
  - 类型：`float`
  - 默认：`0.01`
  - 适用于 `linear / cosine / warmup_cosine / plateau`
  - 表示最小学习率为：`base_lr * min_lr_ratio`

### 4.2 Plateau 专属参数

建议增加：

- `--plateau_metric`
  - 类型：`choices=['val_accuracy', 'val_avg_total_loss', 'val_macro_f1', 'val_auroc', 'val_auprc']`
  - 推荐默认：`val_accuracy`

- `--plateau_factor`
  - 类型：`float`
  - 推荐默认：`0.5`
  - 含义：触发后 `lr = lr * factor`

- `--plateau_patience`
  - 类型：`int`
  - 推荐默认：`10`

- `--plateau_threshold`
  - 类型：`float`
  - 推荐默认：`1e-4`

- `--plateau_cooldown`
  - 类型：`int`
  - 推荐默认：`0`

说明：

- `min_lr` 不再单独暴露，统一用 `cfg['lr'] * min_lr_ratio`
- `mode` 不需要单独暴露
  - 若监控 `val_accuracy`，内部固定 `mode='max'`
  - 若监控 `val_avg_total_loss`，内部固定 `mode='min'`
  - 若监控 `val_macro_f1`，内部固定 `mode='max'`
  - 若监控 `val_auroc`，内部固定 `mode='max'`
  - 若监控 `val_auprc`，内部固定 `mode='max'`

## 5. 各 scheduler 的明确语义

### 5.1 `none`

- 不创建 scheduler
- 训练全程使用固定 `cfg['lr']`

### 5.2 `linear`

- 从初始 `cfg['lr']` 开始
- 按训练总 `step` 数线性衰减
- 训练结束时衰减到 `cfg['lr'] * min_lr_ratio`

### 5.3 `cosine`

- 从初始 `cfg['lr']` 开始
- 按训练总 `step` 数做 cosine decay
- 尾部衰减到 `cfg['lr'] * min_lr_ratio`

### 5.4 `warmup_cosine`

- 先做线性 warmup
- warmup 时长由 `lr_warmup_epochs * steps_per_epoch` 换算得到
- warmup 结束达到 `cfg['lr']`
- 后续继续做 cosine decay，尾部到 `cfg['lr'] * min_lr_ratio`

### 5.5 `plateau`

- 基础学习率初始为 `cfg['lr']`
- 不按固定曲线衰减
- 每个 epoch 验证结束后，根据监控指标调用 `ReduceLROnPlateau.step(metric)`
- 当监控指标在若干轮内未改善时，自动降低 lr
- 最低不会低于 `cfg['lr'] * min_lr_ratio`

## 6. 实现建议

## 6.1 配置层

在 `train.py` 的以下位置新增字段：

- parser
- 默认 `cfg`
- CLI 覆盖逻辑
- `config_json` 兼容读取

新增字段应写入训练产物 `meta.json:cfg`，以保证实验可追溯。

### 6.2 scheduler 构造位置

建议放在：

```python
opt = torch.optim.AdamW(...)
scheduler = ...
```

即优化器创建之后、训练循环开始之前。

### 6.3 scheduler 更新粒度

不同模式需要分开处理：

#### A. `linear / cosine / warmup_cosine`

建议按 `optimizer step` 更新：

```python
loss_total.backward()
opt.step()
scheduler.step()
```

理由：

- 当前代码已经有 `global_step`
- warmup 本质更适合 step-based
- 对长训练更平滑

#### B. `plateau`

必须按 epoch 更新，并且在验证后更新：

```python
metrics = ...
scheduler.step(monitored_metric)
```

因为 `ReduceLROnPlateau` 是基于验证指标是否停滞来降学习率，而不是按 step 走固定曲线。

### 6.4 实现方式建议

#### 固定曲线类 scheduler

建议使用 `LambdaLR` 实现：

- `linear`
- `cosine`
- `warmup_cosine`

优点：

- 不需要复杂组合调度器
- 容易和现有 `global_step` / `total_steps` 逻辑对齐
- 代码集中、可读性更高

#### Plateau

建议直接使用 PyTorch 原生：

- `torch.optim.lr_scheduler.ReduceLROnPlateau`

这样语义最标准，也方便后续用户理解。

## 7. 需要记录哪些学习率信息

当前训练产物没有显式 lr 曲线，这会降低 scheduler 上线后的可解释性。

本轮建议新增：

- `hist['lr']`

记录规则建议如下：

- 每个 epoch 记录一个值
- 记录该 epoch 结束时、下一轮训练将使用的当前 lr
- 由于当前方案只有单一 optimizer + 单一 param group，直接记录：
  - `opt.param_groups[0]['lr']`

同步更新：

- `loss_components.csv` 增加 `lr`
- 训练曲线图增加 lr 曲线

## 8. `ReduceLROnPlateau` 与 early stopping 的关系

这是这次方案里最需要提前说明的地方。

### 8.1 两者的工作机制不同

#### early stopping

- 目标：训练停滞时直接结束训练
- 当前 STOnco 中依据 `val_accuracy`

#### ReduceLROnPlateau

- 目标：训练停滞时先降低学习率，再给模型继续优化的机会
- 不直接结束训练

### 8.2 为什么两者可能冲突

如果两者同时启用，就可能出现下面的情况：

1. 验证指标连续几轮不提升
2. plateau 检测到“停滞”，准备降 lr
3. 但 early stopping 的 patience 也在同时累积
4. 结果模型刚降 lr，还没来得及受益，就先被 early stopping 停掉

也就是说：

- plateau 想“降 lr 后再试一段”
- early stopping 想“既然不涨了就直接停”

这两个策略天然存在一定竞争关系。

### 8.3 本项目中的特别风险

当前 STOnco：

- best model 按 `val_accuracy`
- early stopping 按 `val_accuracy`
- 如果 plateau 也监控 `val_accuracy`

那么三者会高度耦合：

- best checkpoint 选择
- plateau 触发
- early stopping 触发

这会让训练行为更敏感，尤其当 `val_accuracy` 有波动时更明显。

### 8.4 推荐提示策略

本轮文档建议明确写出以下提示：

#### 推荐用法

当 `lr_scheduler='plateau'` 时：

- 建议关闭 early stopping
- 即设置：

```bash
--early_patience 0
```

或：

- `--early_patience <= 0`

#### 若用户坚持同时启用

建议在文档中提醒：

- `early_patience` 应明显大于 `plateau_patience`
- 至少建议满足：

```text
early_patience >= plateau_patience + plateau_cooldown + 3
```

更保守一点可以建议：

```text
early_patience >= 2 * plateau_patience
```

否则很容易出现：

- plateau 刚降 lr
- early stopping 很快就终止训练

### 8.5 是否需要代码里做提示

建议在实现时增加运行时提示，但不强制报错：

- 若 `lr_scheduler='plateau'` 且 `early_patience > 0`
- 打印 warning：
  - 提示 plateau 与 early stopping 可能相互竞争
  - 建议关闭 early stopping 或增大 `early_patience`

这样既不阻塞用户，也能减少误用。

## 9. Plateau 监控哪个指标更合适

### 方案 A：监控 `val_accuracy`（推荐默认）

优点：

- 和当前 best model 选择逻辑一致
- 和当前 early stopping 逻辑一致
- 用户更直观

缺点：

- `accuracy` 常常是台阶式变化，可能比 loss 更噪、更不平滑
- plateau 触发时机会比较“跳”

### 方案 B：监控 `val_avg_total_loss`

优点：

- 通常比 accuracy 更平滑
- plateau 判断更稳定

缺点：

- 和 best model 选择指标不一致
- 可能出现 loss 在下降但 accuracy 不涨，或者反之

### 方案 C：监控 `val_macro_f1`

优点：

- 比 accuracy 更能体现类别不平衡时的整体分类质量
- 和最终分类表现的业务解释通常更接近

缺点：

- 可能比 loss 更波动
- 当前 best model 选择逻辑不是基于它

### 方案 D：监控 `val_auroc`

优点：

- 对分类阈值不敏感
- 更适合衡量排序能力是否继续改善

缺点：

- 可能与最终 0.5 阈值下的 accuracy / F1 不一致
- 用户在日常实验里通常不如 accuracy 直观

### 方案 E：监控 `val_auprc`

优点：

- 在类别不平衡时通常更敏感
- 更能体现正类识别质量变化

缺点：

- 波动可能比 accuracy 更明显
- 与当前 best model 选择逻辑不一致

### 当前建议

本轮建议先采用：

- 默认 `plateau_metric='val_accuracy'`

原因：

- 与项目当前“按验证准确率选最优”的主逻辑更一致

同时保留以下备选指标供显式指定：

- `val_avg_total_loss`
- `val_macro_f1`
- `val_auroc`
- `val_auprc`

## 10. 边界情况处理建议

### 对所有 scheduler

- 若 `lr_scheduler='none'`：不创建 scheduler
- 若 `cfg['lr'] <= 0`：报错
- 若 `min_lr_ratio <= 0`：报错
- 若 `min_lr_ratio > 1`：报错

### 对 `warmup_cosine`

- `lr_warmup_epochs <= 0`
  - 视为无 warmup，直接退化成 `cosine`

- `warmup_steps >= total_steps`
  - 全程只做线性 warmup 到 base lr
  - 不进入 cosine

### 对 `linear / cosine`

- `total_steps <= 0`
  - 直接跳过 scheduler 或退化成固定 lr

### 对 `plateau`

- 如果监控指标是 `val_accuracy`
  - mode 固定为 `max`

- 如果监控指标是 `val_avg_total_loss`
  - mode 固定为 `min`

- 如果监控指标是 `val_macro_f1`
  - mode 固定为 `max`

- 如果监控指标是 `val_auroc`
  - mode 固定为 `max`

- 如果监控指标是 `val_auprc`
  - mode 固定为 `max`

- 若监控值是 `nan`
  - 建议本轮采用“跳过本轮 plateau.step”策略
  - 并打印 warning

这个点也建议在实现时写清楚。

## 11. 对绘图和日志的建议

当前图表主要展示损失和精度。

学习率调度上线后，建议：

- `loss_components.csv` 增加 `lr`
- `train_loss.svg` 或训练指标图增加 lr 面板

这样可以直接回答：

- plateau 什么时候降了 lr
- warmup 是否真的发生了
- cosine / linear 的尾部是否按预期衰减

## 12. 对 HPO 的范围说明

本轮仍建议：

- 不把 HPO 显式纳入第一版范围

说明：

- 由于 `train_hpo.py` 复用 `train_and_validate(...)`
- 若未来通过 `cfg` 传入 scheduler 字段，公共训练函数本身是能兼容的
- 但本轮不建议同时修改：
  - HPO 搜索空间
  - HPO 文档
  - HPO CLI 暴露

否则确认成本会明显增大。

## 13. 推荐的默认值（建议稿）

### 通用 scheduler 参数

- `lr_scheduler='none'`
- `lr_warmup_epochs=10`
- `min_lr_ratio=0.01`

### Plateau 参数

- `plateau_metric='val_accuracy'`
- `plateau_factor=0.5`
- `plateau_patience=10`
- `plateau_threshold=1e-4`
- `plateau_cooldown=0`

### 使用建议

- 若使用 `plateau`，推荐：
  - `--early_patience 0`

## 14. 需要你确认的点

请重点确认以下几项：

### Q1. `plateau` 是作为独立模式加入，而不是做成 `warmup + plateau`

已确认：

- 只加独立 `plateau`

### Q2. `plateau` 监控指标选哪个

已确认：

- 默认监控 `val_accuracy`

同时保留以下备选：

- `val_avg_total_loss`
- `val_macro_f1`
- `val_auroc`
- `val_auprc`

### Q3. `plateau` 与 early stopping 的建议不用写得更强一些

当前建议文案：
- “建议关闭 early stopping 使用 plateau”


### Q4. `linear` 的定义是否确认

当前建议：

- `linear = 从 base lr 线性衰减到 min_lr`

而不是：

- 线性 warmup
- warmup 后线性衰减

确认

## 15. 本轮实施范围结论

若你确认本稿，代码实现阶段建议仅做：

1. `train.py` 新增 scheduler CLI 与 cfg
2. `train_and_validate(...)` 接入：
   - `LambdaLR`
   - `ReduceLROnPlateau`
3. 训练历史增加 `lr`
4. `loss_components.csv` 增加 `lr`
5. 训练图增加 lr 曲线
6. 当 `plateau + early stopping` 同时启用时打印 warning

暂不做：

7. 不同模块不同 lr
8. warmup + plateau 组合
9. HPO 搜索空间更新
