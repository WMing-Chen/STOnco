# Domain Loss 不明显下降的排查与建议（STOnco）

以下基于当前代码路径回顾训练流程与对抗域损失的实现，结合深度学习经验给出可能原因与建议。先给结论：在使用 GRL 的域对抗框架里，域损失不明显下降并不一定是 bug，可能是“特征被混淆”的正常表现；但如果长期不稳定或几乎无变化，也可能是数据/标签/优化设置导致域头学不到有效信号。

## 1. 关键实现位置（用于对照）
- 域对抗 head + GRL：`stonco/core/models.py`（`STOnco_Classifier.forward` + `grad_reverse`）。
- 域损失计算与加权：`stonco/core/train.py`（`_compute_losses` 与训练循环；`avg_batch_domain_loss`/`avg_cancer_domain_loss` 已乘以 `lambda_*`）。
- 域标签构建与兜底：`stonco/core/train.py`（`_load_cancer_labels`、`_resolve_batch_id`、`_resolve_cancer_type`）。

## 2. 现象可能的“合理解释”
- **GRL 目标本身是让域不可分**：域头参数在最小化 CE，但主干在反向最大化 CE；当二者拉扯到平衡，域 loss 会稳定在接近随机水平（约 `log(K)`），并非持续下降。
- **日志里的域 loss 已经乘过权重**：`avg_*_domain_loss` 是 `lambda_* * L_domain`；如果 `lambda_*` 小，曲线自然显得平坦。可优先看原始 CE 曲线 `avg_*_domain_ce`（训练日志/CSV 会保存）。

## 3. 可能原因（按优先级）
### P0. 先确认现象是否“误判”
- 域 loss 是否已经接近随机基线？对于 K 类域，随机 CE 约为 `log(K)`；优先用 `avg_*_domain_ce` 与 `log(K)` 对比（`avg_*_domain_loss` 已乘过 `lambda_*` 且可能为带权版本）。
- 训练是否实际跑满 1000 epoch？早停或中断会导致曲线稀疏或看似“不动”。

### P1. 数据与域标签质量问题（最常见）
- `Batch_id` 缺失时会回退为 `slide_id`：这会造成 **每个切片一个域**，类别数过多且样本极少，域头难学习，CE 难下降。代码位置：`stonco/core/train.py` 中 `_resolve_batch_id`。
- `cancer_type` 缺失时回退到前缀或 `UNKNOWN`：可能引入噪声/错误标签，降低域头可学习性。
- 域类别极度不平衡：虽然训练端已默认按 **graph-frequency 的 sqrt 反频率**做 class-weight（并 clamp + mean-normalize），但若某些域样本过少，域头仍可能学不动或震荡。

### P2. 优化与批大小设置
- `batch_size_graphs` 偏小（默认 2）：单步看到的域类别太少，域头梯度噪声大，loss 不稳定。
- 学习率偏高/无调度：1000 epoch 下 `lr=1e-3` 可能让损失震荡。

### P3. 结构与对抗强度
- 当前实现已将 **alpha（`lambda_*`，域 loss 权重）** 与 **beta（GRL 对抗强度）** 解耦：beta 由 `--grl_beta_mode/--grl_beta_*` 控制。若对抗过强（尤其是 `--grl_beta_mode constant` 且 beta 较大），可能会让域头更快被“混淆”到接近随机水平，从而表现为 loss 不再明显下降。
- 域头容量有限（两层 MLP）：若域信号微弱或类别多，域头很难学到区分能力。

## 4. 建议（按优先级，先低风险）
### 优先级 P0：不改代码的检查
- 检查训练日志是否有 `[Warn] Batch_id missing` 或 `fallback` 提示（说明标签质量有问题）。
- 估算域损失基线：优先用 `avg_*_domain_ce` 与 `log(K)` 对比；若只看 `avg_*_domain_loss`，需要注意它已乘过 `lambda_*` 且 domain loss 可能是带权版本。
- 统计域类别数和每类样本量，确认是否过于稀疏（建议每类至少 5-10 张切片）。

### 优先级 P1：仅改参数的调整
- 若 `Batch_id` 不可靠，先用 `--use_domain_adv_slide 0` 关闭 batch 域对抗。
- 若癌种样本极少或标签不稳定，尝试 `--use_domain_adv_cancer 0` 或下调 `--lambda_cancer`。
- 提升 `--batch_size_graphs`（例如 4/8），让域头每步看到更多域样本。
- 适度降低学习率（如 `1e-4`）或引入简单衰减（若你愿意改代码）。

### 优先级 P2：需要改代码的提升项（建议确认后再动）
- **对抗强度策略**：根据稳定性选择 `--grl_beta_mode dann`（warm-up/平滑增长）或 `constant`（全程恒定）；并联动调小 `--grl_beta_*_target` 或 `--lambda_*`。
- **域 loss 加权策略**：当前默认是 graph-frequency 的 sqrt 反频率（clamp+mean-normalize）；若仍不稳，可考虑调整 clamp 范围、改为按 spot-frequency、或关闭权重对比效果。
- **记录域准确率**：仅靠 loss 不直观，建议加 domain accuracy 曲线辅助判断。

## 5. 结论小结
如果你主要看到 **avg cancer/batch domain loss 不下降**，这不一定是“坏事”，很可能是 GRL 目标在起作用；但若同时 `avg_total_loss` 抖动明显、且域 loss 远高于随机基线，优先检查域标签质量与 batch size，再考虑调整 lambda 和训练策略。
