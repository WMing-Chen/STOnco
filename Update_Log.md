# Update Log

## 2026-04-01 16:20:00 CST

- 更新内容：`train.py` 新增多模式学习率调度，支持 `none / linear / cosine / warmup_cosine / plateau`，并把学习率记录同步写入训练历史、`loss_components.csv` 和训练曲线图。
- 代码影响：
  - `train.py` 新增参数：
    - `--lr_scheduler`
    - `--lr_warmup_epochs`
    - `--min_lr_ratio`
    - `--plateau_metric`
    - `--plateau_factor`
    - `--plateau_patience`
    - `--plateau_threshold`
    - `--plateau_cooldown`
  - 默认配置新增：
    - `lr_scheduler='none'`
    - `lr_warmup_epochs=10`
    - `min_lr_ratio=0.01`
    - `plateau_metric='val_accuracy'`
    - `plateau_factor=0.5`
    - `plateau_patience=10`
    - `plateau_threshold=1e-4`
    - `plateau_cooldown=0`
  - `linear / cosine / warmup_cosine` 使用 step-based `LambdaLR`
  - `plateau` 使用 epoch-based `ReduceLROnPlateau`
  - `plateau_metric` 目前支持：
    - `val_accuracy`
    - `val_avg_total_loss`
    - `val_macro_f1`
    - `val_auroc`
    - `val_auprc`
  - 当 `lr_scheduler='plateau'` 且 `early_patience > 0` 时，训练启动后会打印 warning，提示 plateau 与 early stopping 可能相互竞争，并建议优先关闭早停
  - 当 plateau 监控指标为 `NaN` 时，会跳过该轮 `scheduler.step(...)` 并打印 warning
  - `hist` 新增 `lr`
  - `loss_components.csv` 新增 `lr` 列
  - 学习率曲线改为单独导出 `lr.svg`
  - `train_loss.svg` 和 `train_val_loss.svg` 保持原先 `3x3` 布局，不再混入 lr 子图
- 涉及文件：
  - `stonco/core/train.py`
  - `docs/Tutorial.md`
  - `docs/PLAN_lr_scheduler_multimode_plateau.md`
- 说明：
  - 当前实现只覆盖常规训练入口 `train.py`
  - 本次没有对 `train_hpo.py` 做任何专门适配或搜索空间更新，后续若要支持 HPO 需要单独规划
  - 仍保持单一 optimizer、单一全局 lr scheduler，不做不同模块不同 lr
  - 已完成目标文件语法检查：`python -m py_compile stonco/core/train.py stonco/core/train_hpo.py`

## 2026-04-01 15:45:00 CST

- 更新内容：分类头 `clf_hidden` 从固定 3 层且末层必须为 64，改为支持可变层数和任意正整数宽度；分类头 latent 表示从固定语义 `z64` 升级为通用语义 `z_clf`，并保留对历史 64 维流程的兼容。
- 代码影响：
  - `ClassifierHead` 改为按 `clf_hidden` 动态构建 MLP，`clf_hidden` 现在只要求为非空正整数列表
  - `STOnco_Classifier.forward(..., return_z=True)` 现在统一返回 `out['z_clf']`
  - 当分类头 latent 维度恰好为 64 时，额外兼容返回 `out['z64']`
  - `train.py` 放宽 `--clf_hidden` 与 `cfg['clf_hidden']` 校验，不再要求“恰好 3 个整数且末层 64”
  - 训练产物 `meta.json` 新增 `cfg['clf_latent_dim']`，用于显式记录分类头 latent 维度
  - `export_spot_embeddings.py` 的 `--embed_source` 扩展为 `h|z_clf|z64`
  - `visualize_umap_tsne.py` 的 `--embed_source` 扩展为 `h|z_clf|z64`，并支持 `z_clf_*` / 历史 `z64_*` 双兼容读取
- 涉及文件：
  - `stonco/core/models.py`
  - `stonco/core/train.py`
  - `stonco/core/infer.py`
  - `stonco/utils/visualize_prediction.py`
  - `stonco/utils/export_spot_embeddings.py`
  - `stonco/utils/visualize_umap_tsne.py`
  - `README.md`
  - `docs/Tutorial.md`
- 说明：
  - 新模型推荐使用 `--embed_source z_clf`
  - `--embed_source z64` 仅用于兼容历史 64 维分类头 latent；若当前模型 `clf_latent_dim != 64`，会直接报错并提示改用 `z_clf`
  - 老模型和历史 `z64_*` CSV 仍可继续加载；新导出默认使用 `z_clf_*` 列名
  - 已完成目标文件语法检查：`python -m compileall stonco/core/models.py stonco/core/train.py stonco/core/infer.py stonco/utils/visualize_prediction.py stonco/utils/export_spot_embeddings.py stonco/utils/visualize_umap_tsne.py`

## 2026-04-01 15:05:00 CST

- 更新内容：`train.py` 的 GRL beta 调度新增 `linear` 模式，并为 slide / cancer 两路分别增加 `delay_epochs` 与 `warmup_epochs` 参数；同时 `dann` 模式也支持 `delay_epochs`。
- 代码影响：
  - `--grl_beta_mode` 从 `dann|constant` 扩展为 `dann|constant|linear`
  - 新增参数：
    - `--grl_beta_slide_delay_epochs`
    - `--grl_beta_slide_warmup_epochs`
    - `--grl_beta_cancer_delay_epochs`
    - `--grl_beta_cancer_warmup_epochs`
  - `dann` 行为更新为：delay 之前 `beta=0`，delay 之后在剩余训练过程上重新归一化并走完整 DANN 曲线
  - `linear` 行为定义为：delay 之前 `beta=0`，delay 之后按线性 warm-up 升到 `*_target`
  - 新参数默认值为：slide `1/8`，cancer `3/12`
- 涉及文件：
  - `stonco/core/train.py`
  - `docs/PLAN_GRL_linear_warmup_4params.md`
- 说明：
  - 本次只更新 `train.py`，`train_hpo.py` 未同步这些新参数
  - `grl_beta_gamma` 仅在 `grl_beta_mode=dann` 时生效；`linear` 下忽略
  - 4 个 `*_epochs` 参数要求整数，内部在训练开始后按 `len(train_loader)` 自动换算成 step

## 2026-04-01 14:20:00 CST

- 更新内容：将验证集 `DataLoader` 的 `batch_size` 从固定 `1` 改为复用训练配置 `batch_size_graphs`。
- 代码影响：验证阶段的损失统计仍保持现有 batch-level 口径不变，但验证 batch 现在可能同时包含多个 slide / cancer domain，因此在 `mmd_on='cancer'` 等场景下，`val_avg_mmd_loss` 与 `val_avg_mmd_raw_cancer` 不再因单图验证而恒为 `0`。
- 涉及文件：
  - `stonco/core/train.py`
- 说明：本次仅调整验证 batch 组包方式；`shuffle=False`、`loss_components.csv` 字段、`train_val_loss.svg` 绘图逻辑与训练阶段损失计算均保持不变。

## 2026-04-01 14:05:00 CST

- 更新内容：`train_domain_diagnostics.svg` 的 chance baseline 改为基于训练集实际见过的域数计算，不再受验证集独有域影响；图例文案同步简化为 `Chance (train, ...)` 与 `Chance accuracy (train, ...)`。
- 代码影响：新增训练集 seen-domain 计数 helper，并在单次训练、KFold、LOCO 三个训练入口统一使用 train-only 域数绘制 domain diagnostics 基线。
- 涉及文件：
  - `stonco/core/train.py`
- 说明：domain diagnostics 图仍然展示训练期 domain CE / accuracy 曲线，但 baseline 的解释口径现在与训练阶段实际见过的域类别空间保持一致，更适合用于训练过程解释。

## 2026-04-01 13:40:00 CST

- 更新内容：训练流程新增验证损失统计，并增加 `train_val_loss.svg` 用于同图对比训练集与验证集曲线。
- 代码影响：验证阶段现在同步记录 `avg_total_loss`、`avg_task_loss`、domain loss、MMD loss、`batch_loss_variance` 等验证损失；`loss_components.csv` 新增对应 `val_*` 损失列；训练结束后额外导出 `train_val_loss.svg`。
- 涉及文件：
  - `stonco/core/train.py`
- 说明：`train_loss.svg` 与 `train_val_metrics.svg` 保持原有输出不变；新增图中训练曲线与验证曲线采用不同配色，布局与 `train_loss.svg` 一致。

## 2026-04-01 13:27:22 CST

- 更新内容：将训练指标命名 `Var_risk` / `var_risk` 统一更名为 `batch_loss_variance`。
- 代码影响：更新训练历史键、`loss_components.csv` 列名、训练曲线标题与相关代码注释。
- 涉及文件：
  - `stonco/core/train.py`
  - `docs/Tutorial.md`
  - `docs/PLAN_train_updates.md`
  - `docs/PLAN_MMD_integration_STOnco.md`
- 说明：该更新仅修正命名歧义，指标含义不变，仍表示每个 epoch 内 mini-batch 总损失的方差。
