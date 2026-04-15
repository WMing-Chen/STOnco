# Update Log

## 2026-04-15 22:30:00 CST

- 更新内容：训练流程新增 generated-support Wasserstein barycenter（WB）对齐，用于以连续可学习的 barycenter support 替代/升级当前基于 MMD 的多癌种 latent 对齐。
- 代码影响：
  - 新增 `stonco/core/wb_potentials.py`，包含 `GeneratedSupportMap`、single/dual potential bank 与 `GeneratedSupportWBLoss`
  - `stonco/core/train.py` 新增 `--use_wb_align`、`--wb_loss_type`、`--lambda_wb`、`--wb_anchor_weight`、`--wb_spots_per_graph`、`--wb_support_size`、`--wb_eval_loss`、`--best_metric` 等参数
  - WB 第一版采用 direct-`h` 训练约束：分类 head 仍使用 `h`，训练阶段由 `b=T_phi(h)` 生成 continuous barycenter support，不新增推理时 embedding
  - 支持两种 loss：`euclidean_pairwise` 与 `dual_potential`；均采用 BaryIR-style 交替更新，先更新 potentials，再更新 STOnco 主模型与 support map
  - `prepare_graphs()`、KFold、LOCO 路径均改为在 `use_domain_adv_cancer or use_wb_align` 时返回 `n_domains_cancer`，保证关闭 cancer GRL 但启用 WB 时仍能建立 potential bank
  - `loss_components.csv` 新增 WB 诊断列，并在启用 WB 时额外保存 `wb_train_loss.svg`
  - 训练 artifacts 额外保存 `wb_support_map_last.pt`、`wb_potentials_last.pt` 与 `wb_config.json`
- 涉及文件：
  - `stonco/core/train.py`
  - `stonco/core/wb_potentials.py`
  - `docs/PLAN_continuous_dual_potential_WB_STOnco.md`
  - `docs/Tutorial.md`
- 说明：
  - 默认推荐 `--use_mmd 0 --use_wb_align 1 --wb_loss_type euclidean_pairwise` 作为低计算量 WB-only 起点
  - `dual_potential` 更贴近正规 dual-potential WB objective，但计算量更高，建议降低 `wb_spots_per_graph` 与 `wb_support_size`
  - 已在 `hpc_gpu01` 的 `stonco` 环境完成 smoke test：`euclidean_pairwise` 与 `dual_potential` 均能完成训练、保存模型、WB artifacts、`loss_components.csv` 和 `wb_train_loss.svg`

## 2026-04-10 16:23:02 CST

- 更新内容：训练流程新增 `--save_epoch_checkpoints` 参数，可在常规最优/最后一轮之外额外保留指定 epoch 的模型结果。
- 代码影响：
  - `stonco/core/train.py` 新增 `--save_epoch_checkpoints` 参数解析与归一化校验，支持一次传入多个正整数 epoch
  - 单次训练、KFold 与 LOCO 三条训练路径都会在训练过程中捕获指定 epoch 的 `state_dict`，并在训练结束后统一写出到 `epoch_checkpoints/epoch_XXX/`
  - 根目录 `meta.json` 新增 `saved_epoch_checkpoints` 与 `requested_save_epoch_checkpoints` 字段，用于记录请求轮次与实际成功保存的轮次
  - `docs/Tutorial.md` 补充了该参数的用法、输出目录结构与早停下的行为说明
- 涉及文件：
  - `stonco/core/train.py`
  - `docs/Tutorial.md`
- 说明：
  - 例如 `--epochs 300 --save_epoch_checkpoints 100` 会在正常训练到 300 轮的同时额外保留第 100 轮模型
  - 额外保存的 `epoch_checkpoints/epoch_XXX/` 目录会复制预处理器文件，并生成与该轮对应的 `model.pt`、`meta.json`、训练曲线和 `loss_components.csv`，便于直接按常规 artifacts 目录使用
  - 若开启早停且训练提前结束，未到达的请求轮次不会生成快照，并会在日志中给出提示

## 2026-04-10 15:40:07 CST

- 更新内容：新增 embedding 域统计与局部二维诊断工具 `analyze_spot_embedding_domains.py`，并将其使用方法补充到 `docs/Tutorial.md`。
- 代码影响：
  - 新增 `stonco/utils/analyze_spot_embedding_domains.py`，用于对 `spot_embeddings_*.csv` 计算域均值/方差统计，并生成热图、箱线图、局部维度 KDE、一维 spot 散点分布图与二维联合散点图
  - 支持按组均衡抽样导出二维散点图与对应抽样 CSV，便于对比 `h_7/h_20` 等局部维度组合在不同域上的分布
  - `docs/Tutorial.md` 第 9 章新增 “9.5 embedding 域统计与局部二维诊断” 小节，补充用途、示例命令、默认产物与参数说明
- 涉及文件：
  - `stonco/utils/analyze_spot_embedding_domains.py`
  - `docs/Tutorial.md`
- 说明：
  - 脚本默认面向 `h_*` embedding 列；若分析分类头 latent，可通过 `--embedding_prefix z_clf_` 切换到 `z_clf_*`
  - 该工具偏向诊断与解释，适合定位域偏移残留在哪些维度；正式量化比较仍建议结合 `evaluate_embedding_mixing.py` 的 LISI 指标一起使用

## 2026-04-09 20:29:18 CST

- 更新内容：将 `stonco.core.train` 的 LapPE 默认设置从启用改为关闭。
- 代码影响：
  - `stonco/core/train.py` 默认配置中的 `lap_pe_dim` 从 `16` 改为 `0`
  - 不显式传入 `--lap_pe_dim` 时，训练流程不再默认计算 LapPE
- 涉及文件：
  - `stonco/core/train.py`
- 说明：
  - 本次仅调整 `stonco.core.train` 的默认训练配置
  - 如需重新启用，可在训练命令中显式传入 `--lap_pe_dim` 为正整数

## 2026-04-07 11:35:00 CST

- 更新内容：训练流程新增 sampler 双模式与静态子图训练支持，并将对应说明同步到 `docs/Tutorial.md`。
- 代码影响：
  - `train.py` 新增 `sampler_mode`、`sampler_k_cancers`、`sampler_m_per_cancer`、`sampler_enforce_distinct_batch`、`subgraph_mode`、`subgraph_target_spots`、`subgraph_min_spots`
  - 新增 `stonco/core/sampler.py`，实现 `CancerBalancedBatchSampler`、训练集静态子图展开与 sampler 配置归一化
  - `cancer_balanced_subgraph` 会自动归一化为 `cancer_balanced + static subgraph`
- 涉及文件：
  - `stonco/core/train.py`
  - `stonco/core/sampler.py`
  - `docs/PLAN_sampler_dual_mode.md`
  - `docs/Tutorial.md`
- 说明：
  - 训练集在 train/val split 后再展开子图；验证集保持整图
  - `subgraph_mode=online` 目前保留接口，尚未实现

## 2026-04-07 11:20:00 CST

- 更新内容：完成通用 embedding mixing 评估流程落地，新增 embedding-space / UMAP-space / tSNE-space 的 LISI 量化工具，并补齐一键串联的分析 pipeline。
- 代码影响：
  - 新增公共算法模块：
    - `stonco/utils/embedding_analysis.py`
  - 新增独立评估入口：
    - `stonco/utils/evaluate_embedding_mixing.py`
  - 新增一键分析编排入口：
    - `stonco/utils/run_embedding_analysis_pipeline.py`
  - `visualize_umap_tsne.py` 新增参数：
    - `--out_coords_csv`
  - `visualize_umap_tsne.py` 现可在生成 SVG 的同时导出与图像完全一致的 `umap_1/umap_2`、`tsne_1/tsne_2` 坐标 CSV，供后续 LISI 评估复用
  - `evaluate_embedding_mixing.py` 支持：
    - `--spaces embedding umap tsne`
    - `--group_cols`
    - `--group_roles col:integration|conservation`
    - `--k_values`
    - `--out_spot_csv`
  - `run_embedding_analysis_pipeline.py` 会顺序串联：
    - 导出 embedding CSV
    - 生成 UMAP/t-SNE 图与坐标 CSV
    - 输出 mixing 指标汇总 CSV（以及可选的 per-spot CSV）
- 涉及文件：
  - `stonco/utils/embedding_analysis.py`
  - `stonco/utils/evaluate_embedding_mixing.py`
  - `stonco/utils/run_embedding_analysis_pipeline.py`
  - `stonco/utils/visualize_umap_tsne.py`
  - `docs/PLAN_embedding_mixing_eval_pipeline.md`
  - `docs/Tutorial.md`
- 说明：
  - 正式结论应以 `space=embedding` 的 LISI 为主，`space=umap/tsne` 仅作为配图辅助指标
  - `group_roles` 对常见列支持自动推断：`sample_id/batch_id/slide_id -> integration`，`tumor_label/cell_type/region -> conservation`
  - `cancer_type` 的角色不写死，应按实验目标显式指定

## 2026-04-03 17:57:32 CST

- 更新内容：`visualize_umap_tsne.py` 新增可选参数 `--color_cols`，允许按任意指定 metadata 列生成 UMAP + t-SNE 着色图，同时默认行为保持不变。
- 代码影响：
  - `visualize_umap_tsne.py` 新增参数：
    - `--color_cols`
  - 不传 `--color_cols` 时，仍默认生成按 `tumor_label`、`batch_id`、`cancer_type` 上色的三张图
  - 传入 `--color_cols col1 col2 ...` 时，会改为按给定列列表逐列生成 SVG
  - 输出文件名改为基于列名自动生成并做安全字符清洗，例如 `umap_tsne_h_by_sample_id.svg`
- 涉及文件：
  - `stonco/utils/visualize_umap_tsne.py`
  - `docs/Tutorial.md`
  - `../../STOnco.md`
- 说明：
  - 该参数只控制“按哪几列上色”，不改变 UMAP/t-SNE 所使用的 embedding 来源；embedding 选择仍由 `--embed_source` 控制
  - 若指定列在 CSV 中不存在，会打印 warning 并跳过该列
  - 已完成目标文件语法检查：`python -m compileall stonco/utils/visualize_umap_tsne.py`

## 2026-04-03 17:47:19 CST

- 更新内容：`export_spot_embeddings.py` 的输入模式从“二选一”改为“可组合”，现在支持在同一条命令中混合导出 `--train_npz`、重复 `--npz_glob`，以及新增的可重复 `--npz` 单文件输入。
- 代码影响：
  - `export_spot_embeddings.py` 新增参数：
    - `--npz`
  - `--npz_glob` 改为可重复传入多次
  - 输入校验从“必须且仅能选择一个 `--train_npz` 或 `--npz_glob`”改为“至少提供一个输入源：`--train_npz` 和/或 `--npz` 和/或 `--npz_glob`”
  - 新增单文件/多文件共用的 NPZ 迭代逻辑，支持按输入顺序将多个来源的样本连续写入同一个 `out_csv`
  - `--subset` 语义保持不变，但现在只对 `--train_npz` 生效；对 `--npz` / `--npz_glob` 会忽略
  - 因此可以直接用一条命令完成“训练集子集 + 指定外部验证切片”联合导出，不再需要先导出再 `--append`
- 涉及文件：
  - `stonco/utils/export_spot_embeddings.py`
  - `docs/Tutorial.md`
  - `../../STOnco.md`
- 说明：
  - 新增的 `--npz` 适合少量指定切片，例如 `BRCA1.npz`、`BRCA4.npz`
  - `--npz_glob` 仍适合目录级批量输入，例如 `val_npz/*.npz`
  - 当前导出 CSV 仍不会自动新增 `split=train/val/external` 列；合并后可按 `tumor_label`、`batch_id`、`cancer_type` 上色，但不能直接按来源上色
  - 已完成目标文件语法检查：`python -m compileall stonco/utils/export_spot_embeddings.py`

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
