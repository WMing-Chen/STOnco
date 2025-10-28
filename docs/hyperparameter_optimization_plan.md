# 多阶段超参数优化（HPO）方案 · 与实现完全对齐

本方案文档对当前项目中已实现的“统一三阶段 HPO 流水线”进行详细说明，并与实际代码保持一致：
- 目标指标：验证集 Accuracy（val_accuracy），作为早停与 HPO 的单一优化目标
- 实现位置：<mcfile name="train.py" path="/root/Project/Spotonco/train.py"></mcfile>
  - 统一入口（多阶段）：<mcsymbol name="run_multi_stage_hpo" filename="train.py" path="/root/Project/Spotonco/train.py" startline="401" type="function"></mcsymbol>
  - 单阶段 HPO：<mcsymbol name="run_hyperparameter_optimization" filename="train.py" path="/root/Project/Spotonco/train.py" startline="693" type="function"></mcsymbol>
  - 多阶段复评：<mcsymbol name="run_rescore_multiple_stages" filename="train.py" path="/root/Project/Spotonco/train.py" startline="622" type="function"></mcsymbol>
  - 阶段搜索空间：<mcsymbol name="_get_stage_search_space" filename="train.py" path="/root/Project/Spotonco/train.py" startline="384" type="function"></mcsymbol>
  - 训练/验证与早停：<mcsymbol name="train_and_validate" filename="train.py" path="/root/Project/Spotonco/train.py" startline="230" type="function"></mcsymbol>


一、整体流程（stage1 → stage2 → stage3 + 可选复评）
- 执行顺序：stage1 → stage2 → stage3；每个阶段完成后保存最佳配置，再进入下一阶段
- 参数累积：每一阶段仅搜索该阶段定义的“新参数集合”，并在 trial 构造时自动合并“前序阶段的最佳参数”作为固定前提
- 图构建优化：
  - stage1/2 共用一次准备好的图（GraphBuilder 构造 + LapPE）
  - stage3 涉及 LapPE 超参变动，objective 内对每个 trial 按 trial 配置“重新构建图”，避免无效重复
- 进度展示：
  - 顶层进度条：3 个阶段 + 指定的复评阶段数
  - 子进度条：每阶段的 n_trials 进度（含当前最佳值）
- 复评（可选）：
  - 通过 --rescore_topk 和 --rescore_stages 指定复评 Top-K 与阶段（默认仅复评 stage1）
  - 对 Top-K trial 在多随机种子下重复训练并取均值，以 mean_accuracy 重新排序并更新该阶段最佳
- 最终合并：将各阶段最佳参数累积合并，输出 tuning/best_config.json


二、阶段搜索空间与策略（与实现一致）
- 目标与度量：
  - HPO 优化目标：验证集 Accuracy（maximize）
  - 训练早停：基于验证集 Accuracy 提前停止
  - 其他指标（AUROC/AUPRC/Macro-F1）在训练过程中记录，用于分析但不参与搜索目标

- 阶段1（训练稳定性/收敛）：
  - lr：log-uniform [1e-4, 5e-3]
  - weight_decay：两段式搜索
    - wd_zero ∈ {0,1}；若 1 则 weight_decay=0.0
    - 否则 weight_decay：log-uniform [1e-6, 1e-3]
  - epochs：int ∈ [80, 200]
  - batch_size_graphs：{1, 2, 4}

- 阶段2（网络结构/正则）：
  - hidden：{64, 128, 192, 256}
  - num_layers：int ∈ [2, 5]
  - dropout：float ∈ [0.1, 0.6]
  - 仅当 model=gatv2 时：heads ∈ {2, 4, 6, 8}

- 阶段3（位置编码细化）：
  - concat_lap_pe：{0, 1}（布尔）
  - lap_pe_use_gaussian：{0, 1}（布尔）
  - lap_pe_dim：{8, 12, 16, 20}

说明：上述空间完全由 <mcsymbol name="_get_stage_search_space" filename="train.py" path="/root/Project/Spotonco/train.py" startline="384" type="function"></mcsymbol> 定义，stage1/2/3 各自仅覆盖本阶段相关的键，未覆盖的键来自“累计的前序最佳 + 初始 cfg”。


三、关键实现细节（代码结构与参数传递）
- 统一入口（多阶段）：<mcsymbol name="run_multi_stage_hpo" filename="train.py" path="/root/Project/Spotonco/train.py" startline="401" type="function"></mcsymbol>
  - 准备一次 base 图（用于 stage1/2）
  - 循环阶段：为每阶段创建/复用 SQLite study（Optuna + TPE 采样 + SuccessiveHalving 剪枝）
  - objective：
    - trial_cfg = 累计前序阶段最佳参数 → 覆盖当前阶段搜索空间
    - stage3 内：根据 trial_cfg 重新构图；stage1/2 直接使用 base 图
    - 指标汇报：trial.report(accuracy)，支持剪枝
    - 返回值：best['accuracy']
  - 存档：trials.csv、best_config_stageX.json（将“累计前序最佳 + 当前最佳”写入）
  - 可选复评：读取 stageX study，Top-K × 多 seed 复训，按 mean_accuracy 选择并更新 stage_best_configs；保存 topk_rescore.json 与 best_config_rescored.json
  - 最终：合并所有阶段最佳参数，写入 tuning/best_config.json

- 单阶段 HPO：<mcsymbol name="run_hyperparameter_optimization" filename="train.py" path="/root/Project/Spotonco/train.py" startline="693" type="function"></mcsymbol>
  - 用于仅运行某个阶段（--tune stageX），与多阶段 objective 一致

- 多阶段复评：<mcsymbol name="run_rescore_multiple_stages" filename="train.py" path="/root/Project/Spotonco/train.py" startline="622" type="function"></mcsymbol>
  - 支持通过 --rescore_stages 选择多个阶段（如 1,3）进行 Top-K 多种子复评

- 训练/验证与早停：<mcsymbol name="train_and_validate" filename="train.py" path="/root/Project/Spotonco/train.py" startline="230" type="function"></mcsymbol>
  - 以验证集 Accuracy 作为早停最佳性判据（patience 递增与重置逻辑基于 accuracy）
  - 训练中记录 AUROC/AUPRC/Accuracy/Macro-F1，并输出 1×3 曲线（Loss/AUROC/AUPRC）

- 设备与可重复性：
  - HPO/复评阶段强制 cfg['use_pca']=False，以保证图特征一致性与可复现实验
  - 种子：trial 内部使用 1000+trial.number；复评使用 --seeds 指定列表（默认 42,2023,2024）
  - 设备：通过 --device 指定（cuda 优先），数据加载并行度通过 --num_workers 控制


四、输入参数与默认值（与实现一致）
- 关键 CLI：
  - --tune {stage1,stage2,stage3,all}：选择 HPO 阶段或“一键三阶段”
  - --n_trials：每阶段试验次数
  - --study_name / --storage：Optuna Study 名称与 SQLite 存储路径（若未显式指定 storage，则默认写入 tuning/stageX/study.db）
  - --rescore_topk：对指定阶段 Top-K 进行多种子复评（与 --rescore_stages 联用）
  - --rescore_stages：复评阶段列表（如 "1,3"；默认 "1"）
  - --seeds：复评使用的随机种子列表，默认 "42,2023,2024"
  - --device、--num_workers 等硬件/数据加载开关


五、输出目录结构（与实现一致）
- tuning/
  - stage1|stage2|stage3/
    - study.db（Optuna Study）
    - trials.csv（所有 trial 的参数与目标值）
    - best_config_stageX.json（合并前序最佳 + 当前阶段最佳参数）
    - topk_rescore.json（仅在执行复评时生成）
    - best_config_rescored.json（仅在执行复评时生成）
  - best_config.json（合并 stage1..3 的最终最佳配置）


六、推荐用法与命令示例
- 一键三阶段（默认仅复评 stage1，可按需指定其它阶段）：
  - 示例（CPU/快速验证）：
    python train.py \
      --train_npz /path/to/train_data.npz \
      --artifacts_dir /path/to/artifacts \
      --tune all \
      --n_trials 30 \
      --rescore_topk 3 \
      --rescore_stages 1 \
      --seeds 42,2023,2024 \
      --epochs 100 \
      --early_patience 30 \
      --num_workers 0 \
      --device cpu

- 单阶段 HPO：
  - 以 stage2 为例，使用默认 tuning 路径：
    python train.py \
      --train_npz /path/to/train_data.npz \
      --artifacts_dir /path/to/artifacts \
      --tune stage2 \
      --n_trials 24 \
      --device cuda

- 仅复评（多阶段选择）：
  - 对 stage1 与 stage3 的 Top-2 进行复评：
    python train.py \
      --train_npz /path/to/train_data.npz \
      --artifacts_dir /path/to/artifacts \
      --rescore_topk 2 \
      --rescore_stages 1,3 \
      --seeds 52,2024,2025 \
      --device cuda


七、里程碑与注意事项
- 里程碑：完成各模型（gatv2/sage/gcn）的多阶段 HPO，并使用 mean_accuracy 复评选择稳健的最终配置；合并输出 tuning/best_config.json
- 注意：若 stage3 选择改变 LapPE（dim/是否拼接/是否高斯加权），objective/复评内将按各自配置重新构图，确保搜索有效；阶段1/2 共享 base 图以避免无效开销
- 训练图表（train_metrics.svg）展示 Loss/AUROC/AUPRC，搜索与早停使用 Accuracy（两者用途不同，属刻意设计）


