# 内部验证集按比例划分方案（草案）

## 背景与现状
当前 `stonco/core/train.py` 内部验证集划分有三种模式：
- `--stratify_by_cancer`（默认）：每个癌种随机 1 张作为验证集。
- 不启用分层：最后 1 张作为验证集。
- `--kfold_cancer`：K 折，每折仍是每癌种随机 1 张。

问题：样本量大的癌种验证集过小，比例偏低；样本量小的癌种过拟合风险高；K 折仍为每癌种 1 张，覆盖不足。

## 目标
1. 支持“按比例划分验证集”，并保底每个癌种至少 1 张（若可行）。
2. 可控的总验证集规模，避免“癌种数多 -> 验证集过大”。
3. 对癌种样本数极小的情况有明确处理策略（尤其是 n=1）。
4. 兼容现有 CLI 与默认行为，必要时保留旧策略。

## 方案概述（精简参数版）
新增“按比例 + 保底每癌种 1 张”的划分策略，尽量不扩展参数。核心思路：
- 以总样本数 `N` 计算目标验证集大小 `target = round(N * val_ratio)`。
- 对每个癌种样本数 `n` 计算初始数量：`base = clamp(round(n * val_ratio), 1, n-1)`。
- 若 `n == 1`，无法同时训练/验证，需要明确处理策略（见“需确认项”）。
- 若 `sum(base)` 与 `target` 不一致：
  - **超出**：优先从样本量最大的癌种中回收（每次 -1），但不低于 1。
  - **不足**：优先从样本量最大的癌种中补充（每次 +1），但不超过 `n-1`。
- 最终在每个癌种内随机抽样对应数量。

这能保证：
- 验证集规模大体满足比例；
- 每癌种至少 1 张（只要该癌种 n>=2）；
- 大癌种不会被“固定 1 张”压缩到极小比例。

## CLI 设计（精简）
仅新增 1 个参数：
- `--val_ratio`：float，例 0.1。用于控制“按比例 + 保底 1 张”的比例，默认 0.2。
不新增 `min/max` 或 `strategy`，保持参数最少。

## K 折（按比例逻辑默认）
`--kfold_cancer` 也改为使用同一套“按比例 + 保底 1 张”的分配逻辑，每折独立随机抽样：
- 每折基于相同比例计算各癌种验证数量，再在癌种内随机抽样；
- 维持原先“生成 K 个不同组合”的去重逻辑（若组合重复则继续抽样，最多尝试上限）。
说明：这不是严格的“互斥 K 折”，但与现有实现一致，且覆盖更合理。

## 影响范围与修改点
主要修改：
- `stonco/core/train.py`
  - 新增比例分层函数，如 `_stratified_ratio_split(...)`。
  - `prepare_graphs()` 内调用逻辑更新。
  - `_run_split_test()` 输出新增统计（验证比例、每癌种数量等）。
  - CLI 参数说明与 help 文案更新。
- 可选：`docs/Tutorial.md` 或 README 中补充示例命令。

## 需确认的关键点（已确认）
1. 默认行为：未传 `--val_ratio` 时也按比例划分。
2. `val_ratio` 默认值：0.2。
3. 癌种样本数 `n=1` 的处理：只放训练集，验证不含该癌种。
4. 当保底导致总验证数偏高时，不回收，直接按原始抽样结果即可。

---

# 新增调整方案（待确认）

## 2.1 训练曲线绘图频率改进（建议选一）
目标：减少保存频率，避免每个 epoch 都画图但又保留趋势。

方案 A（固定间隔 + 小规模全量）：
- `epochs <= 100`：每个 epoch 画；
- `epochs > 100`：每隔 10 epoch 画一次；
- 结束时强制补画最后一次。

方案 B（自动分桶，控制总点数）：
- 设定最大绘图点数 `max_points=100`；
- `step = max(1, ceil(epochs / max_points))`；
- 每 `step` epoch 画一次，结束时补画最后一次。

方案 C（数量级分段）：
- `epochs <= 100`：每个 epoch；
- `101-300`：每 5；
- `301-1000`：每 10；
- `>1000`：每 20；
- 结束时补画最后一次。

说明：当前绘图函数 `_plot_train_metrics` 一次性用 `hist` 画全量曲线；若要“稀疏绘图”，建议改为“训练中按间隔保存 PNG/SVG 快照”或“只在训练结束后用下采样后的点画图”。需要你确认倾向（见“需确认点”）。

## 2.2 train_loss.svg / train_val_metrics.svg 顶行 Epoch 轴不显示
修正策略：
- 在 `_plot_train_metrics` 中对第一行子图设置 `ax.tick_params(labelbottom=True)`；
- 或者关闭 `sharex=True`，改为手动同步范围；
- 推荐前者，改动小、兼容现有布局。

## 3. 单次训练/KFold/LOCO 曲线保存开关
新增参数：
- `--save_train_curves`：默认开启，控制是否保存 `train_loss.svg` 与 `train_val_metrics.svg`。

实现点：
- `run_single_training`、`run_kfold_training`、`run_loco_training` 调用 `_plot_train_metrics` 前检查该开关。
- 不影响 `loss_components.csv`（若你希望一起控制，可明确说明）。

## 4. 内部验证集 + 外部验证集 批量预测（预训练模型）
目标：给定预训练模型（`artifacts_dir`），对内部验证集（由 `meta.json` 的 `val_ids` 指定）与外部验证集目录中的 NPZ 同时预测。

建议实现方式（基于现有 `stonco/core/batch_infer.py` 扩展）：
- 新增参数：
  - `--train_npz`：原训练 NPZ（用于取内部验证集的切片内容）。
  - `--external_val_dir`：外部验证集 NPZ 目录（每个 NPZ 为单切片）。
  - 可复用 `--out_csv` 输出合并结果。
- 内部验证集流程：
  - 从 `meta.json` 读取 `val_ids`；
  - 从 `train_npz` 中按 `slide_ids` 过滤出对应切片；
  - 使用 `InferenceEngine` 的 `pp` 进行 transform + 预测；
  - 输出 `sample_id` 与 `source=internal` 标记。
- 外部验证集流程：
  - 遍历 `external_val_dir/*.npz`；
  - 读取 `sample_id`（或文件名）作为 `sample_id`；
  - 输出 `source=external` 标记。
- 合并输出：
  - 复用现有 spot-level 输出字段（`sample_id`, `spot_idx`, `x`, `y`, `p_tumor` 等）；
  - 有 `y` 时保留 `y_true`，否则置空。
  - 目前 `batch_infer.py` 的 spot-level 列为：`sample_id`, `Barcode`, `spot_idx`, `x`, `y`, `p_tumor`, `pred_label`, `y_true`, `threshold`。

样本级（slide-level）汇总表建议列（基于现有输出字段计算）：
- `sample_id`, `source`, `n_spots`, `threshold`
- `accuracy`, `auroc`, `auprc`, `macro_f1`

输出策略：
- spot-level 结果仍输出到 `out_csv`；
- 额外输出样本级汇总 CSV（例如 `out_csv` 同目录下 `batch_preds_summary.csv`），便于快速查看每个样本的整体预测分布与质量指标。

注意点：
- 单次训练 `meta.json` 将与 KFold 的结构一致，写入 `val_ids`、`train_ids`、`metrics` 等字段，保证内部验证集可复现与对齐。
- `train_npz` 仍需命令行提供（`meta.json` 里未记录训练数据路径），用于从原始 NPZ 取回内部验证切片。

## 5. 训练进度条
目标：训练时显示 epoch 进度，且在 KFold/LOCO 场景下能看清当前折/癌种进度。
建议实现：
- 单次训练：使用 `tqdm` 包裹 epoch 循环，`desc='Train'`，`total=epochs`；每个 epoch 后用 `set_postfix` 更新 `train_loss`、`val_acc`（若可计算）。
- KFold：外层再加一层 `tqdm`，`desc='KFold'`、`total=K`，显示当前 fold；每个 fold 内仍用 epoch 进度条，`desc=f'Fold {i}/{K}'`。
- LOCO：外层 `tqdm` 以癌种数为总数，`desc='LOCO'`；每个癌种内 epoch 进度条 `desc=f'LOCO {ct}'`。
- 为避免双层进度条过多，可将外层设置为 `leave=True`，内层 `leave=False`，终端显示更干净。
- 仅增加可视化，不影响训练逻辑与结果。

---

## 需确认的关键点（新增，已确认）
1. 绘图频率选方案 A；只在训练结束后画图，但对点做下采样。
2. `_plot_train_metrics` 保持“训练结束后画一次”，不做中途快照。
3. `--save_train_curves` 只控制 SVG 曲线保存，`loss_components.csv` 仍照常输出。
4. 批量预测内部验证集时：`--train_npz` 由命令行显式提供。
