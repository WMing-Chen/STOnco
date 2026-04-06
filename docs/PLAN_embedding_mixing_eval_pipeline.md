# PLAN: 通用 embedding 混合评估与 UMAP 配套量化方案

## 1. 目标

在当前 STOnco 工具链上新增一套通用的 embedding 混合评估流程，用于量化不同分组在表征空间中的混合程度，并与 UMAP/t-SNE 可视化配套解释。

该方案覆盖两类场景：

- 单癌种去批次/去切片评估
- 多癌种联合训练或跨癌种整合评估

本方案的核心要求：

1. 主评估在原始 embedding 空间完成
2. 允许在 UMAP/t-SNE 降维空间上计算配套的辅助 LISI 指标
3. 支持任意分组列，不把 `sample_id`、`batch_id`、`cancer_type`、`tumor_label` 写死
4. 工具职责清晰拆分，同时提供一个可一键串联三步流程的入口

---

## 2. 背景与问题

当前主要通过如下可视化文件人工判断混合效果：

- `embedding/umap_tsne_h_by_sample_id.svg`
- `embedding/umap_tsne_h_by_batch_id.svg`

这种方式的问题：

1. 缺少可量化指标，难以严谨比较不同实验
2. 图像判断容易受 UMAP/t-SNE 参数、随机种子和视觉主观影响
3. 无法方便地比较“技术域混合改善了多少”以及“生物结构是否被过度洗平”

因此需要新增量化评估模块，并明确区分：

- 原始 embedding 空间中的正式评估
- UMAP/t-SNE 空间中的辅助评估

---

## 3. 指标选择与解释

### 3.1 统一采用 LISI 框架

第一版主指标采用 LISI（Local Inverse Simpson's Index）。

对某个 spot 的 k 近邻中，设某分组列的局部标签分布为：

`p_c = count(label=c in neighbors) / k`

则该 spot 的 LISI 定义为：

`LISI = 1 / sum_c (p_c^2)`

直观解释：

- 邻域几乎全是同一组，LISI 接近 `1`
- 邻域由多个组均匀混合，LISI 变大
- 理论上限接近该列有效组数

---

### 3.2 iLISI 与 cLISI 的区别

数学公式相同，区别在于“标签角色”和“结果解读”。

#### `iLISI`

用于希望被混合的标签列，即 integration 目标。

典型列：

- `sample_id`
- `batch_id`
- `slide_id`
- `cancer_type`（当实验目标是跨癌种整合时）

解释：

- 值越高，表示该标签在局部邻域里越混合
- 是衡量去批次、去切片、跨域整合效果的主指标

#### `cLISI`

用于希望被保留的标签列，即 conservation 目标。

典型列：

- `tumor_label`
- `cell_type`
- `region`
- `cancer_type`（当实验目标是保留癌种差异时）

解释：

- 数值越高表示该生物标签在局部邻域里越混合
- 一般不希望无约束升高，需要与任务目标一起解读

---

### 3.3 本项目中的推荐解读方式

#### 单癌种去批次场景

推荐重点看：

- `iLISI(sample_id)`
- `iLISI(batch_id)`
- `cLISI(tumor_label)`

理想结果：

- `iLISI(sample_id)` 上升
- `iLISI(batch_id)` 上升
- `cLISI(tumor_label)` 不要异常升高

说明：

- 技术域更混合
- 肿瘤/非肿瘤生物结构没有被明显洗平

#### 多癌种联合训练场景

`cancer_type` 的角色不能写死，应由实验目的决定：

- 若目标是跨癌种整合：`cancer_type -> iLISI`
- 若目标是保留癌种差异：`cancer_type -> cLISI`

---

## 4. 为什么同时保留 embedding-space 与 UMAP-space LISI

### 4.1 embedding-space LISI 作为主指标

在原始 embedding 空间（如 `h_*` 或 `z_clf_*`）上计算的 LISI，反映的是模型真实学到的表征结构。

优点：

- 不受降维投影失真影响
- 更适合作为正式比较和结论依据

因此：

- `iLISI_embed`
- `cLISI_embed`

应作为主结论指标。

### 4.2 UMAP-space / tSNE-space LISI 作为辅助指标

在 `UMAP 2D` 或 `t-SNE 2D` 坐标上计算的 LISI，本质上是在量化“图上看起来混不混”。

优点：

- 便于与 SVG 图像联动解释
- 能把肉眼判断转成数字

限制：

- 只反映降维后的局部结构
- 不能替代原始 embedding 空间评估

因此：

- `iLISI_umap`
- `cLISI_umap`
- `iLISI_tsne`
- `cLISI_tsne`

都应明确标注为辅助指标，不作为主结论。

---

## 5. 代码职责划分

为了避免分析逻辑和可视化逻辑相互污染，建议按四层拆分。

### 5.1 `stonco/utils/export_spot_embeddings.py`

职责：模型输出层

只负责：

- 从训练好的模型导出 spot-level embedding
- 输出 metadata 列
- 支持 `h` / `z_clf` / `z64`

不要负责：

- UMAP/t-SNE 计算
- LISI 计算
- 指标汇总

输出仍然是上游标准输入 CSV。

---

### 5.2 `stonco/utils/visualize_umap_tsne.py`

职责：展示层

负责：

- 从 embedding CSV 读取表征
- 计算 UMAP / t-SNE
- 绘制 SVG

新增功能：

- 支持导出降维坐标 CSV，例如 `--out_coords_csv`

该坐标 CSV 应至少包含：

- spot metadata 列
- `embed_source`
- `umap_1`, `umap_2`
- `tsne_1`, `tsne_2`

目的：

- 保证后续在 UMAP/t-SNE 空间上的 LISI 计算与图像完全一致
- 避免评估脚本重新跑降维导致结果与图不一致

---

### 5.3 `stonco/utils/evaluate_embedding_mixing.py`

职责：评估层

新增独立量化入口，负责：

- 读取 embedding CSV 或降维坐标 CSV
- 在指定空间上计算 LISI
- 对任意指定列输出 `iLISI` 或 `cLISI`
- 输出 summary CSV
- 可选输出每个 spot 的局部 LISI 结果 CSV

这是正式指标入口，不负责画图。

---

### 5.4 `stonco/utils/embedding_analysis.py`

职责：公共算法层

新增轻量公共模块，统一封装可复用逻辑：

- 读取 `h_*` / `z_clf_*` / `z64_*`
- 读取 `umap_*` / `tsne_*` 坐标
- 标准化
- 计算 UMAP/t-SNE
- 建立 kNN
- 计算单列 LISI
- 汇总 `mean / median / q25 / q75`

用途：

- `visualize_umap_tsne.py` 复用降维读取与计算
- `evaluate_embedding_mixing.py` 复用空间读取、kNN 和 LISI 逻辑

这样可以避免：

- 两套 embedding 读取逻辑
- 两套 UMAP 参数
- 两套 kNN / LISI 实现

---

## 6. 新增一键串联三步的 pipeline 入口

除了职责分离的三个主工具外，再新增一个编排脚本：

- `stonco/utils/run_embedding_analysis_pipeline.py`

职责：流程编排层

该脚本不实现具体算法，只负责串联：

1. 导出 embedding
2. 生成 UMAP/t-SNE 图和坐标
3. 计算 embedding-space 与 UMAP/t-SNE-space 的 LISI 指标

这样保留了：

- 单工具可单独复用
- 又能满足日常实验中“一条命令跑完整分析”的需求

### 6.1 编排脚本建议职责

负责：

- 解析统一 CLI
- 调用内部函数或子模块完成三步
- 统一管理输出目录和文件命名

不负责：

- 直接复制三套算法代码
- 在脚本里重新实现 UMAP/LISI

### 6.2 pipeline 的默认三步

第 1 步：调用 embedding 导出逻辑

产物：

- `spot_embeddings_{embed_source}.csv`

第 2 步：调用 UMAP/t-SNE 可视化逻辑

产物：

- 多个 SVG 图
- `embedding_coords_{embed_source}.csv`

第 3 步：调用 LISI 量化逻辑

产物：

- `mixing_metrics_{embed_source}.csv`
- 可选 `mixing_metrics_{embed_source}_spot.csv`

### 6.3 pipeline 推荐命名策略

在 `out_dir` 下统一输出：

- `spot_embeddings_h.csv`
- `embedding_coords_h.csv`
- `mixing_metrics_h.csv`
- `mixing_metrics_h_spot.csv`
- `umap_tsne_h_by_sample_id.svg`
- `umap_tsne_h_by_batch_id.svg`
- `umap_tsne_h_by_tumor_label.svg`

---

## 7. 接口设计建议

### 7.1 `evaluate_embedding_mixing.py`

建议参数：

- `--input_csv`
- `--input_kind {embedding,coords}`
- `--space {embedding,umap,tsne,all}`
- `--embed_source {h,z_clf,z64}`
- `--group_cols`
- `--group_roles`
- `--k_values`
- `--out_csv`
- `--out_spot_csv`
- `--max_points`
- `--seed`

其中：

- `input_kind=embedding` 时从 `h_*` / `z_clf_*` / `z64_*` 取向量
- `input_kind=coords` 时从 `umap_1, umap_2` 或 `tsne_1, tsne_2` 取向量

`group_roles` 建议显式指定，格式：

- `sample_id:integration`
- `batch_id:integration`
- `tumor_label:conservation`
- `cancer_type:integration`

输出中统一增加：

- `space`
- `metric_name`
- `group_role`

例如：

- `space=embedding`, `metric_name=iLISI`
- `space=umap`, `metric_name=cLISI`

---

### 7.2 `visualize_umap_tsne.py`

建议新增参数：

- `--out_coords_csv`

如提供，则除生成 SVG 外，再导出每个 spot 的 UMAP/t-SNE 坐标表。

如不提供，则保持原有行为不变，保证兼容。

---

### 7.3 `run_embedding_analysis_pipeline.py`

建议参数按三步合并设计：

- 输入模型与数据：
  - `--artifacts_dir`
  - `--train_npz`
  - `--npz`
  - `--npz_glob`
  - `--subset`
- embedding 导出：
  - `--embed_source`
- 可视化：
  - `--color_cols`
  - `--max_points`
  - `--point_size`
  - `--alpha`
- 量化：
  - `--group_cols`
  - `--group_roles`
  - `--k_values`
  - `--lisi_spaces {embedding,umap,tsne,all}`
- 输出：
  - `--out_dir`
  - `--seed`

默认行为建议：

1. 先导出 embedding CSV
2. 再生成图与坐标 CSV
3. 最后计算 LISI

这样可以确保：

- UMAP 图与 UMAP-space LISI 使用同一份坐标
- 所有中间产物默认落盘，便于复查

---

## 8. 当前场景中的推荐默认方案

针对你现在的 BRCA-only 去批次测试，推荐默认分组角色：

- `sample_id:integration`
- `batch_id:integration`
- `tumor_label:conservation`

推荐默认空间：

- `embedding`
- `umap`

推荐默认 `k`：

- `15`
- `30`
- `50`

解释方式：

- 主看 `iLISI_embed(sample_id/batch_id)`
- 辅看 `iLISI_umap(sample_id/batch_id)`
- 同时监控 `cLISI_embed(tumor_label)`

---

## 9. 计算量与实现原则

按当前 `~39386` 个 spot、`64` 维 embedding 的规模，第一版实现应遵守：

1. 不构建全量 pairwise distance 矩阵
2. 使用 kNN 作为核心计算结构
3. 一次构图复用多个分组列和多个 `k`
4. UMAP-space LISI 复用已保存的 2D 坐标

这样可以在计算量可接受的前提下，同时支持：

- 主空间 LISI
- UMAP/t-SNE 配图空间 LISI

---

## 10. 测试与验收

至少覆盖以下场景：

1. 单癌种 BRCA 输入
   - `sample_id` / `batch_id` / `tumor_label` 正常输出
   - `cancer_type` 若唯一值则跳过
2. 多癌种输入
   - `cancer_type` 可作为 `integration` 或 `conservation`
3. 同一份坐标
   - `visualize_umap_tsne.py` 输出的 UMAP 图与 `evaluate_embedding_mixing.py` 的 `space=umap` 使用完全相同坐标
4. 多 `k`
   - `15/30/50` 都能出汇总结果
5. embedding 与 UMAP 结果同时存在
   - 输出明确标注 `space`
6. 缺失值与唯一值列处理
   - 不导致整体失败
7. sanity check
   - 完全分离数据的 LISI 接近 `1`
   - 完全均匀混合数据的 LISI 明显增大

---

## 11. 最终代码结构建议

- `stonco/utils/export_spot_embeddings.py`
  - 导出 embedding
- `stonco/utils/visualize_umap_tsne.py`
  - 生成 UMAP/t-SNE 图
  - 可选导出降维坐标 CSV
- `stonco/utils/evaluate_embedding_mixing.py`
  - 计算 embedding / UMAP / t-SNE 空间上的 `iLISI` / `cLISI`
- `stonco/utils/embedding_analysis.py`
  - 公共算法层
- `stonco/utils/run_embedding_analysis_pipeline.py`
  - 一键跑三步流程的编排入口

该结构的优点：

1. 上游数据准备、可视化、指标评估职责分离
2. 算法逻辑可复用，不会重复实现
3. 支持单步使用，也支持一键串联使用
4. 既适配当前 BRCA-only 测试，也适配未来多癌种通用评估
