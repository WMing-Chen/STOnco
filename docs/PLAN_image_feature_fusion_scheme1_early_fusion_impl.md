# STOnco 融合图像特征：方案 1（早期融合）实现参考文档

> 本文用于指导后续改代码（暂不实现）。本文已包含方案 1 所需的“数据与预处理契约”（原 `docs/PLAN_image_feature_fusion.md` 的第 2 节），因此可作为方案 1 的完整修改方案参考文档。

---

## 1. 方案 1 的目标与核心约定

### 1.0 快速摘要（来自 `docs/PLAN_image_feature_fusion.md`）

**思路**：把图像特征当作额外的节点特征，直接拼接到 `x`：

- `x = [Xp_gene, Xp_img, img_mask, (LapPE)]`

**代码改动面**（相对最小）：

- `prepare_data.py`：读取并写入 `X_img / X_imgs`、`img_mask / img_masks`、`img_feature_names`
- 训练/推理/批推理：在 `pp.transform(X, gene_names)` 之后，把 `X_img` 经过 `img_pp.transform(...)` 得到 `Xp_img`，再 `np.hstack`
- `meta.json`：记录 `use_image_features / img_use_pca / img_pca_dim` 等

**优点**：

- 改动最小，能快速验证“图像特征有没有增益”
- 对 GNN 类型不敏感（GATv2/GCN/SAGE 都适用）

**风险/注意**：

- 维度膨胀：`X_img=2048`，建议默认 `StandardScaler + PCA` 降到 `img_pca_dim=256` 再拼接
- 解释性：`infer.py` 的 gene attribution 假设“输入前半段是基因特征”；必须固定拼接顺序（`gene → image → mask → LapPE`），并且仅对 gene 段做归因

### 1.1 输入与输出不变的部分

- 任务不变：spot 级 tumor/non-tumor 二分类
- 图不变：仍用 `xy` 构建 KNN 图，边权/ LapPE 逻辑保持现有实现
- gene 预处理不变：继续复用 `stonco/utils/preprocessing.py:Preprocessor`
- 解释性不变：推理阶段只保留现有 gene attribution（`infer.py:compute_gene_attr_for_graph`）

### 1.2 新增的输入（图像特征）

每个样本目录新增：`*_image_features.csv`，包含：

- `Barcode` 列
- `image_feature_1..image_feature_2048` 共 2048 维

CSV → NPZ 的契约与严格校验，按本文 **第 2 节**执行（严格发现、barcode 不重复、NaN/inf 报错、barcode 缺失允许填充+mask+warning）。

### 1.3 早期融合的节点特征拼接顺序（非常重要）

为了保证现有 gene attribution 逻辑不需要大改（它默认“gene 特征在最前段”），本方案固定节点输入特征顺序为：

1. `Xp_gene`（gene 预处理输出）
2. `Xp_img`（图像预处理输出，默认 256 维；也可不 PCA 直接 2048 维）
3. `img_mask`（单列 0/1，float32）
4. `LapPE`（如果 `concat_lap_pe=1` 且 `lap_pe_dim>0`）

即：`x = [Xp_gene, Xp_img, img_mask, (LapPE)]`

> 解释性说明：`infer.py` 的 gene attribution 会把 `x[:, :feat_dim_gene]` 当作 gene 特征段。只要 gene 段始终在最前面，这个假设就成立。

---

## 2. 数据与预处理层的推荐改造点（无论选哪种融合）

> 本节作为“数据契约”：把 **数据读取 → barcode 对齐/缺失 → NPZ 键 → 图像预处理器拟合/保存 → 推理一致性** 一次性确定下来。后续任何融合方案（方案 1/2/3/4）都必须遵守这里的规则。

### 2.1 输入文件（CSV）与严格校验

每个样本目录下必须存在且仅存在 3 个文件（大小写不敏感，均以文件后缀匹配）：

- `*exp.csv`
- `*coordinates.csv`
- `*image_features.csv`（新增，2048 维）

发现规则（严格，避免静默错配）：

- 若某一类文件匹配到 0 个或 >1 个：直接报错，并打印匹配到的文件列表与样本目录名（sample_id）。

`*_image_features.csv` 的校验（严格）：

- 必须包含 `Barcode` 列
- 除 `Barcode` 外必须有且仅有 `2048` 个特征列
- 特征列名/顺序必须一致（推理时也会校验一致性）
- `Barcode` 不允许重复（发现重复直接报错）
- 图像特征中出现 `NaN/inf/-inf`：直接报错并指出样本/列（数据质量问题优先在数据准备阶段暴露）

### 2.2 Barcode 对齐、缺失处理与样本级 warning（数据准备阶段完成）

总体策略：**gene（exp/coord）严格对齐；image 允许缺失并“填充 + mask + 样本级 warning”。**

- 最终 spot 顺序以 `coordinates.csv` 的 barcode 顺序为基准（避免排序导致错位）
- gene 对齐：`barcodes_used = barcodes_coord ∩ barcodes_exp`
  - 不在交集内的 spot 直接丢弃（保持现有 `prepare_data.py` 行为一致）
- image 对齐：将 `image_features.csv` 按 `Barcode` 左连接到 `barcodes_used`
  - 缺失的 barcode：该行 `X_img` 全 0，`img_mask=0`
  - 匹配到的 barcode：写入 2048 维向量，`img_mask=1`
  - image CSV 的额外 barcode（不在 `barcodes_used`）：忽略
- warning（按样本输出，便于排查）：
  - 当 `missing_count > 0` 时：打印 `sample_id=<...> missing image features for k / n_spots (filled with zeros, img_mask=0)`

### 2.3 NPZ 扩展规范（新增键，保持旧格式可用）

图像特征作为“可选键”写入 NPZ（旧 NPZ 缺失这些键时，代码需可退化为 gene-only）。

- 单 slide NPZ：新增
  - `X_img`: `(n_spots, 2048)` `float32`
  - `img_mask`: `(n_spots,)` `uint8`（0/1）
  - `img_feature_names`: `(2048,)` `object`
- 多 slide 训练 NPZ：新增
  - `X_imgs`: `object array`，每个元素 `(n_spots_i, 2048)` `float32`
  - `img_masks`: `object array`，每个元素 `(n_spots_i,)` `uint8`
  - `img_feature_names`: 同上（全体样本一致，存 1 份即可）

兼容性约束：

- 旧 NPZ 没有 `X_img/X_imgs`：训练/推理必须仍可运行（图像分支自动关闭/退化）
- 新 NPZ 有 `X_img/X_imgs`：旧代码不保证兼容，但应给出明确错误提示

### 2.4 图像特征预处理器（训练拟合 + 推理复用，允许关闭 PCA）

原则：图像特征走独立预处理器，不与 gene `Preprocessor` 混用。

处理流水线（`X_img -> Xp_img`）：

1. `StandardScaler`：在训练集所有 spots 上拟合；推理阶段只 `transform`
2. 可选 `PCA`：预处理阶段允许关闭 PCA
   - 默认使用 PCA：`img_pca_dim=256`
   - 允许 `img_use_pca=0`：跳过 PCA，直接使用 2048 维（注意显存/速度成本）
   - `whiten/solver/random_state` 固定在流程中，不暴露为可调超参（你已确认不需要作为模型/CLI 的显式超参）；建议固定为：
     - `whiten=False`
     - `svd_solver='randomized'`
     - `random_state=42`

保存位置：`artifacts_dir/` 根目录，用不同文件名区分（避免与 gene 预处理器文件冲突）：

- `img_feature_names.txt`
- `img_scaler.joblib`
- `img_pca.joblib`（仅当 `img_use_pca=1`）

说明：

- 推理阶段不需要显式读取/依赖 `whiten/solver/random_state`；实际变换以保存的 `img_scaler.joblib` 与（可选）`img_pca.joblib` 为准。
- `img_use_pca/img_pca_dim` 建议记录到训练产物的 `meta.json`（`cfg`）中；不额外新增 `img_preprocess_meta.json` 文件，以减少文件审查量。

### 2.5 “预处理契约”对后续融合方案的约束

- gene attribution 仍基于 gene 分支；任何“拼接到 `g.x`”的方案必须保证 **gene 特征在 `g.x` 的最前段**（保持现有解释逻辑假设）
- `img_mask` 必须写入 NPZ（即使理论上不会缺失），用于：
  - 数据质量检查（missing warning 可追溯）
  - 后期融合（尤其注意力融合）时屏蔽“填充 0”的 image token
- 推理/外部 NPZ 的 `img_feature_names` 必须与训练一致；若不一致，直接报错（避免 silent mismatch）

### 2.6 推理阶段如何使用预处理器（保证训练/推理一致）

- `infer.py` / `batch_infer.py` 推理时会加载训练产物：
  - gene：沿用现有 `Preprocessor.load(artifacts_dir)`
  - image：加载 `img_scaler.joblib` +（可选）`img_pca.joblib`
- 推理阶段只做 `transform`，不做 `fit`
- 推理输入缺少 `X_img` 时（或显式关闭图像分支）：
  - 自动退化为 gene-only，不报错

---

## 3. 需要新增/修改的参数（train 为主，infer 读取 meta.json）

### 3.1 `train.py` 新增配置项（写入 `meta.json:cfg`）

建议新增（或写入 cfg 默认值）：

- `use_image_features`: `bool`（是否启用方案 1 早期融合；默认建议 `False` 保持旧行为）
- `img_use_pca`: `bool`（是否对 2048 维做 PCA；默认 `True`）
- `img_pca_dim`: `int`（默认 `256`；仅当 `img_use_pca=1` 生效）

> `whiten/solver/random_state` 等固定参数不需要推理读取；推理以 `img_pca.joblib` 的实际对象为准（无需写进 `meta.json`）。

### 3.2 CLI 形态建议（只对 train 暴露）

`stonco/core/train.py` 建议新增 CLI：

- `--use_image_features 0/1`（覆盖 `cfg.use_image_features`）
- `--img_use_pca 0/1`（覆盖 `cfg.img_use_pca`）
- `--img_pca_dim 256`（覆盖 `cfg.img_pca_dim`）

`infer.py/batch_infer.py` 默认不新增这些 CLI（避免“推理配置与训练不一致”），只读取 `meta.json` 的 cfg。

---

## 4. 代码改动清单（方案 1 必需）

> 目标：让“训练、推理、批推理、导出 embedding、可视化”在启用 `use_image_features=1` 时都能跑通；否则会出现模型输入维度不一致的问题。

### 4.1 数据准备：`stonco/utils/prepare_data.py`

需要改动：

1. 新增读取函数：`_read_image_features(image_csv) -> (barcodes, feature_names, df_features)`
   - 校验：Barcode 列、2048 维、列名一致、barcode 不重复、所有值 finite
   - 输出：以 barcode 为 index 的 `(n_spots, 2048)` DataFrame
2. 修改 `build_train_npz(...)`
   - 自动发现 `*image_features.csv`
   - gene 对齐：仍用 `bc_inter = bc_coord ∩ bc_exp`（保持现有）
   - image 对齐：对 `bc_inter` 做 reindex：
     - 缺失行填 0，并构造 `img_mask`（缺失为 0，存在为 1）
     - 若缺失数量 >0：按样本打印 warning
   - 保存新增键：
     - `X_imgs`（object array）
     - `img_masks`（object array）
     - `img_feature_names`（object）
3. 修改 `build_single_npz(...)`
   - 同上写入 `X_img/img_mask/img_feature_names`
4. `build_val_npz(...)` 会复用 `build_single_npz`，无需单独改逻辑

NPZ 输出键汇总（启用图像时）：

- train NPZ：`Xs/ys/xys/slide_ids/gene_names/barcodes` + `X_imgs/img_masks/img_feature_names`
- single NPZ：`X/xy/gene_names/(y)/barcodes/sample_id` + `X_img/img_mask/img_feature_names`

### 4.2 图像预处理器：新增到 `stonco/utils/preprocessing.py`（或新文件）

建议新增一个轻量类（示意名）：

- `class ImagePreprocessor:`
  - `fit(X_img_list, img_masks_list=None)`：拟合 `StandardScaler` + 可选 `PCA`
    - 关键：拟合时只使用 `mask==1` 的行（否则缺失填 0 会污染均值/方差）
  - `transform(X_img, img_mask=None)`：返回 `Xp_img`
    - 关键：对 `mask==0` 的行保持输出为 0（不要让 scaler 把“填 0”变成 `(-mean)/std`）
  - `save(artifacts_dir)`：保存到根目录（按第 2 节契约）
    - `img_feature_names.txt`（如需要）
    - `img_scaler.joblib`
    - `img_pca.joblib`（仅当 `img_use_pca=1`）
  - `load(artifacts_dir)`：推理时加载

固定参数（写死在实现里即可）：

- PCA：`whiten=False`, `svd_solver='randomized'`, `random_state=42`

> 说明：这里“预处理器类”不是必须放在 `preprocessing.py`，但放在一起最方便 `train/infer/batch_infer/export` 复用。

### 4.3 训练：`stonco/core/train.py`

关键改动点：

1. CLI + cfg：
   - 加入 `use_image_features/img_use_pca/img_pca_dim`
   - 这些信息写入 `meta.json` 的 `cfg`
2. `prepare_graphs(...)`：
   - 从 train NPZ 读取：
     - 现有：`Xs/ys/xys/slide_ids/gene_names`
     - 新增：`X_imgs/img_masks/img_feature_names`（当 `use_image_features=1` 时必须存在）
   - 拟合并保存图像预处理器：
     - 拟合时机：与 gene `Preprocessor.fit(...)` 一样，在 split 之前，用所有 slides 的 spots 拟合
     - 保存到 `artifacts_dir/`（`img_scaler.joblib` 等）
   - 构图时构造融合节点特征：
     - `Xp_gene = pp.transform(X, gene_names)`
     - `Xp_img = img_pp.transform(X_img, img_mask)`
     - `img_mask_col = img_mask.astype(float32).reshape(-1,1)`
     - 之后交给 `assemble_pyg` 时拼接成 `data.x`
3. `assemble_pyg(...)`（train.py 内部的同名函数）：
   - 当前签名：`assemble_pyg(Xp, xy, y, cfg)`
   - 方案 1 建议改为更明确的签名（示例）：
     - `assemble_pyg(Xp_gene, Xp_img, img_mask, xy, y, cfg)`
   - 在函数内完成 `x = hstack([...])` + LapPE 拼接
4. `in_dim` 计算：
   - 仍然用 `pyg_graphs[0].x.shape[1]` 推断即可

### 4.4 单样本推理：`stonco/core/infer.py`

关键改动点：

1. `InferenceEngine.__init__`：
   - 从 `meta.json` 读取 `cfg.use_image_features/img_use_pca/img_pca_dim`
   - 当 `use_image_features=1` 时加载：
     - `img_scaler.joblib`
     - `img_pca.joblib`（如果 cfg 指定使用 PCA）
2. `main()` 读取 NPZ：
   - 读取 `X/xy/gene_names`
   - 若 NPZ 有 `X_img/img_mask`：使用它们
   - 若缺失：构造 `X_img=zeros`、`img_mask=zeros` 作为兜底（等价于“无图像信息”，但保持输入维度一致）
3. 构造融合节点特征并构图：
   - `Xp_gene = pp_gene.transform(X, gene_names)`
   - `Xp_img = img_pp.transform(X_img, img_mask)`
   - `data_g = assemble_pyg_multimodal(...)`
4. gene attribution：
   - 保持现有 `compute_gene_attr_for_graph` 不动
   - 前提：`g.x` 的最前段仍是 gene 特征段（见 1.3）

### 4.5 批量推理：`stonco/core/batch_infer.py`

关键改动点：

1. `SlideNPZDataset.__getitem__`：
   - 除 `X/xy/gene_names` 外，读取 `X_img/img_mask`（若缺失则填 0 + mask=0 兜底）
   - 构造融合图 `g.x`（与 infer/train 保持完全一致的拼接顺序）
2. `InferenceEngine` 的复用：
   - 目前 batch 推理复用了 `InferenceEngine.compute_gene_attr_for_graph`（它依赖 `g.x` 的维度与顺序）
   - 所以 batch 推理里也必须走同样的 `x` 构造逻辑

### 4.6 导出 embedding：`stonco/utils/export_spot_embeddings.py`

如果启用 `use_image_features=1`，该脚本也需要改：

- 从 NPZ 读取 `X_img/img_mask`（train NPZ 的 `X_imgs/img_masks` 或 single NPZ 的 `X_img/img_mask`）
- 加载图像预处理器（同 infer）
- 构造融合 `g.x`，否则模型输入维度不匹配，无法导出 z64

### 4.7 可视化：`stonco/utils/visualize_prediction.py`

同理：若使用融合模型，该脚本也要读取图像特征并构造融合 `g.x`，否则会因 in_dim 不一致而报错。

---

## 5. 关键逻辑伪代码（帮助实现时对齐）

### 5.1 `Xp_img` 的 transform（避免 mask=0 污染）

```text
Xp_img = zeros(n_spots, img_out_dim)
idx = where(img_mask == 1)
Xp_img[idx] = img_pp.transform(X_img[idx])
```

> 这里的 `img_pp.transform` 指“对 idx 子集做 scaler(+pca)”，mask=0 的行保持为 0。

### 5.2 统一的节点特征拼接（推荐封装成函数）

```text
x = concat([Xp_gene, Xp_img, img_mask[:,None]], axis=1)
if concat_lap_pe and lap_pe_dim>0:
  x = concat([x, LapPE], axis=1)
```

---

## 6. 回归/自检建议（实现后用于验证）

- 数据准备：
  - 生成 single NPZ，检查 keys：`X_img/img_mask/img_feature_names` 是否存在、shape 是否正确
  - 人为删除 image CSV 中部分 barcode，确认 warning 与 mask 行为正确
- 训练：
  - `use_image_features=0`：应与当前行为一致
  - `use_image_features=1`：能正常训练并产出 `img_scaler.joblib`（及可选 `img_pca.joblib`）
- 推理/批推理：
  - 用同一个 artifacts_dir 推理单个/批量 NPZ，确保不报维度错误
  - gene attribution 能正常输出（不要求一致，只要求流程不崩）

---

## 7. 待你确认的问题（确认后再开始实现）

1. **`use_image_features` 默认值**：希望默认 `0` 
2. **`img_mask` 是否作为模型输入的一维特征？**
   - 当前方案 1 设计为“拼进 `g.x`”（更鲁棒，也与后续注意力融合的 mask 语义一致）
   - 如果你希望 `img_mask` 只用于数据质检、不进入模型，需要在本文与后续方案中统一改动（同时更新拼接顺序与 in_dim）
3. **当 `img_use_pca=1` 但训练总 spots 数 < `img_pca_dim`** 时怎么处理？
   - A. 直接报错
4. **图像预处理器拟合范围**：是与 gene `Preprocessor` 保持一致（在 split 前用 train_npz 内所有 slides 的所有 spots 拟合）
