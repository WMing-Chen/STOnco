# STOnco 融合图像特征的设计方案（草案）

> 基于你最新确认：图像特征维度 2048、列名/顺序一致、推理阶段仅保留 gene attribution、默认 backbone 仍以 `gatv2` 为主（但保持 `gcn/sage` 可选）。

## 0. 现有结构快速回顾（基于当前代码）

### 0.1 数据流（CSV → NPZ → PyG 图）

- 数据准备：`stonco/utils/prepare_data.py`
  - 读取 `*_exp.csv`（首列 `Barcode` + 后续基因列）与 `*_coordinates.csv`（`Barcode` + 坐标列 + 可选标签列）
  - 对训练集：遍历多个 slide 目录，做 **union gene**，输出 `train_data.npz`
  - 对单 slide：输出 `{X, xy, gene_names, (y), (barcodes), (sample_id)}` 的 `.npz`
- 训练/推理：`stonco/core/train.py`、`stonco/core/infer.py`、`stonco/core/batch_infer.py`
  - `Preprocessor`（`stonco/utils/preprocessing.py`）对基因表达做 `CP10K + log1p → HVG → StandardScaler → (可选 PCA)`，得到 `Xp`
  - `GraphBuilder` 根据坐标构建 KNN 图与高斯边权 `edge_weight`，并可计算 `LapPE`
  - 节点输入特征：当前为 `x = [Xp, (LapPE)]`（`concat_lap_pe=1` 时拼接）

### 0.2 模型结构（单模态：基因）

`stonco/core/models.py` 中 `STOnco_Classifier`：

- `GNNBackbone(in_dim=...)`：GATv2/GCN/GraphSAGE（WeightedSAGEConv）
- `ClassifierHead`：固定输出 `z64`（用于导出/UMAP）与二分类 `logits`
- 可选双域对抗：`dom_slide` / `dom_cancer`（GRL + DomainHead）

> 关键点：当前代码把“节点特征”视为一个矩阵 `x` 直接送入 GNN，因此最自然的融合方式是让图像特征成为 `x` 的一部分，或作为独立分支在后期与 gene 表示融合。

---

## 1. 目标与新增数据格式

你计划在每个样本目录新增一个文件：

- `BRCA1/BRCA1_image_features.csv`
  - 列：`Barcode, image_feature_1, image_feature_2, ...`
  - 每行对应一个 spot（与 exp/coordinates 一样用 `Barcode` 对齐）

核心问题：**图像特征在 STOnco 中应作为节点特征直接融合，还是作为独立分支做后期融合？**

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


## 3. 几种可选融合方案（从易到难）

下面给出 4 个可落地方案。你可以先选 1 个做 baseline，再逐步升级。

### 方案 1：早期融合（最推荐的 baseline）

**思路**：把图像特征当作额外的节点特征，直接拼接到 `x`：

- `x = [Xp_gene, Xp_img, img_mask, (LapPE)]`

**代码改动面**（相对最小）：

- `prepare_data.py`：读取并写入 `X_img / X_imgs`、`img_mask / img_masks`、`img_feature_names`
- 训练/推理/批推理：在 `pp.transform(X, gene_names)` 之后，把 `X_img` 经过 `img_pp.transform(...)` 得到 `Xp_img`，再 `np.hstack`
- `meta.json`：记录 `use_image_features / img_dim / img_pca_dim / img_feature_names` 等

**优点**：

- 改动最小，能快速验证“图像特征有没有增益”
- 对 GNN 类型不敏感（GATv2/GCN/SAGE 都适用）

**风险/注意**：

- 维度膨胀：你这里 `X_img=2048`，强烈建议先 `StandardScaler + PCA` 降到 `img_pca_dim=256`（见第 2 节契约），再做拼接
- 解释性：`infer.py` 目前的 gene attribution 假设“输入前半段是基因特征”；需要约定拼接顺序（建议始终 `gene → image → LapPE`），并在解释时仅对 gene 维度做归因

---

### 方案 2：晚期融合（双塔：Gene-GNN + Image-MLP）

**思路**：保持现有 gene-GNN 路径不动；对图像特征单独做一个 MLP 编码，然后在分类头之前融合：

- `h_gene = GNN(X_gene, edge_index)`
- `h_img = MLP(X_img)`（可先 PCA 到 `img_pca_dim`；此分支不做 message passing）
- `h_fused = Fuse([h_gene, h_img])`（concat + MLP / gating）
- `logits = Classifier(h_fused)`

**优点**：

- 模态解耦更清晰；图像特征“不会干扰”基因预处理与 GNN 表达学习
- 更容易做缺失模态（没有 `X_img` 时可退化为原模型）

**风险/注意**：

- 需要改 `models.py`（引入第二分支与融合模块），以及保存/加载权重的兼容处理
- 若希望图像信息也能沿图传播，这个方案需要升级为“Image-GNN”（见方案 3 / 方案 4）

---

### 方案 3：双 GNN（Gene-GNN + Image-GNN）

**思路**：两个模态各跑一个 GNN（共享同一张空间图），然后融合：

- `h_gene = GNN_gene(X_gene, edge_index)`
- `h_img = GNN_img(X_img, edge_index)`
- `h_fused = Fuse([h_gene, h_img])`

**优点**：

- 允许图像特征在空间邻域内传播，表达能力更强

**风险/注意**：

- 计算/显存翻倍，训练更慢
- 超参数更多（两套 backbone 的 hidden/layers/dropout 是否共享？）

---

### 方案 4：注意力后期融合（Gene-GNN+MLP + Image-GNN+MLP + Modality Attention）

你提出的思路可以落到一个“每个 spot 两个模态 token”的注意力融合结构上，并且能较好兼容 STOnco 现有的 `ClassifierHead(z64)` 习惯。

**整体数据/图输入**：

- 图仍由 `xy` 构建（KNN + 可选 LapPE），与现有完全一致
- gene 分支输入：`Xp_gene (+LapPE)`（保持现有顺序与 gene attribution 逻辑）
- image 分支输入：`X_img → img_pp(StandardScaler + 可选 PCA)`（默认 `img_pca_dim=256`，也允许 `img_use_pca=0`），并携带 `img_mask`

**编码器设计（建议）**：

- 关键约束：**gene 与 image 两个分支在“二分类前（64 维及之前）”结构尽量统一**，即都采用 `GNNBackbone + MLP(…→64)` 的编码方式。

- Gene 分支（与现有结构对齐）：
  - `h_gene = GNNBackbone_gene(x_gene, edge_index)`
  - `z64_gene = MLP64_gene(h_gene) -> R^{64}`（建议直接复用现有 `ClassifierHead` 的前三层 `[256,128,64]` 作为“64维投影头”，分支内不产出 logits）
- Image 分支（同构）：
  - `x_img = img_pp.transform(X_img)`（`StandardScaler` + 可选 `PCA`；默认 `img_pca_dim=256`，也允许 `img_use_pca=0` 直接 2048 维）
  - `h_img = GNNBackbone_img(x_img, edge_index)`（允许图像特征沿空间传播）
  - `z64_img = MLP64_img(h_img) -> R^{64}`（结构与 gene 分支一致：`[256,128,64]`）

**注意力融合（Modality Attention）**：

把每个 spot 的两个模态表示当作长度为 2 的序列：

- `T = stack([z64_gene, z64_img], dim=1)`，shape 为 `(n_spots, 2, 64)`
- 采用 `MultiheadAttention(64, num_heads)`（或更轻量的“权重注意力”）做融合：
  - Transformer 风格：对 `T` 做 self-attention，然后对 token 维做 pooling（如 mean / 取 gene token / 取注意力输出的 [CLS] 替代）
  - 轻量风格：`α = softmax(MLP([z64_gene; z64_img]))` 得到两个权重，`z64_fused = α1*z64_gene + α2*z64_img`
- 缺失处理（与你的“填充+mask”一致）：当 `img_mask=0` 时，对该 spot 的 image token 置 0，并在注意力里屏蔽 image token（或强制 `α2=0`）

**分类头与 z64 兼容**：

- 融合后直接得到 `z64_fused`，将其作为模型对外的 `z64`（用于 `export_spot_embeddings.py` / UMAP 等）
- 二分类层建议保持与现有一致的“64→1”线性输出：
  - `logits = Linear64To1(z64_fused)`（等价于复用现有 `ClassifierHead` 的最后一层 `fc4`）

**域对抗头的接法**：

- 默认建议接在融合后的 `z64_fused` 上：`dom_head(grad_reverse(z64_fused))`
- 如果担心图像分支引入强域信息，可改为只对 `z64_gene`（或 `h_gene`）做域对抗

**优点**：

- “每个 spot 动态决定更相信哪种模态”，比简单 concat 更灵活
- 图像分支通过 GNN 引入空间传播，符合你希望的建模方式

**风险/注意**：

- 这是当前最重的方案（双 GNN + 注意力），训练成本明显上升
- 需要在 `Data` 中携带 `X_img/img_mask` 或在 `x` 中按约定拼接，并确保推理/批推理/导出 embedding 的代码路径都一致

**结合现有代码的实现落点（便于后续落地）**：

- NPZ：按第 2 节的 `X_img/img_mask` 扩展写入；gene 仍保持原结构
- 图构建：`assemble_pyg(...)` 里保持 `data.x` 为 gene(+LapPE) 特征；把图像分支输入单独挂到 `data.x_img`（以及 `data.img_mask`）
- 模型接口：在 `STOnco_Classifier` 的基础上扩展为“可选图像分支”或新增一个 `STOnco_Multimodal_Classifier`，并保证：
  - 老代码路径（不提供图像特征）依旧可用
  - 推理引擎 `InferenceEngine` 可以从 `meta.json` 判断是否需要读取/使用图像特征

---

## 4. 域对抗（Dual-domain）在多模态下的几个选择

目前域对抗头接在 GNN 输出 `h` 上。加入图像特征后，你可以选择：

1. **域头作用在融合后的表示**（最直接）：`dom_head(grad_reverse(h_fused))`
2. **只对 gene 分支做域对抗**：让基因表征更“去域”，图像分支不受约束（有时更稳）
3. **双分支分别做域对抗**：`dom_gene(h_gene)` + `dom_img(h_img)`，但超参更多

建议先从（1）开始做 baseline；如果发现图像特征强烈携带 batch/癌种信息导致泛化变差，再考虑（2）。

---

## 5. 建议的落地路线（迭代顺序）

1. 先做 **方案 1（早期拼接）**：最快得到“是否有提升”的答案
2. 若提升有限但你相信图像特征有用：
   - 想要更稳、更可控：尝试 **方案 2（晚期双塔）**
   - 认为图像也需要空间传播：尝试 **方案 3（双 GNN）**（先用简单融合）
3. 若你希望“后期融合由模型自动学习每个 spot 的模态权重”：升级到 **方案 4（注意力后期融合）**

---

## 6. 需要你确认的几个关键问题（决定实现细节）

### 6.1 已确认

1. `[sample]_image_features.csv` 特征维度为 2048
2. 所有样本图像特征列名/顺序一致（`image_feature_1..N`）
3. 现实数据不应缺失 barcode；但若发生缺失：采用“填充 + mask”，并在数据处理阶段输出样本级警告
4. 推理阶段解释性：仅保留现有 gene attribution（不新增图像特征归因输出）
5. 优先 backbone：`gatv2`（保留 `gcn/sage` 可选）

### 6.2 待确认 / 我这边有疑问

- 工程范围：第一版先只支持 `prepare_data.py/train.py/infer.py/batch_infer.py`，`train_hpo.py` 后续再补（如你希望第一版就包含 HPO，请明确）
