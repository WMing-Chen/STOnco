# STOnco 残差门控双 GNN 图像融合实现方案

> 目标：在 STOnco 现有结构基础上，将当前 `gene + image` 的 early fusion（输入直接拼接）升级为  
> `Gene GNN + Image GNN + residual gated fusion -> h_fused`，并保持现有 slide/cancer GRL、MMD、WB 组件继续作用在共享表征空间。

---

## 1. 方案结论

本方案的核心定义如下：

```text
Xp_gene --GeneGNN--> h_gene ----------------------.
                                                  \
                                                   -> ResidualGatedFusion -> h_fused
                                                  /
Xp_img  --ImageGNN-> h_img -> ImgProj -> u_img --'

h_fused -> ClassifierHead -> z_clf -> logits
h_fused -> slide/cancer GRL
h_fused -> MMD
h_fused -> WB
```

融合公式采用：

```text
u_img   = Proj(h_img)
g       = img_mask * sigmoid(GateMLP([LN(h_gene), LN(u_img), img_mask]))
h_fused = h_gene + g ⊙ u_img
```

其中：

- `h_gene` 是 gene branch 的图表征；
- `h_img` 是 image branch 的图表征；
- `u_img` 是映射到 gene 空间后的图像残差；
- `g` 是逐 spot 的门控权重，第一版默认 **scalar gate**；
- `h_fused` 是新的共享表征空间，**替代当前代码中的 `h`**。

这意味着：

- 当前代码中所有依赖 `h` 的对齐模块，默认迁移到 `h_fused`；
- 当前 `z_clf` 仍然只是分类头内部表征，不作为主对齐空间；
- gene 分支保持主干地位，image 分支只做增量修正，不采用对称融合。

---

## 2. 为什么选 residual gated fusion

相对当前 early fusion，本方案更适合 STOnco 当前任务设定：

- 多癌种（>11）；
- 多 batch / slide / center；
- 500,000+ spots；
- 图像特征来自 ResNet50 2048 维，天然容易携带癌种/中心/染色 shortcut；
- 现有 STOnco 已有 `slide/cancer GRL`、`MMD`、`WB`，这些模块已围绕 `h` 设计。

采用 residual gated fusion 的原因：

1. **保持 gene 为主干**
   - `h_fused = h_gene + residual` 比 `g*h_gene + (1-g)*h_img` 更稳；
   - 可以显著降低 image branch 直接接管主表征的风险。

2. **与现有对齐模块兼容**
   - 当前对齐模块默认假设存在一个共享表征 `h`；
   - 本方案只需把 `h` 替换为 `h_fused`，语义最连续。

3. **更适合缺失图像**
   - `img_mask=0` 时，只要强制 `g=0`，模型自然退化为 `h_fused=h_gene`；
   - 比对称融合更容易保证行为正确。

4. **更适合多域泛化**
   - image 作为修正项而不是主导项，更利于跨癌种/跨 center 泛化。

---

## 3. 与当前代码的对应关系

当前仓库中与本方案直接相关的现状如下。

### 3.1 已经存在的部分

- 数据准备已支持图像特征 CSV 与 NPZ 扩展：
  - [prepare_data.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/prepare_data.py)
  - 训练 NPZ 已包含：
    - `X_imgs`
    - `img_masks`
    - `img_feature_names`
  - 单 slide NPZ 已包含：
    - `X_img`
    - `img_mask`
    - `img_feature_names`

- 图像预处理器已存在：
  - [preprocessing.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/preprocessing.py)
  - `ImagePreprocessor` 已支持 `StandardScaler + optional PCA`

- 训练/推理/导出链路已经支持读取图像特征，但当前是 early fusion：
  - [train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py)
  - [infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py)
  - [batch_infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/batch_infer.py)
  - [export_spot_embeddings.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/export_spot_embeddings.py)
  - [visualize_prediction.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/visualize_prediction.py)

### 3.2 当前需要替换的关键点

当前 early fusion 的核心是：

```text
Xp_gene + Xp_img + img_mask (+ pe) -> one x -> one GNN -> h
```

具体实现点：

- `build_node_features_early_fusion(...)`
- `assemble_pyg(...)` 里只构建单个 `data.x`
- `STOnco_Classifier.forward(x, edge_index, ...)`

本方案需要把这一条路径改成：

```text
data.x_gene
data.x_img
data.img_mask
data.pe_gene (optional)
-> 双分支前向
-> h_fused
```

---

## 4. 本方案的范围

### 4.1 第一版实现范围

第一版只实现以下内容：

1. Gene GNN 分支；
2. Image GNN 分支；
3. Residual gated fusion；
4. `h_fused` 替代当前 `h`；
5. 现有 `GRL / MMD / WB` 继续默认接 `h_fused`；
6. 推理、批推理、embedding 导出、可视化全部兼容新结构；
7. 现有 early fusion 路径保留，作为兼容和 baseline。

### 4.2 第一版不做的内容

以下内容不进入第一版：

1. 对称 gated fusion；
2. cross-attention / transformer 融合；
3. image branch 单独的 MMD / WB；
4. image branch 辅助分类头；
5. vector gate；
6. image branch 使用 LapPE。

这些内容可以留作第二阶段扩展。

---

## 5. 数据与 NPZ 契约

本方案 **不修改** 当前 `prepare_data.py` 的图像 CSV 和 NPZ 契约，沿用现有设计。

### 5.1 输入文件保持不变

每个样本目录：

- `*exp.csv`
- `*coordinates.csv`
- `*image_features.csv`

### 5.2 NPZ 键保持不变

训练多 slide NPZ：

- `Xs`
- `ys`
- `xys`
- `slide_ids`
- `gene_names`
- `barcodes`
- `X_imgs`
- `img_masks`
- `img_feature_names`

单 slide NPZ：

- `X`
- `xy`
- `gene_names`
- `barcodes`
- `sample_id`
- optional `y`
- `X_img`
- `img_mask`
- `img_feature_names`

### 5.3 改动点仅在图对象组装阶段

变化不发生在 NPZ，而发生在 `assemble_pyg` 阶段：

- 旧：组装单个 `data.x`
- 新：组装 `data.x_gene` / `data.x_img` / `data.img_mask` / `data.pe_gene`

---

## 6. 图对象设计

当前 [train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py) 的 `assemble_pyg(...)` 返回单模态图对象：

```text
data.x
data.edge_index
data.edge_weight
data.y
data.pos
```

本方案建议改为双分支图对象。

### 6.1 建议的 `PyG Data` 字段

统一约定：

```text
data.x_gene       # [N, Dg_in]
data.x_img        # [N, Di_in]
data.img_mask     # [N] or [N,1]
data.edge_index
data.edge_weight
data.y
data.pos
data.slide_id
data.bat_dom
data.cancer_dom
```

可选字段：

```text
data.pe_gene      # [N, Dpe]，若 gene 分支启用 LapPE
```

### 6.2 是否保留 `data.x`

建议：

- **保留 `data.x = data.x_gene` 作为兼容字段**；
- 但模型前向不再依赖 `data.x`；
- 所有新代码使用 `x_gene` / `x_img` / `img_mask`；
- `data.x` 只保留给旧工具链中仍读取 `g.x.shape` 的位置，作为平滑迁移。

这样做的好处：

1. 降低一次性大改带来的影响面；
2. gene attribution 仍然可以把 `data.x_gene` 当成基因输入；
3. 避免部分旧辅助工具立即崩溃。

### 6.3 `assemble_pyg` 的新职责

建议新增一个新的组装函数，例如：

```python
def assemble_pyg_multimodal(Xp_gene, Xp_img, img_mask, xy, y, cfg):
    ...
```

行为：

1. 用 `xy` 构图，生成统一 `edge_index/edge_weight`；
2. gene 分支如启用 LapPE，则计算 `pe_gene`；
3. `data.x_gene = Xp_gene`；
4. `data.x_img = Xp_img`；
5. `data.img_mask = img_mask`；
6. `data.x = Xp_gene` 仅作兼容；
7. `data.num_nodes`, `data.pos`, `data.y` 等照旧设置。

第一版建议：

- `LapPE` 只作用于 gene 分支；
- image 分支先不加 `LapPE`。

---

## 7. 模型设计

当前 [models.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/models.py) 中：

- `GNNBackbone`
- `ClassifierHead`
- `DomainHead`
- `STOnco_Classifier`

本方案建议 **保留这些类的主体结构**，新增两个模块，并扩展 `STOnco_Classifier`。

### 7.1 保留 `GNNBackbone`

`GNNBackbone` 可直接复用，用于：

- `self.gnn_gene`
- `self.gnn_img`

这样可以最少改动地保持 GATv2 / GCN / SAGE 三种 backbone 选择。

### 7.2 新增 `ResidualGatedFusion`

建议新增：

```python
class ResidualGatedFusion(nn.Module):
    ...
```

职责：

1. 将 `h_img` 投影到 gene latent 空间；
2. 基于 `h_gene`、`u_img` 和 `img_mask` 计算门控；
3. 输出 `h_fused`、`gate`、`u_img`。

建议结构：

```text
input:
    h_gene: [N, D]
    h_img:  [N, Di]
    img_mask: [N] or [N,1]

u_img = img_proj(h_img)                     # [N, D]
q_gene = LN(h_gene)
q_img  = LN(u_img)
gate_in = [q_gene, q_img, img_mask_col]    # [N, 2D+1]
g = sigmoid(gate_mlp(gate_in))             # [N,1]
g = g * img_mask_col
h_fused = h_gene + g * u_img
```

### 7.3 第一版 gate 采用 scalar gate

第一版建议：

- `g` 形状为 `[N, 1]`
- 对每个 spot 只预测一个图像残差强度

不建议第一版就用 `[N, D]` 的 vector gate，原因：

- 参数更多；
- 更容易过拟合 batch / center；
- 不利于调试。

### 7.4 `img_mask` 的处理约束

必须保证：

```text
img_mask = 0  =>  g = 0
```

即：

```python
g = g * img_mask_col
```

这样缺失图像的 spot 自然退化为：

```text
h_fused = h_gene
```

### 7.5 gate 初始化

建议 gate 初始偏保守，让训练初期更接近 gene-only：

- `gate_mlp` 最后一层 bias 初始化为负值，例如 `-2.0`
- 这样初始 `sigmoid(bias)` 较小，`g` 接近 0

目的：

- 避免训练初期 image branch 直接影响主表征；
- 降低 shortcut 风险。

### 7.6 是否需要 `Proj`

需要。

因为 `h_gene` 和 `h_img` 不一定同维，即使同维，也建议保留：

```python
self.img_proj = nn.Linear(img_dim, gene_dim)
```

它的作用不仅是对齐维度，也是在数值和语义空间上让 image residual 更容易被 gene 主空间吸收。

---

## 8. `STOnco_Classifier` 的改造建议

### 8.1 不建议新增独立模型类

建议 **继续使用 `STOnco_Classifier` 这个类名**，在内部按配置切换模式，而不是新增一个完全平行的新模型类。

原因：

1. 训练、推理、导出、可视化都已经依赖这个类名；
2. 现有 `load_state_dict(..., strict=False)` 机制已经具备一定兼容性；
3. 代码变更面更小。

### 8.2 新增配置开关

建议增加：

```python
use_image_features: bool
image_fusion_mode: str  # 'early_concat' | 'dual_branch_residual_gate'
```

默认值：

```python
use_image_features = False
image_fusion_mode = 'early_concat'
```

只有当：

```text
use_image_features=1 and image_fusion_mode='dual_branch_residual_gate'
```

时，走本方案。

### 8.3 新的初始化参数

建议增加最少必要配置项：

```python
img_model                 # 默认与 model 相同
img_gnn_hidden            # 默认较轻，如 [128, 64]
img_num_layers            # 默认 2
img_heads                 # 默认与 heads 相同
img_dropout               # 默认与 gnn_dropout 相同
fusion_gate_hidden        # 默认 128
fusion_gate_use_layernorm # 默认 True
fusion_gate_bias_init     # 默认 -2.0
```

第一版不建议暴露过多超参。

### 8.4 `forward(...)` 签名建议

当前签名：

```python
forward(x, edge_index, batch=None, edge_weight=None, ...)
```

建议改成兼容式签名：

```python
forward(
    x=None,
    edge_index=None,
    batch=None,
    edge_weight=None,
    x_gene=None,
    x_img=None,
    img_mask=None,
    pe_gene=None,
    ...
)
```

逻辑：

1. `image_fusion_mode='early_concat'` 时，沿用旧路径，使用 `x`；
2. `image_fusion_mode='dual_branch_residual_gate'` 时，使用 `x_gene/x_img/img_mask`；
3. 最终统一输出：
   - `out['logits']`
   - `out['h'] = h_fused`
   - optional `out['h_gene']`
   - optional `out['h_img']`
   - optional `out['gate']`
   - optional `out['u_img']`
   - `out['z_clf']` 若 `return_z=True`

### 8.5 为什么继续保留 `out['h']`

当前 [train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py) 的 MMD/WB 路径都是通过 `return_h=True` 后读取 `out['h']`。

为了最小改动，建议：

```python
out['h'] = h_fused
out['h_fused'] = h_fused
```

即：

- 老代码继续读 `out['h']`
- 新代码可以明确读 `out['h_fused']`

---

## 9. Gene branch 与 Image branch 的建议配置

### 9.1 Gene branch

第一版：

- 直接复用现有 gene GNN 配置；
- 保留当前 `GNN_hidden`、`num_layers`、`model`、`heads`、`gnn_dropout`；
- gene 分支继续作为主干表征。

### 9.2 Image branch

第一版建议比 gene branch 更轻。

建议默认：

```text
img_use_pca = True
img_pca_dim = 128 or 256
img_num_layers = 2
img_gnn_hidden = [128, 64] 或 [128, 128]
```

原因：

1. image 特征原始维度高，分支更重时显存成本增长明显；
2. image branch 在本方案中只提供 residual，不需要比 gene branch 更强；
3. 多域场景下，较轻 image branch 更利于泛化。

### 9.3 LapPE 的使用

第一版建议：

- `LapPE` 只给 gene branch；
- image branch 不加 `LapPE`。

原因：

1. gene branch 已经是主干；
2. image branch 若再拼 `LapPE`，增加参数和耦合；
3. 当前解释性和预处理逻辑默认围绕 gene 主路径。

---

## 10. 训练图构建改造

### 10.1 `prepare_graphs(...)` 的改造原则

当前 [train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py) 中 `prepare_graphs(...)` 的关键流程是：

1. 读 `Xs/ys/xys`
2. `Preprocessor.fit(...)`
3. `ImagePreprocessor.fit(...)`
4. `assemble_pyg(...)`
5. 返回 `train_graphs, val_graphs, in_dim, ...`

本方案保持 1-3 不变，仅修改 4 和 5。

### 10.2 预处理保持不变

- gene：继续 `Preprocessor`
- image：继续 `ImagePreprocessor`

即：

```python
Xp_gene = pp.transform(...)
Xp_img  = img_pp.transform(...)
```

### 10.3 新的组图方式

旧：

```python
data_g = assemble_pyg(Xp_gene, xy, y, cfg, Xp_img=Xp_img, img_mask=img_mask)
```

新：

```python
data_g = assemble_pyg_multimodal(
    Xp_gene=Xp_gene,
    Xp_img=Xp_img,
    img_mask=img_mask,
    xy=xy,
    y=y,
    cfg=cfg,
)
```

### 10.4 `prepare_graphs(...)` 的返回值

当前返回单个 `in_dim`。

本方案建议改为：

```python
input_dims = {
    'gene_in_dim': int(...),
    'img_in_dim': int(...),   # 若未启用 image，则可为 None
}
```

然后：

```python
return train_graphs, val_graphs, input_dims, n_domains_batch, n_domains_cancer
```

原因：

- 双分支模型不再只有一个输入维度；
- 若继续只返回 `in_dim`，会造成后续模型初始化含糊。

### 10.5 为减少迁移成本的备选方案

如果不希望一次性改太多返回签名，也可以：

- `in_dim` 继续表示 `gene_in_dim`
- 新增 `img_in_dim` 作为额外返回值

但从长期维护看，`input_dims` 字典更清晰。

---

## 11. 训练主流程改造

### 11.1 模型初始化

当前：

```python
model = STOnco_Classifier(in_dim=in_dim, ...)
```

新：

```python
model = STOnco_Classifier(
    in_dim_gene=input_dims['gene_in_dim'],
    in_dim_img=input_dims['img_in_dim'],
    ...
)
```

同时保留兼容：

- early fusion 模式仍使用 `in_dim`
- dual branch 模式使用 `in_dim_gene/in_dim_img`

### 11.2 batch 前向

当前训练循环大多写成：

```python
out = model(
    batch.x,
    batch.edge_index,
    batch=batch.batch,
    edge_weight=getattr(batch, 'edge_weight', None),
    ...
)
```

新模式改为：

```python
out = model(
    x_gene=batch.x_gene,
    x_img=batch.x_img,
    img_mask=batch.img_mask,
    edge_index=batch.edge_index,
    batch=batch.batch,
    edge_weight=getattr(batch, 'edge_weight', None),
    ...
)
```

### 11.3 对齐损失不改接口语义

保持：

```python
out['h'] == h_fused
```

这样：

- `GRL_slide`
- `GRL_cancer`
- `MMD`
- `WB`

原有大部分逻辑无需重写，只要模型前向返回 `h_fused` 即可。

### 11.4 训练监控建议新增

建议在训练日志中新增以下统计项：

- `avg_gate_mean`
- `avg_gate_present`
- `avg_gate_missing`
- `avg_img_residual_norm`

意义：

1. 判断 image branch 是否完全没用；
2. 判断 gate 是否失控过大；
3. 判断缺失图像 spot 是否被正确屏蔽。

---

## 12. 推理、批推理、embedding 导出与可视化改造

本方案不能只改训练，必须同步改：

- [infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py)
- [batch_infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/batch_infer.py)
- [export_spot_embeddings.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/export_spot_embeddings.py)
- [visualize_prediction.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/visualize_prediction.py)

### 12.1 推理组图

和训练一致，推理图对象应改为：

```text
x_gene
x_img
img_mask
edge_index
edge_weight
```

### 12.2 `InferenceEngine.build_model_if_needed(...)`

当前通过 `g.x.shape[1]` 推断输入维度。

本方案应改为：

```python
gene_in_dim = g.x_gene.shape[1]
img_in_dim  = g.x_img.shape[1] if cfg['use_image_features'] else None
```

### 12.3 gene attribution 的改造

当前 [infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py) 的解释逻辑默认输入只有一个 `x`，并通过截取前 `feat_dim_gene` 维来解释 gene 特征。

本方案下，解释逻辑必须改为：

1. 仅对 `x_gene` 求梯度；
2. `x_img`、`img_mask` 保持常量；
3. 前向时调用双分支模型；
4. 最终输出仍回投到 gene 层面。

伪代码：

```python
x_gene = g.x_gene.clone().requires_grad_(True)
x_img = g.x_img.detach()
img_mask = g.img_mask.detach()

out = model(
    x_gene=x_gene,
    x_img=x_img,
    img_mask=img_mask,
    edge_index=...,
    edge_weight=...,
)
loss = out['logits'].sum()
grad = autograd(loss, x_gene)
```

这一步非常关键。  
否则会把 image branch 一起混进 gene attribution，解释性语义会被破坏。

### 12.4 embedding 导出

当前导出脚本可以读取：

- `h`
- `z_clf`

本方案中建议：

- `h` 导出为 `h_fused`
- 可选新增导出：
  - `h_gene`
  - `h_img`
  - `gate`

第一版至少应支持：

```text
embed_source = h_fused / z_clf
```

如果额外输出 `gate`，对排查 image shortcut 很有价值。

### 12.5 预测可视化

`visualize_prediction.py` 应继续从 `logits` 出发，不需要改变可视化主逻辑。

建议可选增加：

- `--save_gate_csv`
- `--save_gate_svg`

用于画出每个 slide 上 gate 的空间分布。

---

## 13. 兼容性策略

### 13.1 训练兼容

建议同时保留两种模式：

1. `early_concat`
2. `dual_branch_residual_gate`

通过配置切换。

这样可以：

- 保持旧实验可重跑；
- 与旧 checkpoint 语义区分；
- 便于直接做 ablation。

### 13.2 推理兼容

推理时根据 `meta.json['cfg']` 自动判断模型模式。

如果：

```text
image_fusion_mode = 'early_concat'
```

则走旧推理路径。

如果：

```text
image_fusion_mode = 'dual_branch_residual_gate'
```

则走双分支推理路径。

### 13.3 checkpoint 兼容

由于 `STOnco_Classifier` 内部结构会增加新模块：

- `gnn_img`
- `img_proj`
- `fusion_gate`

旧 checkpoint 无法严格匹配新模型。

当前仓库已大量使用：

```python
load_state_dict(..., strict=False)
```

这有助于平滑迁移，但仍需在文档和 `meta.json` 中明确：

- 旧 gene-only / early-fusion checkpoint 不能直接当作新双分支模型推理使用；
- 新 checkpoint 必须配合其自身 `cfg` 读取。

---

## 14. 建议新增的配置项

建议在 [train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py) 中新增以下配置项：

```python
image_fusion_mode = 'early_concat'  # or dual_branch_residual_gate
img_model = None                    # None => follow model
img_gnn_hidden = [128, 64]
img_num_layers = 2
img_heads = None                    # None => follow heads
img_dropout = None                  # None => follow gnn_dropout
img_branch_use_lap_pe = False
fusion_gate_hidden = 128
fusion_gate_use_layernorm = True
fusion_gate_bias_init = -2.0
fusion_gate_type = 'scalar'
```

默认策略：

- 不开启双分支时不影响旧行为；
- 开启双分支时只暴露最少超参。

---

## 15. 代码改动清单

### 15.1 必改文件

1. [stonco/core/models.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/models.py)
   - 新增 `ResidualGatedFusion`
   - 扩展 `STOnco_Classifier`

2. [stonco/core/train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py)
   - 新增配置项
   - 改 `assemble_pyg`
   - 改 `prepare_graphs`
   - 改训练/验证前向调用

3. [stonco/core/infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py)
   - 改组图
   - 改模型构建
   - 改 attribution

4. [stonco/core/batch_infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/batch_infer.py)
   - 改组图
   - 改模型调用

5. [stonco/utils/export_spot_embeddings.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/export_spot_embeddings.py)
   - 改组图
   - 改 forward
   - 可选导出 gate

6. [stonco/utils/visualize_prediction.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/visualize_prediction.py)
   - 改组图
   - 改 forward

### 15.2 通常无需改动的文件

1. [stonco/utils/prepare_data.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/prepare_data.py)
   - 当前图像 CSV 与 NPZ 契约已足够

2. [stonco/utils/preprocessing.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/preprocessing.py)
   - `ImagePreprocessor` 已可复用

---

## 16. 第一版推荐的具体默认值

建议默认配置：

```text
use_image_features = 1
image_fusion_mode = dual_branch_residual_gate

gene branch:
    model = 当前主干配置
    GNN_hidden = 当前配置
    num_layers = 当前配置

image branch:
    img_use_pca = 1
    img_pca_dim = 128 或 256
    img_model = 跟随主干
    img_gnn_hidden = 128,64
    img_num_layers = 2
    img_dropout = gnn_dropout

fusion:
    fusion_gate_hidden = 128
    fusion_gate_use_layernorm = 1
    fusion_gate_bias_init = -2.0
    fusion_gate_type = scalar
```

对齐：

```text
GRL_slide  -> h_fused
GRL_cancer -> h_fused
MMD        -> h_fused
WB         -> h_fused
```

解释性：

```text
仅对 x_gene 做 attribution
```

---

## 17. 实现顺序建议

### Phase 1: 模型与组图改造

1. 新增 `ResidualGatedFusion`
2. 扩展 `STOnco_Classifier`
3. 新增双分支 `assemble_pyg_multimodal`
4. 跑通训练前向

### Phase 2: 推理与导出改造

1. 改 `infer.py`
2. 改 `batch_infer.py`
3. 改 `export_spot_embeddings.py`
4. 改 `visualize_prediction.py`

### Phase 3: 解释性与诊断

1. 重写 gene attribution 路径
2. 新增 gate 统计
3. 可选导出 gate 空间图

### Phase 4: 实验与消融

至少比较：

1. gene-only
2. early concat
3. dual branch + residual gate
4. dual branch + residual gate + GRL
5. dual branch + residual gate + WB

---

## 18. 验证与测试清单

### 18.1 结构正确性

1. `img_mask=0` 时，`gate==0`
2. `img_mask=0` 时，`h_fused == h_gene`
3. batch 后 `x_gene/x_img/img_mask` shape 正确

### 18.2 训练行为

1. `avg_gate_mean` 初期较小
2. 随训练推进，`avg_gate_present` 可上升，但不应迅速饱和到 1
3. `avg_gate_missing` 应接近 0

### 18.3 推理一致性

1. 训练和推理对同一 slide 输出形状一致
2. `export_spot_embeddings` 能导出 `h_fused`
3. `visualize_prediction` 能正常画图

### 18.4 解释性

1. gene attribution 仅依赖 `x_gene`
2. image 分支值变化不会改变 gene attribution 的维度定义

---

## 19. 风险与规避

### 19.1 image 分支失效

表现：

- `gate≈0`
- `u_img` 范数很小

处理：

- 检查 `img_pca_dim` 是否过小；
- 检查 image GNN 是否太弱；
- 观察 image-only baseline 是否本身没有信息。

### 19.2 image 分支接管

表现：

- `gate` 很快升高；
- `h_fused` 域信息明显增强；
- LOCO / 跨 center 泛化下降。

处理：

- 保持 residual 融合；
- 维持负 bias 初始化；
- 必要时降低 image branch 容量；
- 后续再考虑对 `h_img` 加弱 slide GRL。

### 19.3 对齐与任务冲突

表现：

- 加图像后 GRL/MMD/WB 把分类性能一起拉低。

处理：

- 先只把主对齐接到 `h_fused`；
- 不在第一版对 `h_img` 直接做 MMD/WB；
- 重新检查 `lambda_slide/lambda_cancer/lambda_wb`。

---

## 20. 最终推荐

对于当前 STOnco，第一版建议明确采用以下策略：

```text
1. 保留 prepare_data.py 和 ImagePreprocessor 的现有契约
2. 用双分支图对象替换单 x 图对象
3. 用 Gene GNN + Image GNN + residual gated fusion 得到 h_fused
4. 保持 h_fused 作为新的共享表征空间
5. 现有 GRL / MMD / WB 默认继续接 h_fused
6. z_clf 不作为主对齐空间
7. gene attribution 只对 x_gene 求导
8. 早期融合路径保留，作为兼容和 baseline
```

一句话总结：

**当前 STOnco 的最佳迁移方式不是把 image 再拼进 `x`，而是把现有的单一共享表征 `h` 升级为由 gene 主导、image 残差修正的 `h_fused`，并让现有对齐模块继续围绕这个新共享空间工作。**

---

## 21. 实现前需确认的事项

下面这些点是结合当前文档方案与 STOnco 现有代码后，真正会影响实现路径的确认项。  
这些点建议你先逐条确认；确认后再开始改代码。

### 21.1 是否保留 early fusion 作为并行模式

当前代码已经完整支持 `use_image_features=1` 的 early fusion 路径，训练、推理、embedding 导出和可视化都已经打通。

需要确认：

1. 是否要在本次实现中 **保留** 现有 early fusion，作为：
   - baseline
   - 兼容旧实验
   - 配置可切换模式
2. 还是直接用双分支方案替换现有 image fusion 路径？

推荐：

- **保留**
- 增加 `image_fusion_mode`：
  - `early_concat`
  - `dual_branch_residual_gate`

### 21.2 双分支模式下，`h_fused` 是否作为唯一主对齐空间

当前 STOnco 的：

- slide/cancer GRL
- MMD
- WB

都是围绕当前 `h` 设计的。

需要确认：

1. 双分支方案中，是否明确规定：
   - `out['h'] = h_fused`
   - 所有现有对齐模块默认继续吃 `h_fused`
2. 第一版是否完全 **不** 对 `h_img` 单独做 GRL / MMD / WB？

推荐：

- **是**
- 第一版只对 `h_fused` 做主对齐；
- 不对 `h_img` 单独加 MMD/WB；
- `h_img` 的单独对齐如果要做，放到第二阶段。

### 21.3 `ClassifierHead` 是否继续保持现有结构，只把输入从 `h` 改成 `h_fused`

当前 [models.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/models.py) 的分类头是：

```text
h -> ClassifierHead -> z_clf -> logits
```

需要确认：

1. 是否保持 `ClassifierHead` 内部结构不变？
2. 只是将输入从原来的 `h` 改为 `h_fused`？

推荐：

- **保持不变**
- 只改输入，不改分类头结构；
- `z_clf` 继续保持当前语义和导出逻辑。

### 21.4 image branch 的 backbone 是否与 gene branch 同类型

当前 `GNNBackbone` 支持：

- `gatv2`
- `gcn`
- `sage`

需要确认：

1. image branch 是否默认与 gene branch 使用同一种 GNN 类型？
2. 还是允许 image branch 独立指定 `img_model`？

推荐：

- 第一版实现上 **允许独立配置**
- 默认值设为：`img_model = model`

这样：

- 默认情况下实现简单；
- 后续若需要 image 用更轻模型，不必重构接口。

### 21.5 image branch 的规模是否固定为“更轻”

如果 image branch 直接复制 gene branch 的宽度和深度，显存和过拟合风险都会上升。

需要确认：

1. 第一版是否明确把 image branch 设为比 gene branch 更轻？
2. 是否接受默认：
   - `img_num_layers = 2`
   - `img_gnn_hidden = [128, 64]`

推荐：

- **是**
- 第一版就固定 image branch 更轻。

### 21.6 LapPE 是否只给 gene branch

当前代码中的 `LapPE` 是围绕单输入 `x` 拼接设计的。

双分支后有两个选择：

1. 只给 gene branch
2. gene/image 两个分支都给

需要确认：

是否同意第一版：

- gene branch 可继续使用 `LapPE`
- image branch 不使用 `LapPE`

推荐：

- **同意**
- 这是第一版最稳的做法。

### 21.7 组图对象是否改为显式双输入字段

当前很多代码路径默认 `data.x` 是唯一输入。

双分支实现时有两个方案：

1. 继续把两路信息塞进一个 `data.x`，在模型里再拆
2. 图对象显式保存：
   - `data.x_gene`
   - `data.x_img`
   - `data.img_mask`

需要确认：

是否接受第二种显式方案？

推荐：

- **接受**
- 并且暂时保留 `data.x = data.x_gene` 作为兼容字段

这能显著降低后续维护难度，也更适合 attribution。

### 21.8 `prepare_graphs(...)` 的返回值是否允许改

当前 `prepare_graphs(...)` 返回：

```text
train_graphs, val_graphs, in_dim, n_domains_batch, n_domains_cancer
```

双分支后不再只有一个输入维度。

需要确认：

1. 是否允许把返回值改成：
   - `input_dims = {'gene_in_dim': ..., 'img_in_dim': ...}`
2. 还是你希望尽量不改返回签名，只新增 `img_in_dim`？

推荐：

- **允许改成 `input_dims` 字典**

这是比较干净的接口。

### 21.9 gene attribution 是否继续“只解释 gene”

当前 [infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py) 的 attribution 实际上是假设 gene 特征在输入前段。

双分支后，需要明确：

1. 是否继续只输出 gene attribution？
2. 是否不做 image attribution？

推荐：

- **继续只做 gene attribution**
- 实现方式改为：只对 `x_gene` 求导，`x_img` 和 `img_mask` 视为常量输入

这和你之前的要求最一致，也能保持现有解释性语义。

### 21.10 embedding 导出时，是否只保留 `h_fused` 与 `z_clf`

当前导出脚本主要围绕：

- `h`
- `z_clf`

双分支后可选导出更多中间量：

- `h_gene`
- `h_img`
- `gate`

需要确认：

第一版你希望：

1. 只保留最小集：
   - `h_fused`
   - `z_clf`
2. 还是同时把 `h_gene/h_img/gate` 也做成可选导出？

推荐：

- 第一版最少应支持：
  - `h_fused`
  - `z_clf`
- 另外建议 **增加可选** 的 `gate` 导出

因为 `gate` 对诊断 image shortcut 非常有价值。

### 21.11 是否在训练日志里加入 gate 监控

双分支 residual gate 最常见的问题有两个：

1. `gate≈0`，image branch 白做
2. `gate≈1`，image branch 过强

需要确认：

是否同意在训练日志和 `hist` 中增加这些指标：

- `avg_gate_mean`
- `avg_gate_present`
- `avg_gate_missing`
- `avg_img_residual_norm`

推荐：

- **同意**

否则实现后不容易判断模型到底有没有正确使用图像分支。

### 21.12 是否接受“第一版不做 image branch 辅助 loss”

有时为了防止 image branch 学空，会给 `h_img` 加一个弱辅助分类头。

但这样会扩大实现面，并引入新的权重超参。

需要确认：

第一版是否接受：

- 不加 image auxiliary head
- 不加 image auxiliary classification loss

推荐：

- **接受**
- 先看 residual gate + 主任务 + 现有对齐是否足够。

### 21.13 gate 的具体形式是否确认采用 scalar residual gate

当前文档默认：

```text
g: [N, 1]
h_fused = h_gene + g ⊙ u_img
```

需要确认：

1. 第一版是否固定采用 scalar gate？
2. 是否同意 gate MLP 最后一层 bias 负初始化，例如 `-2.0`？

推荐：

- **固定采用 scalar gate**
- **同意负 bias 初始化**

这是当前最稳的第一版。

### 21.14 对缺失图像的 hard masking 是否做成强约束

当前文档默认：

```text
img_mask = 0 => g = 0 => h_fused = h_gene
```

需要确认：

是否把这个逻辑作为实现中的**强约束**，而不是让模型自己学？

推荐：

- **作为强约束**

否则会引入不必要的不确定性。

### 21.15 是否按“分阶段实现”推进

结合当前代码体量，我建议实现顺序是：

1. `models.py + train.py`
2. `infer.py + batch_infer.py`
3. `export_spot_embeddings.py + visualize_prediction.py`
4. attribution 与 gate 诊断

需要确认：

是否按这个顺序推进，而不是一次性铺开全部改动？

推荐：

- **按阶段推进**

---

### 建议你最终确认的最小决策集

如果你想先快速拍板，至少请确认以下 8 条：

1. 保留 early fusion 作为并行模式，新增 `image_fusion_mode`
2. 第一版只把 `GRL / MMD / WB` 放在 `h_fused`
3. `ClassifierHead` 结构不改，只改输入为 `h_fused`
4. image branch 比 gene branch 更轻
5. `LapPE` 只给 gene branch
6. 图对象改为 `x_gene/x_img/img_mask` 显式字段，并保留 `data.x=data.x_gene` 兼容
7. attribution 继续只解释 gene
8. gate 采用 scalar residual gate，并对缺失图像做 hard masking

待你确认这些点后，再开始代码实现。
