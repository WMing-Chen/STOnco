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

### 1.1 实现硬约束

结合 STOnco 当前代码结构，本方案新增以下硬约束。后续实现必须满足这些约束后，才认为双分支方案真正接入完成。

1. **新增统一 `model.encode(...)`**
   - 所有共享表征 `h` 的来源必须走模型实例的 `model.encode(...)`；
   - 外部训练、推理、导出、WB/MMD 代码禁止直接调用 `model.gnn(...)` 获取 `h`；
   - early fusion 与 dual branch 都由 `encode(...)` 屏蔽内部差异；
   - `forward(...)` 内部也必须通过 `encode(...)` 得到 `h` 后再接分类头和域头。

2. **优化器参数组必须覆盖新模块**
   - dual branch 模式下，optimizer 必须明确加入：
     - `gnn_gene`
     - `gnn_img`
     - `fusion`
     - `clf`
     - `dom_slide/dom_cancer`
   - 不能继续只依赖 `model.gnn.parameters()`。

3. **`sampler.py` 必须支持所有 node-level 字段切片**
   - static subgraph 训练必须同步切片：
     - `x`
     - `x_gene`
     - `x_img`
     - `img_mask`
     - `pe_gene`
     - `y`
     - `pos`
   - clone full graph 时也必须保留这些字段。

4. **image branch 内部必须做 mask propagation**
   - `img_mask=0` 不仅要让最终 `gate=0`；
   - image branch 的输入和每层 hidden 都应按 `img_mask` 清零，避免缺失图像节点通过 GNN message passing 影响其他节点；
   - 第一版采用节点级 hard mask，不做 image GNN 的边级 valid-valid 过滤。

5. **checkpoint 必须按 `image_fusion_mode` 严格构建和校验**
   - 默认使用 `strict=True` 加载 checkpoint；
   - 不依赖 `strict=False` 静默兼容新旧模型；
   - 推理、导出、可视化必须根据 `meta.json['cfg']['image_fusion_mode']` 构建对应结构；
   - 模式不匹配、关键模块缺失或 unexpected/missing keys 异常时，应给出明确错误。

6. **改动范围必须覆盖重复训练/评估路径**
   - 除主训练路径外，`train.py` 内部的 kfold/LOCO 重复路径也必须同步；
   - `eval_loco_checkpoints.py` 也必须同步走新模型构建与 `encode(...)` 路径；
   - 本轮方案暂不把 `train_hpo.py` 纳入必改范围，后续另行处理。

---

## 2. 为什么选 residual gated fusion

相对当前 early fusion，本方案更适合 STOnco 当前任务设定：

- 多癌种（>11）；
- 多 batch / slide / center；
- 500,000+ spots；
- 图像特征来自外部图像模型，例如 UNI、ResNet50 等，天然容易携带癌种/中心/染色 shortcut；
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

图像特征输入维度不固定。  
`X_img` 的原始维度由外部图像模型决定，例如 UNI、ResNet50 或其他 encoder；模型实际接收的是 `ImagePreprocessor.transform(...)` 后的 `Xp_img`。

因此：

```text
img_in_dim = Xp_img.shape[1] = img_pp.out_dim()
```

而不是写死为 2048、1024、768 或某个固定值。

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

### 6.2 `data.x` 兼容策略

已确认：

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

但仅在 fusion 末端令 `g=0` 还不够。由于 image branch 使用 GNN，缺失图像节点即使输入为 0，也可能通过卷积 bias、归一化或邻居聚合产生非零 hidden，并进一步参与 message passing。

因此第一版实现必须增加 image branch mask propagation：

```text
x_img_input = x_img * img_mask
每层 image hidden 后继续 h_img_layer = h_img_layer * img_mask
fusion 前 h_img = h_img * img_mask
u_img = u_img * img_mask
g = g * img_mask
```

这样至少保证缺失图像节点自身不会产生 image residual。第一版不做 image GNN 的边级 valid-valid 过滤；如果后续实验发现缺失模式仍明显影响有图像节点，再单独扩展。

### 7.5 gate 初始化

建议 gate 初始偏保守，让训练初期更接近 gene-only：

- `gate_mlp` 最后一层 bias 初始化为负值，例如 `-2.0`
- 这样初始 `sigmoid(bias)` 较小，`g` 接近 0

目的：

- 避免训练初期 image branch 直接影响主表征；
- 降低 shortcut 风险。

### 7.6 保留 `ImgProj`

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
2. `meta.json['cfg']['image_fusion_mode']` 可以作为模型结构选择的唯一来源；
3. 代码变更面更小。

注意：继续使用同一个类名不等于允许 checkpoint 静默混用。  
新旧模型必须通过 `image_fusion_mode` 明确区分，加载权重时也必须校验关键模块是否匹配。

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

### 8.6 新增统一 `encode(...)` 作为共享表征唯一出口

双分支实现后，`STOnco_Classifier` 必须新增：

```python
def encode(
    self,
    x=None,
    edge_index=None,
    edge_weight=None,
    x_gene=None,
    x_img=None,
    img_mask=None,
    pe_gene=None,
    return_aux=False,
):
    ...
```

语义：

1. `early_concat` 模式：
   - 使用旧路径：`h = self.gnn(x, edge_index, edge_weight=edge_weight)`；
   - `return_aux=True` 时至少返回 `{'h': h}`。
2. `dual_branch_residual_gate` 模式：
   - 使用 `x_gene/x_img/img_mask/pe_gene`；
   - gene branch 得到 `h_gene`；
   - image branch 在内部执行 mask propagation 后得到 `h_img`；
   - fusion 得到 `h_fused`；
   - 返回的主 `h` 必须等于 `h_fused`。

`forward(...)` 内部必须先调用 `encode(...)`：

```python
h, enc_aux = self.encode(..., return_aux=True)
logits, z_clf = self.clf(h, return_z=return_z)
dom_logits = domain_heads(grad_reverse(h, ...))
```

这样：

- classifier；
- slide/cancer GRL；
- MMD；
- WB；
- embedding 导出；
- 其他显式需要 `h` 的路径；

都能通过同一个入口拿到一致的共享表征。

实现后，外部代码不应再通过 `model.gnn(...)` 获取 `h`。当前 WB potential 更新中直接调用 `model.gnn(batch.x, ...)` 的位置必须改为 `model.encode(...)`。

建议模型暴露统一维度字段：

```python
self.encoder_out_dim = ...
```

外部所有原来读取 `model.gnn.out_dim` 的地方，应改为读取 `model.encoder_out_dim`。

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
img_use_pca = False
img_pca_dim = 256  # 仅当显式开启 img_use_pca=1 时生效
img_num_layers = 2
img_gnn_hidden = [128, 64] 或 [128, 128]
```

原因：

1. image 特征原始维度由外部图像模型决定，默认保留预处理后的完整图像特征，避免默认 PCA 改变外部图像模型表征；
2. image branch 在本方案中只提供 residual，不需要比 gene branch 更强；
3. 若显存或速度压力较大，可显式开启 `img_use_pca=1` 并设置 `img_pca_dim=128/256`。

实现要求：

- 不在模型里固定 `in_dim_img`；
- `in_dim_img` 必须来自 `Xp_img.shape[1]`、`img_pp.out_dim()` 或 `prepare_graphs(...)` 返回的 `input_dims['img_in_dim']`；
- `img_pca_dim` 只是 PCA 开启时的推荐输出维度，不代表原始图像特征维度。

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
- 图像输入维度由图像预处理输出动态决定，不应由配置或代码写死。

推荐取值：

```python
gene_in_dim = int(Xp_gene.shape[1])
img_in_dim = int(Xp_img.shape[1]) if use_image_features else None
```

或者在 image preprocessor 已 fit 后：

```python
img_in_dim = int(img_pp.out_dim())
```

### 10.5 为减少迁移成本的备选方案

如果不希望一次性改太多返回签名，也可以：

- `in_dim` 继续表示 `gene_in_dim`
- 新增 `img_in_dim` 作为额外返回值

但从长期维护看，`input_dims` 字典更清晰。

### 10.6 subgraph sampler 必须同步支持双分支字段

当前 `sampler.py` 的 static subgraph 逻辑只复制单输入字段：

```text
x
edge_index
edge_weight
y
pos
slide/domain meta
```

双分支图对象上线后，`split_graph_into_subgraphs(...)` 与 `_clone_full_graph_as_subgraph(...)` 必须同步处理所有 node-level 字段：

```text
x
x_gene
x_img
img_mask
pe_gene
y
pos
```

要求：

1. 对节点级 tensor 使用相同 `node_idx_t` 切片；
2. 对图级 domain/meta 字段继续直接复制；
3. 如果某个字段不存在，保持兼容，不报错；
4. 切子图后 `data.x` 仍保持兼容语义，dual branch 下建议为 `data.x_gene`；
5. 验证 batch 后 `batch.x_gene/batch.x_img/batch.img_mask` 的第一维与 `batch.y` 一致。

否则在 `subgraph_mode=static` 下，训练会退回旧 `data.x` 语义，双分支模型无法稳定运行。

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

建议训练侧封装一个小 helper，例如：

```python
def model_forward(model, batch, cfg, **kwargs):
    if cfg['image_fusion_mode'] == 'dual_branch_residual_gate':
        return model(
            edge_index=batch.edge_index,
            edge_weight=getattr(batch, 'edge_weight', None),
            batch=getattr(batch, 'batch', None),
            x_gene=batch.x_gene,
            x_img=batch.x_img,
            img_mask=batch.img_mask,
            pe_gene=getattr(batch, 'pe_gene', None),
            **kwargs,
        )
    return model(
        batch.x,
        batch.edge_index,
        edge_weight=getattr(batch, 'edge_weight', None),
        batch=getattr(batch, 'batch', None),
        **kwargs,
    )
```

主训练、验证、外部验证、kfold、LOCO 中所有前向都应复用同一 helper，避免局部路径遗漏。

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

但 WB 有一个额外约束：当前 potential 更新阶段直接调用 `model.gnn(...)` 计算 detached `h`。该路径必须改为：

```python
with torch.no_grad():
    h_detached = model.encode(
        x=batch.x,
        x_gene=getattr(batch, 'x_gene', None),
        x_img=getattr(batch, 'x_img', None),
        img_mask=getattr(batch, 'img_mask', None),
        pe_gene=getattr(batch, 'pe_gene', None),
        edge_index=batch.edge_index,
        edge_weight=getattr(batch, 'edge_weight', None),
    )
```

也就是说，WB potential loss、WB model loss、MMD、GRL、classifier 必须全部基于同一个 `encode(...)` 产出的共享表征。

### 11.4 optimizer 参数组

当前 optimizer 的参数组围绕 `model.gnn/model.clf/domain/support_map` 构建。双分支后必须显式覆盖新模块。

建议：

```text
early_concat:
    gnn      -> model.gnn
    clf      -> model.clf
    dom      -> model.dom_slide/model.dom_cancer

dual_branch_residual_gate:
    gnn_gene -> model.gnn_gene
    gnn_img  -> model.gnn_img
    fusion   -> model.fusion
    clf      -> model.clf
    dom      -> model.dom_slide/model.dom_cancer
```

WB 的 `support_map` 与 `wb_module` 参数组保持现有设计，但其输入维度应从 `model.encoder_out_dim` 读取，而不是从 `model.gnn.out_dim` 读取。

### 11.5 训练监控建议新增

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

这里的 `img_in_dim` 仍然是预处理后的动态维度。推理时必须复用训练产物中的 `img_scaler.joblib` 和可选 `img_pca.joblib`，让 `g.x_img.shape[1]` 与训练时一致。

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

旧 checkpoint 无法严格匹配新模型。正式训练产物、推理、导出和可视化默认都应使用 `strict=True` 加载。这里不能把 `strict=False` 当作兼容策略，否则旧 checkpoint 可能在新双分支结构中静默加载，导致 image/fusion 模块随机初始化后继续推理。

要求：

- 旧 gene-only / early-fusion checkpoint 不能直接当作新双分支模型推理使用；
- 新 checkpoint 必须配合其自身 `cfg` 读取。
- 推理、批推理、导出、可视化、LOCO checkpoint 评估必须先读取 `meta.json['cfg']['image_fusion_mode']`；
- 构建模型后，默认 `model.load_state_dict(state_dict, strict=True)`；
- 如果 key 不匹配，直接报错，要求使用与 checkpoint 同一 `image_fusion_mode` 的模型结构；
- `strict=False` 只允许用于明确的迁移/调试脚本，不进入常规推理链路。

建议新增一个统一加载 helper：

```python
def load_stonco_model_from_artifacts(artifacts_dir, cfg, input_dims, device):
    model = build_model_from_cfg(cfg, input_dims)
    model.load_state_dict(state_dict, strict=True)
    return model
```

这样所有入口共享同一套 checkpoint 加载逻辑，避免某些脚本仍然静默加载错误结构。

如果以后确实需要旧模型到新模型的迁移，应单独写显式 migration helper，并在 helper 内清楚打印或保存 missing/unexpected keys，而不是复用常规推理加载路径。

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
   - 新增统一 `encode(...)`
   - 扩展 `STOnco_Classifier`
   - 新增 `encoder_out_dim`

2. [stonco/core/train.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train.py)
   - 新增配置项
   - 改 `assemble_pyg`
   - 改 `prepare_graphs`
   - 改训练/验证前向调用
   - 改 optimizer param groups
   - 改 WB potential 更新，禁止直接 `model.gnn(...)`
   - 同步单次训练、kfold、LOCO 内部重复路径

3. [stonco/core/sampler.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/sampler.py)
   - static subgraph 切片必须保留 `x_gene/x_img/img_mask/pe_gene`
   - full graph clone 也必须保留双分支字段

4. [stonco/core/infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/infer.py)
   - 改组图
   - 改模型构建
   - 改 checkpoint 校验
   - 改 attribution

5. [stonco/core/batch_infer.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/batch_infer.py)
   - 改组图
   - 改模型调用
   - 改 checkpoint 校验

6. [stonco/utils/export_spot_embeddings.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/export_spot_embeddings.py)
   - 改组图
   - 改 forward
   - 改 checkpoint 校验
   - 可选导出 gate

7. [stonco/utils/visualize_prediction.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/visualize_prediction.py)
   - 改组图
   - 改 forward
   - 改 checkpoint 校验

8. [stonco/utils/eval_loco_checkpoints.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/eval_loco_checkpoints.py)
   - 改组图和模型调用
   - 复用 `InferenceEngine` 或统一模型加载 helper
   - 禁止继续假设 `data_g.x` 是唯一输入

### 15.2 通常无需改动的文件

1. [stonco/utils/prepare_data.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/prepare_data.py)
   - 当前图像 CSV 与 NPZ 契约已足够

2. [stonco/utils/preprocessing.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/utils/preprocessing.py)
   - `ImagePreprocessor` 已可复用

3. [stonco/core/train_hpo.py](/apps/users/sky_luozhihui/STOnco/model/STOnco/stonco/core/train_hpo.py)
   - 本轮先不纳入必改范围
   - 后续若需要 HPO 支持双分支，再单独补齐返回值与配置搜索空间

---

## 16. 第一版推荐的具体配置

全局默认值仍建议保持兼容：

```text
use_image_features = 0
image_fusion_mode = early_concat
```

原因：

- 旧训练命令和旧 NPZ 可以继续运行；
- 不会要求所有数据都立刻具备 `X_imgs/img_masks/img_feature_names`；
- early fusion 仍可作为 baseline。

启用双分支实验时，建议使用以下配置：

```text
use_image_features = 1
image_fusion_mode = dual_branch_residual_gate

gene branch:
    model = 当前主干配置
    GNN_hidden = 当前配置
    num_layers = 当前配置

image branch:
    img_use_pca = 0
    img_pca_dim = 256  # 仅当显式开启 img_use_pca=1 时生效
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
2. 新增模型实例方法 `model.encode(...)`
3. 扩展 `STOnco_Classifier`
4. 新增双分支 `assemble_pyg_multimodal`
5. 改 `sampler.py` 的 static subgraph 字段切片
6. 跑通训练前向

### Phase 2: 训练主路径与 checkpoint 改造

1. 改 optimizer param groups
2. 改 WB potential 更新，统一走 `model.encode(...)`
3. 改 `model.encoder_out_dim` 相关调用
4. 新增/复用 checkpoint 严格校验 helper
5. 同步 `train.py` 单次训练、kfold、LOCO 重复路径

### Phase 3: 推理与导出改造

1. 改 `infer.py`
2. 改 `batch_infer.py`
3. 改 `export_spot_embeddings.py`
4. 改 `visualize_prediction.py`
5. 改 `eval_loco_checkpoints.py`

### Phase 4: 解释性与诊断

1. 重写 gene attribution 路径
2. 新增 gate 统计
3. 可选导出 gate 空间图

### Phase 5: 实验与消融

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
4. `model.forward(..., return_h=True)['h']` 与 `model.encode(...)` 输出一致
5. 代码中不再存在用于获取共享表征的外部 `model.gnn(...)` 调用

### 18.2 训练行为

1. `avg_gate_mean` 初期较小
2. 随训练推进，`avg_gate_present` 可上升，但不应迅速饱和到 1
3. `avg_gate_missing` 应接近 0
4. dual branch 模式下 optimizer 参数组包含 image branch 与 fusion 参数
5. WB potential loss 与 WB model loss 使用同一个 `h_fused` 空间

### 18.3 推理一致性

1. 训练和推理对同一 slide 输出形状一致
2. `export_spot_embeddings` 能导出 `h_fused`
3. `visualize_prediction` 能正常画图
4. `eval_loco_checkpoints.py` 能按 checkpoint 自身 `image_fusion_mode` 正确加载模型
5. 模式不匹配 checkpoint 会明确报错，而不是静默随机初始化新模块

### 18.4 解释性

1. gene attribution 仅依赖 `x_gene`
2. image 分支值变化不会改变 gene attribution 的维度定义

### 18.5 sampler 与重复训练路径

1. `subgraph_mode=static` 下，子图保留 `x_gene/x_img/img_mask/pe_gene`
2. kfold 路径能完成一次最小训练和保存
3. LOCO 路径能完成一个癌种的最小训练和 per-slide 评估

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

## 21. 已确认实现决策

本节记录最终确认的实现标准。后续代码实现以本节为准；如果前文仍有“建议”措辞，遇到冲突时以本节为准。

### 21.1 模型与融合

1. 保留现有类名 `STOnco_Classifier`，本轮不改名为 `STOnco`。
2. 新增模型实例方法 `model.encode(...)`，作为共享表征 `h` 的唯一出口。
3. `forward(...)` 内部也必须先调用 `encode(...)` 得到 `h`，再进入分类头和域头。
4. 所有外部路径禁止直接调用 `model.gnn(...)` 获取共享表征。
5. 双分支模式下：
   - `out['h'] = h_fused`
   - `out['h_fused'] = h_fused`
   - `h_fused` 是 GRL、MMD、WB 的唯一主对齐空间
6. 第一版不对 `h_img` 单独加 GRL、MMD、WB。
7. `ClassifierHead` 结构保持不变，只把输入从旧 `h` 改为 `h_fused`。
8. gate 固定采用 scalar residual gate：

```text
g: [N, 1]
h_fused = h_gene + g * u_img
```

9. gate MLP 最后一层 bias 负初始化，默认 `fusion_gate_bias_init = -2.0`。
10. 第一版不做 image auxiliary head，也不做 image auxiliary classification loss。

### 21.2 图像特征与 image branch

1. 图像特征来自外部图像模型，例如 UNI、ResNet50 或其他 encoder。
2. 原始 `X_img` 维度不固定，不能在模型里写死 2048、1024、768 等维度。
3. 模型实际使用的图像输入维度由图像预处理结果动态决定：

```text
img_in_dim = Xp_img.shape[1] = img_pp.out_dim()
```

4. `img_pca_dim` 只是 PCA 开启时的输出维度配置，不代表原始图像特征维度。
5. image branch 默认比 gene branch 更轻：

```text
img_num_layers = 2
img_gnn_hidden = [128, 64]
```

6. image branch 允许独立配置 `img_model`，默认 `img_model = model`。
7. LapPE 第一版只给 gene branch；image branch 不使用 LapPE。
8. 保留 `ImgProj`，即使 `h_gene` 与 `h_img` 同维也保留投影层。

### 21.3 缺失图像与 mask

1. `img_mask` 是图像模态有效性标记：

```text
img_mask = 1  表示该 spot 有有效图像特征
img_mask = 0  表示该 spot 缺失图像特征
```

2. 对缺失图像执行节点级 hard masking，并作为强约束：

```text
img_mask = 0 => gate = 0
img_mask = 0 => image residual = 0
img_mask = 0 => h_fused = h_gene
```

3. image branch 内部必须做 mask propagation：

```text
x_img_input = x_img * img_mask
每层 image hidden 后继续 h_img_layer = h_img_layer * img_mask
fusion 前 h_img = h_img * img_mask
u_img = u_img * img_mask
g = g * img_mask
```

4. 第一版不做 image GNN 的边级 valid-valid 过滤。缺失节点自身必须被 hard mask；valid-invalid 边过滤留作后续增强。

### 21.4 数据对象与接口

1. 双分支图对象显式保存：

```text
data.x_gene
data.x_img
data.img_mask
data.pe_gene  # optional
```

2. 保留 `data.x = data.x_gene` 作为兼容字段，但新模型前向不依赖 `data.x`。
3. `prepare_graphs(...)` 返回值改为 `input_dims` 字典：

```python
input_dims = {
    'gene_in_dim': int(...),
    'img_in_dim': int(...) or None,
}
```

4. `sampler.py` 的 static subgraph 和 full graph clone 必须保留所有 node-level 字段：

```text
x
x_gene
x_img
img_mask
pe_gene
y
pos
```

### 21.5 训练、导出与解释性

1. optimizer 参数组必须显式覆盖：

```text
gnn_gene
gnn_img
fusion
clf
dom_slide/dom_cancer
```

2. WB potential 更新必须改走 `model.encode(...)`，不能再直接调用 `model.gnn(...)`。
3. WB、MMD、GRL、classifier、embedding 导出必须使用同一个 `h_fused` 空间。
4. embedding 导出第一版至少支持：

```text
h_fused
z_clf
```

5. `gate` 做成可选导出，用于诊断 image shortcut。
6. gene attribution 继续只解释 gene：
   - 只对 `x_gene` 求导
   - `x_img` 和 `img_mask` 作为常量输入
   - 第一版不做 image attribution
7. 训练日志和 `hist` 增加 gate 诊断指标：

```text
avg_gate_mean
avg_gate_present
avg_gate_missing
avg_img_residual_norm
```

### 21.6 兼容与 checkpoint

1. 保留 early fusion 作为并行 baseline，通过 `image_fusion_mode` 切换：

```text
early_concat
dual_branch_residual_gate
```

2. 全局默认保持兼容：

```text
use_image_features = 0
image_fusion_mode = early_concat
```

3. 启用双分支时显式配置：

```text
use_image_features = 1
image_fusion_mode = dual_branch_residual_gate
```

4. 常规训练产物、推理、导出、可视化默认使用：

```python
model.load_state_dict(state_dict, strict=True)
```

5. 旧 early-fusion/gene-only checkpoint 只能按其自身 `image_fusion_mode` 加载，不自动升级为 dual branch。
6. `strict=False` 只允许用于单独的迁移/调试 helper，不进入常规链路。

### 21.7 本轮范围

本轮必须覆盖：

1. `models.py`
2. `train.py`
3. `sampler.py`
4. `infer.py`
5. `batch_infer.py`
6. `export_spot_embeddings.py`
7. `visualize_prediction.py`
8. `eval_loco_checkpoints.py`

`train.py` 内部的单次训练、kfold、LOCO 重复路径都必须同步。

本轮明确不处理：

```text
train_hpo.py
```

因此本轮不保证 HPO 路径在 dual branch 模式下可用，后续单独适配。

### 21.8 实现顺序

按阶段推进：

1. `models.py + train.py + sampler.py`
2. 统一 checkpoint 加载校验
3. `infer.py + batch_infer.py`
4. `export_spot_embeddings.py + visualize_prediction.py + eval_loco_checkpoints.py`
5. attribution 与 gate 诊断
