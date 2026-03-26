# PLAN: 将 UMAP/t-SNE 从 `z64` 切换为直接使用 `h`

## 1. 目标与范围

目标：把当前可视化链路从

- `export_spot_embeddings.py` 导出 `z64_0...z64_63`
- `visualize_umap_tsne.py` 读取 `z64_*` 做 UMAP/t-SNE

切换为优先使用 `h`（GNN 输出表示）做 UMAP/t-SNE，可用于更直接地观察域对抗分支作用效果。

本方案只描述代码修改，不改训练主干逻辑。

---

## 2. 当前实现现状（需要改动的原因）

### 2.1 导出脚本当前只导出 `z64`

文件：`stonco/utils/export_spot_embeddings.py`

关键点：

- 模型前向使用 `return_z=True`，只读取 `out['z64']`：
  - `out = model(..., return_z=True)`
  - `z64 = out['z64']`
- 强校验 `z64.shape[1] == 64`
- 最终写入列名 `z64_0...z64_63`

问题：

- 代码没有导出 `h`；
- 导致后续可视化只能基于 `z64`，无法直接观察域对抗作用层（当前域头作用在 `h`）。

### 2.2 可视化脚本当前只识别 `z64_*`

文件：`stonco/utils/visualize_umap_tsne.py`

关键点：

- `_get_z64(df)` 函数硬编码读取前缀 `z64_`
- 并且强制要求列数为 64

问题：

- `h` 维度不是固定 64（默认 GATv2 下通常为 512）；
- 当前读取逻辑无法兼容 `h_*`。

---

## 3. 设计原则

1. 最小侵入：不改变训练与推理核心，只改导出和可视化工具。
2. 向后兼容：默认行为尽量不破坏老流程（仍可读 `z64_*`）。
3. 显式选择：允许用户通过参数选择 `h` 或 `z64`。
4. 维度自适应：`h` 维度可变，不做固定 64 的硬编码。

---

## 4. 具体修改方案（逐文件）

> 下面的“位置锚点”基于当前仓库版本（你机器上我查看到的行号）。后续若文件有改动，优先按函数名定位。

## 4.1 `stonco/utils/export_spot_embeddings.py`

### 修改 A：新增导出表示类型参数

位置锚点：

- `main()` 参数区，当前在 `parser.add_argument('--num_threads'...)` 后（约第 157 行）

具体改法（新增）：

```python
parser.add_argument(
    '--embed_source',
    choices=['h', 'z64'],
    default='h',
    help='Embedding source for export: h (GNN output) or z64 (classifier latent).'
)
```

新增 CLI 参数：

- `--embed_source`，候选：`h` / `z64`
- 默认建议：`h`（满足你当前目标）

原因：

- 让导出脚本直接切换表示来源；
- 避免写死在 `z64`。

### 修改 B：模型前向与取值逻辑

当前逻辑：

- 只 `return_z=True`，仅读 `out['z64']`

建议改为：

- 当 `embed_source=z64`：保持 `return_z=True`，取 `out['z64']`
- 当 `embed_source=h`：`return_z=False` 即可，从 `out['h']` 取 embedding

注意：

- 现有 `STOnco_Classifier.forward()` 默认未返回 `h`，需要配合 4.3 一并修改模型输出。

位置锚点：

- 现有前向和取值在 `with torch.no_grad():` 内（约第 268-278 行）

当前代码（需要替换）：

```python
out = model(..., return_z=True)
logits = out['logits'].detach().cpu().numpy()
z64 = out['z64'].detach().cpu().numpy()
p_tumor = 1.0 / (1.0 + np.exp(-logits))
```

建议替换为：

```python
need_z = (args.embed_source == 'z64')
need_h = (args.embed_source == 'h')
out = model(
    g.x,
    g.edge_index,
    batch=getattr(g, 'batch', None),
    edge_weight=getattr(g, 'edge_weight', None),
    return_z=need_z,
    return_h=need_h,
)
logits = out['logits'].detach().cpu().numpy()
p_tumor = 1.0 / (1.0 + np.exp(-logits))

if args.embed_source == 'z64':
    emb = out['z64'].detach().cpu().numpy()
    emb_prefix = 'z64_'
else:
    emb = out['h'].detach().cpu().numpy()
    emb_prefix = 'h_'
```

### 修改 C：统一列名前缀与维度写出

当前固定写法：

- `for j in range(64): df[f'z64_{j}'] = ...`

建议改为动态写法：

- `emb = selected_embedding`，shape 为 `[N, D]`
- 前缀按 source 决定：`h_` 或 `z64_`
- `for j in range(D): df[f'{prefix}{j}'] = emb[:, j]`

原因：

- `h` 维度可变，必须动态导出。

位置锚点：

- 现有 `z64` 强校验和写列在约第 280-308 行

当前代码（需要替换）：

```python
if z64.ndim != 2 or z64.shape[1] != 64:
    raise ValueError(...)
n_spots = int(z64.shape[0])
...
for j in range(64):
    df[f'z64_{j}'] = z64[:, j]
```

建议替换为：

```python
if emb.ndim != 2:
    raise ValueError(f'Expected embedding shape [N,D], got {emb.shape} for sample {sample_id}')
n_spots, emb_dim = int(emb.shape[0]), int(emb.shape[1])
if emb_dim < 2:
    raise ValueError(f'Embedding dim must be >=2, got {emb_dim} for sample {sample_id}')
...
df['embed_source'] = [args.embed_source] * n_spots
df['embed_dim'] = [emb_dim] * n_spots
for j in range(emb_dim):
    df[f'{emb_prefix}{j}'] = emb[:, j]
```

### 修改 D：输出元信息（建议）

建议在 CSV 额外写两列常量：

- `embed_source`（`h` 或 `z64`）
- `embed_dim`（D）

原因：

- 防止后续混用文件时不清楚来源与维度。

---

## 4.2 `stonco/utils/visualize_umap_tsne.py`

### 修改 A：读取函数泛化

当前函数：

- `_get_z64(df)`，仅识别 `z64_`，且强制 64 列

建议改为：

- `_get_embedding(df, prefix)`，根据前缀读取：
  - `prefix='h'` -> 读取 `h_*`
  - `prefix='z64'` -> 读取 `z64_*`
- 按末尾索引排序后取 numpy
- 不再固定 64 列，只要求 `>=2` 维

原因：

- 同时支持 `h` 和 `z64`；
- 兼容不同 backbone 下不同 `h` 维度。

位置锚点：

- 当前 `_get_z64(df)` 在约第 17-22 行

当前代码（需要替换整个函数）：

```python
def _get_z64(df: pd.DataFrame) -> np.ndarray:
    z_cols = [c for c in df.columns if c.startswith('z64_')]
    if len(z_cols) != 64:
        raise ValueError(...)
    z_cols = sorted(z_cols, key=lambda x: int(x.split('_')[-1]))
    return df[z_cols].to_numpy(dtype=float)
```

建议替换为：

```python
def _get_embedding(df: pd.DataFrame, embed_source: str) -> np.ndarray:
    prefix = f'{embed_source}_'
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        raise ValueError(f'Expected >=2 columns starting with {prefix}, got {len(cols)}')
    cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))
    return df[cols].to_numpy(dtype=float)
```

### 修改 B：新增可视化输入参数

新增 CLI 参数：

- `--embed_source`，候选 `h` / `z64`，默认 `h`

读取逻辑：

- `Z = _get_embedding(df, args.embed_source)`

位置锚点：

- 参数定义区在约第 153-160 行
- 读取 embedding 在约第 164 行

具体改法：

1) 在参数区新增：

```python
parser.add_argument(
    '--embed_source',
    choices=['h', 'z64'],
    default='h',
    help='Which embedding columns to visualize: h_* or z64_*.'
)
```

2) 把

```python
Z = _get_z64(df)
```

替换为

```python
Z = _get_embedding(df, args.embed_source)
```

### 修改 C：输出文件名包含来源（建议）

当前文件名固定：

- `umap_tsne_by_tumor.svg` 等

建议改为：

- `umap_tsne_h_by_tumor.svg`
- `umap_tsne_z64_by_tumor.svg`

原因：

- 避免覆盖与混淆，便于 A/B 对照。

位置锚点：

- `plots = [...]` 定义在约第 185-189 行

当前：

```python
plots = [
    ('tumor_label', 'umap_tsne_by_tumor.svg'),
    ('batch_id', 'umap_tsne_by_batch.svg'),
    ('cancer_type', 'umap_tsne_by_cancer.svg'),
]
```

建议替换为：

```python
plots = [
    ('tumor_label', f'umap_tsne_{args.embed_source}_by_tumor.svg'),
    ('batch_id', f'umap_tsne_{args.embed_source}_by_batch.svg'),
    ('cancer_type', f'umap_tsne_{args.embed_source}_by_cancer.svg'),
]
```

---

## 4.3 `stonco/core/models.py`

### 修改：`forward()` 增加可选返回 `h`

当前 `forward()` 输出 dict 包含：

- `logits`
- `dom_logits_slide`
- `dom_logits_cancer`
- （可选）`z64`

建议新增参数：

- `return_h=False`

并在 `return_h=True` 时：

- `out['h'] = h`

原因：

- 导出脚本获取 `h` 最稳妥的方式应通过模型 forward 的显式输出；
- 比在导出脚本中绕过分类头重复写 `model.gnn(...)` 更安全，避免未来改动造成分歧。

兼容性：

- 默认 `return_h=False`，不影响现有训练与推理调用。

位置锚点：

- `STOnco_Classifier.forward()` 在约第 170 行

当前签名（需要替换）：

```python
def forward(self, x, edge_index, batch=None, edge_weight=None, grl_beta_slide=1.0, grl_beta_cancer=1.0, return_z=False):
```

建议改为：

```python
def forward(
    self,
    x,
    edge_index,
    batch=None,
    edge_weight=None,
    grl_beta_slide=1.0,
    grl_beta_cancer=1.0,
    return_z=False,
    return_h=False,
):
```

并在输出 dict 组装处（约第 180-183 行）新增：

```python
if return_h:
    out['h'] = h
```

最终与现有 `return_z` 并存：

```python
if return_z:
    out['z64'] = z64
if return_h:
    out['h'] = h
```

---

## 4.4 一次性检查点（防漏改）

1. `export_spot_embeddings.py` 中不应再出现固定 `for j in range(64)` 写 embedding 列。  
2. `visualize_umap_tsne.py` 中不应再调用 `_get_z64(...)`。  
3. `models.py` forward 输出在 `return_h=True` 时必须包含 `out['h']`。  
4. 用 `rg -n "z64_|return_h|embed_source|_get_z64"` 做回归检查，确认新旧分支都可用。

---

## 5. 推荐修改顺序

1. 先改 `models.py` 支持 `return_h`。
2. 再改 `export_spot_embeddings.py` 支持 `--embed_source` 与动态列导出。
3. 最后改 `visualize_umap_tsne.py` 支持按前缀读取并命名输出文件。
4. 用同一批样本分别导出 `h` / `z64` 两套 CSV，做并排可视化对照。

---

## 6. 验证清单（必须做）

1. 导出 `h`：
   - 命令可运行；
   - 输出 CSV 存在 `h_0...h_{D-1}`；
   - `embed_dim` 与实际列数一致。
2. 导出 `z64`（回归测试）：
   - 仍能输出 `z64_0...z64_63`；
   - 与旧脚本结果维度一致。
3. 可视化 `h`：
   - `visualize_umap_tsne.py --embed_source h` 正常产图。
4. 可视化 `z64`：
   - `--embed_source z64` 正常产图，且不破坏历史使用习惯。
5. 稳定性：
   - 固定 `--seed` 下重复运行，结果可复现（UMAP/TSNE 内部随机性受控）。

---

## 7. 潜在风险与应对

风险 1：`h` 维度较高（默认可达 512），TSNE 更慢。

- 应对：使用已有 `--max_points` 下采样；必要时先 PCA 到 50 维再 TSNE（可后续加开关）。

风险 2：老脚本/老分析默认假设只有 `z64_*`。

- 应对：保留 `--embed_source z64`，并让可视化脚本双兼容。

风险 3：误把“域混合好”当作“模型好”。

- 应对：同时看 `tumor_label` 分离与 `batch/cancer` 混合，不单看一个视角。

---

## 8. 最小可执行命令（改完后）

导出 `h`：

```bash
python -m stonco.utils.export_spot_embeddings \
  --artifacts_dir artifacts \
  --train_npz data/train.npz \
  --out_csv artifacts/spot_embeddings_h.csv \
  --embed_source h
```

画 `h` 的 UMAP/t-SNE：

```bash
python -m stonco.utils.visualize_umap_tsne \
  --embeddings_csv artifacts/spot_embeddings_h.csv \
  --out_dir artifacts \
  --embed_source h \
  --max_points 50000
```

---

## 9. 为什么这是当前任务下的合理方案

你的目标是检查域对抗是否促使模型学到跨癌种/跨批次可泛化特征。当前域对抗分支直接作用于 `h`，因此把可视化输入切换到 `h` 能更直接反映机制层效果；同时保留 `z64` 作为对照可判断“机制效果是否真正传递到最终判别表示”，两者结合比单看 `z64` 更可靠。
