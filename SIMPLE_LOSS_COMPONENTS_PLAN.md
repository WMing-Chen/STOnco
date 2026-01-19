# STOnco Loss组件记录 - 极简实现方案

**目标**: 实现训练中Loss组件记录，仅增加一个参数，最小化代码改动

---

## 一、核心设计

### 1.1 仅增加一个参数

```bash
python -m stonco.core.train \
    --train_npz data/train.npz \
    --artifacts_dir output \
    --save_loss_components 1    # 新增：保存Loss组件（0/1, 默认: 0）
```

### 1.2 数据结构（最小化）

```python
hist = {
    # 原有字段（保持不变）
    'train_loss': [],        # 总损失（保持兼容）
    'val_auroc': [],
    'val_auprc': [],

    # 仅新增3个字段（分开存储，不计算总域损失）
    'task_loss': [],         # Task_Loss
    'slide_domain_loss': [],  # λ₁×Slide_Domain_Loss
    'cancer_domain_loss': [], # λ₂×Cancer_Domain_Loss
}
```

### 1.3 存储（单文件）

- **文件名**: `loss_components.csv`
- **格式**: CSV（直接可用）
```csv
epoch,train_loss,task_loss,slide_domain_loss,cancer_domain_loss,val_auroc,val_auprc
1,0.854,0.632,0.111,0.111,0.78,0.65
2,0.743,0.521,0.111,0.111,0.82,0.70
...
```

---

## 二、代码改动（最小化）

### 2.1 训练循环修改（约15行）

```python
# train_and_validate() 第409-437行
for epoch in range(1, cfg['epochs']+1):
    model.train()
    tot_loss = 0.0
    tot_task_loss = 0.0          # 新增
    tot_slide_dom_loss = 0.0     # 新增（分开存储）
    tot_cancer_dom_loss = 0.0    # 新增（分开存储）
    num_batches = 0

    for batch in train_loader:
        # ... forward pass ...
        loss = loss_cls

        if out.get('dom_logits_slide'):
            slide_loss = cfg['lambda_slide'] * loss_dom_slide
            loss = loss + slide_loss
            tot_slide_dom_loss += slide_loss.item()  # 新增

        if out.get('dom_logits_cancer'):
            cancer_loss = cfg['lambda_cancer'] * loss_dom_cancer
            loss = loss + cancer_loss
            tot_cancer_dom_loss += cancer_loss.item()  # 新增

        loss.backward(); opt.step()
        tot_loss += loss.item()
        tot_task_loss += loss_cls.item()  # 新增
        num_batches += 1

    # 记录
    avg_loss = tot_loss / num_batches
    hist['train_loss'].append(avg_loss)  # 兼容

    if args.save_loss_components:         # 新增判断
        hist['task_loss'].append(tot_task_loss / num_batches)
        hist['slide_domain_loss'].append(tot_slide_dom_loss / num_batches)
        hist['cancer_domain_loss'].append(tot_cancer_dom_loss / num_batches)
```

### 2.2 参数解析（3行）

```python
# train.py 第198行附近
parser.add_argument('--save_loss_components', type=int,
                    choices=[0,1], default=0,
                    help='保存Loss组件到CSV (0/1, 默认: 0)')
```

### 2.3 存储功能（18行）

```python
# run_single_training() 第485行后
if args.save_loss_components and hist.get('task_loss'):
    import pandas as pd

    # 根据history长度创建DataFrame
    n_epochs = len(hist['train_loss'])

    df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'train_loss': hist['train_loss'],
        'task_loss': hist.get('task_loss', [float('nan')] * n_epochs),
        'slide_domain_loss': hist.get('slide_domain_loss', [float('nan')] * n_epochs),
        'cancer_domain_loss': hist.get('cancer_domain_loss', [float('nan')] * n_epochs),
        'val_auroc': hist['val_auroc'],
        'val_auprc': hist['val_auprc'],
    })
    df.to_csv(os.path.join(args.artifacts_dir, 'loss_components.csv'),
              index=False, float_format='%.6f')
```

---

## 三、可视化（无改动）

**保持原有3面板图不变**（完全兼容）

仅CSV包含详细数据，用户可自行分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/loss_components.csv')

# 绘制Loss组件（分开存储，独立分析）
plt.figure(figsize=(12, 6))

# 子图1: Task vs Slide
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['task_loss'], label='Task Loss', linewidth=2)
plt.plot(df['epoch'], df['slide_domain_loss'], label='Slide Domain Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Task vs Slide Domain')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: Task vs Cancer
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['task_loss'], label='Task Loss', linewidth=2)
plt.plot(df['epoch'], df['cancer_domain_loss'], label='Cancer Domain Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Task vs Cancer Domain')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_components.png', dpi=150, bbox_inches='tight')
```

---

## 四、已确认的设计

✅ **参数命名**: `--save_loss_components` (已确认)

✅ **domain_loss定义**: 分开存储，不计算总域损失 (已确认)
```python
hist['slide_domain_loss'] = []   # λ₁×Slide_Domain_Loss
hist['cancer_domain_loss'] = []  # λ₂×Cancer_Domain_Loss
```

**设计特点**:
- 保持最简，不预计算聚合值
- 用户可自行分析各组件贡献
- 灵活性高，便于后续扩展

---

## 五、实施计划

**工作量**: 约1小时

### Phase 1: 核心实现（30分钟）
1. 修改 `train_and_validate()` - 记录task和domain loss
2. 添加命令行参数 `--save_loss_components`
3. 修改 `run_single_training()` - 保存CSV
4. 测试单折训练

### Phase 2: 模式兼容（15分钟）
5. 验证K-fold/LOCO模式正常（通过参数传递）
6. 测试backward compatibility

### Phase 3: 文档（15分钟）
7. 更新README参数说明
8. 添加使用示例

**总计**: 1小时完成

---

## 六、示例输出

### CSV文件

```csv
epoch,train_loss,task_loss,slide_domain_loss,cancer_domain_loss,val_auroc,val_auprc
1,0.854321,0.632145,0.111088,0.111088,0.782345,0.651234
2,0.743210,0.521034,0.111088,0.111088,0.823456,0.702345
3,0.692345,0.470159,0.111093,0.111093,0.851234,0.734567
...
42,0.456789,0.234567,0.111111,0.111111,0.923456,0.856789
```

### 分析示例

```python
import pandas as pd

# 加载数据
df = pd.read_csv('output/loss_components.csv')

# 计算总域损失（用户自行计算）
df['total_domain_loss'] = df['slide_domain_loss'] + df['cancer_domain_loss']

# 计算各组件占比
df['slide_ratio'] = df['slide_domain_loss'] / df['train_loss']
df['cancer_ratio'] = df['cancer_domain_loss'] / df['train_loss']

# 查看后10轮均值
last_10 = df.tail(10)
print(f"最后10轮:")
print(f"  Task Loss: {last_10['task_loss'].mean():.4f}")
print(f"  Slide Domain: {last_10['slide_domain_loss'].mean():.4f} "
      f"({last_10['slide_ratio'].mean():.1%})")
print(f"  Cancer Domain: {last_10['cancer_domain_loss'].mean():.4f} "
      f"({last_10['cancer_ratio'].mean():.1%})")
```

输出：
```
最后10轮:
  Task Loss: 0.2346
  Slide Domain: 0.1111 (24.3%)
  Cancer Domain: 0.1111 (24.3%)
```

---

## 七、向后兼容保证

✅ **100%向后兼容**:
- 默认 `save_loss_components=1`（启用）
- 原有代码无需任何修改
- 生成的文件与原流程完全一致
- 旧模型加载无影响

✅ **文件增加**:
- 仅增加1个可选CSV文件
- 不修改任何现有文件结构
- 不增加JSON/其他存储

✅ **性能影响**:
- 不启用时：性能影响 **0%**（代码不执行）
- 启用时：性能影响 **<1%**（仅累加3个float变量：task + slide + cancer）

✅ **代码侵入性**:
- 仅修改 `train.py` 1个文件
- 约40行新增代码
- 不改变任何函数签名
- 不影响训练和推理流程

---

## 八、实施状态

✅ **已确认**：
1. 参数名: `--save_loss_components`
2. domain_loss定义: 分开存储（slide和cancer独立）

**准备开始编码**（约1小时完成全部实现）

---

**方案状态**: 已确认，等待实施
