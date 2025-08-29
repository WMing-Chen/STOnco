# 🎯 双域自适应方案（Cancer Domain + Slide Domain）

## 1. 目标

* **癌种域对抗 (Cancer-type adversarial)**
  抑制模型依赖癌种特异性信号，学到跨癌种共享的“肿瘤 vs 非肿瘤”特征。
* **样本域对抗 (Slide-id adversarial)**
  抑制批次效应/切片特异性信号，提升对新样本的泛化。

---

## 2. 数据划分策略

1. **主要划分**

   * 用 **癌种分层划分 (stratified by cancer)** 或 **分层 k-fold** → 保证训练集和验证集都覆盖主要癌种。
2. **稀有癌种处理**

   * 若某癌种 ≤2 张切片 → 强制划入训练集，不参与验证。
   * 在 domain adversarial 时将其合并到 `"Other"`，避免 domain classifier 崩溃。
3. **额外评估**

   * 做 **leave-one-cancer-out** 验证 → 检查跨癌种泛化能力。

---

## 3. 模型结构

基于你已有的 `SpotoncoGNNClassifier`，扩展为 **双域对抗**：

```
               ┌──────────────┐
Input (X,xy) → │  GNN Encoder │ ──────────────┐
               └──────────────┘               │
                      │                        │
                 Task Head                 GRL (λ1)
               (肿瘤分类器)                   │
                      │                       │
                      └──── Cancer Domain Head (癌种分类器)
                                               │
                                          GRL (λ2)
                                               │
                                          Slide Domain Head (切片分类器)
```

* **Task Head**: 主任务 (肿瘤 vs 非肿瘤)。
* **Cancer Domain Head**: 对抗癌种 → λ1 控制 loss 权重。
* **Slide Domain Head**: 对抗切片 → λ2 控制 loss 权重。

---

## 4. Loss 设计

```python
Total_Loss = Task_Loss 
           + λ1 * Cancer_Domain_Loss 
           + λ2 * Slide_Domain_Loss
```

* **Task\_Loss**: BCE (二分类)
* **Cancer\_Domain\_Loss**: CrossEntropy（癌种分类）
* **Slide\_Domain\_Loss**: CrossEntropy（切片分类）
* **λ1, λ2**: 可调权重（推荐 λ1=0.3, λ2=0.1 起步，根据验证效果调节）

---

## 5. 训练流程

1. **输入构建**：

   * 每个 graph 加上 `slide_id`、`cancer_type`。
2. **正向传播**：

   * GNN Encoder → Task Head 得到肿瘤预测；
   * 经过 GRL → Cancer Domain Head 预测癌种；
   * 经过 GRL → Slide Domain Head 预测切片。
3. **计算 loss**：

   * 任务 loss + 两个 domain loss（加权）。
4. **反向传播**：

   * GRL 自动反向梯度，迫使 encoder 学到“与癌种/切片无关”的表征。

---

## 6. 推理阶段

* **只用 Task Head** 做预测。
* **癌种/切片标签完全不需要**，域分类器仅在训练时作为正则化。

---

## 7. 超参数调优建议

* **分阶段**：

  * 阶段 1：固定 λ1=0, λ2=0（无对抗），先调 GNN 主干和学习率。
  * 阶段 2：开启癌种域对抗（λ1>0, λ2=0），找到合适 λ1。
  * 阶段 3：再加入 slide 域对抗，调整 λ2。
* **交叉验证**：

  * 最终超参数确定后，用癌种分层 k-fold cross-validation 评估稳健性。
  * 同时做 leave-one-cancer-out，检查跨癌种泛化。

---

## 8. 评估指标

* **常规指标**：AUROC、AUPRC、Accuracy、Macro-F1。
* **跨癌种指标**：

  * 在 leave-one-cancer-out 验证时，单独汇报每个癌种的指标。
  * 看模型在未见癌种上的 AUROC/AUPRC。

---

✅ **最终效果**：

* 训练时用双域对抗（癌种 + 切片），强制模型学到跨癌种、跨批次的稳健特征。
* 验证时既保证常规划分指标稳定，又能在 leave-one-cancer-out 测试泛化到新癌种的能力。
* 推理时只需输入基因表达和空间坐标，不需要任何域标签。


