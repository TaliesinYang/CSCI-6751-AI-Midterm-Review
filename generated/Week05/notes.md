# Week05 — Model Evaluation Metrics

> 基于合并转录（共 1691 segments, 约 100 分钟）。本周围绕 **模型评估（Model Evaluation）**，重点讲解 **回归指标（Regression Metrics）** 和 **分类指标（Classification Metrics）**，以及如何使用 **交叉验证（Cross-Validation）** 评估模型性能。

---

## 1) Validation Set vs Test Set

### 核心区别

**Validation Set（验证集）**：
- **来源**：从训练数据中划分出来
- **用途**：在训练过程中评估模型，调整超参数
- **重要性**：用于选择最佳模型，但**不用于最终测试**

**Test Set（测试集）**：
- **来源**：完全独立于训练数据
- **用途**：最终评估模型的泛化能力
- **重要性**：**只能使用一次**，避免过拟合到测试集

**教授强调**：
> "Validation is different from test data. It's a part of our train data, but it's used to evaluate the training model itself."

**典型划分**：
- Training: 60-70%
- Validation: 15-20%
- Test: 15-20%

---

## 2) 回归指标（Regression Metrics）

### 2.1 MAE (Mean Absolute Error)

**公式**：
```
MAE = (1/n) × Σ|y - ŷ|
```

**含义**：
- 预测值与真实值的**绝对误差的平均值**
- **单位与原始数据相同**

**例子**：
- y = [10, 20, 30], ŷ = [12, 18, 28]
- MAE = (|10-12| + |20-18| + |30-28|) / 3 = (2 + 2 + 2) / 3 = 2

**优点**：
- 易于理解
- 对异常值（outliers）**不敏感**

**缺点**：
- 无法反映误差的方差

---

### 2.2 MSE (Mean Squared Error)

**公式**：
```
MSE = (1/n) × Σ(y - ŷ)²
```

**特点**：
- 误差**平方**，放大大误差的影响
- **单位是原始数据的平方**（如果 y 单位是米，MSE 单位是平方米）

**为什么用平方？**
1. **消除负号**：避免正负误差抵消
2. **惩罚大误差**：比 MAE 更关注离群值

**教授例子**：
- Model A: MAE = 2, MSE = 10
- Model B: MAE = 2, MSE = 4
- **结论**：Model B 更稳定（方差小）

---

### 2.3 RMSE (Root Mean Squared Error)

**公式**：
```
RMSE = √MSE = √[(1/n) × Σ(y - ŷ)²]
```

**为什么需要 RMSE？**

**问题**：MSE 的单位是平方，难以解释

**解决**：开平方根，**单位与原始数据相同**

**教授强调**：
> "When you want to compare model one with model two, RMSE is more interpretable than MSE because it's in the same unit as your target variable."

**比较**：
- MSE = 100 → RMSE = 10
- 如果目标变量是"房价（万元）"，RMSE = 10 表示平均误差 10 万元

---

### 2.4 R² (R-Squared / Coefficient of Determination)

**公式**：
```
R² = 1 - (SSres / SStot)

其中：
SSres = Σ(y - ŷ)²  (残差平方和)
SStot = Σ(y - ȳ)²  (总平方和, ȳ = 均值)
```

**含义**：
- **模型解释了多少百分比的方差**
- **范围**：0 到 1（可能为负）
  - R² = 1：完美拟合
  - R² = 0：模型和预测均值一样差
  - R² < 0：模型比均值还差

**解释**：
- R² = 0.85 → 模型解释了 85% 的方差

---

## 3) 分类指标（Classification Metrics）

### 3.1 混淆矩阵（Confusion Matrix）

**场景**：二分类问题（如：患者是否有风险？邮件是否为垃圾邮件？）

**四种情况**：

|  | **Predicted Positive (1)** | **Predicted Negative (0)** |
|---|---|---|
| **Actual Positive (1)** | True Positive (TP) | False Negative (FN) |
| **Actual Negative (0)** | False Positive (FP) | True Negative (TN) |

**教授例子**：
- **TP**：模型预测 risky，实际也是 risky ✓
- **FP**：模型预测 risky，实际是 non-risky ✗
- **TN**：模型预测 non-risky，实际也是 non-risky ✓
- **FN**：模型预测 non-risky，实际是 risky ✗

**重要性**：
- **FP (Type I Error)**：错误地拉响警报（如：误诊健康人为患者）
- **FN (Type II Error)**：漏掉真正的问题（如：漏诊患者）

---

### 3.2 Precision（精确率）

**公式**：
```
Precision = TP / (TP + FP)
```

**含义**：
- 在**所有预测为 Positive 的样本中**，真正是 Positive 的比例
- **问题**：模型说是"有风险"，有多少真的有风险？

**例子**：
- TP = 80, FP = 20
- Precision = 80 / (80 + 20) = 0.8 (80%)
- 解释：模型预测为"有风险"的患者中，80% 真的有风险

**应用场景**：
- **垃圾邮件过滤**：避免误判正常邮件为垃圾邮件（FP 成本高）

---

### 3.3 Recall（召回率 / 灵敏度）

**公式**：
```
Recall = TP / (TP + FN)
```

**含义**：
- 在**所有真实 Positive 的样本中**，模型找到了多少
- **问题**：所有真正"有风险"的患者中，模型识别出了多少？

**例子**：
- TP = 80, FN = 20
- Recall = 80 / (80 + 20) = 0.8 (80%)
- 解释：所有"有风险"的患者中，80% 被模型识别出来

**应用场景**：
- **癌症筛查**：不能漏掉任何患者（FN 成本高）

**教授强调**：
> "Recall is important when missing a positive case is costly. That's why recall is important in medical diagnosis."

---

### 3.4 Precision vs Recall 权衡

**矛盾**：
- 提高 Recall → 降低阈值 → 更多 FP → Precision 下降
- 提高 Precision → 提高阈值 → 更多 FN → Recall 下降

**例子**：
- **极端 Recall = 100%**：把所有样本都预测为 Positive（但 Precision 很低）
- **极端 Precision = 100%**：只预测最确定的样本为 Positive（但 Recall 很低）

---

### 3.5 F1 Score

**公式**：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**含义**：
- Precision 和 Recall 的**调和平均数**（Harmonic Mean）
- **范围**：0 到 1（越高越好）

**为什么用调和平均数？**
- **普通平均**：(0.9 + 0.1) / 2 = 0.5
- **调和平均**：2 × (0.9 × 0.1) / (0.9 + 0.1) = 0.18
- **优势**：惩罚极端情况，要求 Precision 和 Recall 都高

**教授例子**：
- Precision = 0.8, Recall = 0.5
- F1 = 2 × (0.8 × 0.5) / (0.8 + 0.5) = 0.615

**应用**：
- 当 Precision 和 Recall **同等重要**时使用

---

## 4) Cross-Validation（交叉验证）

### 4.1 K-Fold Cross-Validation

**问题**：
- 单次划分 train/test 可能不稳定
- 数据量小时，测试集太小不可靠

**解决**：
1. 将数据划分为 K 份（chunks）
2. 每次用 1 份作为验证集，其余 K-1 份作为训练集
3. 重复 K 次，每份都当过验证集
4. 最终结果 = K 次结果的**平均值**

**教授例子（K=8）**：
```
Iteration 1: Train on [1,2,3,4,5,6,7], Validate on [8]
Iteration 2: Train on [1,2,3,4,5,6,8], Validate on [7]
Iteration 3: Train on [1,2,3,4,5,7,8], Validate on [6]
...
Iteration 8: Train on [2,3,4,5,6,7,8], Validate on [1]
```

**优点**：
- 每个样本都被用于验证
- 结果更稳定、可靠

**缺点**：
- 计算成本高（需要训练 K 次）

**常用 K 值**：
- K = 5 或 10（常见）
- K = n（Leave-One-Out，数据量很小时）

---

## 5) 重要概念总结

| 指标 | 公式 | 用途 | 单位 |
|------|------|------|------|
| **MAE** | Σ\|y - ŷ\| / n | 平均绝对误差 | 与 y 相同 |
| **MSE** | Σ(y - ŷ)² / n | 平均平方误差 | y 的平方 |
| **RMSE** | √MSE | 平均误差（可解释） | 与 y 相同 |
| **R²** | 1 - SSres/SStot | 方差解释比例 | 无单位 (0-1) |
| **Precision** | TP / (TP + FP) | 预测为正的准确率 | 比例 (0-1) |
| **Recall** | TP / (TP + FN) | 真实为正的覆盖率 | 比例 (0-1) |
| **F1** | 2PR / (P + R) | 平衡 Precision & Recall | 比例 (0-1) |

---

## 6) 实际应用建议

### 选择合适的指标

**回归问题**：
- **MAE**：对异常值不敏感
- **RMSE**：关注大误差
- **R²**：解释模型拟合程度

**分类问题**：
- **Accuracy**：类别平衡时
- **Precision**：避免误报（如：垃圾邮件过滤）
- **Recall**：避免漏报（如：癌症筛查）
- **F1**：平衡两者

**教授建议**：
> "Don't just look at one metric. Always consider the context and the cost of different types of errors."

---

## 7) 课堂例子回顾

### 回归例子

**数据**：
- y = [10, 20, 30, 40]
- ŷ = [12, 18, 28, 42]

**计算**：
- MAE = (2 + 2 + 2 + 2) / 4 = 2
- MSE = (4 + 4 + 4 + 4) / 4 = 4
- RMSE = √4 = 2

### 分类例子

**数据**：
- TP = 80, FP = 20
- FN = 10, TN = 90

**计算**：
- Precision = 80 / (80 + 20) = 0.8
- Recall = 80 / (80 + 10) = 0.889
- F1 = 2 × (0.8 × 0.889) / (0.8 + 0.889) = 0.842

---

## 附加资源

- Scikit-learn metrics documentation
- Cross-validation best practices
- ROC curve and AUC（下周可能讲解）
