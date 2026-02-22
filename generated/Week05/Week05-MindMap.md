# Week 05 Model Evaluation Metrics

## Validation vs Test Set

### Validation Set（验证集）
- 来自训练数据
- 调整超参数
- 选择最佳模型

### Test Set（测试集）
- 完全独立
- 只用一次
- 评估泛化能力

### 典型划分
- Train 60-70%
- Validation 15-20%
- Test 15-20%

## 回归指标（Regression Metrics）

### MAE (Mean Absolute Error)
- (1/n) × Σ|y - ŷ|
- 单位与原始数据相同
- 对异常值不敏感

### MSE (Mean Squared Error)
- (1/n) × Σ(y - ŷ)²
- 惩罚大误差
- 单位是数据的平方

### RMSE (Root MSE)
- √MSE
- 单位与原始数据相同
- 最可解释的误差指标

### R² (R-Squared)
- 1 - (SSres / SStot)
- 模型解释方差的比例
- 范围 0-1（可能为负）

## 分类指标（Classification Metrics）

### Confusion Matrix（混淆矩阵）
- TP: 预测正，实际正 ✓
- FP: 预测正，实际负 ✗ (Type I)
- TN: 预测负，实际负 ✓
- FN: 预测负，实际正 ✗ (Type II)

### Precision（精确率）
- TP / (TP + FP)
- 预测为正中真正为正的比例
- 重要场景：垃圾邮件过滤（避免 FP）

### Recall（召回率）
- TP / (TP + FN)
- 真实为正中被找到的比例
- 重要场景：癌症筛查（避免 FN）

### Precision vs Recall 权衡
- 提高 Recall → Precision 下降
- 提高 Precision → Recall 下降

### F1 Score
- 2 × (P × R) / (P + R)
- 调和平均数
- 惩罚极端不平衡

## Cross-Validation（交叉验证）

### K-Fold CV
- 数据分 K 份
- 每份轮流做验证集
- 结果取平均

### 常用 K 值
- K = 5 或 10（常见）
- K = n（Leave-One-Out）

### 优缺点
- 优：每个样本都被验证，结果稳定
- 缺：计算成本高（训练 K 次）

## 应用建议

### 回归问题
- MAE: 异常值不敏感
- RMSE: 关注大误差
- R²: 拟合程度

### 分类问题
- Accuracy: 类别平衡时
- Precision: 避免误报
- Recall: 避免漏报
- F1: 平衡两者

## 课程资源

### 笔记
- [[Week05/notes|完整笔记]]
- [[Week05/vocabulary|术语表]]
- [[Week05/qa-summary|QA总结]]
- [[Week05/Reminders|重点提醒]]
