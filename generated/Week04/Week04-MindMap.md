# Week 04 Overfitting & Regularization

## Gradient Descent 回顾

### 核心思想
- 找函数最小值（optimal point）
- 比喻：雾天山顶找下山路

### 算法步骤
- 随机初始点
- 计算梯度
- 负梯度方向移动
- X_next = X_current - α × gradient

### Learning Rate (α)
- 太大 → 跳过最小值
- 太小 → 收敛太慢
- Hyperparameter

## Overfitting vs Underfitting

### Underfitting（欠拟合）
- 模型太简单
- Train error 高, Test error 高
- 例：直线拟合曲线数据

### Overfitting（过拟合）
- 模型太复杂，记住噪声
- Train error 低, Test error 高
- 例：9 次多项式拟合 10 个点

### 如何判断
- Train Error vs Test Error
- 欠拟合：都高
- 良好：都低且接近
- 过拟合：Train 低 Test 高

### 教授强调
- Test error 才是关键
- 训练误差低不代表模型好

## Model Complexity（模型复杂度）

### Polynomial Degree
- Degree 1: 欠拟合
- Degree 2-3: 拟合较好
- Degree 6-9: 过拟合
- Degree 是 Hyperparameter

### 选择方法
- Cross-Validation
- Regularization

## Regularization（正则化）

### 核心思想
- 在损失函数加惩罚项
- minimize: MSE + λ × penalty

### L2 (Ridge Regression)
- 惩罚系数平方和
- 系数变小但不为 0
- 几何：圆形约束

### L1 (Lasso Regression)
- 惩罚系数绝对值和
- 系数可能为 0（特征选择）
- 几何：菱形约束

### Lambda (λ)
- λ 大 → 模型简单（underfitting）
- λ 小 → 模型复杂（overfitting）
- λ = 0 → 无正则化
- Grid Search + CV 选择

## Train-Test Split

### 数据划分
- Train Set (80%)
- Test Set (20%)

### 原则
- Never train on test set
- random_state 保证可重现

## Hyperparameters 总结

### 本周涉及
- Learning Rate (α)
- Polynomial Degree
- Lambda (λ)

### 特点
- 不是从数据学的
- 通过 Cross-Validation 选择

## 课程资源

### 笔记
- [[Week04/notes|完整笔记]]
- [[Week04/vocabulary|术语表]]
- [[Week04/qa-summary|QA总结]]
- [[Week04/Reminders|重点提醒]]
