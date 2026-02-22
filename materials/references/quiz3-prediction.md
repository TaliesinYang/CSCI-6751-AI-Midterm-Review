# Quiz #3 预测分析

## 📊 已考内容回顾

### **Quiz #1（Oct 14, 2025）**
- ✅ Question 1 (25分): **Gradient Descent 手算**（Linear Regression）
- ✅ Question 2 (25分): **Fuzzy Logic**（隶属函数 + Centroid）

### **Quiz #2（Dec 2, 2025）**
- ✅ Question 1 (40分): **Classification Metrics**（Confusion Matrix, Precision, Recall, F1）
- ✅ Question 2 (60分): **Neural Network**（Forward Pass + Backpropagation）

### **Midterm（Oct 14, 2025）**
- ✅ Question 1 (25分): **Multivariate Linear Regression + GD + L2 Regularization**
- ✅ Question 2 (25分): **Fuzzy Logic**（Coffee Maker System）
- ✅ MCQ (50分): Overfitting/Underfitting, L1/L2 Regularization, 矩阵维度等

---

## 🎯 Quiz 模式分析

### **时间间隔**
- Quiz #1 → Quiz #2: **约7周**（Oct 14 → Dec 2）
- Quiz #2 → 现在: **约9周**（Dec 2 → Feb 8）
- **结论**: 如果按照这个节奏，下周很可能有 Quiz #3！

### **题型规律**
1. **每次2道大题**（总分50-100分）
2. **考最近2-3周的重点内容**
3. **重复但不同形式**：
   - Gradient Descent 在 Quiz #1 和 Midterm 都考了（但 Midterm 加了 Regularization）
   - Fuzzy Logic 在 Quiz #1 和 Midterm 都考了（但题目不同）
4. **逐渐递进**：
   - Quiz #1 → Quiz #2 → 难度递增
   - 手算 → 概念理解 → 综合应用

---

## 🔮 Quiz #3 预测（可能性排序）

### **高可能性（90%+）**

#### **预测 1: Regularization 深入考察**

**为什么？**
- Midterm 考了 L2 Regularization（只是计算 Cost）
- L1 vs L2 在 MCQ 考了概念，但没有深入计算题
- Week04 重点讲了 Regularization，但 Quiz #2 没考

**可能的题型：**

**题型 A：L1 vs L2 对比计算**
```
给定数据和参数：
- 数据点：(x, y) = [(1,3), (2,5), (3,7)]
- 模型：y = ax + b
- 参数：a = 2.5, b = 0.8
- λ = 0.5

任务：
(a) 计算 L2 Regularized Cost: J = MSE + λΣw²
(b) 计算 L1 Regularized Cost: J = MSE + λΣ|w|
(c) 解释两者的区别
```

**题型 B：带 Regularization 的 Gradient Descent**
```
给定：
- 数据：(1,3), (2,5)
- 初始：a=1, b=0
- Learning rate: η=0.1
- L2 Regularization: λ=0.5

任务：计算一次 GD 迭代（包含正则化项）
梯度公式：
∂J/∂a = (1/n)Σ(ŷ-y)x + λa
∂J/∂b = (1/n)Σ(ŷ-y)
```

---

#### **预测 2: Overfitting/Underfitting 判断与调整**

**为什么？**
- Week04 核心内容
- 下周作业要演示 Overfitting 识别
- Midterm MCQ 考了概念，但没有实际计算题

**可能的题型：**

**题型 A：给数据判断模型状态**
```
给定三个模型在同一数据集上的表现：

| 模型 | Polynomial Degree | Train MSE | Test MSE |
|------|-------------------|-----------|----------|
| A    | 1                 | 45        | 48       |
| B    | 3                 | 12        | 15       |
| C    | 9                 | 2         | 85       |

任务：
(a) 哪个模型 Underfitting？为什么？
(b) 哪个模型 Overfitting？为什么？
(c) 哪个模型最好？为什么？
(d) 如何改进模型 C？（至少3种方法）
```

**题型 B：调整超参数**
```
你训练了一个 Ridge Regression 模型：
- Degree = 6
- λ = 0.01
- Train MSE = 5, Test MSE = 80

观察：模型严重 Overfitting

任务：
(a) 列出3种减少 Overfitting 的方法
(b) 如果增加 λ 到 10，Train MSE 和 Test MSE 会如何变化？
(c) 如果减少 Degree 到 2，会如何变化？
```

---

### **中等可能性（60-70%）**

#### **预测 3: Normal Equation 手算**

**为什么？**
- Week03 重点内容
- Midterm 只在 MCQ 考了维度（没有手算）
- 是 Linear Regression 的核心方法之一

**可能的题型：**

```
给定数据：
x    y
1    2
2    4
3    5

任务：使用 Normal Equation θ = (X^T X)^(-1) X^T y 求解 y = ax + b

步骤：
(a) 构建设计矩阵 X 和目标向量 y
(b) 计算 X^T X
(c) 计算 (X^T X)^(-1)（2×2 矩阵求逆）
(d) 计算 X^T y
(e) 得到最终参数 θ = [b, a]
```

---

#### **预测 4: Feature Encoding 实际应用**

**为什么？**
- Week02 讲过，但从未考过
- 是实际 ML 中的重要步骤
- 可能作为小题或选择题出现

**可能的题型：**

```
给定数据集：

| Car Type | Color | Price |
|----------|-------|-------|
| SUV      | Red   | 30000 |
| Sedan    | Blue  | 25000 |
| Truck    | Red   | 40000 |

任务：
(a) 对 "Car Type" 进行 One-Hot Encoding
(b) 对 "Color" 进行 Dummy Encoding（避免 Dummy Variable Trap）
(c) 解释 One-Hot vs Dummy 的区别
(d) 为什么线性回归要用 Dummy Encoding？
```

---

### **低可能性（30-40%）**

#### **预测 5: Neural Network 更深入的题目**

**为什么？**
- Quiz #2 刚考过
- 可能不会连续两次 Quiz 考同样的内容
- 除非课程继续深入 NN（如 CNN, RNN）

**如果考，可能的形式：**
- 不同网络结构（3层变4层）
- 不同激活函数（ReLU, Tanh）
- 完整的 Backpropagation（不只是最后一层）

---

#### **预测 6: Classification Metrics 扩展**

**为什么？**
- Quiz #2 刚考过
- 可能不会重复

**如果考，可能的形式：**
- ROC Curve / AUC
- Precision-Recall Trade-off
- Multi-class Classification

---

## 🎯 最可能的组合（Top 3）

### **组合 1（最高可能性 ⭐⭐⭐⭐⭐）**
- **Question 1 (40-50分)**: Regularization 深入计算（L1 vs L2，或带正则化的 GD）
- **Question 2 (50-60分)**: Overfitting/Underfitting 判断与调整

**理由**：
- Week04 核心内容
- Midterm 只浅显考了，没有深入
- 下周作业演示正好是 Overfitting 相关

---

### **组合 2（高可能性 ⭐⭐⭐⭐）**
- **Question 1 (40-50分)**: Normal Equation 手算（完整推导）
- **Question 2 (50-60分)**: Regularization + Overfitting 综合题

**理由**：
- Normal Equation 是重点但没考过手算
- Regularization 是 Week04 核心
- 两者可以结合（用 Normal Equation 求解 Ridge Regression）

---

### **组合 3（中等可能性 ⭐⭐⭐）**
- **Question 1 (30-40分)**: Feature Encoding 实际应用
- **Question 2 (60-70分)**: Overfitting/Underfitting + Hyperparameter Tuning

**理由**：
- Feature Encoding 从未考过
- Overfitting 是当前重点
- 可能作为热身题 + 主要题的组合

---

## 📝 复习优先级（基于预测）

### **必须掌握（P0 - 最高优先级）**

#### 1️⃣ **Regularization（L1 vs L2）**
- [ ] L2 (Ridge) 公式：J = MSE + λΣw²
- [ ] L1 (Lasso) 公式：J = MSE + λΣ|w|
- [ ] 带正则化的梯度：
  - ∂J/∂w = (∂MSE/∂w) + λw（L2）
  - ∂J/∂w = (∂MSE/∂w) + λ·sign(w)（L1）
- [ ] 能解释两者区别（L2收缩，L1稀疏）
- [ ] 能手算一次带 L2 的 GD 迭代

**练习题**：
- 给定参数和 λ，计算 Regularized Cost
- 给定数据，手算一次 Ridge GD 迭代

---

#### 2️⃣ **Overfitting/Underfitting 判断**
- [ ] 定义：
  - Underfitting: Train error 高 + Test error 高
  - Overfitting: Train error 低 + Test error 高
  - Good fit: Train ≈ Test（都低）
- [ ] 解决方法：
  - Overfitting → 增加 λ, 减少 degree, 增加数据, Early Stopping
  - Underfitting → 减少 λ, 增加 degree, 增加特征
- [ ] Hyperparameters 的影响：
  - Degree ↑ → 更复杂 → 更容易 Overfit
  - λ ↑ → 更简单 → 更容易 Underfit

**练习题**：
- 给定多个模型的 Train/Test MSE，判断状态
- 给定一个 Overfitting 的模型，列出改进方法

---

### **重要掌握（P1 - 高优先级）**

#### 3️⃣ **Normal Equation 手算**
- [ ] 公式：θ = (X^T X)^(-1) X^T y
- [ ] 步骤：
  1. 构建设计矩阵 X（第一列全是1）
  2. 计算 X^T X（2×2 或 3×3 矩阵）
  3. 求逆 (X^T X)^(-1)（2×2 矩阵求逆公式）
  4. 计算 X^T y
  5. 矩阵乘法得到 θ
- [ ] 2×2 矩阵求逆公式：
  ```
  A = [a b]     A^(-1) = 1/(ad-bc) × [ d  -b]
      [c d]                          [-c   a]
  ```

**练习题**：
- 给定2-3个数据点，用 Normal Equation 求解 y = ax + b
- 验证结果（代入数据检查）

---

#### 4️⃣ **Feature Encoding**
- [ ] One-Hot Encoding：
  - 每个类别变成一个二进制列
  - K 个类别 → K 列
- [ ] Dummy Encoding：
  - K 个类别 → K-1 列
  - 避免 Dummy Variable Trap（多重共线性）
- [ ] 何时用哪个：
  - 树模型 → One-Hot
  - 线性模型 → Dummy

**练习题**：
- 给定分类特征，手动编码
- 解释为什么线性回归要用 Dummy

---

### **了解即可（P2 - 中等优先级）**

#### 5️⃣ **Gradient Descent 复习**
- 虽然 Quiz #1 考过，但可能和 Regularization 结合再考
- 确保记住公式和步骤

#### 6️⃣ **Neural Network 基础**
- Quiz #2 刚考过，短期内可能不会再考
- 但要保持熟悉度

---

## 📚 推荐复习资料

### **自己写的 PDF 教程**
1. ✅ `Gradient-Descent-Tutorial.pdf`（已有）
2. ✅ `Fuzzy-Logic-Tutorial.pdf`（已有）
3. ✅ `Linear-Algebra-Basics.pdf`（已有）
4. ✅ `CSCI-6751-Formula-Cheatsheet.pdf`（已有）

### **Week 笔记**
- Week03: Linear Regression, GD, Normal Equation
- Week04: Overfitting, Regularization, Hyperparameters

### **真题练习**
- Quiz #1, Quiz #2, Midterm（都有答案）

---

## ⏰ 7天复习计划（针对 Quiz #3）

### **Day 1-2（周末）: Regularization 深入学习**
- [ ] 复习 Week04 笔记（Regularization 部分）
- [ ] 理解 L1 vs L2 的区别（公式 + 几何解释）
- [ ] 练习计算 Regularized Cost
- [ ] 练习带正则化的 GD 手算（自己出2-3道题）

### **Day 3（周一）: Overfitting/Underfitting 判断**
- [ ] 复习 Week04 笔记（Overfitting 部分）
- [ ] 总结判断标准（Train/Test MSE 对比）
- [ ] 列出所有解决方法（至少6种）
- [ ] 做假想题：给定场景，如何调整参数

### **Day 4（周二）: Normal Equation 手算**
- [ ] 复习 Week03 笔记（Normal Equation）
- [ ] 复习 `Linear-Algebra-Basics.pdf`
- [ ] 练习 2×2 矩阵求逆（至少3道）
- [ ] 完整做一道 Normal Equation 手算题

### **Day 5（周三）: Feature Encoding**
- [ ] 复习 Week02 笔记（Data Encoding 部分）
- [ ] 练习 One-Hot 和 Dummy Encoding
- [ ] 理解 Dummy Variable Trap
- [ ] 做几道编码练习题

### **Day 6（周四）: 综合复习**
- [ ] 做 Quiz #1 和 Quiz #2 的题目（限时）
- [ ] 检查所有公式是否记住
- [ ] 复习 `Formula-Cheatsheet.pdf`
- [ ] 查漏补缺

### **Day 7（周五）: 模拟考试 + 放松**
- [ ] 自己出2道题，限时做
- [ ] 复习错题
- [ ] 整理考试用的公式卡片
- [ ] 早点睡觉，保持状态

---

## 🎯 考试当天策略

### **时间分配（假设 50 分钟，100 分）**
- **读题 + 分配时间**：5 分钟
- **Question 1**：20-22 分钟
- **Question 2**：20-25 分钟
- **检查 + 补充**：3-5 分钟

### **做题顺序**
1. **先做简单的**（Regularization Cost 计算）
2. **再做熟悉的**（Overfitting 判断）
3. **最后做复杂的**（Normal Equation 手算）

### **必带物品**
- ✅ 计算器（如果允许）
- ✅ 公式卡片（手写的，复习用）
- ✅ 铅笔 + 橡皮（方便修改）
- ✅ 水 + 纸巾

---

## 📋 公式速查表（考前必看）

### **Regularization**
```
L2 (Ridge):
J = (1/n)Σ(y - ŷ)² + λΣw²
∂J/∂w = (2/n)Σ(ŷ-y)x + 2λw

L1 (Lasso):
J = (1/n)Σ(y - ŷ)² + λΣ|w|
∂J/∂w = (2/n)Σ(ŷ-y)x + λ·sign(w)
```

### **Gradient Descent**
```
w_new = w_old - η × ∂J/∂w

∂J/∂a = (2/n)Σ(ŷ-y)x
∂J/∂b = (2/n)Σ(ŷ-y)
```

### **Normal Equation**
```
θ = (X^T X)^(-1) X^T y

2×2 矩阵求逆：
A^(-1) = 1/(ad-bc) × [ d  -b]
                      [-c   a]
```

### **Overfitting 判断**
```
Underfitting: Train ↑ + Test ↑
Overfitting:  Train ↓ + Test ↑
Good Fit:     Train ≈ Test ↓
```

---

## 💡 最后建议

1. **不要过度紧张**：Quiz 是检验学习效果，不是刁难
2. **重视手算练习**：考试可能没有 Python
3. **理解 > 记忆**：理解原理比背公式更重要
4. **时间管理**：不要在一道题上花太久
5. **检查单位/符号**：特别是梯度的正负号
6. **写清楚步骤**：即使答案错了，步骤对也有分

**祝你 Quiz #3 顺利！🎉**
