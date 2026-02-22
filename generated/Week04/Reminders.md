# Week04 Reminders（AI）

## ⚠️ 下周要求（Week05）

**📍 来源**：Recording 2, 约 34:56 (2096s)

### 📝 作业演示 + 随机提问

**任务**：
1. ✅ **完成作业代码**
   - 能在自己电脑上打开并运行
   - 不需要完全解决，但要能演示
   
2. ✅ **可以用 Google、任何 IDE**
   - **不是闭卷**（可以查资料）
   - 可以用 Jupyter Notebook、Colab、PyCharm 等
   
3. ✅ **准备回答 1-2 个随机问题**
   - 关于代码理解
   - 关于 overfitting 的观察

### 🎯 核心考点

**主题：识别和调整 Overfitting**

**可能被问到的问题**：
- 你的数据中有 overfitting 吗？如何判断？
- Train error 和 Test error 分别是多少？
- 如果增加 polynomial degree，会发生什么？
- 如何通过调整参数来减少 overfitting？
- Lambda（正则化参数）的作用是什么？

**教授原话**：
> "And find out like the reason. Like check that you can see any clear overfitting there in the data. Increasing number of ones (parameters), or reduce the number of ones, probably."

**复习重点**：
- [ ] 理解 Overfitting vs Underfitting
- [ ] 会计算 Train MSE 和 Test MSE
- [ ] 知道如何调整 polynomial degree
- [ ] 知道 Ridge regression（L2）的作用
- [ ] 能解释 lambda 参数的影响

---

## 📚 Quiz 真题分析（往年题目）

### 🔥 **Quiz #1 题型（50分，40分钟）**

#### **Question 1: Gradient Descent 手算（25分）**

**题目类型**：
- 给定：简单线性回归 `y = ax + b`
- 给定：数据点（2-3个）
- 给定：初始参数（a=0, b=0），学习率 η
- 要求：手算**一次**梯度下降迭代

**需要计算**：
1. ✅ **Predictions（预测值）**：ŷ = ax + b
2. ✅ **Error（误差）**：ŷ - y
3. ✅ **Gradients（梯度）**：
   - ∂E/∂a = (2/n) × Σ(ŷᵢ - yᵢ)xᵢ
   - ∂E/∂b = (2/n) × Σ(ŷᵢ - yᵢ)
4. ✅ **Updated parameters（更新参数）**：
   - aₙₑw = aₒₗd - η × ∂E/∂a
   - bₙₑw = bₒₗd - η × ∂E/∂b

**例题（2025年10月）**：
```
数据：(1,3), (2,5)
初始：a=0, b=0, η=0.1
MSE: E = (1/n) × Σ(yᵢ - ŷᵢ)²

步骤：
1. ŷ₁=0, ŷ₂=0
2. Error: -3, -5
3. ∂E/∂a = -13, ∂E/∂b = -8
4. aₙₑw = 1.3, bₙₑw = 0.8
```

---

#### **Question 2: Fuzzy Logic（25分）**

**题目类型**：
- 给定：温度的模糊集（Low/Medium/High，三角函数）
- 给定：输出规则（风扇速度 Slow/Medium/Fast）
- 给定：IF-THEN 规则
- 要求：(a) 计算隶属度 (b) 去模糊化（Centroid method）

**需要计算**：
1. ✅ **Membership degree（隶属度）**：
   - 用三角函数公式：μ(T) = ?
   - 对 Low, Medium, High 分别计算
2. ✅ **Defuzzification（去模糊化）**：
   - Centroid: Output = Σ(μᵢ × valueᵢ) / Σμᵢ

**例题（2025年10月）**：
```
Fuzzy sets:
- Low: (0, 0, 25)
- Medium: (20, 30, 40)
- High: (35, 50, 50)

Fan speeds: Slow=20, Medium=50, Fast=80

给定 T=30°C：
1. μ(Low)=0, μ(Medium)=1, μ(High)=0
2. Output = (0×20 + 1×50 + 0×80) / (0+1+0) = 50
```

---

### 🔥 **Quiz #2 题型（100分，50分钟）**

#### **Question 1: Classification Metrics（40分）**

**题目类型**：
- 给定：二分类问题（如垃圾邮件检测）
- 给定：预测结果统计（预测了多少，实际多少）
- 要求：(a) 混淆矩阵 (b) 计算指标 (c) 解释哪个指标最重要

**需要计算**：
1. ✅ **Confusion Matrix（混淆矩阵）**：
   ```
   TP（真正例）, FP（假正例）
   FN（假负例）, TN（真负例）
   ```
2. ✅ **Metrics（指标）**：
   - **Precision** = TP / (TP + FP)
   - **Recall** = TP / (TP + FN)
   - **Accuracy** = (TP + TN) / Total
   - **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
3. ✅ **解释**：哪个指标最重要？为什么？

**例题（2025年12月）**：
```
200封邮件：
- 预测为垃圾邮件：70封
  - 真的是垃圾：50封（TP）
  - 不是垃圾：20封（FP）
- 预测为正常邮件：130封
  - 真的是垃圾：20封（FN）
  - 真的正常：110封（TN）

Precision = 50/(50+20) = 0.71
Recall = 50/(50+20) = 0.71
...

答案：Precision 最重要（避免把真邮件标记为垃圾）
```

---

#### **Question 2: Neural Network Backpropagation（60分）**

**题目类型**：
- 给定：小型神经网络（输入层→隐藏层→输出层）
- 给定：权重、偏置、训练样本
- 要求：(a) 前向传播计算输出 (b) 计算梯度

**需要计算**：
1. ✅ **Forward Pass（前向传播）**：
   - 隐藏层：z = W·x + b
   - 激活：h = σ(z)（sigmoid）
   - 输出：ŷ = v·h + bₒ
2. ✅ **Loss（损失）**：
   - MSE: L = (1/2)(ŷ - y_true)²
3. ✅ **Gradients（梯度）**：
   - ∂L/∂v₁, ∂L/∂v₂, ∂L/∂bₒ

**例题（2025年12月）**：
```
网络：3 inputs → 2 hidden (sigmoid) → 1 output
样本：x=[0.5, -1.0, 2.0], y_true=1.5

计算：
1. z₁, z₂（隐藏层）
2. h₁=σ(z₁), h₂=σ(z₂)
3. ŷ = v₁h₁ + v₂h₂ + bₒ
4. L = (1/2)(ŷ - 1.5)²
5. 梯度...
```

---

## 📋 Quiz 准备清单（基于真题）

### ✅ **必须会手算的**

#### 1️⃣ **Gradient Descent（梯度下降）**
- [ ] 给定数据和初始参数，计算预测值
- [ ] 计算误差（error = ŷ - y）
- [ ] 计算梯度（偏导数）：
  - ∂E/∂a = (2/n) × Σ(ŷ - y)x
  - ∂E/∂b = (2/n) × Σ(ŷ - y)
- [ ] 更新参数：aₙₑw = aₒₗd - η × gradient
- [ ] **复习：Week03 笔记 + Week04 开头**

#### 2️⃣ **Fuzzy Logic（模糊逻辑）**
- [ ] 三角隶属函数计算：
  - 给定 (a, b, c) 和输入 x，计算 μ(x)
- [ ] 去模糊化（Centroid method）：
  - Output = Σ(μᵢ × valueᵢ) / Σμᵢ
- [ ] **复习：Week01 笔记 + 你发的图片例题**

#### 3️⃣ **Classification Metrics（分类指标）**
- [ ] 从描述中构建混淆矩阵（TP/FP/FN/TN）
- [ ] 计算 Precision, Recall, Accuracy, F1
- [ ] **公式必须记住**：
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1 = 2PR / (P + R)
- [ ] 能解释不同场景下哪个指标最重要
- [ ] **新内容，需要补充学习**

#### 4️⃣ **Neural Network（神经网络）**
- [ ] 前向传播：
  - 计算隐藏层输入：z = Wx + b
  - 激活函数：σ(z) = 1/(1+e⁻ᶻ)
  - 输出层计算
- [ ] MSE 损失：L = (1/2)(ŷ - y)²
- [ ] 梯度计算（链式法则）
- [ ] **新内容，需要补充学习**

---

## 📚 从 Week01-Week04 累积的必会内容

### ✅ **Week01: AI 基础 + Fuzzy Logic**
- [ ] Turing Test 定义
- [ ] Expert Systems vs AI
- [ ] **Fuzzy Logic（✨Quiz 必考）**
  - 三角隶属函数计算
  - Centroid 去模糊化
- [ ] Neural Networks 基本结构

### ✅ **Week02: Supervised Learning**
- [ ] Regression vs Classification（看输出类型）
- [ ] Feature Encoding（One-Hot vs Dummy）
- [ ] Statistics vs Machine Learning
- [ ] **Classification Metrics 可能在这周讲过**

### ✅ **Week03: Linear Regression**
- [ ] **必背公式**：`θ = (XᵀX)⁻¹Xᵀy`（Normal Equation）
- [ ] **Gradient Descent（✨Quiz 必考）**
  - 迭代公式：θₙₑw = θₒₗd - η × ∇E
- [ ] 线性代数基础（Transpose, Inverse）
- [ ] **考试可能没有 Python，需要手算**

### ✅ **Week04: Overfitting & Regularization**
- [ ] Overfitting vs Underfitting
- [ ] L1 vs L2 Regularization
- [ ] Hyperparameters（Learning Rate, Degree, Lambda）
- [ ] Train-Test Split 的重要性
- [ ] **下周作业演示**（overfitting 识别）

---

## 🎯 Quiz 考试策略

### 📝 **时间分配**

**Quiz #1（40分钟，50分）**：
- Gradient Descent: 20分钟（25分）
- Fuzzy Logic: 20分钟（25分）

**Quiz #2（50分钟，100分）**：
- Classification Metrics: 15分钟（40分）
- Neural Network: 35分钟（60分）

### 🧠 **复习优先级**

**高优先级（必考）**：
1. ✅ **Gradient Descent 手算**（Week03 + Week04）
2. ✅ **Fuzzy Logic 计算**（Week01 + 你的图片）
3. ✅ **Normal Equation 公式**（Week03）

**中优先级（很可能考）**：
4. ✅ **Classification Metrics**（需补充）
5. ✅ **Neural Network 前向传播**（需补充）
6. ✅ **Overfitting 判断**（Week04 + 下周作业）

**低优先级（概念题）**：
7. Regression vs Classification
8. Feature Encoding
9. Regularization 原理

---

## ⚠️ 需要补充学习的内容

### 📌 **Classification Metrics（Quiz #2 第一题）**

**还没在 Week01-04 讲过**，需要自学或等待后续课程：
- 混淆矩阵（TP/FP/FN/TN）
- Precision（精确率）
- Recall（召回率）
- Accuracy（准确率）
- F1-Score

**学习资源**：
- 查看教材相关章节
- 搜索 "confusion matrix precision recall"
- 做几道练习题

### 📌 **Neural Network Backpropagation（Quiz #2 第二题）**

**Week01 只讲了基础，需要深入学习**：
- 前向传播（详细计算）
- Sigmoid 激活函数：σ(z) = 1/(1+e⁻ᶻ)
- MSE 损失函数
- 梯度计算（链式法则）

**学习资源**：
- 等待后续课程（可能 Week05-06）
- 复习 Week01 的 Neural Networks 部分
- 做手算练习

---

## 📅 重要日期

| 事项 | 日期 | 备注 |
|------|------|------|
| **下周作业演示** | Week05 | Overfitting 相关，准备 1-2 个问题 |
| **Midterm Exam** | February 24, 2026 | 混合格式（MCQ + 短答） |
| **Final Exam** | March 8, 2026 | 混合格式 |
| **Quizzes** | 不定期 | 参考往年：10月、12月 |

---

## 🎯 本周关键 Takeaway

1. **Quiz 题型已明确**：Gradient Descent + Fuzzy Logic（Quiz #1），Classification + Neural Network（Quiz #2）
2. **必须会手算**：梯度下降、模糊逻辑去模糊化
3. **需要补充**：Classification Metrics, Neural Network Backpropagation
4. **下周作业**：演示 overfitting 识别和调整
5. **考试可能没有 Python**：理解原理，能手算

---

## 📝 每日复习计划建议

**Day 1-2**: Gradient Descent 手算练习（Week03 内容）  
**Day 3**: Fuzzy Logic 计算练习（Week01 + 图片例题）  
**Day 4**: Classification Metrics 学习（补充内容）  
**Day 5**: Neural Network 基础复习（Week01）  
**Day 6**: 综合练习（做往年真题）  
**Day 7**: 查漏补缺 + Overfitting 作业准备
