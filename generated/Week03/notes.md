# Week03 — Linear Regression（监督学习回归）

> 基于两段录音合并转录（共 2067 segments，约 124 分钟）。本周主要围绕 **监督学习（Supervised Learning）** 的回归任务，重点讲 **线性回归 Linear Regression**、两类解法（Closed-form / Gradient-based），以及考试需要记住的矩阵公式。

---

## 1) 特征维度与数据可视化（Feature Dimension）

### 特征选择 vs 特征降维

**Feature Selection（特征选择）**：
- 从原始特征中选取部分特征用于可视化或建模
- 数据本身**不变**（originality is not changed）
- 例：1000 维数据中选取 3 个最重要的特征来绘图

**Feature Reduction（特征降维）**：
- 将高维特征**转换**为新的低维特征
- 数据值会**改变**（values change）
- 例：PCA 将 100 维压缩到 20 维

**关键区别**：
- Selection 保留原始特征，Reduction 创建新特征
- 人类最多直观展示 3D（X, Y, Z），超过 3 维必须降维

---

## 2) Supervised Learning：Regression vs Classification

- **如果输出是连续数值（quantity）** → **Regression（回归）**
  - 例：预测住院天数（length of stay）、房价、冰激凌销量
- **如果输出是类别（category）** → **Classification（分类）**
  - 例：垃圾邮件/非垃圾邮件、是否违约、疾病类别

### 课堂例子 1：住院时长预测（回归）
- 输入特征（features）：年龄、性别、入院类型（emergency/elective）、诊断类别
- 目标（target）：住院天数（如 4.6 天、2.6 天 — 连续数值）

### 课堂例子 2：保险风险评分
- 评分 1-5（如果看作离散类别 → Classification）
- 分类 vs 回归取决于如何**定义输出**

> **教授强调**：关键是看输出是连续值还是离散类别，而不是看数据本身的特征类型。

---

## 3) Linear Regression：要做什么？

### 3.1 直观目标
给定一组训练点（observations / data points），找到一个函数去拟合它们：
- 输入 **X**（independent variables / features / predictors）
- 输出 **Y**（dependent variable / target）
- **Y depends on X**

### 3.2 线性函数形式

**一元线性（Simple Linear Regression）**：
```
ŷ = α + β·x
```
- **α（intercept/截距）**：直线与 Y 轴的交点
- **β（slope/斜率）**：直线的陡峭程度

**多元线性（Multivariate Linear Regression）**：
```
ŷ = α₀ + α₁·x₁ + α₂·x₂ + ... + αₙ·xₙ
```
- 2D → line；3D → plane；更高维 → hyperplane

**多项式回归（Polynomial Regression）**：
```
ŷ = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ
```
- 把 x, x², x³... 当作不同特征扩展
- degree 越高，曲线越复杂

> 提到 "degree（阶数）" 与 "hyperparameter（超参数）"，将在后续课程详细讲解。

### 3.3 课堂例子

**例子 1 — 冰激凌销售预测**：
- X = 温度，Y = 冰激凌销量
- 温度 12°C → 销量 200；温度 26°C → 销量 600
- 拟合直线后，预测 25°C 时销量约 500

**例子 2 — 二手车价格预测（多变量）**：
- 特征：age（年龄）、distance（行驶距离）、weight（重量）
- 目标：price（价格）
- `price = α₀ + α₁·age + α₂·distance + α₃·weight`
- 目标：找到最优 α₀, α₁, α₂, α₃ 使误差最小

---

## 4) 误差函数（Loss Function）

### 4.1 Sum of Squared Errors (SSE)

**公式**：
```
E = Σ(yᵢ - ŷᵢ)²
```
- `yᵢ`：ground truth（真实值）
- `ŷᵢ`：model prediction（模型预测值）

**为什么用平方？**
1. 消除负号（避免正负误差抵消）
2. 惩罚大误差更强

**也提到**：Sum of Absolute Errors (SAE)：`E = Σ|yᵢ - ŷᵢ|`，但 SSE 更常用

### 4.2 最小化误差的数学原理

**核心思路**：函数最小值处，导数（斜率）= 0

1. 对误差函数 E 分别对 α 和 β 求偏导
2. 令 `∂E/∂α = 0` 和 `∂E/∂β = 0`
3. 使用链式法则（Chain Rule）：`d/dx[f(g(x))] = f'(g(x))·g'(x)`
4. 代入 `ŷ = α + β·x` 后求导
5. 联立两个方程求解 α 和 β

---

## 5) Linear Regression 的两类求解方法（本周重点）

### A. Closed-form / Least Squares（闭式解 / 最小二乘）

#### 一元线性回归的 OLS 公式

**斜率（slope）**：
```
β = Cov(X, Y) / Var(X)
```

**截距（intercept）**：
```
α = Ȳ - β·X̄
```

其中：
- `Var(X) = (1/(n-1)) × Σ(xᵢ - X̄)²`（方差）
- `Cov(X, Y) = (1/(n-1)) × Σ(xᵢ - X̄)(yᵢ - Ȳ)`（协方差）

#### 数值计算示例（课堂演示）

**数据**：5 个点，x = [1, 2, 3, 4, 5]

**计算过程**：
1. X̄ = (1+2+3+4+5)/5 = **3**
2. Ȳ = 2.46（给定）
3. Var(X) = [(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²] / 4 = [4+1+0+1+4]/4 = **2.5**
4. β = Cov(X,Y) / Var(X) ≈ **0.5**
5. α = 2.46 - 0.5×3 ≈ **0.96**
6. 最终直线：斜率约 0.5，截距约 1

#### 矩阵形式（考试必背公式 — Normal Equation）

教授口头强调要 "memorize this formula"：

> **θ = (Xᵀ X)⁻¹ Xᵀ y**

- **适用范围**：simple linear、multivariate、polynomial，所有线性回归
- **一个公式解决所有问题**："If you know this one, all the linear regressions can be solved by this."

#### 矩阵运算基础（考试需要）

需要掌握的线性代数操作：
- **Transpose（转置）**：行变列
- **Inverse（逆矩阵）**：`A⁻¹` 使得 `A × A⁻¹ = I`
- **Determinant（行列式）**：用于求逆

**矩阵求逆步骤**（课堂演示 3×3 矩阵）：
1. 计算行列式（determinant）
2. 对每个元素，删除所在行和列，计算子矩阵行列式
3. 交替添加正/负号（cofactor expansion）
4. 除以原矩阵行列式

> 教授建议："Just provide this matrix to AI and ask what would be the inverse."（练习时可以用 AI 验证答案）

> **考试提示**：考试题规模通常是 1-2 degrees，数据量很小，考试时没有 Python，需要手算。

### B. Gradient-based solution（梯度法 / Gradient Descent）

- 思路：定义损失（error / objective），用梯度逐步更新参数
- 本周提到 "gradient solution / gradient descent" 作为另一条路径
- 更适合大数据、复杂模型
- **详细讲解将在 Week04 展开**

---

## 6) 方差与协方差的直觉理解

### 方差（Variance）
- **直觉**：数据点距均值的平均"离散程度"
- 方差高 → 数据分散（如班上同学成绩差异大）
- 方差低 → 数据集中（如所有人得分相同 → 方差为 0）
- `Var(X) = (1/(n-1)) × Σ(xᵢ - X̄)²`

### 协方差（Covariance）
- **直觉**：两个变量**共同变化**的程度
- Cov > 0 → X 增大时 Y 也增大（正相关）
- Cov < 0 → X 增大时 Y 减小（负相关）
- Cov ≈ 0 → X 和 Y 无线性关系
- `Cov(X,Y) = (1/(n-1)) × Σ(xᵢ - X̄)(yᵢ - Ȳ)`

---

## 7) 过拟合预告（Overfitting Preview）

### 场景
- 8 个散点数据，实际关系是线性的，但有 2 个噪声点（outliers）
- **Linear fit（degree 1）**：有误差，但合理
- **Polynomial degree 5**：通过所有点，误差 = 0，但...

### Overfitting 定义
- 模型对训练数据表现完美（100% 准确）
- 但对**新数据（unseen data）表现很差**
- 原因：模型学习了**噪声**（noise/outliers）

> "We ask our model to learn the noise. When you learn the noise, for the new points, the model will make a mistake."

### 教授的类比
> "We have some materials in the class. Some of the materials are not correct. If you learn those incorrect materials, you're going to make a mistake."

### 防止思路（preview）
- 选择"中等" degree，不要太高
- "Simple models is the best because that can perform good for training data and for test data is also okay."
- 详细方法（regularization 等）将在 Week04 讲解

> **面试高频题**：教授特别指出 overfitting 是面试中最常见的问题之一。需要能回答：① 什么是 overfitting；② 如何防止 overfitting。

---

## 8) 机器学习框架回顾

### ML 三大组件
每个机器学习算法都有：
1. **Representation（表示）**：数据如何表达（one-hot encoding、feature engineering）
2. **Optimization（优化）**：如何找最优参数（least squares、gradient descent）
3. **Evaluation（评估）**：如何评价模型（MSE、accuracy、precision、recall）

### ML Pipeline
1. 理解问题 → 2. 收集数据 → 3. 训练模型 → 4. 评估结果 → 5. 若不好则回到第 2 步

### Train-Test Split
- 典型划分：**80% training + 20% test**
- Training = fitting the line
- Prediction = using the trained model for new X
- **Never train on test set!**

### 数据编码提醒
- 决策树可以直接处理类别数据
- 神经网络必须使用 **one-hot encoding**

---

## 9) 神经网络结构简介

课堂提到神经网络作为另一种模型类型：

```
[Input Layer] → [Hidden Layer(s)] → [Output Layer]
```

- Input：特征（年龄、诊断等）
- Hidden：中间层（1层到多层均可）
- Output：预测值
- 模型本质是一个 **hypothesis（假设函数）**，将 X 映射到 Y
- 层数太多 → Deep Network

---

## 10) 你需要能讲清楚的概念对照

- **Instance / observation / data point**：一条样本
- **Feature / independent variable / predictor**：输入变量（X）
- **Target / label / dependent variable**：输出（Y）
- **Regression**：预测连续值
- **Classification**：预测离散类别
- **Intercept (α)**：截距，直线与 Y 轴交点
- **Slope (β)**：斜率，直线陡峭程度
- **SSE**：误差平方和
- **OLS**：Ordinary Least Squares（最小二乘法）
- **Normal Equation**：闭式解矩阵公式
- **Overfitting**：模型记住噪声，泛化能力差

---

## 11) 考试导向提醒（教授反复强调）

### 必须记住的公式
1. **Normal Equation（闭式解矩阵公式）**：`θ = (Xᵀ X)⁻¹ Xᵀ y`
2. **OLS 一元公式**：`β = Cov(X,Y)/Var(X)`，`α = Ȳ - β·X̄`
3. **SSE**：`E = Σ(yᵢ - ŷᵢ)²`

### 考试形式提示
- 考试时**没有 Python**，需要理解公式和手算
- 题目规模：1-2 个特征，少量数据点
- 可能给你数据，要求手动计算最优参数
- 线性代数基础要复习：transpose、inverse、determinant

### 面试重要性
- Overfitting 是面试中最常见的 AI/ML 问题
- 必须能回答：什么是过拟合 + 如何防止

---

## 12) 小结（1段话）

本周把监督学习的"回归"任务落地到 **线性回归**：给定训练数据（X, Y），学习一个函数用于预测连续目标。详细讲解了误差函数（SSE）和最小化原理，推导了一元 OLS 公式（β = Cov/Var, α = Ȳ - βX̄）并用数值例子演示计算过程。求解上强调两条路线：**最小二乘闭式解（Normal Equation `θ = (XᵀX)⁻¹Xᵀy`，需要矩阵逆）** 与 **梯度法（Gradient Descent，下周详解）**。还预告了 overfitting 问题及其重要性。考试重点是"会区分回归/分类 + 记住闭式解公式 + 具备基本矩阵求逆能力 + 理解过拟合概念"。
