# Week03 Vocabulary — Linear Regression

## Core Terms

- **Supervised Learning（监督学习）**：给定带标签数据（X, Y）学习映射函数，用于预测未见样本。
- **Regression（回归）**：输出是连续数值（quantity），如住院天数、价格。
- **Classification（分类）**：输出是离散类别（category），如“是/否”、类别标签。

## Data & Variables

- **Observation / Instance / Data point（样本/观测）**：一条数据记录。
- **Feature（特征）**：输入变量（X）。
- **Target / Label（目标/标签）**：输出变量（Y）。
- **Independent variable（自变量）**：X；不依赖其它变量（在模型语境下作为输入）。
- **Dependent variable（因变量）**：Y；依赖 X 的输出。

## Linear Regression

- **Linear Regression（线性回归）**：假设输出 Y 与输入 X 存在线性关系；通过拟合找到参数。
- **Simple Linear Regression（一元线性回归）**：一个主要特征（单变量）场景。
- **Multivariate Linear Regression（多元线性回归）**：多个特征输入（X1, X2, …）。
- **Polynomial Regression（多项式回归）**：使用 x、x²、x³…等特征扩展进行拟合（仍可写成线性参数形式）。
- **Degree（阶数）**：多项式的最高次数；常被视为模型复杂度/超参数的一部分。

## Optimization / Solutions

- **Least Squares（最小二乘）**：通过最小化残差平方和得到参数。
- **Closed-form Solution（闭式解）**：可直接用公式算出解（不需要迭代）。
- **Gradient-based Solution（基于梯度的解法）**：通过梯度下降等迭代优化参数。
- **Gradient Descent（梯度下降）**：沿负梯度方向迭代更新参数以降低损失。

## Matrix / Linear Algebra

- **Transpose（转置，Xᵀ）**：矩阵行列互换。
- **Inverse（逆矩阵，A⁻¹）**：满足 A·A⁻¹ = I 的矩阵；计算闭式解需要。
- **Determinant（行列式）**：用于判断矩阵可逆性等；求逆常用到。
- **Normal Equation（正规方程）**：线性回归闭式解公式：
  - **θ = (Xᵀ X)⁻¹ Xᵀ y**

## Model / Training

- **Training data（训练数据）**：用来学习模型参数的数据。
- **Unseen data（未见数据）**：训练时未出现的新输入，用于预测。
- **Hyperparameter（超参数）**：训练前设定的参数（如多项式阶数、学习率等）。

## Classroom Example Terms

- **Length of stay（住院时长）**：回归目标示例。
- **Admission type: emergency（急诊入院类型）**：输入特征示例。
- **Diagnosis category（诊断类别）**：输入特征示例。
