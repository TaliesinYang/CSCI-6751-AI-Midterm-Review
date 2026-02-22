# Week 03 Linear Regression

[[Week03/notes|📖 完整笔记]]

## 核心概念

### Regression vs Classification

- Regression: 连续数值输出（住院天数、价格）
- Classification: 离散类别输出（垃圾邮件检测）
- [[Week03/notes#1) Supervised Learning：Regression vs Classification|📝 详细说明]]

### Linear Regression 任务

- 目标: 学习函数预测连续值
- 数据: 输入 X (features), 输出 Y (target)
- 最小化预测误差
- [[Week03/notes#2) Linear Regression：要做什么？|📝 详细说明]]

## 求解方法

[[Week03/notes#3) Linear Regression 的两类求解方法（本周重点）|📝 两种方法对比]]

### Normal Equation 闭式解

- 一步到位计算参数
- 需要矩阵可逆
- 必背公式: θ = (XᵀX)⁻¹Xᵀy
- 需要掌握: 矩阵转置、求逆、行列式
- [[Week03/notes#A. Closed-form / Least Squares（闭式解 / 最小二乘）|📝 详细内容]]
- [[Week03/notes#考试必背公式（Normal Equation）|🎯 公式详解]]

### Gradient Descent 梯度下降

- 迭代优化，逐步逼近最优解
- 不需要求逆
- Learning Rate α 控制步长
- 三种类型: Batch GD, SGD, Mini-batch
- [[Week03/notes#B. Gradient-based solution（梯度法 / Gradient Descent）|📝 详细内容]]
- [[Week03/gradient-descent-handout.pdf|📄 PDF讲义]]

## 必背公式

[[Week03/notes#考试必背公式（Normal Equation）|🎯 笔记中的公式]]

### Normal Equation

- θ = (XᵀX)⁻¹Xᵀy

### Gradient Descent 更新规则

- θ := θ - α∇J(θ)

### Linear Regression 梯度

- ∇J(θ) = (1/m)Xᵀ(Xθ - y)

### MSE 损失函数

- J(θ) = (1/2m)Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²

## 考试重点

[[Week03/notes#5) 本周"考试导向"提醒（教授反复提到）|🎯 完整要求]]

### 必须会的

- 区分 Regression vs Classification（看输出类型）
- 记住 Normal Equation 公式
- 理解两种求解方法差异
- 掌握基本矩阵运算（转置、求逆、行列式）
- 理解梯度下降原理

### 教授强调

- 考试可能没有 Python
- 需要会手算
- 矩阵运算很重要

## 核心术语

[[Week03/vocabulary|📖 完整术语表]] | [[Week03/notes#4) 你需要能讲清楚的概念对照|📝 概念对照]]

### 数据相关

- Instance / Observation: 样本
- Feature: 输入变量 (X)
- Target / Label: 输出 (Y)

### 模型相关

- Hypothesis: 假设函数
- Cost Function: 损失函数
- Gradient: 梯度
- Convergence: 收敛

## 课程资源

### 笔记文件

- [[Week03/notes|完整笔记]]
- [[Week03/vocabulary|术语表]]
- [[Week03/qa-summary|QA总结]]
- [[Week03/Reminders|重点提醒]]

### 讲义文件

- [[Week03/gradient-descent-handout.pdf|Gradient Descent PDF讲义]]

## 总结

[[Week03/notes#6) 小结（1段话）|📝 一句话总结]]

- 监督学习的回归任务落地到线性回归
- 两条求解路线: Normal Equation (闭式解) vs Gradient Descent (梯度法)
- 考试重点: 区分回归/分类 + 记住公式 + 矩阵运算
