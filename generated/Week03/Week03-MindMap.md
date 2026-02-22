# Week 03 Linear Regression

## 核心概念

### Regression vs Classification
#### Regression
- 连续数值输出
- 例：住院天数、价格
#### Classification
- 离散类别输出
- 例：垃圾邮件检测

### Linear Regression 任务
#### 目标
- 学习函数 ŷ = Xθ
- 最小化预测误差
#### 数据
- 输入 X (features)
- 输出 Y (target)

## 求解方法

### Normal Equation
#### 闭式解
- 一步到位
- 需要矩阵可逆
#### 必背公式
- θ = (XᵀX)⁻¹Xᵀy
#### 需要掌握
- 矩阵转置
- 矩阵求逆
- 行列式

### Gradient Descent
#### 梯度下降
- 迭代优化
- 不需要求逆
#### 更新规则
- θ := θ - α∇J(θ)
#### 梯度公式
- ∇J(θ) = (1/m)Xᵀ(Xθ - y)
#### 组件
- α: Learning Rate
- J(θ): Cost Function
#### 类型
- Batch GD: 全部数据
- SGD: 单个样本
- Mini-batch: 小批量

## 必背公式

### Normal Equation
- θ = (XᵀX)⁻¹Xᵀy

### Gradient Descent
- θ := θ - α∇J(θ)

### Gradient for Linear Regression
- ∇J(θ) = (1/m)Xᵀ(Xθ - y)

### MSE Loss
- J(θ) = (1/2m)Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²

## 考试重点

### 必须会
- 区分 Regression vs Classification
- 记住 Normal Equation 公式
- 理解两种求解方法差异
- 掌握基本矩阵运算
- 理解梯度下降原理

### 教授强调
- 考试可能没有 Python
- 需要会手算
- 矩阵运算很重要

## 核心术语

### 数据相关
- Instance / Observation: 样本
- Feature: 输入变量
- Target / Label: 输出

### 模型相关
- Hypothesis: 假设函数
- Cost Function: 损失函数
- Gradient: 梯度
- Convergence: 收敛

## 课程资源

### 笔记
- [[Week03/notes|完整笔记]]
- [[Week03/vocabulary|术语表]]
- [[Week03/qa-summary|QA总结]]

### 讲义
- [[Week03/gradient-descent-handout.pdf|PDF讲义]]
- [[Week03/Reminders|重点提醒]]
