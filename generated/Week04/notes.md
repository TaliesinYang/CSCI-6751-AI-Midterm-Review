# Week04 — Overfitting, Underfitting, and Regularization

> 基于两段录音合并转录（共 2275 segments, 约 139 分钟）。本周围绕 **模型复杂度（Model Complexity）**，重点讲解 **Overfitting（过拟合）vs Underfitting（欠拟合）**、**Regularization（正则化）**，以及如何评估模型的泛化能力。

---

## 1) Gradient Descent 回顾

### 核心思想
- **目标**：找到函数的最小值点（optimal point）
- **比喻**：在雾天的山顶找下山路，每次选择**最陡的方向**（steepest descent）

### 算法步骤
1. 从随机点开始（X_current）
2. 计算当前点的斜率（gradient / derivative）
3. 向**负梯度方向**移动：`X_next = X_current - α × gradient`
   - α (alpha) = **Learning Rate（学习率）**
4. 重复步骤 2-3，直到收敛

### 重要概念
- **Learning Rate (α)**：
  - 太大 → 可能跳过最小值点
  - 太小 → 收敛太慢
  - 是一个 **Hyperparameter（超参数）**

---

## 2) Overfitting vs Underfitting（核心概念）

### 2.1 定义

**Underfitting（欠拟合）**：
- 模型**太简单**，无法捕捉数据的真实模式
- **训练误差高，测试误差也高**
- 例：用一条直线拟合明显的曲线数据

**Overfitting（过拟合）**：
- 模型**太复杂**，记住了训练数据的噪声
- **训练误差很低，但测试误差很高**
- 例：用 9 次多项式拟合 10 个点（完美拟合训练数据，但泛化能力差）

### 2.2 如何判断？

**关键指标：Train Error vs Test Error**

| 情况 | Train Error | Test Error | 判断 |
|------|-------------|------------|------|
| 欠拟合 | 高 | 高 | 模型太简单 |
| 良好 | 低 | 低（接近 train error） | 泛化能力好 |
| 过拟合 | 很低 | 高（远大于 train error） | 记住了噪声 |

**教授强调的例子**：
- 模型 A：Train error = 10, Test error = 200 → **过拟合**
- 模型 B：Train error = 30, Test error = 30 → **更好**（稳定）

**重要原则**：
> "We don't care only about training error. **Test error** is what matters!"  
> 训练误差低不代表模型好，**泛化能力（test performance）才是关键**。

---

## 3) 模型复杂度（Model Complexity）

### 3.1 Polynomial Degree（多项式阶数）

**实验观察**（课堂演示）：
- **Degree = 1**（直线）：欠拟合，误差大
- **Degree = 2-3**：拟合较好
- **Degree = 6-9**：过拟合，测试误差飙升

**结论**：
- Degree 越高 → 模型越复杂 → 越容易 overfitting
- **Degree 是一个 Hyperparameter**

### 3.2 如何选择合适的 Degree？

**方法 1：Cross-Validation**
- 尝试不同 degree（1, 2, 3, ..., 10）
- 比较 test error
- 选择 test error 最低的

**方法 2：Regularization**（后文详述）

---

## 4) Regularization（正则化）

### 4.1 核心思想

**问题**：高阶多项式会产生很大的系数（coefficients），导致模型对噪声敏感。

**解决**：在损失函数中加入**惩罚项（penalty term）**，限制系数大小。

### 4.2 正则化公式

**原始目标**（只最小化误差）：
```
minimize: MSE = Σ(y - ŷ)²
```

**加正则化后**：
```
minimize: MSE + λ × (惩罚项)
```

**两部分的作用**：
1. **第一项（MSE）**：最小化预测误差
2. **第二项（λ × penalty）**：防止模型过于复杂

---

## 5) L1 vs L2 Regularization

### 5.1 L2 Regularization (Ridge Regression)

**公式**：
```
minimize: Σ(y - ŷ)² + λ × Σ(w_i²)
```

**特点**：
- 惩罚系数的**平方和**
- 倾向于让所有系数都**变小**，但不会变成 0
- **几何解释**：在参数空间中，约束区域是一个**圆形**（circle）

**Python 实现**：
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)  # alpha = λ
model.fit(X_train, y_train)
```

### 5.2 L1 Regularization (Lasso Regression)

**公式**：
```
minimize: Σ(y - ŷ)² + λ × Σ|w_i|
```

**特点**：
- 惩罚系数的**绝对值和**
- 倾向于让某些系数**变成 0**（特征选择）
- **几何解释**：约束区域是一个**菱形**（diamond）

**区别总结**：
- **L2（Ridge）**：系数缩小但不为 0
- **L1（Lasso）**：系数可能为 0（稀疏解）

---

## 6) Lambda (λ) 超参数

### 6.1 Lambda 的作用

**Lambda 大**：
- 更重视惩罚项 → 模型更简单
- 可能导致 underfitting

**Lambda 小**：
- 更重视误差项 → 模型更复杂
- 可能导致 overfitting

**Lambda = 0**：
- 没有正则化 → 等同于普通线性回归

### 6.2 如何选择 Lambda？

**方法：Grid Search + Cross-Validation**
1. 尝试一系列 λ 值（0.01, 0.1, 1, 10, 100...）
2. 对每个 λ，用交叉验证评估 test error
3. 选择 test error 最低的 λ

**Lambda 也是 Hyperparameter**

---

## 7) Train-Test Split（数据划分）

### 7.1 Why Split?

**问题**：如果只用全部数据训练，无法评估泛化能力。

**解决**：把数据分成两部分：
- **Train Set (80%)**：用于训练模型
- **Test Set (20%)**：用于评估性能

### 7.2 Python 实现（课堂演示）

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 评估
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
```

**重要提示**：
- 用 `random_state` 保证可重现性（reproducibility）
- **Never train on test set!** Test set 只用于最终评估

---

## 8) 课堂实验：Polynomial Regression + Ridge

### 8.1 实验设置

**数据**：合成数据（synthetic data）
```python
import numpy as np
np.random.seed(42)  # 可重现性

X = np.random.rand(1000) * 2
Y = 4 + 3*X + np.random.randn(1000) * 0.5  # 加噪声
```

**目标**：比较不同 degree 和不同 λ 的效果

### 8.2 观察结果

**Degree = 2（低阶）**：
- Train MSE ≈ Test MSE（接近）
- 模型稳定

**Degree = 9（高阶，无正则化）**：
- Train MSE 很低
- Test MSE 很高 → **Overfitting!**

**Degree = 9 + Ridge (λ=0.1)**：
- Train MSE 略高
- Test MSE 降低 → **正则化有效**

---

## 9) 关键概念总结

### 9.1 Hyperparameters（超参数）

本周提到的超参数：
1. **Learning Rate (α)**：梯度下降的步长
2. **Polynomial Degree**：模型复杂度
3. **Lambda (λ)**：正则化强度

**特点**：
- 不是从数据中学习的，需要人为设定
- 通过 Cross-Validation 选择最优值

### 9.2 评估模型的准则

**错误的标准**：
- ❌ 只看 train error

**正确的标准**：
- ✅ 看 train error vs test error 的差距
- ✅ 更关注 test error（泛化能力）
- ✅ 稳定性很重要（train ≈ test 最好）

---

## 10) 实际应用场景（课堂例子）

### 例子 1：温度预测
- 输入：历史温度数据
- 输出：预测下周温度
- 评估：模型在训练数据上准确不够，必须在未来数据（test set）上准确

### 例子 2：建筑物价格预测
- 问题：训练误差 = 10，测试误差 = 200
- 原因：模型记住了训练数据的噪声，对新建筑预测失败
- 解决：降低模型复杂度或加正则化

---

## 11) 下周要求（重要！）

### 11.1 作业演示 + 随机提问

**时间**：34:56 (2096s) 提到

**要求**：
1. **完成作业代码**（可以在自己电脑上运行）
2. **可以用 Google、任何 IDE**（不是闭卷）
3. **准备回答 1-2 个随机问题**

### 11.2 核心考点

**Overfitting 相关**：
- 如何识别数据中的 overfitting？
- 如何调整参数？
  - **Increasing degree** → more overfitting
  - **Increasing lambda** → less overfitting

**教授原话**：
> "Find out the reason. Check that you can see any clear overfitting there in the data. Increasing number of ones (parameters), or reduce the number of ones (parameters), probably."

---

## 12) 期中考试讨论（学生提问）

### 学生问：期中会怎么考？是选择题吗？

**教授回答**（2077s）：
> "We're gonna talk about that."（之后再讨论）

**猜测考试形式**（基于 Week01）：
- 混合形式：**选择题 + 短答题/解题**
- 需要理解概念，能解释原理
- 可能需要手算一些简单的例子

---

## 小结（1段话）

本周深入讨论 **过拟合与欠拟合** 的本质：模型复杂度与泛化能力的权衡。核心方法是 **正则化（L1/L2）**，通过惩罚项约束模型复杂度。评估模型时，**不能只看训练误差，必须关注测试误差和稳定性**。下周要演示作业代码，准备回答关于 overfitting 的问题。期中考试形式待定（可能混合选择题和解题）。
