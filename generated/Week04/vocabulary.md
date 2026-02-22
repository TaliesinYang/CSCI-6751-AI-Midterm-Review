# Week04 Vocabulary — Overfitting & Regularization

> Key terms and definitions from Week04 lectures.

---

## Model Evaluation & Generalization

**Overfitting（过拟合）**  
Model is too complex and memorizes training data including noise, resulting in low train error but high test error.  
模型过于复杂，记住了训练数据的噪声，导致训练误差低但测试误差高。

**Underfitting（欠拟合）**  
Model is too simple to capture the underlying pattern in data, resulting in high error on both train and test sets.  
模型过于简单，无法捕捉数据的真实模式，导致训练和测试误差都高。

**Generalization（泛化能力）**  
Model's ability to perform well on unseen data (test set).  
模型在未见过的数据（测试集）上的表现能力。

**Train-Test Split（训练-测试划分）**  
Dividing dataset into training set (e.g., 80%) for learning and test set (e.g., 20%) for evaluation.  
将数据集分为训练集（如 80%）用于学习，测试集（如 20%）用于评估。

**Train Error**  
Model's error/loss on training data.  
模型在训练数据上的误差。

**Test Error**  
Model's error/loss on test data (unseen during training). More important than train error for evaluating generalization.  
模型在测试数据上的误差（训练时未见过）。比训练误差更重要，用于评估泛化能力。

---

## Regularization（正则化）

**Regularization（正则化）**  
Technique to prevent overfitting by adding a penalty term to the loss function that constrains model complexity.  
通过在损失函数中添加惩罚项来约束模型复杂度，从而防止过拟合的技术。

**L1 Regularization / Lasso Regression**  
Regularization using sum of absolute values of coefficients: `λ × Σ|w_i|`.  
Tends to produce sparse solutions (some coefficients become exactly 0).  
使用系数绝对值和的正则化：`λ × Σ|w_i|`。倾向于产生稀疏解（某些系数变为 0）。

**L2 Regularization / Ridge Regression**  
Regularization using sum of squared coefficients: `λ × Σ(w_i²)`.  
Shrinks all coefficients but doesn't make them exactly 0.  
使用系数平方和的正则化：`λ × Σ(w_i²)`。缩小所有系数但不会让它们变为 0。

**Lambda (λ) / Regularization Parameter**  
Hyperparameter controlling the strength of regularization.  
- Large λ → strong regularization → simpler model → risk of underfitting  
- Small λ → weak regularization → complex model → risk of overfitting  
控制正则化强度的超参数。

**Penalty Term（惩罚项）**  
Additional term in loss function that penalizes large coefficients.  
损失函数中惩罚大系数的额外项。

---

## Model Complexity

**Model Complexity（模型复杂度）**  
Measure of how flexible/expressive a model is. Higher complexity → more risk of overfitting.  
模型灵活性/表达能力的度量。复杂度越高 → 过拟合风险越大。

**Polynomial Degree（多项式阶数）**  
In polynomial regression, the highest power of input variable. Higher degree → more complex model.  
Example: degree=2 is `y = w₀ + w₁x + w₂x²`  
多项式回归中输入变量的最高次幂。阶数越高 → 模型越复杂。

**Coefficient / Weight（系数/权重）**  
Parameters in regression model (θ, w, β). Large coefficients often indicate overfitting.  
回归模型中的参数。大系数通常表明过拟合。

---

## Hyperparameters（超参数）

**Hyperparameter（超参数）**  
Parameters not learned from data but set before training. Examples: learning rate, lambda, polynomial degree.  
Must be tuned using cross-validation.  
不是从数据中学习的参数，而是在训练前设定的。例如：学习率、lambda、多项式阶数。

**Learning Rate (α / alpha)（学习率）**  
Step size in gradient descent. Controls how much to update parameters in each iteration.  
- Too large → may overshoot minimum  
- Too small → slow convergence  
梯度下降中的步长。控制每次迭代更新参数的幅度。

**Grid Search**  
Method to find optimal hyperparameters by trying multiple values and comparing performance.  
通过尝试多个值并比较性能来找到最优超参数的方法。

---

## Gradient Descent

**Gradient Descent（梯度下降）**  
Iterative optimization algorithm to find minimum of a function by moving in the direction of steepest descent (negative gradient).  
通过向最陡下降方向（负梯度）移动来找到函数最小值的迭代优化算法。

**Gradient / Derivative（梯度/导数）**  
Slope of function at a point. Indicates direction and rate of steepest increase.  
函数在某点的斜率。表示最陡增长的方向和速率。

**Convergence（收敛）**  
When gradient descent reaches a point where updates become negligible (close to minimum).  
当梯度下降到达一个点，更新变得微不足道（接近最小值）。

**Steepest Descent（最陡下降）**  
Direction of maximum decrease in function value (opposite of gradient direction).  
函数值最大减少的方向（梯度的反方向）。

---

## Python / Sklearn Terms

**train_test_split**  
Sklearn function to randomly split data into train and test sets.  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Ridge**  
Sklearn implementation of Ridge Regression (L2 regularization).  
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)  # alpha = λ
```

**Lasso**  
Sklearn implementation of Lasso Regression (L1 regularization).  
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

**MSE (Mean Squared Error)（均方误差）**  
Common metric for regression: average of squared differences between predictions and actual values.  
```
MSE = (1/n) × Σ(y - ŷ)²
```

**random_state**  
Seed for random number generator to ensure reproducibility of train-test split.  
随机数生成器的种子，确保训练-测试划分的可重现性。

---

## Evaluation Metrics

**Error / Loss（误差/损失）**  
Difference between predicted and actual values. Used to measure model performance.  
预测值与实际值之间的差异。用于衡量模型性能。

**Train MSE**  
Mean Squared Error on training set.  
训练集上的均方误差。

**Test MSE**  
Mean Squared Error on test set. **More important** for evaluating generalization.  
测试集上的均方误差。对于评估泛化能力**更重要**。

**Prediction (ŷ / y_pred)（预测值）**  
Model's output for a given input.  
模型对给定输入的输出。

---

## Concepts from Previous Weeks (Referenced)

**Closed-form Solution / Normal Equation（闭式解）**  
Direct calculation of optimal parameters: `θ = (XᵀX)⁻¹Xᵀy`  
Limitation: Requires matrix inversion (not always possible).  
直接计算最优参数的方法。局限：需要矩阵求逆（不总是可行）。

**Least Squares Method（最小二乘法）**  
Method to find best-fit line by minimizing sum of squared residuals.  
通过最小化残差平方和来找到最佳拟合线的方法。

**Feature（特征）**  
Input variable (X) used to make predictions.  
用于预测的输入变量。

**Target / Label（目标/标签）**  
Output variable (Y) we want to predict.  
我们想要预测的输出变量。

---

## Key Takeaways

1. **Overfitting = memorizing noise** → Use regularization or reduce model complexity
2. **Test error > Train error** → Generalization is key, not just training performance
3. **Hyperparameters** (λ, degree, α) → Tune with cross-validation
4. **Ridge (L2)** vs **Lasso (L1)** → Different ways to constrain coefficients
5. **Always split data** → Never evaluate on training set
