# Week04 Q&A Summary — Overfitting & Regularization

> 从课堂互动中提取的问答/讨论（按主题归类）。

---

## 1) 期中考试格式（重要！）

**时间**：Recording 2, 约 34:00 (2032s)

**学生问**：期中考试会怎么考？是选择题（MCQ）吗？

**教授回答**：
- "We're gonna talk about that." （之后再详细讨论）
- 没有明确说明具体格式

**补充信息（来自 Week01）**：
- 期中和期末：**混合格式**（选择题 + 短答题/解题）
- Quiz：**不是选择题**，15分钟，1-2题短答

**要点**：
- 考试可能**没有 Python**，需要理解概念和手算
- 需要会解释原理，不只是记代码

---

## 2) 下周要求：作业演示 + 随机提问

**时间**：Recording 2, 约 34:56 (2096s)

**学生问**：下周有什么要求？

**教授回答**：
> "So for next week, complete this assignment. And you don't need to solve it, but be able to complete and open on your machine. You can Google it, use any IDE you have. And then we're gonna ask you one of, maybe one, two questions. It's gonna be random. OK. You need to be ready for that."

**翻译**：
- ✅ **完成作业**（能在自己电脑上运行）
- ✅ **可以查 Google、用任何 IDE**（不是闭卷）
- ✅ **准备回答 1-2 个随机问题**（关于代码理解）

**核心考点**：
- 识别数据中的 **overfitting**
- 通过 **增加/减少参数数量** 来调整
- 能解释为什么会 overfitting/underfitting

**教授原话**：
> "And find out like the reason. Like check that you can see any clear overfitting there in the data. Increasing number of ones (parameters), or reduce the number of ones, probably."

---

## 3) Gradient Descent：为什么要减去梯度？

**隐含问题**（课堂推导）：为什么更新公式是 `X_next = X_current - α × gradient`？

**教授解释**（用登山比喻）：

**场景**：你在山顶，想下到平地（找最小值），但是雾很大看不清路。

**策略**：
1. 每次选择**最陡的下坡路**（steepest descent）
2. 如果坡度是负的（往下），你往那个方向走
3. 如果坡度是正的（往上），你往反方向走

**数学解释**：
- 在 X=10 处，斜率 = -2（负的，往下）
- 如果直接加 -2：`10 + (-2) = 8` → 往左走（错误！）
- 正确做法：减去负梯度：`10 - (-2) = 12` → 往右走（正确）

**公式逻辑**：
- Gradient 指向**增长最快的方向**
- 我们要找**最小值**，所以要往**负梯度方向**走
- 因此是 `-α × gradient`（减去梯度）

**Why Important**：这是 gradient descent 的核心原理，理解这个才能理解为什么所有优化算法都基于梯度。

---

## 4) 如何判断好的模型？

**隐含问题**（课堂讨论）：两个模型，哪个更好？
- 模型 A：Train error = 10, Test error = 200
- 模型 B：Train error = 30, Test error = 30

**学生可能的误解**：模型 A 的 train error 更低，应该更好？

**教授回答**：
> "I personally prefer a model which train and test are stable. For example, 10 and 200 for train and test is much worse than error of 30 for both training and test."

**重要原则**：
1. ❌ **不要只看 train error**
2. ✅ **看 train vs test error 的差距**
3. ✅ **稳定性很重要**：
   - Train=10, Test=10 > Train=2, Test=100
   - 即使 train error 高一点，只要 test error 接近，模型更可靠

**现实例子（教授举的）**：
- 预测建筑物价格：训练数据上准确 100%，但测试数据（新建筑）预测失败
- 预测下周天气：训练数据上完美，但实际预测（未来）失败

**Why Important**：这是评估模型的核心标准，面试和实际工作中常考。

---

## 5) Closed-form Solution 的局限性

**问题**（从Week03延续）：为什么不总是用 closed-form solution `θ = (XᵀX)⁻¹Xᵀy`？

**教授回答**（Week04 开头回顾）：
- **局限**：有些矩阵不可逆（not inversible）
- **解决**：用 Gradient Descent（迭代优化）

**对比**：
| 方法 | 优点 | 缺点 |
|------|------|------|
| Closed-form | 一步直接求解，精确 | 需要矩阵可逆，大数据时计算慢 |
| Gradient Descent | 通用，适用于任何可微函数 | 需要调学习率，迭代多次 |

**Why Important**：理解不同方法的适用场景，不是所有问题都有闭式解。

---

## 6) Regularization 的几何解释

**问题**（课堂讨论）：为什么 L1 和 L2 的约束区域不同？

**教授解释**（带图示）：

**L2 (Ridge)**：
- 约束区域是**圆形**（circle）
- 公式：`w₁² + w₂² ≤ C`
- 最优点：通常不在坐标轴上
- 结果：系数缩小但不为 0

**L1 (Lasso)**：
- 约束区域是**菱形**（diamond）
- 公式：`|w₁| + |w₂| ≤ C`
- 最优点：常在坐标轴上（某个 w=0）
- 结果：稀疏解（某些系数为 0）

**Why Important**：几何直觉帮助理解为什么 L1 能做特征选择，L2 不能。

---

## 7) Lambda 怎么选？

**问题**（隐含）：正则化参数 λ 怎么确定？

**教授建议**（课堂演示）：
1. **Grid Search**：尝试一系列值（0.01, 0.1, 1, 10, 100...）
2. **Cross-Validation**：对每个 λ，评估 test error
3. **选择 test error 最低的 λ**

**重要点**：
- λ = 0 → 没有正则化（可能过拟合）
- λ 很大 → 模型太简单（可能欠拟合）
- λ 是 **Hyperparameter**，不是从数据学习的

**Python 示例**（课堂提到）：
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_lambda = grid_search.best_params_['alpha']
```

**Why Important**：实际应用中必须知道如何调参，不能随便设 λ=1。

---

## 8) 为什么要分 Train/Test？

**问题**（隐含）：为什么不能用全部数据训练，然后评估？

**教授解释**（通过例子）：
- 如果只在训练数据上评估，你不知道模型能否泛化
- **Test set = 模拟未来的新数据**
- 例：预测下周温度，不能用"下周的数据"来训练模型

**重要原则**：
> "For those trained data, of course, it works perfect, right? But what matters is unseen data."

**实践建议**：
- 80% train, 20% test（常见比例）
- 用 `random_state` 保证可重现
- **Never train on test set!**（这是作弊）

**Why Important**：这是机器学习的基本准则，违反这个会导致过于乐观的评估。

---

## 课堂关键结论（教授反复强调）

1. **Test error > Train error** in importance
2. **Stability matters**：train ≈ test 比 train 很低但 test 很高更好
3. **Hyperparameters**（λ, degree, α）需要通过 cross-validation 调优
4. **Regularization** 是防止 overfitting 的有效方法
5. **下周要演示作业 + 回答 overfitting 相关问题**
