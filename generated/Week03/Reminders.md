# Week03 Reminders（AI）

## ✅ 本周必会（考试导向）

1. **先判断是 Regression 还是 Classification**
   - 输出是连续数值（quantity）→ Regression
   - 输出是类别（category）→ Classification

2. **Linear Regression 两类解法要会说清楚**
   - **Closed-form / Least Squares（闭式解）**
   - **Gradient-based（梯度法 / Gradient Descent）**

3. **闭式解公式必须背下来（教授强调：要 memorize）**
   - **θ = (Xᵀ X)⁻¹ Xᵀ y**

4. **线性代数基础要复习（闭式解依赖）**
   - transpose（转置）
   - inverse（逆矩阵）
   - determinant（行列式）

---

## 🧠 复习清单（建议按顺序）

- [ ] 用一句话解释：什么是 supervised learning（训练数据→学模型→预测未见数据）
- [ ] 用“输出类型”区分 regression vs classification
- [ ] 会解释：X（independent/features）与 Y（dependent/target）
- [ ] 会写并解释 normal equation：θ = (XᵀX)⁻¹Xᵀy
- [ ] 复习矩阵求逆（2×2 / 3×3 基础）与行列式
- [ ] 知道：simple / multivariate / polynomial regression 的差别（本质上都是特征表示不同）

---

## 📝 课堂例子（可用于自测）

- 输入：年龄、性别、入院类型（emergency）、诊断类别…
- 输出：住院天数（length of stay，连续数值）
- 问题：这是 regression 还是 classification？为什么？

---

## ⚠️ 重要提醒

- 教授提到考试场景可能 **没有 Python**：需要你理解公式和计算过程。
- 闭式解要求 (XᵀX) 可逆；不满足时通常需要正则化/其它方法（后续可能会讲）。
