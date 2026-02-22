# CSCI 6751 AI Midterm — 复习计划 + AI 提示词

> **考试日期**: 2026-02-24 (周二)
> **格式**: 计算题 50分(50min) + MCQ 50分(15min) = 100分, 65分钟
> **覆盖范围**: Week01-Week05

---

## 一、复习计划（按优先级刷真题）

### Step 1: Gradient Descent 手算 (25分 — 每年必考)

**看什么**:
- `materials/references/Past-Exam-Questions.pdf` — Quiz#1 Q1 + Midterm Q1 真题
- `materials/references/Exam-Formula-Cheatsheet.pdf` — GD 公式 + 6步手算流程
- `generated/Week03/notes.md` — 线性回归完整笔记
- `generated/Week04/notes.md` — GD 回顾 + L2 正则化梯度

**做什么**:
1. 看一遍公式速查表的 GD 部分
2. 做 Quiz#1 真题 (简单 GD, y=ax+b, 2数据点)
3. 做 Midterm 真题 (多变量 GD + L2, y=θ₀+θ₁x₁+θ₂x₂)
4. 换数字再做一遍（用 AI 生成新题）

---

### Step 2: Fuzzy Logic 手算 (25分 — 每年必考)

**看什么**:
- `materials/references/Past-Exam-Questions.pdf` — Quiz#1 Q2 + Midterm Q2 真题
- `materials/references/Exam-Formula-Cheatsheet.pdf` — Fuzzy Logic 公式 (trimf + trapmf)
- `materials/references/Fuzzy-Logic-Tutorial.pdf` — 模糊逻辑详细教程
- `generated/Week01/notes.md` — Fuzzy Logic 基础

**做什么**:
1. 看公式速查表的 Fuzzy 部分（特别是 trapmf 五个区间判断）
2. 做 Quiz#1 真题 (三角形 triangular, 单输入)
3. 做 Midterm 真题 (梯形 trapezoidal, 双输入, 3条规则)
4. 换数字再做一遍

---

### Step 3: MCQ 刷题 (50分 — 10题选择)

**看什么**:
- `materials/references/Past-Exam-Questions.pdf` — Midterm MCQ 真题
- `materials/references/past-exams/Midterm-MCQ_CSCI 6751_Fall 2025_G1 - Solutions.pdf`
- `materials/references/past-exams/Midterm-MCQ_CSCI 6751_Fall 2025_G2 - Solutions.pdf`

**做什么**:
1. 做完所有 MCQ 真题（G1 + G2 两套）
2. 每题搞懂为什么，不只是记答案
3. 用 AI 生成同类型新题练习

---

### Step 4: 对照思维导图扫盲

**看什么**:
- `generated/CSCI-6751-Midterm-Review.canvas` — 考试专用思维导图（含考点映射）
- 每周 `generated/WeekXX/Reminders.md` — 各周考试重点清单
- 每周 `generated/WeekXX/vocabulary.md` — 术语对照

**做什么**:
1. 打开 Canvas，逐个节点过，心虚的标红
2. 对心虚的内容回去看对应 notes.md
3. 重点看 "Common Mistakes" 节点

---

### Step 5: 考前最后过一遍

**看什么**:
- `materials/references/Exam-Formula-Cheatsheet.pdf` — 公式速查表
- `materials/references/CSCI-6751-Formula-Cheatsheet.pdf` — 另一份公式表
- `generated/CSCI-6751-Midterm-Review.canvas` — 思维导图快速浏览

**做什么**:
1. 手写一遍所有公式
2. 看一遍常见错误清单
3. 早点睡

---

## 二、文件索引

### 真题（最重要）

| 文件 | 内容 |
|------|------|
| `materials/references/Past-Exam-Questions.pdf` | 往年真题汇总 + 解答 |
| `materials/references/past-exams/Midterm-CSCI 6751_Fall 2025_G1 - Solutions.pdf` | Midterm 计算题 G1 + 解答 |
| `materials/references/past-exams/Midterm-CSCI 6751_Fall 2025_G2 - Solutions.pdf` | Midterm 计算题 G2 + 解答 |
| `materials/references/past-exams/Midterm-MCQ_CSCI 6751_Fall 2025_G1 - Solutions.pdf` | Midterm MCQ G1 + 解答 |
| `materials/references/past-exams/Midterm-MCQ_CSCI 6751_Fall 2025_G2 - Solutions.pdf` | Midterm MCQ G2 + 解答 |
| `materials/references/past-exams/Quiz-CSCI 6751_Fall 2025_G1-Solutions.pdf` | Quiz#1 + 解答 |
| `materials/references/past-exams/Quiz-CSCI 6751_Fall 2025_G2 - Solutions.pdf` | Quiz#1 G2 + 解答 |
| `materials/references/past-exams/Quiz#2-CSCI 6751_Fall 2025_G1 - Solutions.pdf` | Quiz#2 + 解答 |

### 公式表

| 文件 | 内容 |
|------|------|
| `materials/references/Exam-Formula-Cheatsheet.pdf` | 考试公式速查 (GD 6步 + Normal Eq + Fuzzy + Metrics + 常见错误) |
| `materials/references/CSCI-6751-Formula-Cheatsheet.pdf` | 课程公式参考 |
| `materials/references/Formula-Reference-Clean.pdf` | 干净版公式参考 |

### 教程与参考

| 文件 | 内容 |
|------|------|
| `materials/references/Gradient-Descent-Tutorial.pdf` | 梯度下降详解 |
| `materials/references/Fuzzy-Logic-Tutorial.pdf` | 模糊逻辑教程 |
| `materials/references/Linear-Algebra-Basics.pdf` | 线性代数基础 (矩阵/转置/逆) |
| `materials/references/Matrix-Inverse-Practice.pdf` | 矩阵求逆练习 |

### 课堂笔记（AI生成）

| 文件 | 内容 |
|------|------|
| `generated/Week01/notes.md` | AI基础 + Fuzzy Logic + Expert Systems + Neural Networks |
| `generated/Week02/notes.md` | ML基础 + Supervised/Unsupervised/RL + Encoding |
| `generated/Week03/notes.md` | 线性回归 + Normal Equation + Gradient Descent |
| `generated/Week04/notes.md` | Overfitting/Underfitting + L1/L2 Regularization |
| `generated/Week05/notes.md` | MAE/MSE/RMSE/R² + Confusion Matrix + Precision/Recall/F1 |

### 思维导图

| 文件 | 内容 |
|------|------|
| `generated/CSCI-6751-Midterm-Review.canvas` | **考试复习专用** (57节点, 含考点映射 + 常见错误) |
| `generated/CSCI-6751-Knowledge-Map.canvas` | 课程知识总览 (Week01-04) |

---

## 三、AI 提示词（最重要 — 直接复制给 AI）

### Prompt 0: 初始化（先喂这个）

```
我正在准备 CSCI 6751 Artificial Intelligence 的期中考试。

考试格式：
- 计算题 2道 (各25分, 共50分钟)
- MCQ 10道 (各5分, 共15分钟)

考试内容覆盖 5 周：
1. Week01: AI基础 + Fuzzy Logic (三角/梯形隶属函数, 去模糊化)
2. Week02: ML基础 (Supervised/Unsupervised/RL, Regression vs Classification)
3. Week03: 线性回归 (Normal Equation, Gradient Descent)
4. Week04: Overfitting/Underfitting + L1/L2 Regularization
5. Week05: 评估指标 (MAE/MSE/RMSE/R², Precision/Recall/F1)

往年考试固定题型：
- Q1 (25分): Gradient Descent 多变量 + L2 正则化, 手算一次迭代
- Q2 (25分): Fuzzy Logic 梯形隶属函数, 多输入变量, 计算 firing strength + centroid 去模糊化
- MCQ: GD vs Normal Equation, L1 vs L2, 矩阵维度, Overfitting 判断, 评估指标

请用苏格拉底式提问法帮我复习。规则：
1. 一次只问一个问题
2. 不要直接给答案，通过提示引导我思考
3. 如果我答错了，告诉我哪里错了，再给我一次机会
4. 如果我答对了，追问一个更深的相关问题
5. 每答完一组问题，给我打分并指出薄弱点
```

---

### Prompt 1: Gradient Descent 苏格拉底复习

```
现在复习 Gradient Descent。请基于以下真题格式考我：

真题格式：给定多变量线性回归 y = θ₀ + θ₁x₁ + θ₂x₂，3个数据点，初始参数全为0，learning rate η，带 L2 正则化 (lambda)。要求手算一次迭代。

需要我完成的步骤：
1. 计算每个数据点的预测值 ŷ
2. 计算误差 e = ŷ - y
3. 计算 MSE
4. 计算每个参数的梯度 (θ₀不加正则化, θ₁和θ₂要加 +2λθⱼ)
5. 更新参数 θ_new = θ_old - η × gradient
6. 计算 L2 regularized cost

请给我出一道新题（不要用真题原题的数字），一步步引导我做。每一步让我先算，你再检查。

关键公式供参考：
- ∂J/∂θ₀ = (2/n)Σ(ŷᵢ - yᵢ)
- ∂J/∂θⱼ = (2/n)Σ(ŷᵢ - yᵢ)·xⱼᵢ + 2λθⱼ
- θ_new = θ_old - η · gradient
- J_ridge = (1/n)Σ(ŷᵢ - yᵢ)² + λΣθⱼ²
```

---

### Prompt 2: Fuzzy Logic 苏格拉底复习

```
现在复习 Fuzzy Logic。请基于以下真题格式考我：

真题格式：给定一个模糊控制系统，2个输入变量（各有3个模糊集，用梯形 trapmf(a,b,c,d) 定义），1个输出变量（也有3个模糊集），3条 IF-THEN 规则。给定具体输入值，要求：
(a) 计算每个输入在每个模糊集的隶属度
(b) 计算每条规则的 firing strength (AND = MIN)
(c) 用 centroid 方法去模糊化

请给我出一道新题，一步步引导我做。

关键公式供参考：
- trapmf(a,b,c,d): x≤a→0, a<x<b→(x-a)/(b-a), b≤x≤c→1, c<x<d→(d-x)/(d-c), x≥d→0
- trimf(a,b,c): x≤a→0, a<x≤b→(x-a)/(b-a), b<x<c→(c-x)/(c-b), x≥c→0
- AND: Firing Strength = min(μ₁, μ₂)
- Centroid: Output = Σ(FSᵢ × Outputᵢ) / Σ(FSᵢ)

常见陷阱（帮我注意）：
- 判断 x 落在哪个区间容易搞错（上升/平顶/下降/外部）
- AND 应该用 MIN 不是 MAX
- 梯形平顶区间 [b,c] 隶属度 = 1
- Centroid 分子分母别算反
```

---

### Prompt 3: MCQ 模拟测试

```
请给我出 10 道 MCQ 选择题，模拟真实考试。每题 4 个选项，覆盖以下主题：

1. GD vs Normal Equation（什么时候用哪个？特征>1000时？）
2. L1 vs L2 Regularization（哪个做特征选择？哪个系数变0？）
3. Normal Equation 矩阵维度（X 是 m×(n+1), θ 是？）
4. Overfitting vs Underfitting（Train Error 和 Test Error 的关系？）
5. Bias vs Variance（High bias = ? High variance = ?）
6. Lambda 的影响（λ太大→? λ太小→?）
7. Precision vs Recall（什么场景优先哪个？）
8. MAE vs MSE vs RMSE（单位区别？对异常值的敏感度？）
9. R² 解释（R²=0.85 意味着什么？R²可以为负吗？）
10. Cross-Validation（K-fold 是什么？validation set 和 test set 区别？）

规则：
- 先出所有 10 题，让我做完
- 我交卷后再公布答案
- 每题给出详细解释（为什么对，为什么其他选项错）
- 最后给总分和薄弱点分析
```

---

### Prompt 4: Normal Equation 手算练习

```
请给我出一道 Normal Equation 手算题：

题目要求：
- 给定 2-3 个数据点 (x, y)
- 要求用 Normal Equation θ = (XᵀX)⁻¹Xᵀy 求解参数
- 需要我手算：构建 X 矩阵(加截距列) → XᵀX → 求逆 → Xᵀy → 最终 θ

一步步引导我做，每步让我先算你再检查。

关键公式：
- 2×2 矩阵求逆: A⁻¹ = (1/det)[d,-b;-c,a], det = ad-bc
- θ 的维度 = (n+1) × 1 (n是特征数)
```

---

### Prompt 5: Classification Metrics 练习

```
请给我出一道分类指标计算题：

题目格式（基于往年 Quiz#2）：
- 给定一个二分类场景（如疾病检测/垃圾邮件）
- 给定预测结果的描述（不直接给 TP/FP/FN/TN，需要我自己提取）
- 要求：
  (a) 构建混淆矩阵
  (b) 计算 Precision, Recall, Accuracy, F1
  (c) 回答：在这个场景中哪个指标最重要？为什么？

一步步引导我做。

关键公式：
- Precision = TP / (TP + FP) — 预测为正的有多少真正
- Recall = TP / (TP + FN) — 真正为正的找到了多少
- F1 = 2 × P × R / (P + R)
- Accuracy = (TP + TN) / Total

记忆技巧：
- Precision 看 Predicted Positives (分母含 FP)
- Recall 看 Real Positives (分母含 FN)
```

---

### Prompt 6: 薄弱点诊断 + 针对性出题

```
我刚做完了 CSCI 6751 的复习，以下是我的自测结果：

[在这里填写你的情况，例如：]
- Gradient Descent: 能做对简单的，多变量+L2还不太熟
- Fuzzy Logic: trapmf 的区间判断有时搞混
- MCQ: L1 vs L2 的区别不太清楚
- Normal Equation: 矩阵求逆容易算错
- Metrics: Precision 和 Recall 的公式记混

请根据我的薄弱点：
1. 用最简洁的方式帮我理清这些概念
2. 针对每个薄弱点出 2 道练习题
3. 用苏格拉底式提问引导我做
```

---

### Prompt 7: 考前最后冲刺

```
CSCI 6751 AI 期中考试明天就考了。请帮我做考前最后冲刺：

1. 列出所有必须记住的公式（不要解释，直接列）
2. 列出 5 个最容易犯的错误
3. 给我出一道 "闪电轮" 快速问答（20个概念题，每题用一句话回答）
4. 最后给我考试策略建议（时间分配、做题顺序）

格式要求：简洁、直接、不废话。
```

---

## 四、常见错误速查（考前必看）

### Gradient Descent
- e = **ŷ - y** (不是 y - ŷ)
- 更新用 **减号**: θ_new = θ_old **-** η·gradient
- 别忘除以 **n**: (2/**n**)Σ...
- 梯度对 θⱼ 要乘 **xⱼᵢ**: (2/n)Σ(eᵢ · **xⱼᵢ**)
- L2 正则化梯度加 **+2λθⱼ** (但 **θ₀ 不加**)
- Learning Rate **η** 别漏

### Normal Equation
- X 第一列全是 **1** (截距项)
- 行列式: det = ad **-** bc (不是 ad+bc)
- 矩阵求逆: 对角**交换**、副对角**取负**、除以行列式
- 矩阵相乘前检查**维度匹配**

### Fuzzy Logic
- 先判断 x 在哪个**区间** (上升/平顶/下降/外部)
- AND 用 **MIN** (不是 MAX)
- 梯形平顶区间 [b,c] 的 μ = **1**
- Centroid 公式: **分子** = Σ(FS × Output), **分母** = Σ(FS)

### Classification Metrics
- Precision = TP/(TP+**FP**) — **P**recision 看 **P**redicted
- Recall = TP/(TP+**FN**) — **R**ecall 看 **R**eal
- F1 用**调和平均** (不是算术平均)
- Type I Error = **FP**, Type II Error = **FN**

### Overfitting / Underfitting
- Underfitting: Train **高** + Test **高** = High **Bias**
- Overfitting: Train **低** + Test **高** = High **Variance**
- λ **太大** → Underfitting, λ **太小** → Overfitting
