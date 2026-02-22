# Week01 Reminders（AI）

## ⚠️ 课程基本信息（必须知道）

### 📝 评分结构

| Component | Weight |
|-----------|--------|
| Participation | 2% |
| Weekly Assignments | 10% |
| Midterm Exam | **50%** |
| Final Project (2-3人) | 38% |

### 📅 重要日期

| 事项 | 日期 | 备注 |
|------|------|------|
| **Midterm** | February 24, 2026 | 混合格式（MCQ + 短答） |
| **Final Exam** | March 8, 2026 | 混合格式 |
| **Quizzes** | 不定期 | 每节课都要准备 |

### 📌 教授提醒
- Quiz **不会提前通知**（但有时会给 hint）
- Quiz 不是 MCQ，是 1-2 道题，15 分钟
- Midterm 和 Final 有 MCQ + 短答
- 上课前**阅读讲义**（提前 1-2 天上传）
- 邮件要**简短清楚**（2-3 句话）

---

## ✅ 本周必会（考试导向）

1. **AI 的定义与五大能力**
   - Learning, Reasoning, Problem Solving, Perception, Language Understanding
   - AI 是计算机科学的子领域，创造能执行类人任务的智能机器

2. **Turing Test 定义（必须能解释）**
   - 如果人类无法区分 AI 与人类的回答 → AI 通过测试
   - 1950 年 Alan Turing 提出

3. **AI 历史三次浪潮**
   - 1950s-60s: 第一次 AI 热潮（Turing Test）
   - 1980s: 第二次 AI 热潮（Expert Systems）
   - 2010s-now: 第三次 AI 热潮（Deep Learning, LLMs）
   - **1956 Dartmouth Conference**：AI 作为学术领域诞生

4. **Expert Systems 架构**
   - Knowledge Base + Inference Engine + Explanation System
   - 基于 IF-THEN 规则，**没有学习能力**
   - 例子：MYCIN（医疗诊断系统）

---

## 🧠 Fuzzy Logic（模糊逻辑）— ✨Quiz 可能考

### 核心概念
- 扩展经典 True/False 到 **0-1 之间的隶属度**
- Membership degree（隶属度）：0.0 到 1.0

### 必须会的计算

**三角隶属函数**：给定 (a, b, c) 和输入 x，计算 μ(x)

**去模糊化（Defuzzification）**：
- **Centroid method**: `Output = Σ(μᵢ × valueᵢ) / Σμᵢ`

### 应用例子
- 电饭煲、ABS 刹车、风扇速度控制器
- 任何需要基于模糊输入平滑控制的系统

---

## 🧠 Neural Networks（基础概念）

### 结构
```
[Input Layer] → [Hidden Layer(s)] → [Output Layer]
```

- Fully Connected（全连接）：每个节点连接到下一层所有节点
- Deep Neural Networks：多个隐藏层
- 需要 **GPU**（NVIDIA 主导）

### 关键人物
- **Geoffrey Hinton**: "Godfather of AI"，2024 Nobel Prize (Physics)
- **Yann LeCun**: CNN (Convolutional Neural Networks) 发明者

---

## 🧠 Machine Learning 基础

### ML 三大类型
- [ ] **Supervised Learning（监督学习）**：有标签数据
- [ ] **Unsupervised Learning（无监督学习）**：无标签，找模式
- [ ] **Reinforcement Learning（强化学习）**：奖惩反馈

### Classification vs Regression（关键区分）
- [ ] **Classification**：输出是离散类别（Cat vs Dog, 0/1）
- [ ] **Regression**：输出是连续数值（房价, 温度）

### ML Pipeline
1. 定义问题 → 2. 收集数据 → 3. 预处理 → 4. 选模型 → 5. 训练 → 6. 评估 → 7. 部署

---

## 📋 复习清单

### ✅ 概念类（必须能口头解释）
- [ ] 什么是 AI？与 Computer Science 的关系？
- [ ] Turing Test 是什么？怎么判断机器有智能？
- [ ] Expert Systems 的架构和局限性
- [ ] Fuzzy Logic 与经典逻辑的区别
- [ ] Neural Networks 的基本结构（Input → Hidden → Output）
- [ ] Supervised vs Unsupervised 的区别
- [ ] Classification vs Regression 的区别

### ✅ 计算类（可能出题）
- [ ] Fuzzy Logic 隶属度计算（三角函数）
- [ ] Fuzzy Logic 去模糊化（Centroid method）
- [ ] 基本 Python 操作（NumPy 数组、列表切片）

---

## 📚 从 Week01 需要带到后续课程的知识

### 为 Week02 准备
- ML 三大类型的区别
- Supervised Learning 的定义和例子
- Classification vs Regression 区分

### 为 Week03 准备
- Linear Regression 基本概念
- Loss function / error 的直觉理解

---

## 🎯 本周关键 Takeaway

1. **AI** 是关于创造能学习、推理、解决问题的智能机器
2. **Machine Learning** 是 AI 的子集，专注于从数据中学习
3. **Deep Learning** 是 ML 的子集，使用多层神经网络
4. **Supervised vs Unsupervised**：有无标签是关键区别
5. **Fuzzy Logic**：从 True/False 扩展到 0-1 连续值
6. **Expert Systems**：基于规则，没有学习能力
7. Python 是本课程的主要编程工具
