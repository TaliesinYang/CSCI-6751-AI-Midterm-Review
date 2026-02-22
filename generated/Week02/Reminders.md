# Week02 Reminders（AI）

## ✅ 本周必会（考试导向）

1. **Machine Learning 的正式定义（Tom Mitchell）**
   - **T (Task)**：具体要解决的问题（分类、回归等）
   - **P (Performance)**：性能度量指标
   - **E (Experience)**：从数据中学到的经验/模式

2. **ML vs Statistics 区别**
   - **Statistics**：发现并**解释**关系（如：吸烟是否导致癌症？）
   - **Machine Learning**：基于模式的**准确预测**（如：根据吸烟量预测寿命）
   - ML 是统计学的高层应用

3. **Supervised Learning 两种任务**
   - **Regression**：输出是连续数值（float/integer）
   - **Classification**：输出是离散类别（class A/B/C）
   - 关键区分：分数 0-100 → Regression；等级 A/B/C → Classification

4. **Data Encoding 三种方法（必须会区分）**
   - **Label Encoding** ❌：创建虚假数值关系，**只用于 Target**
   - **One-Hot Encoding** ✅：K 列 K 类别，无排序假设
   - **Dummy Encoding** ✅：K-1 列，避免多重共线性

---

## 🧠 复习清单（建议按顺序）

### ML 核心概念
- [ ] 用 T/P/E 三要素解释什么是 Machine Learning
- [ ] ML vs Statistics：用一个例子说明区别
- [ ] 用"输出类型"区分 Regression vs Classification

### Supervised Learning
- [ ] 能举出 3 个 Regression 例子（房价、温度、保费）
- [ ] 能举出 3 个 Classification 例子（肿瘤、风险评估、鸢尾花）
- [ ] 理解：数据有标签 → Supervised；无标签 → Unsupervised

### Unsupervised + Reinforcement Learning
- [ ] Clustering（聚类）的应用场景（社交网络、市场细分）
- [ ] PCA / Dimensionality Reduction 的目的（降低维度，保留信息）
- [ ] Reinforcement Learning 四要素：States, Actions, Rewards, Policy
- [ ] RL 与 Supervised Learning 的区别（奖惩 vs 标签）

### Feature Encoding（考试重点）
- [ ] 为什么 Label Encoding 用于特征会出问题？（创建虚假大小关系）
- [ ] One-Hot vs Dummy 的区别？（K 列 vs K-1 列）
- [ ] 什么时候用 One-Hot？（树模型、神经网络）
- [ ] 什么时候用 Dummy？（线性回归，避免多重共线性）

---

## 📝 课堂例子（可用于自测）

### 例子 1：Iris 鸢尾花分类
- **Features (X)**：Sepal length/width, Petal length/width
- **Target (Y)**：Species (Setosa, Versicolor, Virginica)
- **150 样本**，每种 50 个
- **问题**：这是 Regression 还是 Classification？为什么？
- **答案**：Classification，因为输出是 3 个离散类别

### 例子 2：车险保费预测
- **Features (X)**：年龄(20/25/30/40)、车型(Toyota/Honda/BMW)、驾龄、位置
- **Target (Y)**：保费金额 ($2,000/$2,500...)
- **问题**：车型特征需要怎么编码？
- **答案**：One-Hot 或 Dummy Encoding，不能用 Label Encoding

### 例子 3：考试成绩
- 分数 0-100 → **Regression**（连续数值）
- 等级 A/B/C/D/F → **Classification**（5 个离散类别）
- **同一个问题，定义不同就会变成不同任务**

---

## 📚 AI 层级图（需要理解）

```
AI (Artificial Intelligence)
├── Expert Systems
├── Fuzzy Computing
├── Robotics
├── Natural Language Processing
└── Machine Learning
    ├── Traditional ML (Linear Regression, Decision Trees, SVM, K-NN)
    └── Deep Learning (CNN, RNN, Transformer)
```

### Deep Learning 适用场景
- ✅ 图像（Face Recognition, Object Detection）
- ✅ 语音（Speech Recognition）
- ✅ 文本（NLP, Transformer）
- ❌ 表格数据 → 用传统 ML 更好

---

## 🔥 Neural Networks 工作方式（Face Recognition 例子）

### 层级特征提取
- **Layer 1（低级）**：检测边缘和线条
- **Layer 2（中级）**：组合成圆形、矩形等形状
- **Layer 3（高级）**：组合成眼、鼻、嘴等特征
- **Output**：分类识别（Person A / B / C）

### Image as Input
- 图像 = 像素矩阵
- 每个像素 = RGB 值 (0-255 per channel)
- 60×60 image = 60×60×3 array

---

## 📋 从 Week01-Week02 累积的必会内容

### ✅ Week01 回顾
- [ ] Turing Test 定义
- [ ] Expert Systems vs AI（规则 vs 学习）
- [ ] **Fuzzy Logic（✨Quiz 可能考）**
  - 三角隶属函数计算
  - Centroid 去模糊化
- [ ] Neural Networks 基本结构

### ✅ Week02 新增
- [ ] ML 三要素 (T/P/E)
- [ ] Supervised vs Unsupervised（看有无标签）
- [ ] Regression vs Classification（看输出类型）
- [ ] Feature Encoding（One-Hot vs Dummy）
- [ ] Deep Learning 适用场景

---

## 🎯 本周关键 Takeaway

1. **ML 定义**：T (Task) + P (Performance) + E (Experience)
2. **有标签 → Supervised**；**无标签 → Unsupervised**
3. **连续输出 → Regression**；**离散输出 → Classification**
4. **Feature Encoding 很重要**：Label Encoding ❌ for features；One-Hot/Dummy ✅
5. **Deep Learning**：用于图像/语音/文本；表格数据用传统 ML
6. **下周预告**：Linear Regression 详细讲解（公式推导、两种求解方法）
