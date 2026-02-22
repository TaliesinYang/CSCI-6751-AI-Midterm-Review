# Week 02 Machine Learning Fundamentals

## ML 定义

### Tom Mitchell 定义
- T (Task): 具体问题
- P (Performance): 性能度量
- E (Experience): 从数据学习

### ML vs Statistics
- Statistics: 发现和解释关系
- ML: 基于模式的准确预测
- ML 是统计学的高层应用

## 为什么用 ML？

### 四大场景
- 人类专业知识不存在（火星导航）
- 人类无法解释（语音识别）
- 需要定制化（医疗诊断）
- 数据量巨大（基因分析）

## ML 分类

### Supervised Learning（监督学习）
#### Regression（回归）
- 连续/浮点输出
- 房价、温度、海冰水平
- 最小化预测误差
#### Classification（分类）
- 类别输出
- 肿瘤分类、客户风险、鸢尾花
- 找决策边界

### Unsupervised Learning（无监督学习）
#### Clustering（聚类）
- 相似数据分组
- 社交网络、市场细分
#### Dimensionality Reduction（降维）
- PCA 降维
- 1000 特征 → 20 特征

### Reinforcement Learning（强化学习）
- Agent + Environment
- Reward / Penalty
- States, Actions, Policy
- 应用：游戏、机器人

### Semi-Supervised
- 混合标签和非标签数据
- 80% 有标签 + 20% 无标签

## AI 层级

### 结构
- AI → Expert Systems, Fuzzy, Robotics, NLP, ML
- ML → Traditional ML + Deep Learning
- Deep Learning → CNN, RNN, Transformer

### Deep Learning 适用场景
- 图像、语音、文本 → 用 DL
- 表格数据 → 用传统 ML

## Neural Networks 工作方式

### Face Recognition 例子
- Layer 1: 检测边缘
- Layer 2: 组合成形状
- Layer 3: 组合成特征（眼、鼻）
- Output: 分类识别

### Image as Input
- 图像 = 像素矩阵
- RGB: 0-255 per channel
- 60×60 图像 = 60×60×3 array

## Data Types & Encoding

### Feature Types
- Numeric: 实数、整数
- Categorical: 类别
- Binary: 二元

### Encoding Methods
#### Label Encoding ❌
- 直接数字编码
- 创建错误关系
- 只用于 Target
#### One-Hot Encoding ✅
- K 列 K 类别
- 无虚假排序
- 用于：树模型、神经网络
#### Dummy Encoding ✅
- K-1 列 K 类别
- 避免多重共线性
- 用于：线性回归

## 实际例子

### Iris 鸢尾花分类
- 4 特征 → 3 类别
- 150 样本

### 车险保费预测
- 年龄、车型、经验、位置
- 回归问题

### 房价预测
- 面积、卧室数、房龄
- 回归问题

## 课程资源

### 笔记
- [[Week02/notes|完整笔记]]
- [[Week02/vocabulary|术语表]]
- [[Week02/qa-summary|QA总结]]
