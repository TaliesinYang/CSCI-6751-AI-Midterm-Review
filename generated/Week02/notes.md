# Week 02: Machine Learning Fundamentals

**Date**: Week 2
**Topic**: Machine Learning Basics
**Recording**: Recording_4 (English)

---

## 📚 Overview

This lecture covers the fundamental concepts of machine learning, including supervised, unsupervised, and reinforcement learning paradigms. The session explores the relationship between AI, machine learning, and deep learning, along with practical applications and data encoding techniques.

---

## 🎯 Key Concepts

### 1. Machine Learning Definition

**Machine Learning** is the study of algorithms that improve performance on a given task through experience:
- **T (Task)**: The specific problem to solve (e.g., classification, regression)
- **P (Performance)**: Measure of how well the task is performed
- **E (Experience)**: Data and patterns learned from past examples

**ML vs Statistics**:
- **Statistics**: Finding relationships and interpreting them (e.g., does smoking cause cancer?)
- **Machine Learning**: Accurate prediction based on patterns (e.g., predicting lifespan based on smoking habits)
- Machine learning is a higher-level application of statistics

---

### 2. Why Use Machine Learning?

Machine learning is valuable when:

1. **Human Expertise Doesn't Exist**
   - Example: Mars navigation - no human has experienced Mars terrain
   - Solution: Train models on Mars images to detect obstacles and safe paths

2. **Human Cannot Explain Expertise**
   - Example: Speech recognition - humans recognize words but can't explain frequency patterns
   - Solution: Machines can analyze frequency characteristics and learn patterns

3. **Customization Required**
   - Example: Medical diagnosis - different patients with similar symptoms may need different treatments
   - Solution: ML can customize predictions patient-by-patient based on DNA, genes, etc.

4. **Huge Amount of Data**
   - Example: Genetic analysis with millions of characteristics
   - Solution: ML models can process and find patterns in massive datasets

---

### 3. Machine Learning Categories

#### A. Supervised Learning

Data has **labels** (known target values)

**Types:**

**1. Regression**
- Target is a **continuous/float number**
- Examples:
  - House price prediction ($100,000, $250,000, etc.)
  - Arctic sea ice level prediction
  - Temperature forecasting
- Goal: Find a function that maps input features to continuous output
- Minimize error between predictions and ground truth

**2. Classification**
- Target is a **category/class**
- Examples:
  - Tumor classification: malignant vs benign
  - Customer risk: risky vs not risky
  - Iris flower species: setosa, versicolor, virginica
- Goal: Find decision boundaries to separate classes

**Key Distinction**:
- Exam score 0-100? → **Regression** (continuous values)
- Exam grade A/B/C/D/F? → **Classification** (5 discrete categories)

---

#### B. Unsupervised Learning

Data has **NO labels** - only features

**Types:**

**1. Clustering**
- Group similar data points together
- Examples:
  - Social network: group users by interests
  - Market segmentation: group customers by behavior
  - Astronomical data: group similar stars
- Similarity can be defined by: distance, interests, characteristics

**2. Dimensionality Reduction (PCA)**
- Reduce number of features while keeping important information
- Example: Dataset with 1,000 features → reduce to 20 important features
- Benefits: Reduce computational cost, remove noise

---

#### C. Reinforcement Learning

Learning from **rewards and penalties**

**Concept**:
- **Agent** performs actions in an **environment**
- Each action receives **reward** (positive) or **penalty** (negative)
- Goal: Learn optimal policy to maximize cumulative rewards
- Process: Trial and error → adjust behavior based on feedback

**Example**: Island survival
- Try touching objects → get burned (penalty) → avoid next time
- Try eating fruit → tastes good (reward) → seek again

**Applications**:
- Game playing (Chess, Go)
- Robotics
- Autonomous systems
- Path finding

**Components**:
- **States (S)**: Set of possible situations
- **Actions (A)**: Set of possible moves
- **Rewards (R)**: Feedback values
- **Policy**: Strategy for selecting optimal actions

---

#### D. Semi-Supervised Learning

- Mix of labeled and unlabeled data
- Example: 80% of records have labels, 20% don't
- Less common than supervised/unsupervised

---

### 4. AI Hierarchy

```
AI (Artificial Intelligence)
├── Expert Systems (old approach)
├── Fuzzy Computing
├── Robotics
├── Natural Language Processing
└── Machine Learning
    ├── Traditional ML
    │   ├── Linear Regression
    │   ├── Decision Trees
    │   ├── SVM
    │   └── Clustering
    └── Deep Learning
        ├── Neural Networks
        ├── CNNs (Images)
        ├── RNNs (Sequential data)
        └── Transformers (NLP)
```

**Deep Learning**:
- Subset of machine learning
- Best for: Images, speech, text
- Not ideal for: Tabular data (use traditional ML instead)
- Inspired by brain neural networks

---

### 5. Neural Networks - How They Work

**Example: Face Recognition**

**Layer 1 (Low-level)**:
- Detect edges and lines
- Basic shapes

**Layer 2 (Mid-level)**:
- Combine edges into objects
- Create circles, rectangles from edges

**Layer 3 (High-level)**:
- Combine objects into meaningful features
- Eyes, nose, mouth positions

**Output Layer**:
- Classify: Is it Person A, B, or C?
- Or: Is it smiling/crying/neutral?

**Image as Input**:
- Image = Matrix of pixels
- Each pixel = RGB values (0-255 for each channel)
- 60x60 image = 60x60x3 array
- Neural network processes this array through multiple layers

**Applications**:
- Face detection and recognition
- Object detection (cars, chairs, animals)
- Speech recognition
- Autonomous vehicles

---

### 6. Data Types and Encoding

#### Feature Types

**1. Numeric**
- **Real values**: 2.5, 100.75, -3.14
- **Integer values**: 1, 2, 50, 100

**2. Categorical**
- Classes/Categories: Red, Blue, Green
- Car brands: Toyota, Honda, BMW
- Cities: Toronto, Vancouver, Calgary

**3. Binary**
- Two classes only: Yes/No, 0/1, True/False

---

#### Encoding Categorical Data

**Problem**: Most ML algorithms need numeric input, not text categories

**Methods**:

**1. Label Encoding** ❌ (Avoid for features)
```
Toyota → 1
Honda → 2
BMW → 3
Mercedes → 4
```
**Issue**: Model interprets as numeric values (Mercedes = 4x Toyota)
**Creates false relationships** and incorrect weights
**✓ OK for target variable only**

**2. One-Hot Encoding** ✅
```
Color: Red, Green, Blue

Red   → [1, 0, 0]
Green → [0, 1, 0]
Blue  → [0, 0, 1]
```
- Create K columns for K categories
- Each record has 1 in its category column, 0 elsewhere
- No false ordering or relationships
- **Use for**: Tree-based models, neural networks

**3. Dummy Encoding** ✅
```
Color: Red, Green, Blue

Red   → [1, 0]
Green → [0, 1]
Blue  → [0, 0]  (implied by others being 0)
```
- Create K-1 columns for K categories
- Remove one category (it's implied)
- Avoids multicollinearity issue
- **Use for**: Linear regression, logistic regression

**When to use which:**
- **One-hot**: Tree-based models (more complex models)
- **Dummy**: Linear models (to avoid multicollinearity)
- Dummy reduces number of features: K-1 instead of K

---

### 7. Real-World Examples

#### Example 1: Iris Flower Classification
**Features** (X):
- Sepal length
- Sepal width
- Petal length
- Petal width

**Target** (Y):
- Species: Setosa, Versicolor, Virginica

**Dataset**: 150 samples (50 of each species)
**Task**: Classify new flower based on measurements

#### Example 2: Auto Insurance Premium Prediction
**Features** (X):
- Age: 20, 25, 30, 40 (numeric)
- Car model: Toyota, Honda, BMW (categorical - needs encoding)
- Experience: 0, 2, 5, 8, 10 years (numeric)
- Location: Risky/Quiet area (categorical)

**Target** (Y):
- Premium amount: $2,000, $2,500, etc. (continuous)

**Task**: Regression problem to predict premium

#### Example 3: House Price Prediction
**Features** (X):
- Square footage (numeric)
- Number of bedrooms (integer)
- Age of house (numeric)
- Location (categorical)

**Target** (Y):
- Price (continuous - regression)

---

## 💡 Important Takeaways

1. **Supervised vs Unsupervised**:
   - Supervised: Have labels → prediction
   - Unsupervised: No labels → find structure/patterns

2. **Regression vs Classification**:
   - Regression: Predict continuous values (prices, temperature)
   - Classification: Predict categories (classes, types)

3. **Feature Engineering**:
   - Proper encoding of categorical variables is crucial
   - Wrong encoding (label encoding for features) creates false relationships
   - Choose encoding method based on model type

4. **Deep Learning Context**:
   - Use for images, speech, text
   - Traditional ML often better for tabular data
   - Deep learning requires more data and computational resources

5. **Model Training**:
   - Learn patterns from training data
   - Goal: Minimize error/distance between predictions and ground truth
   - Model should generalize to new, unseen data

---

## 🔑 Key Terms

- **Features/Predictors**: Input variables (X)
- **Target/Label**: Output variable (Y)
- **Training Data**: Dataset used to learn patterns
- **Ground Truth**: Actual correct values
- **Error/Loss**: Distance between prediction and ground truth
- **Model/Function**: Mathematical representation learned from data
- **Multicollinearity**: When features are dependent on each other

---

## 📝 Questions for Review

1. What are the three components (T, P, E) of machine learning?
2. When should you use regression vs classification?
3. Why is label encoding problematic for input features?
4. What's the difference between one-hot and dummy encoding?
5. When is deep learning preferred over traditional ML?
6. What is the main difference between supervised and unsupervised learning?
7. How does reinforcement learning differ from supervised learning?

---

## 🎓 Next Steps

- Practice with Iris dataset for classification
- Implement one-hot encoding in Python/Pandas
- Understand linear regression and decision trees
- Explore neural network architectures for image data
