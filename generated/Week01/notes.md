# CSCI 6751 - Lecture 01: Introduction to Artificial Intelligence

**Date**: January 14, 2026
**Topic**: Introduction to AI, Machine Learning Fundamentals, Python Basics

---

## Course Overview

### Learning Objectives
- Understand core AI principles
- Learn different types of machine learning (supervised vs unsupervised)
- Understand neural networks and their role in modern AI
- Natural language processing fundamentals
- Reinforcement learning strategies

### Prerequisites
- Programming experience (Java or similar)
- Python knowledge helpful (will review basics)

### Assessment Structure
| Component | Weight |
|-----------|--------|
| Participation | 2% |
| Weekly Assignments | 10% |
| Midterm Exam | 50% |
| Final Project (Groups of 2-3) | 38% |

**Important Dates**:
- Midterm: February 24, 2026
- Final Exam: March 8, 2026
- Quizzes: Unannounced (be prepared every session)

### Tools & Software
- Python with libraries: scikit-learn, Keras, PyTorch
- OpenAI APIs for NLP
- IDEs: PyCharm, Jupyter Notebook, VS Code, Google Colab

---

## What is Artificial Intelligence?

### Definition
AI is a subset of computer science focused on **creating machines that are intelligent** - able to perform tasks similar to humans through:
1. **Learning** - Acquiring knowledge from data
2. **Reasoning** - Making inferences from learned knowledge
3. **Problem Solving** - Handling complex tasks
4. **Perception** - Sensing environment (for some AI systems)
5. **Language Understanding** - NLP capabilities (for some AI systems)

### The Turing Test (1950)
**Alan Turing's definition**: If a human cannot reliably distinguish between responses from a human and a machine, then the machine is said to have intelligence.

- Human interrogator communicates with two hidden entities (one human, one AI)
- If interrogator cannot tell which is which, the AI passes the test

---

## History of AI

### Timeline Overview

| Era | Key Developments |
|-----|------------------|
| 1950 | Turing Test proposed |
| 1956 | **Dartmouth Conference** - Birth of AI as academic field |
| 1980s | Expert Systems era |
| 1980s-90s | Neural Networks invented |
| 1990s | Machine Learning rise |
| 2000s | Big Data era begins |
| 2012 | Deep Learning breakthrough |
| 2020 | GPT-3 released |
| 2022-23 | GPT-4, ChatGPT |
| 2024 | Current AI boom |

### AI Winters and Booms
The field experienced three major "booms" with periods of reduced interest ("winters") in between:
1. **First AI Boom** (1950s-60s): Initial excitement, Turing Test
2. **Second AI Boom** (1980s): Expert Systems
3. **Third AI Boom** (2010s-present): Deep Learning, LLMs

---

## Expert Systems (1980s)

### Architecture
```
[User Interface] → [Inference Engine] → [Knowledge Base]
                         ↓                    ↑
                   [Explanation System]  [Human Expert]
                         ↓
                      [Output]
```

### Key Components
1. **Knowledge Base**: Facts and rules provided by human expert
2. **Inference Engine**: Matches inputs against rules
3. **Explanation System**: Explains why a decision was made
4. **Knowledge Acquisition System**: Interface to add rules

### Example: MYCIN
- Medical diagnosis system
- Doctor provides rules: "If symptoms X, Y, Z → prescribe medication A"
- Patient inputs symptoms → system matches rules → provides diagnosis

### Limitation
- **No learning** - Just if-then-else rules
- Not a true AI model, but important first real-world application

### Modern Applications
- Customer support systems
- Financial advising
- Engineering diagnostics

---

## Fuzzy Logic

### Concept
Extends classical true/false logic to **degrees of truth** (0 to 1).

### Example: Temperature Classification
| Temperature | "Warm" Membership |
|-------------|-------------------|
| 0°C | 0.0 |
| 10°C | 0.5 |
| 20°C | 0.8 |
| 25°C | 1.0 |

Instead of answering "Is 20°C warm?" with yes/no, we say: "20°C is warm with membership degree 0.8"

### Applications
- Rice cookers
- ABS brakes
- Fan speed controllers
- Any system requiring smooth control based on imprecise inputs

---

## Neural Networks

### Biological Inspiration
- Human brain has millions of **neurons** connected to each other
- Neurons pass messages to perform tasks
- Sense input → process through layers → generate output

### Artificial Neural Networks (ANN)
```
[Input Layer] → [Hidden Layer(s)] → [Output Layer]
```

- **Fully Connected (Dense) Networks**: Every node connects to all nodes in next layer
- **Deep Neural Networks**: Many hidden layers (sometimes thousands)

### Hardware Requirements
- Complex neural networks require **GPUs** (Graphics Processing Units)
- **NVIDIA** became dominant due to AI/deep learning demand
- Software: PyTorch, TensorFlow, Keras

### Key Figures
- **Geoffrey Hinton** - "Godfather of AI", Nobel Prize 2024 (Physics)
- **Yann LeCun** - Introduced CNNs (Convolutional Neural Networks)

---

## Machine Learning Fundamentals

### Definition
Taking data and **finding patterns** within it to make predictions on new, unseen data.

### The ML Pipeline
1. **Define the problem/goal**
2. **Gather training data** (labeled or unlabeled)
3. **Clean/preprocess data**
4. **Choose model type**
5. **Train the model**
6. **Evaluate accuracy**
7. **Deploy for predictions**

### Types of Machine Learning

#### 1. Supervised Learning
- Training data is **labeled** (has known outcomes)
- Examples:
  - Spam email classification (spam/not spam)
  - Bank risk prediction (risky/not risky)
  - House price prediction

#### 2. Unsupervised Learning
- Data is **not labeled**
- Goal: Find patterns, groupings
- Example: Clustering customers by behavior

#### 3. Reinforcement Learning
- Agent learns through trial and error
- Example: Pathfinding, game playing

### Classification vs Regression

| Type | Output | Example |
|------|--------|---------|
| **Classification** | Discrete categories (0, 1, 2...) | Cat vs Dog |
| **Regression** | Continuous values | House price |

### Linear Regression
- Fit a line to predict continuous values
- Will have some **error** (loss) if relationship is non-linear
- Goal: Minimize total error across all data points

---

## Image Classification Example (Cat vs Dog)

### Process
1. **Collect images** of cats and dogs
2. **Label each image** (cat=0, dog=1)
3. **Preprocess**:
   - Resize to uniform dimensions (e.g., 64x64)
   - Convert to numerical values (RGB: 0-255)
   - Result: 64 x 64 x 3 tensor per image
4. **Data augmentation** if needed:
   - Rotate images
   - Flip horizontally
   - Create variations
5. **Train model** to recognize patterns
6. **Test on unseen images**

---

## Deep Learning Applications

- **Face Recognition**: Unlock phones, security
- **Self-Driving Cars**: Object detection (Waymo, Tesla)
- **Robots**: Industrial automation
- **Generative AI**: Image generation (DALL-E, Midjourney)
- **Language Models**: ChatGPT, conversational AI

---

## Python Fundamentals

### IDEs for Python
- PyCharm
- Jupyter Notebook
- VS Code
- Google Colab (cloud-based, free)

### Basic Operations
```python
# Variables
x = 2
y = 3

# Arithmetic
x + y      # 5 (addition)
x - y      # -1 (subtraction)
x / y      # 0.66 (division)
x // y     # 0 (integer division)
x % y      # 2 (modulo/remainder)
x ** y     # 8 (power)

# Strings
s = "hello"
s.upper()  # "HELLO"
s.lower()  # "hello"
"hello" + " world"  # "hello world"
```

### Lists
```python
numbers = [-2, -5, 10, 20, 25, -6, 0]

numbers[0]      # -2 (first element)
numbers[2]      # 10 (third element)
numbers[1:4]    # [-5, 10, 20] (slice, end excluded)
numbers[2:]     # [10, 20, 25, -6, 0] (from index 2 to end)
numbers.append(100)  # Add element
```

### NumPy Arrays
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

arr + 10       # [11, 12, 13, 14, 15]
arr * 2        # [2, 4, 6, 8, 10]
arr ** 2       # [1, 4, 9, 16, 25]

np.sum(arr)    # 15
np.mean(arr)   # 3.0
np.std(arr)    # standard deviation

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
```

### Loops and Conditionals
```python
# For loop
for num in numbers:
    print(num + 5)

# While loop
while condition:
    # do something

# If-else
if x % 2 == 0:
    print("even")
else:
    print("odd")

# Function definition
def my_function(param):
    return param * 2
```

---

## Key Takeaways

1. **AI** is about creating intelligent machines that can learn, reason, and solve problems
2. **Machine Learning** is a subset of AI focused on learning from data
3. **Deep Learning** is a subset of ML using neural networks with many layers
4. **Supervised learning** uses labeled data; **unsupervised** finds patterns in unlabeled data
5. **Neural networks** are inspired by biological neurons and require powerful hardware (GPUs)
6. Modern AI breakthroughs (GPT, image generation) are built on **transformer architectures** and deep learning
7. Python with NumPy, scikit-learn, and PyTorch are essential tools for AI development

---

## Next Session Preview
- More details on machine learning algorithms
- Clustering techniques (K-means, DBSCAN, Hierarchical)
- Regression in depth
