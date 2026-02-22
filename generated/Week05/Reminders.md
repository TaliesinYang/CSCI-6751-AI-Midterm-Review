# Week05 Reminders — Model Evaluation Metrics

> Important points, exam preparation, and action items from Week05

---

## 🎯 Exam Focus Areas

### Must-Know Formulas

**Memorize these - will definitely appear on exam:**

1. **MAE**: `(1/n) × Σ|y - ŷ|`
2. **MSE**: `(1/n) × Σ(y - ŷ)²`
3. **RMSE**: `√MSE`
4. **R²**: `1 - (SSres / SStot)`
   - SSres = `Σ(y - ŷ)²`
   - SStot = `Σ(y - ȳ)²`
5. **Precision**: `TP / (TP + FP)`
6. **Recall**: `TP / (TP + FN)`
7. **F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)`

**Practice Tip**: Create flashcards for each formula and practice calculating them with sample data.

---

### Confusion Matrix - Know This Cold

```
                Predicted
              Pos    Neg
Actual  Pos   TP     FN
        Neg   FP     TN
```

**Exam Strategy**: 
- Given TP, FP, TN, FN → calculate Precision, Recall, F1
- Practice this calculation at least 10 times before exam

**Common mistakes to avoid**:
- ❌ Precision = TP / (TP + FN) ← **WRONG**
- ✅ Precision = TP / (TP + FP) ← **CORRECT**

---

### Key Conceptual Questions

**Be ready to explain:**

1. **Validation Set vs Test Set**
   - What's the difference?
   - Why can't we use test set during training?
   - When do we use each one?

2. **Precision vs Recall Tradeoff**
   - Why can't we have both at 100%?
   - When to prioritize Precision? (Give examples)
   - When to prioritize Recall? (Give examples)

3. **Why RMSE instead of MSE?**
   - Unit interpretation
   - Which is more interpretable for stakeholders?

4. **R² Interpretation**
   - What does R² = 0.85 mean in plain English?
   - What does R² = 0 mean?
   - Can R² be negative? (Yes! When model is worse than predicting mean)

---

## 🔥 Professor's Emphasis Points

### "Don't just look at training error!"

**Why**: Training error can be misleading. A model with perfect training error (0) might be overfitting.

**Exam Question Type**: 
> "Model A: Train error = 10, Test error = 200  
> Model B: Train error = 30, Test error = 30  
> Which is better and why?"

**Answer**: Model B, because it generalizes better (test error is stable)

---

### "Context matters for metrics"

**Why**: Professor emphasized repeatedly that no metric is universally "good" or "bad."

**Examples to remember**:
- MAE = 8.5 → Good or bad? **Depends on scale**
- Precision = 0.8 → Good or bad? **Depends on cost of FP**
- Recall = 0.6 → Good or bad? **Depends on cost of FN**

**Exam Tip**: If asked "Is this metric value good?", always mention **context** and **domain requirements**.

---

### "Recall is important in medical diagnosis"

**Why**: Missing a disease (FN) is more costly than false alarm (FP)

**Exam Application**:
- Medical diagnosis → prioritize **Recall**
- Spam filtering → prioritize **Precision**
- General purpose → use **F1 Score**

**Memorize these use cases** - they appear frequently in exam questions.

---

### "Harmonic mean penalizes extremes"

**Why F1 uses harmonic mean**:
- Arithmetic mean of 0.9 and 0.1 = 0.5 (misleading)
- Harmonic mean (F1) ≈ 0.18 (reflects poor performance)

**Exam Insight**: If Precision and Recall are very imbalanced, F1 will be low, showing that the model isn't truly balanced.

---

## ⚠️ Common Pitfalls to Avoid

### 1. Confusing Precision and Recall

**Wrong**:
- Precision = TP / (TP + FN) ❌
- Recall = TP / (TP + FP) ❌

**Correct**:
- Precision = TP / (TP + **FP**) ✅ (Of predictions, how many correct?)
- Recall = TP / (TP + **FN**) ✅ (Of actuals, how many found?)

**Memory Trick**: 
- Precision cares about **Predicted** Positives (TP + **FP**)
- Recall cares about **Real** Positives (TP + **FN**)

---

### 2. Forgetting Unit Differences

| Metric | Unit |
|--------|------|
| MAE | Same as y |
| MSE | y² (squared) |
| RMSE | Same as y |
| R² | Unitless (0-1) |

**Why it matters**: Exam might ask "Which metric has the same unit as the target variable?" → MAE and RMSE, **not MSE**

---

### 3. Misunderstanding Cross-Validation

**Wrong thinking**: "Cross-validation splits data into train and test"

**Correct understanding**: "Cross-validation splits **training data** into K folds to **validate** the model before final testing on test set"

**Key**: Cross-validation happens **before** final test set evaluation.

---

### 4. Type I vs Type II Error Confusion

| Error | Definition | Also Called | Example |
|-------|------------|-------------|---------|
| Type I | False Positive | False Alarm | Healthy person diagnosed as sick |
| Type II | False Negative | Missed Detection | Sick person diagnosed as healthy |

**Exam Tip**: If confused, remember FP = **F**alse **P**ositive = Type **I** (alphabetically first)

---

## 📊 Practice Problems to Work Through

### Problem 1: Regression Metrics
Given predictions and actuals, calculate MAE, MSE, RMSE, R²

**Sample Data**:
- y = [10, 20, 30, 40, 50]
- ŷ = [12, 18, 32, 38, 51]

**Practice calculating all 4 metrics by hand**

---

### Problem 2: Classification Metrics
Given confusion matrix, calculate Precision, Recall, F1

**Sample Matrix**:
- TP = 85, FP = 15
- FN = 20, TN = 80

**Calculate**:
1. Precision = ?
2. Recall = ?
3. F1 = ?
4. Which metric is higher and why?

---

### Problem 3: Model Comparison
Compare two models:
- Model A: Precision = 0.9, Recall = 0.5, F1 = ?
- Model B: Precision = 0.7, Recall = 0.7, F1 = ?

**Questions**:
1. Calculate F1 for both
2. Which is better for medical diagnosis?
3. Which is better for spam filtering?

---

## 📝 Assignment Reminders

### Check if Assignment Due Soon
- Review assignment requirements for Week05
- Likely to involve implementing/calculating metrics
- May require using scikit-learn's metrics module

### Common Assignment Tasks:
1. Split data into train/validation/test
2. Train models and calculate metrics
3. Compare models using multiple metrics
4. Implement K-fold cross-validation
5. Write justification for metric choice

---

## 🧪 Lab/Practical Tips

### Scikit-learn Functions to Know

```python
from sklearn.metrics import (
    mean_absolute_error,      # MAE
    mean_squared_error,       # MSE (pass squared=True)
    r2_score,                 # R²
    confusion_matrix,         # Confusion matrix
    precision_score,          # Precision
    recall_score,             # Recall
    f1_score,                 # F1
    classification_report     # All classification metrics
)
```

**Tip**: `classification_report()` gives you Precision, Recall, F1 all at once - very useful!

---

## 🔄 Cross-Validation Implementation

### Remember the pattern:
```python
from sklearn.model_selection import cross_val_score

# K-fold cross-validation (K=5)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Average across folds
mean_score = scores.mean()
```

**Exam Tip**: Know that `cv=5` means 5-fold cross-validation.

---

## 🎓 Study Strategy

### Before Exam:
1. ✅ Memorize all formulas (write them 10 times each)
2. ✅ Practice confusion matrix calculations (at least 5 problems)
3. ✅ Understand when to use which metric (make a decision tree)
4. ✅ Review Professor's examples from slides
5. ✅ Do practice problems with actual calculations

### During Exam:
1. ✅ Write down key formulas on scratch paper immediately
2. ✅ Draw confusion matrix structure if classification question
3. ✅ Check units (MSE vs RMSE/MAE)
4. ✅ Consider context before answering "Is this metric good?"

---

## 💡 Quick Reference Card

### Create a cheat sheet with:
- 7 key formulas (MAE, MSE, RMSE, R², Precision, Recall, F1)
- Confusion matrix layout
- When to use which metric (Medical→Recall, Spam→Precision)
- Type I = FP, Type II = FN
- RMSE and MAE have same unit as y; MSE is y²

**Time-saving tip**: Create this now, review it daily before exam.

---

## 🚀 Action Items

- [ ] **Today**: Review all formulas, create flashcards
- [ ] **This Week**: Complete practice problems (at least 10 calculations)
- [ ] **Before Next Class**: Implement metrics in Python for assignment
- [ ] **Before Exam**: Review confusion matrix calculations, know use cases by heart

---

## 📌 Final Reminder

**Professor's most important advice**:
> "Don't just look at one metric. Always consider the context and the cost of different types of errors."

**This philosophy will appear in exam questions** - be ready to justify metric choices based on domain/context!

---

**Good luck! 🍀**
