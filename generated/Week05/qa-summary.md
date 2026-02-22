# Week05 Q&A Summary

> Important questions and discussions from Week05 lecture on model evaluation metrics.

---

## 1. Validation Set vs Test Set

### Q: What's the difference between validation set and test set?

**A**: Validation set is **part of the training data** that we split off to evaluate the model during training. Test set is **completely separate** and should only be used once for final evaluation.

**Why Important**: Many students confuse these two concepts. Understanding this prevents "data leakage" where test data influences training.

**Key Point**: 
- Validation → used multiple times during training to tune hyperparameters
- Test → used **only once** for final evaluation

---

## 2. When is MAE = 8.5 considered "good"?

### Q: Professor calculated MAE = 8.5 for a model. How do we know if 8.5 is good or bad?

**A**: "It depends on the context. Sometimes 8 is good, but in some systems, 8 is a very big number. You need to look at the scale of your target variable."

**Why Important**: Error metrics are **meaningless without context**. Need to compare:
1. Scale of target variable
2. Business requirements
3. Baseline model performance
4. Domain constraints

**Example**:
- Predicting house prices in millions: MAE = 8.5 (万元) is excellent
- Predicting temperature in Celsius: MAE = 8.5°C is terrible

---

## 3. Why RMSE instead of MSE?

### Q: Why do we need RMSE when we already have MSE?

**A**: "When you want to compare model one with model two, RMSE is more interpretable because it's in the same unit as your target variable. MSE is in squared units."

**Why Important**: Interpretability matters for:
- Communicating results to non-technical stakeholders
- Understanding prediction error magnitude
- Setting realistic performance targets

**Example**:
- Target: House price (万元)
- MSE = 100 (万元²) ← Hard to interpret
- RMSE = 10 (万元) ← "Average error is 10万元"

---

## 4. MSE vs MAE - Which is better?

### Q: When should we use MSE vs MAE?

**A**: Professor explained through example:
- Model A: MAE = 2, MSE = 10
- Model B: MAE = 2, MSE = 4
- **Conclusion**: Model B is more stable (smaller variance)

**Why Important**: 
- **MAE**: Treats all errors equally (robust to outliers)
- **MSE**: Penalizes large errors more (sensitive to outliers)

**Use Cases**:
- Use **MAE** when outliers should not dominate (e.g., housing prices with rare mansions)
- Use **MSE/RMSE** when large errors are particularly costly (e.g., medical dosage predictions)

---

## 5. Understanding Underfitting

### Q: What does "underfit means that it's useless" mean?

**A**: An underfit model is too simple to capture the pattern in data. It performs poorly on both training and test data, so it has no practical value.

**Why Important**: Students often focus only on overfitting and forget that underfitting is equally problematic.

**Signs of Underfitting**:
- High training error AND high test error
- Model predictions close to mean/median (not learning patterns)
- Increasing model complexity improves both train and test performance

---

## 6. Precision vs Recall Tradeoff

### Q: Can we have both high Precision and high Recall?

**A (Implicit)**: There's usually a tradeoff. Professor emphasized different contexts:
- **Medical diagnosis**: Recall is more important (don't miss diseases)
- **Spam filtering**: Precision is more important (don't block legitimate emails)

**Why Important**: Understanding this tradeoff is crucial for:
- Setting appropriate decision thresholds
- Choosing the right metric for evaluation
- Aligning model performance with business goals

**Key Insight**: F1 Score tries to balance both, but in practice, you often need to prioritize one over the other based on cost of errors.

---

## 7. What is F1 Score trying to achieve?

### Q: Why use harmonic mean for F1 instead of arithmetic mean?

**A (Demonstrated through example)**:
- Arithmetic mean of 0.9 and 0.1: (0.9 + 0.1) / 2 = 0.5
- Harmonic mean (F1): 2 × (0.9 × 0.1) / (0.9 + 0.1) ≈ 0.18

**Why Important**: Harmonic mean **penalizes extreme imbalance**. If either Precision or Recall is very low, F1 will be low, forcing the model to balance both.

**Use Case**: When you need **both** Precision and Recall to be reasonably high, not just their average.

---

## 8. True Positive vs False Positive

### Q: How to remember the difference between TP, FP, TN, FN?

**A (Professor's explanation)**: 
- **First word** (True/False) = Is the prediction correct?
- **Second word** (Positive/Negative) = What did the model predict?

**Why Important**: Confusion matrix is fundamental to classification metrics. Easy to mix up under exam pressure.

**Memory Trick**:
- **True** Positive → Prediction is **correct**, predicted Positive
- **False** Positive → Prediction is **wrong**, predicted Positive (but actually Negative)

---

## 9. Type I vs Type II Error

### Q: What's the difference between Type I and Type II errors?

**A**: 
- **Type I Error** = False Positive = False alarm (saying there's a problem when there isn't)
- **Type II Error** = False Negative = Missed detection (saying there's no problem when there is)

**Why Important**: Different domains have different tolerance for these errors:
- **Medical**: Type II Error is worse (missing a disease)
- **Legal**: Type I Error is worse (false conviction)

**Example from lecture**:
- Predicting "risky patient" but they're actually healthy → **Type I Error**
- Predicting "non-risky patient" but they actually have disease → **Type II Error**

---

## 10. Cross-Validation Implementation

### Q: How exactly does K-fold cross-validation work?

**A (Professor's example with K=8)**:
```
Iteration 1: Train on chunks [1,2,3,4,5,6,7], Validate on [8]
Iteration 2: Train on chunks [1,2,3,4,5,6,8], Validate on [7]
Iteration 3: Train on chunks [1,2,3,4,5,7,8], Validate on [6]
...
Iteration 8: Train on chunks [2,3,4,5,6,7,8], Validate on [1]
```

**Why Important**: Cross-validation provides more robust evaluation than single train/test split, especially with small datasets.

**Key Benefit**: Every sample gets to be in the validation set once, reducing variance in performance estimation.

---

## 11. R² Interpretation

### Q: What does R² = 0.85 mean?

**A (Implicit from formula)**: The model explains 85% of the variance in the target variable. The remaining 15% is due to factors not captured by the model.

**Why Important**: R² gives an intuitive percentage of "goodness of fit."

**Range**:
- R² = 1.0 → Perfect fit
- R² = 0.0 → Model is as good as predicting the mean
- R² < 0.0 → Model is worse than predicting the mean

---

## 12. When to use which metric?

### Q: With so many metrics, which should we report?

**A (Professor's advice)**: "Don't just look at one metric. Always consider the context and the cost of different types of errors."

**Why Important**: Real-world model evaluation requires multiple metrics:

**For Regression**:
- Report: MAE, RMSE, R²
- MAE for interpretability
- RMSE for penalizing large errors
- R² for overall fit quality

**For Classification**:
- Report: Precision, Recall, F1, Confusion Matrix
- Choose primary metric based on business needs
- Always show confusion matrix for transparency

---

## 13. Calculating F1 from Precision and Recall

### Q: Student asked about F1 calculation with Precision=0.5, Recall=0.02

**A (Professor demonstrated)**:
- F1 = 2 × (0.5 × 0.02) / (0.5 + 0.02)
- F1 = 2 × 0.01 / 0.52
- F1 ≈ 0.038

**Why Important**: Shows how F1 is very low when one metric is extremely low, even if the other is decent. This demonstrates the "harmonic mean" property.

---

## Key Takeaways for Exam

1. **Understand context**: No metric is "always the best"
2. **Know formulas**: Especially Precision, Recall, F1, MAE, MSE, RMSE, R²
3. **Confusion matrix**: Be able to calculate all metrics from it
4. **Tradeoffs**: Precision vs Recall, Bias vs Variance
5. **Units matter**: MAE and RMSE have same units as target; MSE is squared
6. **Cross-validation**: Know how K-fold works and why it's useful

---

**Note**: These Q&As are extracted from the most educationally valuable discussions. Routine clarifications are omitted.
