# Week05 Vocabulary — Model Evaluation Metrics

> Key terms and definitions from Week05 lectures on regression and classification metrics.

---

## Model Evaluation Concepts

**Validation Set（验证集）**  
A subset of training data used to evaluate the model during training and tune hyperparameters.  
从训练数据中划分出来的子集，用于在训练过程中评估模型并调整超参数。

**Test Set（测试集）**  
Independent dataset used only once for final model evaluation, never seen during training.  
独立的数据集，仅用于最终模型评估一次，训练时从未见过。

**Generalization（泛化能力）**  
Model's ability to perform well on unseen data (test set), not just training data.  
模型在未见过的数据（测试集）上的表现能力，而不仅仅是训练数据。

**Overfitting（过拟合）**  
Model memorizes training data including noise, resulting in low train error but high test error.  
模型记住了训练数据包括噪声，导致训练误差低但测试误差高。

**Underfitting（欠拟合）**  
Model is too simple to capture the underlying pattern, resulting in high error on both train and test sets.  
模型过于简单，无法捕捉潜在模式，导致训练和测试集上的误差都很高。

---

## Regression Metrics

**MAE (Mean Absolute Error)（平均绝对误差）**  
Average of absolute differences between predictions and actual values: `MAE = (1/n) × Σ|y - ŷ|`.  
预测值与实际值之间绝对差异的平均值。

**MSE (Mean Squared Error)（均方误差）**  
Average of squared differences between predictions and actual values: `MSE = (1/n) × Σ(y - ŷ)²`.  
预测值与实际值之间平方差异的平均值。

**RMSE (Root Mean Squared Error)（均方根误差）**  
Square root of MSE: `RMSE = √MSE`. Same unit as target variable.  
MSE 的平方根。与目标变量单位相同。

**SSres (Sum of Squared Residuals)（残差平方和）**  
Sum of squared differences between actual and predicted values: `Σ(y - ŷ)²`.  
实际值与预测值之间平方差异的总和。

**SStot (Total Sum of Squares)（总平方和）**  
Sum of squared differences between actual values and their mean: `Σ(y - ȳ)²`.  
实际值与其均值之间平方差异的总和。

**R² (R-Squared / Coefficient of Determination)（决定系数）**  
Proportion of variance in target variable explained by model: `R² = 1 - (SSres / SStot)`.  
模型解释目标变量方差的比例。Range: 0 to 1 (higher is better).

**Residual（残差）**  
Difference between actual value and predicted value: `residual = y - ŷ`.  
实际值与预测值之间的差异。

---

## Classification Metrics

**Confusion Matrix（混淆矩阵）**  
Table showing counts of TP, FP, TN, FN for binary classification evaluation.  
显示二分类评估中 TP、FP、TN、FN 计数的表格。

**True Positive (TP)（真阳性）**  
Model correctly predicts positive class (prediction = 1, actual = 1).  
模型正确预测正类（预测 = 1，实际 = 1）。

**False Positive (FP)（假阳性）**  
Model incorrectly predicts positive when actual is negative (prediction = 1, actual = 0). Also called **Type I Error**.  
模型错误地预测为正类但实际为负类（预测 = 1，实际 = 0）。也称为**第一类错误**。

**True Negative (TN)（真阴性）**  
Model correctly predicts negative class (prediction = 0, actual = 0).  
模型正确预测负类（预测 = 0，实际 = 0）。

**False Negative (FN)（假阴性）**  
Model incorrectly predicts negative when actual is positive (prediction = 0, actual = 1). Also called **Type II Error**.  
模型错误地预测为负类但实际为正类（预测 = 0，实际 = 1）。也称为**第二类错误**。

**Precision（精确率 / 查准率）**  
Proportion of predicted positives that are actually positive: `Precision = TP / (TP + FP)`.  
在所有预测为正的样本中，真正为正的比例。

**Recall（召回率 / 查全率 / Sensitivity）**  
Proportion of actual positives that are correctly predicted: `Recall = TP / (TP + FN)`.  
在所有真实为正的样本中，被正确预测出来的比例。

**Sensitivity（灵敏度）**  
Same as Recall. Measures model's ability to detect positive cases.  
与召回率相同。衡量模型检测正类的能力。

**Specificity（特异度）**  
Proportion of actual negatives correctly predicted: `Specificity = TN / (TN + FP)`.  
在所有真实为负的样本中，被正确预测出来的比例。

**F1 Score（F1 分数）**  
Harmonic mean of Precision and Recall: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`.  
精确率和召回率的调和平均数。Balances precision and recall.

**Harmonic Mean（调和平均数）**  
Average that gives more weight to smaller values: `HM = n / (Σ(1/x_i))`.  
对较小值给予更多权重的平均数。Used in F1 score to penalize extreme imbalance.

**Type I Error（第一类错误）**  
False Positive - rejecting true null hypothesis (false alarm).  
假阳性 - 拒绝真实的零假设（误报）。

**Type II Error（第二类错误）**  
False Negative - failing to reject false null hypothesis (missing detection).  
假阴性 - 未能拒绝错误的零假设（漏报）。

---

## Cross-Validation

**Cross-Validation（交叉验证）**  
Technique to assess model performance by training and evaluating on different data splits.  
通过在不同数据划分上训练和评估来评估模型性能的技术。

**K-Fold Cross-Validation（K 折交叉验证）**  
Split data into K chunks; train on K-1 chunks and validate on 1 chunk, repeat K times.  
将数据分为 K 份；在 K-1 份上训练，在 1 份上验证，重复 K 次。

**Fold（折）**  
One partition of the dataset in K-fold cross-validation.  
K 折交叉验证中数据集的一个分区。

**Leave-One-Out Cross-Validation (LOOCV)（留一法交叉验证）**  
K-fold where K = n (number of samples). Each sample is validation set once.  
K 折交叉验证中 K = n（样本数）。每个样本都作为验证集一次。

---

## Model Comparison

**Baseline Model（基线模型）**  
Simple model used as reference point for comparison (e.g., predicting mean value).  
用作比较参考点的简单模型（如预测均值）。

**Hyperparameter（超参数）**  
Parameter set before training (e.g., learning rate, K in K-fold, polynomial degree).  
在训练前设置的参数（如学习率、K 折中的 K、多项式阶数）。

**Model Complexity（模型复杂度）**  
Measure of model's capacity to fit data (e.g., polynomial degree, number of parameters).  
模型拟合数据能力的度量（如多项式阶数、参数数量）。

---

## Application Context

**Medical Diagnosis（医学诊断）**  
Classification task where FN (missing disease) is more costly than FP (false alarm).  
分类任务，其中 FN（漏诊）比 FP（误诊）代价更高。Prioritize high Recall.

**Spam Filtering（垃圾邮件过滤）**  
Classification task where FP (blocking legitimate email) is more costly than FN.  
分类任务，其中 FP（拦截正常邮件）比 FN 代价更高。Prioritize high Precision.

**Risk Assessment（风险评估）**  
Predicting whether entity (patient, customer, transaction) is risky or not.  
预测实体（患者、客户、交易）是否有风险。

---

## Statistical Terms

**Variance（方差）**  
Measure of spread in data: how far values are from the mean.  
数据离散程度的度量：值与均值的距离。

**Bias（偏差）**  
Difference between model's expected prediction and true value.  
模型的期望预测值与真实值之间的差异。

**Bias-Variance Tradeoff（偏差-方差权衡）**  
Tradeoff between model's bias (underfitting) and variance (overfitting).  
模型偏差（欠拟合）和方差（过拟合）之间的权衡。

---

## Example Contexts from Lecture

**Risky vs Non-Risky（有风险 vs 无风险）**  
Binary classification: predicting whether patient/transaction is risky (positive) or not (negative).  
二分类：预测患者/交易是否有风险（正类）。

**Spam vs Non-Spam（垃圾邮件 vs 正常邮件）**  
Email classification: predicting whether email is spam (positive) or legitimate (negative).  
邮件分类：预测邮件是垃圾邮件（正类）还是正常邮件（负类）。

---

**Note**: Terms are ordered by category and importance for exam preparation.
