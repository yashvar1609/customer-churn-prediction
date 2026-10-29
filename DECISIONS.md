# Project Decisions Log

This log documents the key design decisions, engineering choices, and modeling results for the Telco Customer Churn Prediction project.

---

## Phase 1: Data Analysis & Cleaning

### Decision: Impute missing TotalCharges with 0.0
* **Reason:** All 11 missing `TotalCharges` values belong exclusively to customers with `tenure == 0`. Since these are brand-new customers who have not completed their first billing cycle, their total accumulated charge is mathematically and logically \$0.00. Imputing as 0.0 preserves these rows for training without injecting artificial charge values.
* **Alternative Considered:** Dropping the rows (rejected because it removes training data unnecessarily) or imputing with mean/median (rejected because it falsely inflates spend for brand-new customers).

### Decision: Drop customerID
* **Reason:** `customerID` is a unique identifier with a cardinality of 7,043. It carries zero predictive value and keeping it would add unnecessary high-cardinality noise, risking overfitting.

### Decision: Perform duplicate checks post-customerID drop
* **Reason:** Checking for duplicates before dropping `customerID` would yield 0 duplicates because the ID is unique for every row. Checking after dropping the ID ensures we search for duplicate observations in the actual feature space. (No duplicates were found).

### Decision: Retain statistical outliers
* **Reason:** The IQR method flagged zero outliers in `tenure` and `MonthlyCharges`. Even if high values were flagged (such as in `TotalCharges`), they represent genuine high-value, long-term customer accounts rather than data entry anomalies. Removing them would discard critical signals about our highest-paying customers.

---

## Phase 3 & 4: Feature Engineering & Selection

### Decision: Drop `TotalCharges` and `avg_monthly_spend` only for Logistic Regression
* **Reason:** Multi-collinearity analysis revealed that `TotalCharges` is highly correlated with `tenure` (0.83) and `MonthlyCharges` (0.65). Furthermore, our derived feature `avg_monthly_spend` is extremely redundant with `MonthlyCharges` (0.957 correlation). These relationships result in extremely high VIF scores (~40-50+). Multi-collinearity causes severe instability in linear model coefficients. We will drop these two columns at training time specifically for Logistic Regression.
* **Reason for keeping them for Tree Models:** Tree ensembles (Random Forest, XGBoost) perform splits based on feature partition impurity (like Gini or Entropy) and are natively robust to multicollinearity. We keep them to allow tree models to capture any subtle interaction terms.

### Decision: Keep `num_services`
* **Reason:** The correlation between `num_services` and `MonthlyCharges` is 0.720. While strong, it represents behavioral engagement (how many products they use) rather than a pure price metric. It is not a mathematical redundancy and can be used by tree models to distinguish customers.

### Decision: Fit `StandardScaler` strictly on training data
* **Reason:** To prevent data leakage. Fitting the scaler on the full dataset leaks information about the test set's mean and standard deviation into the training set, which artificially inflates evaluation metrics and violates true cross-validation.

---

## Phase 5, 6, & 7: Modeling, Tuning, & Evaluation

### Decision: Class Weighting chosen over SMOTE and Baseline
* **Reason:** Across all model families, baseline models without imbalance handling suffered from very poor Recall (~0.38 - 0.50), failing to catch the majority of churners. Applying **Class Weighting** (balanced class weights in sklearn / `scale_pos_weight` in XGBoost) provided the best overall performance, significantly boosting Recall (to 0.75 - 0.79) while keeping Precision above 0.51. In comparison, SMOTE was effective but yielded slightly lower F1 scores and ROC-AUC for our top-performing tree ensembles.

### Decision: Selected Tuned XGBoost (with Class Weighting) as the final model
* **Reason:** Tuned XGBoost achieved the highest **ROC-AUC of 0.8427** on the untouched test set. It offers an excellent balance of **Recall (0.7930)** and **Precision (0.5122)**, giving an F1-score of **0.6224**. 
* **Business Justification:** A high recall of 79.3% means we capture nearly 80% of actual churners, allowing the business to proactively run retention campaigns. The precision of 51.2% indicates that about half of our promotional offers will go to customers who would have stayed anyway. Since the cost of an unnecessary promotion (False Positive, ~$10-20 discount) is much lower than the cost of losing a customer entirely (False Negative, losing customer lifetime value of hundreds of dollars), this trade-off is highly profitable for a telecom business.
* **Tuning Method & Best Parameters:** RandomizedSearchCV with 5-fold cross-validation optimizing for `roc_auc`.
  * `subsample`: 0.6
  * `n_estimators`: 100
  * `max_depth`: 3
  * `learning_rate`: 0.05
  * `colsample_bytree`: 0.8
  * `scale_pos_weight` (imbalance handling ratio): 2.7758
* **Model Serialization:** Persisted to [models/best_churn_model.pkl](models/best_churn_model.pkl).

### Final Model Comparison (Ranked by ROC-AUC)

| Rank | Model | Strategy | ROC-AUC | Recall | F1 | Accuracy | Precision | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Tuned XGBoost** | **Class Weight** | **0.8427** | **0.7930** | **0.6224** | **0.7452** | **0.5122** | **2.5283** |
| 2 | Logistic Regression | Baseline | 0.8399 | 0.5000 | 0.5750 | 0.8043 | 0.6764 | 0.1534 |
| 3 | Random Forest | Class Weight | 0.8394 | 0.7527 | 0.6292 | 0.7651 | 0.5405 | 1.1312 |
| 4 | Logistic Regression | Class Weight | 0.8392 | 0.7742 | 0.6115 | 0.7395 | 0.5053 | 0.0761 |
| 5 | Tuned Random Forest | Class Weight | 0.8387 | 0.7849 | 0.6226 | 0.7480 | 0.5159 | 1.8797 |
| 6 | Random Forest | SMOTE | 0.8380 | 0.7043 | 0.6187 | 0.7701 | 0.5516 | 1.4786 |
| 7 | Random Forest | Baseline | 0.8357 | 0.4919 | 0.5631 | 0.7979 | 0.6583 | 1.0473 |
| 8 | Logistic Regression | SMOTE | 0.8347 | 0.6720 | 0.6127 | 0.7751 | 0.5631 | 0.1524 |
| 9 | Decision Tree | Class Weight | 0.8263 | 0.7796 | 0.6042 | 0.7295 | 0.4932 | 0.0491 |
| 10 | XGBoost | Class Weight | 0.8227 | 0.6425 | 0.5865 | 0.7601 | 0.5395 | 0.2460 |
| 11 | XGBoost | Baseline | 0.8207 | 0.4704 | 0.5335 | 0.7822 | 0.6162 | 0.3412 |
| 12 | Decision Tree | SMOTE | 0.8186 | 0.7500 | 0.6078 | 0.7438 | 0.5110 | 0.0592 |
| 13 | Decision Tree | Baseline | 0.8183 | 0.3844 | 0.4881 | 0.7865 | 0.6682 | 0.0614 |
| 14 | XGBoost | SMOTE | 0.8067 | 0.5484 | 0.5543 | 0.7665 | 0.5604 | 0.3582 |
