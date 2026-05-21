# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview

This project focuses on building a **fraud detection system** using machine learning on a highly imbalanced dataset of **6.3 million transactions**. The goal is to identify fraudulent transactions while minimizing the risk of missing actual fraud cases.

---

## Objective

* Detect fraudulent transactions with **high recall**
* Handle **class imbalance effectively**
* Build a model that is **realistic and production-ready**

---

## Dataset

* Total records: **6.3 million**
* Fraud cases: **~0.13% (highly imbalanced)**

---

## Key Steps

### 🔹 1. Data Preprocessing

* Handled missing values
* Converted categorical variables using one-hot encoding
* Performed train-test split with stratification

---

### 🔹 2. Feature Engineering

Created meaningful features such as:

* `amount_to_balance_ratio`
* `hourly_velocity`
* `amount_deviation`
* `is_night`, `high_amount`

⚠️ Removed leakage-prone features to ensure model generalization.

---

### 🔹 3. Model Selection

Tested multiple models:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost ✅ (Final Model)

---

### 🔹 4. Handling Imbalance

* Used `scale_pos_weight` in XGBoost
* Focused on **Recall as primary metric**

---

### 🔹 5. Threshold Tuning

Adjusted probability threshold to **0.9** to balance:

* High recall (catch frauds)
* Acceptable precision

---

## 📈 Final Results

### 🔥 Classification Report

| Metric            | Value      |
| ----------------- | ---------- |
| Recall (Fraud)    | **~0.997** |
| Precision (Fraud) | **~0.52**  |
| F1 Score          | **~0.69**  |

---

### 🔍 Confusion Matrix

* True Positives: **1639**
* False Negatives: **4** ✅ (very low)
* False Positives: **1483**

---

### Key Insight

> The model successfully captures almost all fraud cases while maintaining a manageable number of false positives.

---

## Feature Importance Insights

Top features:

* `amount_to_balance_ratio` (strongest signal)
* `hourly_velocity` (burst fraud detection)
* Transaction types (`TRANSFER`, `CASH_OUT`)
* Time-based features (`hour`, `day`)

---

## Business Trade-off

* ✅ High Recall → Minimal financial loss
* ⚠️ Moderate Precision → Some false alerts (acceptable in banking)

---

## Model Saving

```python
joblib.dump(model, "fraud_model.pkl")
```

---

## Future Improvements

* Deploy using FastAPI
* Real-time fraud detection system
* Improve precision using advanced features

---

## Conclusion

This project demonstrates:

* Handling large-scale imbalanced data
* Avoiding data leakage
* Making business-driven ML decisions

---

⭐ If you like this project, consider giving it a star!
