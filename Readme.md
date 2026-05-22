# 📊 Fraud Detection Monitor

A **Streamlit dashboard** for predicting fraudulent transactions, monitoring model activity, and explaining predictions using **SHAP** — built as a companion to the [Fraud Detection ML System](https://github.com/Prakruthisb/financial-fraud-detection-streamlit).

---

## 🖥️ Live Demo

> 🔗 [streamlit-app-url] *(https://financial-fraud-detection-app-nbgymfqvphqknnlggptj9p.streamlit.app/)*

---

## 📌 Overview

This dashboard provides a complete interface for interacting with the fraud detection model — from entering a transaction and getting a prediction, to understanding *why* the model flagged it, to monitoring all past predictions over time.

---

## ✨ Features

### 🔍 Tab 1 — Prediction

* Input form accepting raw transaction details:
  * Step, Transaction Type, Amount
  * Sender/Receiver balances and IDs
* Runs the transaction through the trained **XGBoost pipeline** (`fraud_pipeline.pkl`)
* Applies a **0.9 probability threshold** for fraud classification
* Displays result as:
  * 🚨 `FRAUD` with confidence score
  * ✅ `LEGIT` with confidence score
* Every prediction is **automatically logged** for monitoring

### 🔍 SHAP Explanation (inside Prediction tab)

* After each prediction, the dashboard generates a **SHAP explanation** for that specific transaction
* Shows which input features pushed the model toward or away from fraud
* Also generates a **plain English explanation** — no ML knowledge needed to interpret results
* Makes the model transparent and auditable for analysts

### 📊 Tab 2 — Monitoring

* Displays a full monitoring dashboard of all logged predictions
* Tracks fraud vs. legitimate transaction volumes over time
* Helps identify patterns, model drift, or sudden spikes in flagged activity

---

## 🛠️ Tech Stack

| Component        | Technology              |
| ---------------- | ----------------------- |
| Dashboard UI     | Streamlit               |
| ML Pipeline      | XGBoost + scikit-learn  |
| Explainability   | SHAP                    |
| Prediction Log   | Custom logger module    |
| Data Handling    | Pandas                  |
| Visualizations   | Matplotlib              |

---

## 🗂️ Project Structure

```
fraud-detection-monitor/
│
├── app/
│   └── streamlit_app.py              # Main Streamlit app (2 tabs)
├── SHAP_explanation/
│   └── fraud_explanability.py      # SHAP + plain English explanation logic
├── fraud_pipeline.pkl              # Trained XGBoost pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Prakruthisb/financial-fraud-detection-streamlit
cd fraud-detection-monitor

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`.

---

## 🔍 How SHAP Explanation Works

For every prediction, the dashboard calls `explain_transaction_streamlit()` which:

1. Extracts the preprocessing + model steps from the pipeline
2. Computes SHAP values for the input transaction
3. Returns feature-level contributions (which features increased or decreased fraud probability)

Then `explain_in_plain_english()` converts those SHAP values into a human-readable summary — for example:

> *"This transaction was flagged primarily because the amount is very high relative to the sender's balance, and it was a TRANSFER type which is commonly associated with fraud."*

---

## 🔗 Related Repository

This dashboard connects to the core fraud detection model:

👉 [Fraud Detection ML System](https://github.com/Prakruthisb/financial-fraud-detection-ml) — XGBoost model trained on 6.3M transactions with FastAPI deployment

---

## 🔮 Future Improvements

* Email/SMS alerts when high-probability fraud is detected
* Integration with live transaction stream (e.g., Kafka)
* Role-based access for analysts vs. admins

---

⭐ If you find this useful, consider giving it a star!