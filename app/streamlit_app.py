import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 🔥 NEW: Monitoring imports
from app.monitoring.dashboard import render_monitoring_tab
from app.monitoring.logger import log_prediction
from SHAP_explanation.fraud_explanability import explain_transaction_streamlit, explain_in_plain_english

# Load pipeline
pipeline = joblib.load("fraud_pipeline.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Transaction Fraud Detection System")

# 🔥 CREATE TABS
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Monitoring"])

# =============================================================================
# 🔍 TAB 1 — PREDICTION
# =============================================================================
with tab1:

    st.markdown("Enter transaction details to predict fraud probability.")

    # -----------------------------
    # Input Form
    # -----------------------------
    with st.form("fraud_form"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step", min_value=0, value=1)
            type_ = st.selectbox("Transaction Type", 
                                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
        
        with col2:
            oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
            oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=5000.0)
            nameOrig = st.text_input("Sender ID", "C123")
            nameDest = st.text_input("Receiver ID", "C456")

        submit = st.form_submit_button("Predict")

    # -----------------------------
    # Prediction
    # -----------------------------
    if submit:
        
        input_data = pd.DataFrame([{
            "step": step,
            "type": type_,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "oldbalanceDest": oldbalanceDest,
            "nameOrig": nameOrig,
            "nameDest": nameDest
        }])

        prob = pipeline.predict_proba(input_data)[:, 1][0]
        pred = prob > 0.9

        # 🔥 LOG PREDICTION (IMPORTANT)
        log_prediction(
            raw_transaction=input_data.iloc[0].to_dict(),
            fraud_probability=float(prob),
            predicted_fraud=bool(pred),
            threshold=0.9
        )

        st.subheader("Prediction Result")

        if pred:
            st.error(f"🚨 FRAUD (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ LEGIT (Confidence: {1 - prob:.2f})")

        # -----------------------------
        # SHAP Explanation
        # -----------------------------
        st.subheader("Model Explanation (SHAP)")

        try:
            df = input_data.copy()
            result = explain_transaction_streamlit(pipeline, df)
            st.write(result)
            st.subheader("\nPlain English explanation:")
            st.write(explain_in_plain_english(result))

        except Exception as e:
            import traceback
            st.error("SHAP explanation failed")
            st.code(traceback.format_exc())   # 🔥 shows FULL traceback in UI


# =============================================================================
# 📊 TAB 2 — MONITORING
# =============================================================================
with tab2:
    render_monitoring_tab()